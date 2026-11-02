#!/usr/bin/env python3
"""Exp3-B1: GT-score MBC bootstrap, encoder recovery, then low-LR joint tuning."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
import numpy as np
import torch
import yaml

from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.training.model import SourceFirstDynamicModel
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, fallback_geometry, geometry_from_manifest
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.exp2c4a_gt_score_mbc_only import _render_with_gt_score
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import EPSILON, _ENCODER_PREFIXES, _build_intervention_model, _full_image_uv, _git_sha, _gt_respiratory_scores, _respiratory_only_motion, _score_values, _set_seed, _validate_observations
from experiments.c4_diagnostics.exp2c4h_oracle_field_fullbatch_reconstruction import _fixed_pixel_grid, _observation_audit
from experiments.c4_diagnostics.exp3a_joint_fullbatch_reconstruction_recovery import _aggregate_dvf, _dynamic_dvf_values, _metrics, _prediction_loss_terms
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom

_LEVEL0 = "respiratory_mbc.levels.0."; _ALLOWED = (*_ENCODER_PREFIXES, _LEVEL0)

def _prefixes(phase: str) -> tuple[str, ...]:
    return (_LEVEL0,) if phase == "A" else _ENCODER_PREFIXES if phase == "B" else _ALLOWED if phase == "C" else (_ for _ in ()).throw(ValueError("phase must be A, B, or C"))

def _configure_phase(model, phase: str, *, film_lr: float, respiratory_mbc_lr: float) -> torch.optim.Adam:
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    encoder = [p for n, p in model.named_parameters() if n.startswith(_ENCODER_PREFIXES) and phase in {"B", "C"}]
    mbc = [p for n, p in model.named_parameters() if n.startswith(_LEVEL0) and phase in {"A", "C"}]
    for parameter in encoder + mbc: parameter.requires_grad_(True)
    model.respiratory_mbc.set_active_levels(1)
    groups = ([{"params": encoder, "lr": film_lr}] if encoder else []) + ([{"params": mbc, "lr": respiratory_mbc_lr}] if mbc else [])
    optimizer = torch.optim.Adam(groups); owned = {id(p) for group in optimizer.param_groups for p in group["params"]}; expected = {id(p) for n, p in model.named_parameters() if n.startswith(_prefixes(phase))}
    if owned != expected or any(p.requires_grad != (id(p) in expected) for p in model.parameters()): raise RuntimeError(f"Exp3-B1 phase {phase} ownership audit failed")
    return optimizer

def _phase_loss(model, observation, phase: str, motion, uv):
    target = observation.image[0, uv[:, 1].long(), uv[:, 0].long()]
    if phase == "A":
        return (_render_with_gt_score(model, observation, uv, motion) - target).square().mean()
    return _prediction_loss_terms(model, observation, uv)["data_mse"]

def _accumulate_full_batch(model, observations: list, *, phase: str, motion, pixel_samples: int, accumulation_chunk_size: int) -> dict[str, Any]:
    total = observations[0].image.new_zeros(()); ids: list[int] = []
    for start in range(0, len(observations), accumulation_chunk_size):
        losses = []
        for observation in observations[start:start + accumulation_chunk_size]:
            losses.append(_phase_loss(model, observation, phase, motion, _fixed_pixel_grid(observation, count=pixel_samples))); ids.append(int(observation.dynamic_frame_id)); total += losses[-1].detach()
        (torch.stack(losses).sum() / len(observations)).backward()
    return {"full_batch_reconstruction_mse": total / len(observations), "observation_ids": ids}

@torch.inference_mode()
def _phase_a_metrics(model, groups, motion, grid_shape):
    grid, _ = SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, grid_shape); rows = {}
    for location, sequence in groups.items():
        full_p=[]; full_g=[]; center_p=[]; center_g=[]; scores=[]
        for observation in sequence:
            stamp = grid.new_tensor([observation.timestamp_s]); score = _gt_respiratory_scores(motion, stamp); field = ScoreWeightedMBCField(model.respiratory_mbc, score); uv = _full_image_uv(observation); centers=SourceFirstDynamicModel._pixel_world(observation, uv)
            full_p.append(field(grid)[0]); full_g.append(motion.respiratory_dvf_mm(grid, stamp)[0]); center_p.append(field(centers[None])[0]); center_g.append(motion.respiratory_dvf_mm(centers[None], stamp)[0]); scores.append(score[0])
        value=torch.stack(scores); rows[location]={**_score_values(value,value), **{f"full_fov_{k}":v for k,v in _dynamic_dvf_values(torch.stack(full_p),torch.stack(full_g)).items()}, **{f"pixel_center_{k}":v for k,v in _dynamic_dvf_values(torch.stack(center_p),torch.stack(center_g)).items()}}
    return {"per_location": rows, "aggregate": aggregate_location_metrics(rows), "overall_score_metrics": _score_values(torch.cat([torch.tensor([[0.]])]), torch.cat([torch.tensor([[0.]])]))}

def _escape(metric): return metric["full_fov_dynamic_cosine"] >= .90 and .5 <= metric["full_fov_dynamic_rms_ratio"] <= 1.5 and metric["full_fov_dynamic_normalized_epe"] <= .75
def _strict(metric): return metric["full_fov_dynamic_cosine"] >= .95 and .8 <= metric["full_fov_dynamic_rms_ratio"] <= 1.2 and metric["full_fov_dynamic_normalized_epe"] <= .50
def _classify_stages(a, b, c):
    causal = "mbc_bootstrap_failure" if not _escape(a) else "encoder_given_learned_mbc_failure" if not _escape(b) else "joint_unfreeze_regression" if not _escape(c) else "staged_bootstrap_strict_recovery" if _strict(c) else "staged_bootstrap_coldstart_escape"
    return {"causal_classification":causal,"strict_recovery":_strict(c),"phase_a_escape":_escape(a),"phase_a_strict":_strict(a),"phase_b_escape":_escape(b),"phase_b_strict":_strict(b),"phase_c_escape":_escape(c),"phase_c_strict":_strict(c)}

def _load_checkpoint(path: Path): return torch.load(path, map_location="cpu", weights_only=True)
def _restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
def _save(path: Path, **state):
    rng={"torch":torch.get_rng_state()};
    if torch.cuda.is_available(): rng["cuda"]=torch.cuda.get_rng_state_all()
    torch.save({**state,"rng_state":rng,"git_sha":_git_sha()},path)
def _changed(before, model, allowed):
    changed=[n for n,p in model.named_parameters() if not torch.equal(before[n],p.detach().cpu())]; return {"changed":changed,"unexpected":sorted(set(changed)-set(allowed))}
def _snapshot_changed(before, after, allowed):
    changed=[name for name in before if not torch.equal(before[name],after[name])]
    return {"changed":changed,"unexpected":sorted(set(changed)-set(allowed))}
def _norm(model,prefix,grad=False):
    values=[(p.grad if grad else p.detach()).reshape(-1) for n,p in model.named_parameters() if n.startswith(prefix) and (not grad or p.grad is not None)]; return float(torch.cat(values).norm()) if values else 0.
def _assert_expected_gradients(model, phase):
    return all(any(n.startswith(prefix) and p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm()) > 0 for n,p in model.named_parameters()) for prefix in _prefixes(phase))
def _thresholds_update(values, metric, step):
    values = dict(values); c,r,n=metric["full_fov_dynamic_cosine"],metric["full_fov_dynamic_rms_ratio"],metric["full_fov_dynamic_normalized_epe"]
    for name,ok in {"first_full_fov_cosine_ge_0p80_step":c>=.8,"first_full_fov_cosine_ge_0p90_step":c>=.9,"first_full_fov_cosine_ge_0p95_step":c>=.95,"first_full_fov_rms_ratio_in_0p5_1p5_step":.5<=r<=1.5,"first_full_fov_rms_ratio_in_0p8_1p2_step":.8<=r<=1.2,"first_full_fov_nepe_le_0p75_step":n<=.75,"first_full_fov_nepe_le_0p5_step":n<=.5,"first_coldstart_escape_step":_escape(metric),"first_strict_recovery_step":_strict(metric)}.items():
        if ok and values.get(name) is None: values[name]=step
    return values

def run(args):
    device=torch.device(args.device); resume=getattr(args,"resume_from",None); schedule={"A":args.phase_a_steps,"B":args.phase_b_steps,"C":args.phase_c_steps}; total=sum(schedule.values())
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None or (args.locations_per_view,args.frames_per_location,args.pixel_mode,args.pixel_samples)!=(3,50,"fixed_grid",256)): raise ValueError("formal Exp3-B1 requires prescribed geometry and fixed 256-pixel grid")
    if resume is None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite: raise FileExistsError("output directory is non-empty")
        args.output_dir.mkdir(parents=True,exist_ok=True)
    _set_seed(args.seed); config=yaml.safe_load(args.source_config.read_text()); synthetic=yaml.safe_load(args.experiment_config.read_text())["synthetic"]; domain=_load_domain(args.canonical_domain); motion=_respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=synthetic["respiratory_frequency_hz"],cardiac_frequency_hz=synthetic["cardiac_frequency_hz"],respiratory_amplitude_mm=synthetic["respiratory_amplitude_mm"],cardiac_amplitude_mm=synthetic["cardiac_amplitude_mm"]))
    shape,psf=((8,8),1) if args.smoke else (tuple(synthetic["image_shape"]),synthetic["psf_samples"]); frames=geometry_from_manifest(args.manifest,args.qc_table,locations_per_view=args.locations_per_view,frames_per_location=args.frames_per_location,device=device) if args.manifest else fallback_geometry(locations_per_view=args.locations_per_view,frames_per_location=args.frames_per_location,image_shape=shape,motion=motion,device=device); observations=_build_observations(frames,image_shape=shape,phantom=AnalyticCanonicalPhantom(domain).to(device),motion=motion,psf_samples=psf,seed=args.seed); groups=_validate_observations(observations,locations_per_view=args.locations_per_view,frames_per_location=args.frames_per_location,smoke=args.smoke); audit=_observation_audit(observations); uv=_fixed_pixel_grid(observations[0],count=args.pixel_samples); pixel={"pixel_mode":"fixed_grid","requested_pixel_samples":args.pixel_samples,"actual_pixel_count":len(uv),"fixed_uv":uv.long().cpu().tolist()}
    model=_build_intervention_model(config,domain,n_dynamic_frames=len(observations),seed=args.seed,device=device); protocol={"objective":"reconstruction_image_data_mse_only","exact_full_batch":True,"active_respiratory_levels":1,"phase_schedule":schedule}; low=5e-5; checkpoint=_load_checkpoint(resume) if resume else None; start=int(checkpoint["global_step"]) if checkpoint else 0
    if checkpoint and (checkpoint["observation_audit"]!=audit or checkpoint["pixel_protocol"]!=pixel or checkpoint["protocol"]!=protocol): raise ValueError("incompatible Exp3-B1 resume checkpoint")
    def phase_for(step): return "A" if step<=schedule["A"] else "B" if step<=schedule["A"]+schedule["B"] else "C"
    next_phase=phase_for(start+1) if start<total else "C"; optimizer=_configure_phase(model,next_phase,film_lr=5e-4 if next_phase!="C" else low,respiratory_mbc_lr=5e-4 if next_phase=="A" else low)
    if checkpoint:
        model.load_state_dict(checkpoint["model"]); _restore_rng(checkpoint["rng_state"])
        if checkpoint["phase"]==next_phase: optimizer.load_state_dict(checkpoint["optimizer"])
        curve=checkpoint["curve"]; initial=checkpoint["initial_metrics"]; phase_metrics=checkpoint["phase_metrics"]; snapshots=checkpoint["snapshots"]; thresholds=checkpoint.get("thresholds", {})
    else:
        initial,_=_metrics(model,groups,motion,grid_shape=tuple(args.evaluation_grid_shape)); curve=[]; phase_metrics={}; snapshots={"start":parameter_snapshot(model)}; thresholds={}
    calls={"A":0,"B":0,"C":0}; renderer=[0]; mbc=[0]; card=[0]; unc=[0]; hooks=[model.psf.register_forward_hook(lambda *_:renderer.__setitem__(0,renderer[0]+1)),model.respiratory_mbc.register_forward_hook(lambda *_:mbc.__setitem__(0,mbc[0]+1)),model.cardiac_mbc.register_forward_hook(lambda *_:card.__setitem__(0,card[0]+1)),model.uncertainty.register_forward_hook(lambda *_:unc.__setitem__(0,unc[0]+1))]
    try:
        for step in range(start+1,total+1):
            phase=phase_for(step)
            if step==1 or phase!=phase_for(step-1): optimizer=_configure_phase(model,phase,film_lr=5e-4 if phase!="C" else low,respiratory_mbc_lr=5e-4 if phase=="A" else low)
            optimizer.zero_grad(set_to_none=True); before_mbc=mbc[0]; batch=_accumulate_full_batch(model,observations,phase=phase,motion=motion if phase=="A" else None,pixel_samples=args.pixel_samples,accumulation_chunk_size=args.accumulation_chunk_size); calls[phase]+=mbc[0]-before_mbc
            frozen_grads = [n for n,p in model.named_parameters() if not n.startswith(_prefixes(phase)) and p.grad is not None]
            if batch["observation_ids"] != [o.dynamic_frame_id for o in observations] or frozen_grads: raise RuntimeError("full-batch or frozen-gradient audit failed: " + ", ".join(frozen_grads))
            if not _assert_expected_gradients(model, phase): raise RuntimeError("missing expected phase gradient")
            optimizer.step(); boundary = step in {schedule["A"],schedule["A"]+schedule["B"],total}
            if boundary or step % args.eval_every == 0:
                metrics=_phase_a_metrics(model,groups,motion,tuple(args.evaluation_grid_shape)) if phase=="A" else _metrics(model,groups,motion,grid_shape=tuple(args.evaluation_grid_shape))[0]; dvf=_aggregate_dvf(metrics); thresholds=_thresholds_update(thresholds,dvf,step); curve.append({"global_step":step,"phase":phase,"phase_step":step-(0 if phase=="A" else schedule["A"] if phase=="B" else schedule["A"]+schedule["B"]),"full_batch_reconstruction_mse":float(batch["full_batch_reconstruction_mse"]),**dvf,"encoder_parameter_norm":_norm(model,"film_encoder."),"respiratory_mbc_level0_parameter_norm":_norm(model,_LEVEL0),"encoder_grad_norm":_norm(model,"film_encoder.",True),"respiratory_mbc_level0_grad_norm":_norm(model,_LEVEL0,True),"film_lr":optimizer.param_groups[0]["lr"] if phase!="A" else 0.,"respiratory_mbc_lr":optimizer.param_groups[-1]["lr"] if phase!="B" else 0.})
            if boundary:
                key={schedule["A"]:"A",schedule["A"]+schedule["B"]:"B",total:"C"}[step]; phase_metrics[key]=metrics; snapshots[key]=parameter_snapshot(model); _save(args.output_dir/f"checkpoint_phase_{key.lower()}_end.pt",model=model.state_dict(),optimizer=optimizer.state_dict(),global_step=step,phase=phase,protocol=protocol,pixel_protocol=pixel,observation_audit=audit,curve=curve,initial_metrics=initial,phase_metrics=phase_metrics,snapshots=snapshots,thresholds=thresholds)
            if step%args.checkpoint_every==0:
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),global_step=step,phase=phase,protocol=protocol,pixel_protocol=pixel,observation_audit=audit,curve=curve,initial_metrics=initial,phase_metrics=phase_metrics,snapshots=snapshots,thresholds=thresholds); _save(args.output_dir/f"checkpoint_{step}.pt",**state); _save(args.output_dir/"checkpoint_last.pt",**state)
    finally:
        for hook in hooks: hook.remove()
    required_phases = {phase_for(step) for step in range(start + 1, total + 1)}
    if renderer[0] <= 0 or any(calls[key] <= 0 for key in required_phases) or card[0] != 0 or unc[0] != 0: raise RuntimeError("Exp3-B1 runtime module-call audit failed")
    stage_audit={"A":_snapshot_changed(snapshots["start"],snapshots["A"],[n for n in snapshots["start"] if n.startswith(_LEVEL0)]),"B":_snapshot_changed(snapshots["A"],snapshots["B"],[n for n in snapshots["A"] if n.startswith(_ENCODER_PREFIXES)]),"C":_snapshot_changed(snapshots["B"],snapshots["C"],[n for n in snapshots["B"] if n.startswith(_ALLOWED)])}
    if any(value["unexpected"] for value in stage_audit.values()): raise RuntimeError("Exp3-B1 phase parameter-change audit failed")
    a,b,c=(_aggregate_dvf(phase_metrics[key]) for key in "ABC"); classification=_classify_stages(a,b,c); final_snapshot=snapshots["C"]; frozen={"film_encoder_card_unchanged":all(torch.equal(snapshots["start"][n],final_snapshot[n]) for n in snapshots["start"] if n.startswith("film_encoder.card.")),"canonical_unchanged":all(torch.equal(snapshots["start"][n],final_snapshot[n]) for n in snapshots["start"] if n.startswith("canonical.")),"respiratory_mbc_higher_levels_unchanged":all(torch.equal(snapshots["start"][n],final_snapshot[n]) for n in snapshots["start"] if n.startswith("respiratory_mbc.levels.1.") or n.startswith("respiratory_mbc.levels.2.")),"cardiac_mbc_unchanged":all(torch.equal(snapshots["start"][n],final_snapshot[n]) for n in snapshots["start"] if n.startswith("cardiac_mbc.")),"uncertainty_unchanged":all(torch.equal(snapshots["start"][n],final_snapshot[n]) for n in snapshots["start"] if n.startswith("uncertainty."))}
    with (args.output_dir/"training_curve.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)
    for key,name in (("A","metrics_phase_a_end.json"),("B","metrics_phase_b_end.json"),("C","metrics_final.json")): (args.output_dir/name).write_text(json.dumps(phase_metrics[key],indent=2)+"\n")
    (args.output_dir/"metrics_initial.json").write_text(json.dumps(initial,indent=2)+"\n"); (args.output_dir/"stage_transition_audit.json").write_text(json.dumps(stage_audit,indent=2)+"\n"); factor={"encoder_parameter_norm":curve[-1]["encoder_parameter_norm"],"respiratory_mbc_level0_parameter_norm":curve[-1]["respiratory_mbc_level0_parameter_norm"],"basis_field_metric_supported":False}; (args.output_dir/"factor_diagnostics.json").write_text(json.dumps(factor,indent=2)+"\n"); np.savez_compressed(args.output_dir/"recovery_arrays.npz",initial_dvf_metrics=np.asarray(list(_aggregate_dvf(initial).values())),phase_a_dvf_metrics=np.asarray(list(a.values())),phase_b_dvf_metrics=np.asarray(list(b.values())),phase_c_dvf_metrics=np.asarray(list(c.values())))
    _save(args.output_dir/"checkpoint_last.pt",model=model.state_dict(),optimizer=optimizer.state_dict(),global_step=total,phase="C",protocol=protocol,pixel_protocol=pixel,observation_audit=audit,curve=curve,initial_metrics=initial,phase_metrics=phase_metrics,snapshots=snapshots,thresholds=thresholds)
    summary={"experiment":"Exp3-B1 staged bootstrap recovery","scientific_question":"Can staged MBC then encoder training escape joint cold-start?","git_sha":_git_sha(),"seed":args.seed,"observation_audit":audit,"pixel_protocol":pixel,"scientific_protocol":protocol,"phase_schedule":{"phase_a_steps":schedule["A"],"phase_b_steps":schedule["B"],"phase_c_steps":schedule["C"]},"phase_a":{"oracle_use":"GT respiratory score as coefficient only","trainable_modules":[_LEVEL0],"lr":5e-4,"final_dvf":a,"escape":_escape(a),"strict":_strict(a)},"phase_b":{"oracle_use":"evaluation only","trainable_modules":list(_ENCODER_PREFIXES),"film_lr":5e-4,"final_dvf":b,"escape":_escape(b),"strict":_strict(b)},"phase_c":{"oracle_use":"evaluation only","film_lr":low,"respiratory_mbc_lr":low,"final_dvf":c,"escape":_escape(c),"strict":_strict(c)},**classification,"frozen_module_audit":frozen,"stage_parameter_change_audit":stage_audit,"renderer_calls":{"total":renderer[0]},"mbc_calls":calls,"cardiac_mbc_calls":card[0],"uncertainty_calls":unc[0],"threshold_steps":thresholds,"best_metrics":min(curve,key=lambda x:x["full_batch_reconstruction_mse"]),"interpretation_contract":"Raw score amplitude/sign are gauge-dependent diagnostics. DVF metrics determine recovery."}; (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); return summary

def main():
 p=argparse.ArgumentParser(); p.add_argument("--source-config",type=Path,default=PROJECT_ROOT/"configs/source_first_change4_paperaligned.yaml"); p.add_argument("--experiment-config",type=Path,default=PROJECT_ROOT/"experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml"); p.add_argument("--manifest",type=Path); p.add_argument("--qc-table",type=Path); p.add_argument("--canonical-domain",type=Path); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cpu"); p.add_argument("--overwrite",action="store_true"); p.add_argument("--seed",type=int,default=0); p.add_argument("--locations-per-view",type=int,default=3); p.add_argument("--frames-per-location",type=int,default=50); p.add_argument("--phase-a-steps",type=int,default=2000); p.add_argument("--phase-b-steps",type=int,default=4000); p.add_argument("--phase-c-steps",type=int,default=4000); p.add_argument("--eval-every",type=int,default=250); p.add_argument("--pixel-mode",default="fixed_grid"); p.add_argument("--pixel-samples",type=int,default=256); p.add_argument("--accumulation-chunk-size",type=int,default=5); p.add_argument("--checkpoint-every",type=int,default=1000); p.add_argument("--resume-from",type=Path); p.add_argument("--evaluation-grid-shape",type=lambda x:tuple(map(int,x.split(","))),default=(8,8,8)); p.add_argument("--smoke",action="store_true"); run(p.parse_args())
if __name__=="__main__": main()
