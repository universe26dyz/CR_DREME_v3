"""功能：由有效动态帧构建 Stage-1B 的固定位置 temporal-mean observations。
论文来源：NeSVoR static slice-domain refinement + S2V-DREME temporal averaging。
输入：authoritative manifest、完整 QC table 和 runtime DICOM sidecar。
输出：mean-slice ``.npy`` 文件与 PHI-free mean_slice_manifest.csv。
主要步骤：每个 view/slice_id 对 qc_valid frames 做算术平均；whole-location 无有效帧时省略 observation。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary hybrid adaptation；不会裁剪 canonical domain。
命令行使用示例：python -m cardioresp4d.reference.build_mean_slices --manifest results/dicom_manifest.csv --output-dir results/reference
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from cardioresp4d.data.dataset import CardioRespDataset

def build_mean_slices(manifest_path: str | Path, output_dir: str | Path, *, qc_table_path: str | Path | None = None) -> Path:
    """Write only observed valid-location means; missing GT never creates a zero-filled slice."""
    dataset = CardioRespDataset(manifest_path, valid_only=False, qc_table_path=qc_table_path)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset._rows): groups[(row['view'].upper(), row['slice_id'])].append(index)
    out = Path(output_dir); images = out / 'mean_slices'; images.mkdir(parents=True, exist_ok=True)
    fields = ('mean_slice_id','view','slice_id','image_file','geometry_json','contributing_frame_uids','n_valid','n_total','timestamp_min_s','timestamp_max_s','timestamp_mean_s')
    rows: list[dict[str, str]] = []
    for (view, slice_id), indices in sorted(groups.items()):
        valid = [i for i in indices if dataset._rows[i]['qc_valid'] not in ('0','false','False')]
        if not valid: continue
        samples = [dataset[i] for i in valid]
        mean = np.mean(np.stack([sample['image'] for sample in samples]), axis=0, dtype=np.float64).astype(np.float32)
        if not np.isfinite(mean).all(): raise RuntimeError(f'Non-finite temporal mean for {view}/{slice_id}')
        stable = hashlib.sha256(f'mean-slice-v1:{view}:{slice_id}'.encode()).hexdigest()[:24]
        image_file = images / f'{stable}.npy'; np.save(image_file, mean)
        timestamps = np.asarray([sample['timestamp_s'] for sample in samples], dtype=float)
        rows.append({'mean_slice_id': stable, 'view': view, 'slice_id': slice_id, 'image_file': str(image_file.relative_to(out)),
                     'geometry_json': json.dumps(samples[0]['geometry'], sort_keys=True),
                     'contributing_frame_uids': json.dumps([sample.get('source_file_token') or str(i) for i, sample in zip(valid, samples)]),
                     'n_valid': str(len(valid)), 'n_total': str(len(indices)), 'timestamp_min_s': repr(float(timestamps.min())),
                     'timestamp_max_s': repr(float(timestamps.max())), 'timestamp_mean_s': repr(float(timestamps.mean()))})
    manifest = out / 'mean_slice_manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser(description='Build qc-valid temporal mean slices for Stage 1B static PSF supervision.')
    parser.add_argument('--manifest', required=True, type=Path); parser.add_argument('--output-dir', required=True, type=Path); parser.add_argument('--qc-table', type=Path)
    args = parser.parse_args(); print(build_mean_slices(args.manifest, args.output_dir, qc_table_path=args.qc_table))
if __name__ == '__main__': main()
