"""DICOM-world pixel, cardiac ROI, and balanced three-view sampling primitives."""
from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np
import torch

def pixel_indices_to_world(rows, cols, geometry):
    r=np.asarray(rows,dtype=float); c=np.asarray(cols,dtype=float); o=np.asarray(geometry['image_position_patient'],float); d=np.asarray(geometry['image_orientation_patient'],float); s=np.asarray(geometry['pixel_spacing'],float)
    return o+c[...,None]*s[1]*d[:3]+r[...,None]*s[0]*d[3:]
def roi_pixel_pool(geometry, box_min_mm, box_max_mm, margin_mm=15.):
    rr,cc=np.indices((int(geometry['rows']),int(geometry['columns']))); world=pixel_indices_to_world(rr.ravel(),cc.ravel(),geometry); inside=np.all((world>=np.asarray(box_min_mm)-margin_mm)&(world<=np.asarray(box_max_mm)+margin_mm),1); return {'rows':rr.ravel(),'cols':cc.ravel(),'roi':np.flatnonzero(inside),'intersects_roi':bool(inside.any())}
def sample_pixels(pool,n,roi_fraction,generator):
    total=len(pool['rows']); global_i=torch.randint(total,(n,),generator=generator)
    if not pool['intersects_roi']: return pool['rows'][global_i],pool['cols'][global_i],0
    nr=round(n*roi_fraction); roi=torch.as_tensor(pool['roi']); chosen=roi[torch.randint(len(roi),(nr,),generator=generator)]; index=torch.cat((chosen,global_i[:n-nr])); return pool['rows'][index],pool['cols'][index],nr
@dataclass
class BalancedViewEpochSampler:
    by_view:dict; seed:int=0
    def __post_init__(self): self.steps_per_epoch=max(map(len,self.by_view.values()))
    def epoch(self,epoch=0):
        g=torch.Generator().manual_seed(self.seed+epoch); orders={v:torch.randperm(len(x),generator=g).tolist() for v,x in self.by_view.items()}
        for i in range(self.steps_per_epoch): yield {v:self.by_view[v][orders[v][i%len(orders[v])]] for v in self.by_view}
