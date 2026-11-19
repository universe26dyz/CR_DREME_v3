import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cardioresp4d.data.dataset import CardioRespDataset
from cardioresp4d.geometry.world_geometry import DicomPlane
from cardioresp4d.roi.cardiac_box import CardiacBox
from cardioresp4d.roi.roi_qc import box_plane_intersection
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map
root=Path('/data2/dengyz/dataset/CR_DREME_v3'); out=root/'exp/simple4d_roi_cardiac_residual_v1/00_roi_audit/candidate_160'; out.mkdir(exist_ok=False)
audit=json.loads((out.parent/'auto_cardiac_box.json').read_text()); center=np.array(audit['center_mm']); box=CardiacBox(center,[160,160,140]); core=CardiacBox(center,[130,130,110]);
m=root/'v1/phase1/dicom_manifest.csv'; q=m.parent/'acquisition_qc/acquisition_qc.csv'; st=json.loads((root/'exp1_s2vdreme_reference_v1/02_static_nesvor/static_reference_manifest.json').read_text()); poses=load_corrected_pose_map(st['corrected_pose_manifest']['path']); d=CardioRespDataset(m,valid_only=True,qc_table_path=q,normalization_mode='per_series')
result={}
for v in ('SAX','2CH','4CH'):
 groups={}
 for i,r in enumerate(d._rows):
  if r['view'].upper()==v: groups.setdefault(r['slice_id'],[]).append(i)
 stack=np.array(audit['views'][v]['stack_normal_acquisition']); ranked=sorted((float(np.dot(poses[(v,k)].corrected_geometry['center_mm'],stack)),k) for k in groups)
 # physical quartiles plus median
 chosen=[ranked[round((len(ranked)-1)*q)] for q in (.25,.5,.75)]; info=[]
 fig,axs=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
 for ax,(coord,k) in zip(axs,chosen):
  g=poses[(v,k)].corrected_geometry; row=np.array(g['row_direction']); col=np.array(g['column_direction']); sp=np.array(g['pixel_spacing_mm']); _,h,w=g['image_shape']; origin=np.array(g['center_mm'])-(w-1)/2*sp[1]*row-(h-1)/2*sp[0]*col; plane=DicomPlane(origin,row,col,sp,h,w,g['slice_thickness_mm']); mean=np.mean([d[i]['image'] for i in groups[k]],axis=0); ax.imshow(mean,cmap='gray')
  for b,color in ((box,'red'),(core,'cyan')):
   p=box_plane_intersection(b,plane)['pixel'];
   if len(p)>=3:
    p=np.vstack([p,p[0]]);ax.plot(p[:,0],p[:,1],color=color,lw=1.3)
  p=box_plane_intersection(box,plane)['pixel']; cp=plane.world_to_pixel(center); ax.scatter(*cp,c='yellow',marker='+'); ax.set(xlim=(-.5,w-.5),ylim=(h-.5,-.5),title=f'{v} {k} ({coord:.1f} mm)'); info.append({'slice_id':k,'coordinate_mm':coord,'frames':len(groups[k]),'intersection_extent_px':np.ptp(p,axis=0).tolist() if len(p) else None,'polygon_px':p.tolist(),'center_px':cp.tolist()})
 fig.suptitle('160×160×140 mm support (red); 15 mm taper core (cyan), not segmentation');fig.savefig(out/f'qc_{v.lower()}.png',dpi=130);plt.close(fig);result[v]=info
(out/'audit.json').write_text(json.dumps({'center_mm':center.tolist(),'size_mm':[160,160,140],'taper_mm':15,'views':result},indent=2)+'\n')
