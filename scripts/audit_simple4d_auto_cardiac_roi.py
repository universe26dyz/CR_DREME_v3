import csv,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cardioresp4d.data.dataset import CardioRespDataset
from cardioresp4d.geometry.world_geometry import DicomPlane
from cardioresp4d.roi.cardiac_box import CardiacBox
from cardioresp4d.roi.roi_qc import box_plane_intersection
from cardioresp4d.roi.auto_cardiac import central_plane,intersect_planes
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map
root=Path('/data2/dengyz/dataset/CR_DREME_v3'); out=root/'exp/simple4d_roi_cardiac_residual_v1/00_roi_audit'; out.mkdir(parents=True,exist_ok=False)
m=root/'v1/phase1/dicom_manifest.csv'; q=m.parent/'acquisition_qc/acquisition_qc.csv'; static=root/'exp1_s2vdreme_reference_v1/02_static_nesvor/static_reference_manifest.json'
d=CardioRespDataset(m,valid_only=True,qc_table_path=q,normalization_mode='per_series'); st=json.loads(static.read_text()); poses=load_corrected_pose_map(st['corrected_pose_manifest']['path'])
selected={}; details={}; centers=[]; normals=[]
for view in ('SAX','2CH','4CH'):
 indices={}
 for i,r in enumerate(d._rows):
  if r['view'].upper()==view: indices.setdefault(r['slice_id'],[]).append(i)
 raw=DicomPlane.from_geometry(d._rows[next(iter(indices.values()))[0]]); stack=raw.normal
 obs=[]
 for key in indices:
  g=poses[(view,key)].corrected_geometry
  obs.append(SimpleNamespace(view=view,slice_id=key,qc_valid=True,center_mm=np.array(g['center_mm']),normal=np.array(g['normal'])))
 (coord,key,ob),bounds=central_plane(obs,stack_normal=stack); g=poses[(view,key)].corrected_geometry
 row=np.array(g['row_direction']); col=np.array(g['column_direction']); spacing=np.array(g['pixel_spacing_mm']); _,h,w=g['image_shape']; origin=np.array(g['center_mm'])-(w-1)/2*spacing[1]*row-(h-1)/2*spacing[0]*col
 plane=DicomPlane(origin,row,col,spacing,h,w,g['slice_thickness_mm']); selected[view]=(indices[key],plane)
 details[view]={'count':len(indices),'slice_id':key,'stack_normal_acquisition':stack.tolist(),'coordinate_mm':coord,'stack_coordinate_range_mm':list(bounds),'origin_mm':origin.tolist(),'row_direction':row.tolist(),'column_direction':col.tolist(),'normal':plane.normal.tolist(),'fov_mm':[float(w*spacing[1]),float(h*spacing[0])],'pixel_shape':[h,w]}
 centers.append(g['center_mm']); normals.append(g['normal'])
center,res,cond=intersect_planes(centers,normals); candidates=[[100,100,80],[120,120,100],[140,140,120]]
report={'center_mm':center.tolist(),'rank':int(np.linalg.matrix_rank(normals)),'condition_number':cond,'signed_plane_residual_mm':dict(zip(('SAX','2CH','4CH'),res.tolist())),'views':details,'candidate_size_mm':candidates,'selection':'pending anatomical review','coordinate_system':'corrected patient-world LPS, baseline S05PROP; stack order by corrected center projected onto original acquisition normal; even count lower median','inputs':{'manifest':str(m),'qc_table':str(q),'corrected_pose_manifest':st['corrected_pose_manifest']['path']}}
with (out/'geometry_residuals.tsv').open('w') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['view','slice_id','coordinate_mm','residual_mm']);[w.writerow([v,details[v]['slice_id'],details[v]['coordinate_mm'],r]) for v,r in zip(('SAX','2CH','4CH'),res)]
for view,(indices,plane) in selected.items():
 mean=np.mean([d[i]['image'] for i in indices],axis=0); fig,axs=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
 for ax,size in zip(axs,candidates):
  box=CardiacBox(center,size); p=box_plane_intersection(box,plane)['pixel']; cp=plane.world_to_pixel(center); ax.imshow(mean,cmap='gray',origin='upper');
  if len(p)>=3:
   pp=np.vstack([p,p[0]]);ax.plot(pp[:,0],pp[:,1],color='red',lw=1.5)
  ax.scatter(*cp,marker='+',s=85,color='cyan');ax.set_title(f'{view} {details[view]["slice_id"]} | {size} mm');ax.set_xlim(-.5,plane.columns-.5);ax.set_ylim(plane.rows-.5,-.5)
  details[view].setdefault('candidate_projection',[]).append({'size_mm':size,'extent_px':(np.ptp(p,axis=0).tolist() if len(p) else None),'polygon_px':p.tolist(),'center_px':cp.tolist()})
 fig.suptitle('Automatic geometry-derived cardiac support box — not segmentation');fig.savefig(out/f'qc_{view.lower()}.png',dpi=130);plt.close(fig)
(out/'auto_cardiac_box.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
