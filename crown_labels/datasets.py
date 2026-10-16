"""Dense training crops, at a fixed 0.1 m grid; model resize is separate."""
from pathlib import Path
import json,hashlib
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from affine import Affine

def sha(path):
 with open(path,'rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('wb')as f:np.savez_compressed(f,**arrays)
 tmp.replace(path)
def write_manifest(root,rows,layout):
 m={'schema':'crown_labels_dense_manifest_v1','samples':rows,'train_crops':len(rows),'instance_cost':512,'rgb_grid_m':.1,'crop_size':512,'field_m':51.2,'model_input_size':1008,'layout':layout}
 (root/'manifest.json').write_text(json.dumps(m,indent=2));return m
def paired(run_root,template,out):
 """Keep a provided historical RGB/D/height template; replace only instance IDs."""
 root=Path(run_root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 template=json.loads(Path(template).read_text());plan=json.loads((root/'plan.json').read_text())
 results={x['window']['scene']:json.loads((root/'runs'/x['window']['scene']/'result.json').read_text())for x in plan['selected']}
 rows=[];cache={}
 keep=('rgb','rgb_valid','valid_pixels','height','height_valid','height_bin')
 for row in template['samples']:
  name=Path(row['new_source']).stem;res=results[name]
  assert res['converged']and res['active_instance_cost']==512
  if name not in cache:
   assert sha(res['native_path'])==res['native_sha256']
   with np.load(res['native_path'],allow_pickle=False)as z:cache[name]=(z['instance_ids'].copy(),z['valid_pixels'].copy())
  with np.load(row['native_path'],allow_pickle=False)as z:a={k:z[k].copy()for k in keep}
  c=round((row['transform'][2]-res['transform'][2])/.1);r=round((res['transform'][5]-row['transform'][5])/.1)
  ids=cache[name][0][r:r+512,c:c+512].copy();nv=cache[name][1][r:r+512,c:c+512]
  assert ids.shape==(512,512)and np.all(~a['valid_pixels']|nv),'Paired domain changes'
  a.update(schema=np.array('crown_labels_dense_v1'),instance_ids=ids,instance_keys=np.unique(ids[ids>0]),reliable_background=a['valid_pixels']&(ids==0),ignore_pixels=~a['valid_pixels'])
  targets=len(np.unique(ids[(ids>0)&a['valid_pixels']]))
  if targets>200:raise ValueError('More than 200 targets; never truncate')
  dest=out/'native'/(row['name']+'.npz');save(dest,a)
  rows.append(dict(name=row['name'],uav=row['uav'],split='train',origin=row['origin'],native_path=str(dest.resolve()),native_sha256=sha(dest),targets=targets,transform=row['transform'],crs=row['crs']))
 return write_manifest(out,rows,'provided historical paired positions; RGB, D and H unchanged')
def build(run_root,out,edge=8):
 root=Path(run_root);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];excluded=[]
 plan=json.loads((root/'plan.json').read_text())
 for item in plan['selected']:
  w=item['window'];res=json.loads((root/'runs'/w['scene']/'result.json').read_text())
  assert res['converged']and res['file_audit_passed']and res['active_instance_cost']==512
  with np.load(res['native_path'],allow_pickle=False)as z:full={k:z[k].copy()for k in ['instance_ids','valid_pixels','height','height_valid','height_bin']}
  local=[];reasons=[]
  for letter,(r,c)in zip('ABCD',[(0,0),(0,512),(512,0),(512,512)]):
   tf=Affine(*res['transform'])*Affine.translation(c,r);sl=np.s_[r:r+512,c:c+512]
   with rasterio.open(w['dom'])as ds:
    alpha=[i+1 for i,x in enumerate(ds.colorinterp)if x.name=='alpha']
    with WarpedVRT(ds,crs=ds.crs,transform=tf,width=512,height=512,src_nodata=None,resampling=Resampling.bilinear)as vrt:
     rgb=vrt.read([1,2,3]).transpose(1,2,0);rv=vrt.read(alpha[0],resampling=Resampling.nearest)>0 if alpha else np.all(vrt.read_masks([1,2,3],resampling=Resampling.nearest)>0,axis=0)
   rgb[~rv]=0;valid=rv&full['valid_pixels'][sl]
   if edge:valid[:edge]=False;valid[-edge:]=False;valid[:,:edge]=False;valid[:,-edge:]=False
   ids=full['instance_ids'][sl].copy();targets=len(np.unique(ids[(ids>0)&valid]))
   if rv.mean()<.98:reasons.append(letter+': RGB-valid < .98')
   if valid.mean()<.90:reasons.append(letter+': D < .90')
   if targets>200:reasons.append(letter+': targets > 200')
   a=dict(schema=np.array('crown_labels_dense_v1'),rgb=rgb,rgb_valid=rv,instance_ids=ids,instance_keys=np.unique(ids[ids>0]),valid_pixels=valid,ignore_pixels=~valid,reliable_background=valid&(ids==0),height=full['height'][sl],height_valid=full['height_valid'][sl],height_bin=full['height_bin'][sl])
   name=w['scene']+'_'+letter;dest=out/'native'/(name+'.npz');save(dest,a)
   local.append(dict(name=name,parent=w['scene'],uav=w['uav'],split='train',origin=[r,c],native_path=str(dest.resolve()),native_sha256=sha(dest),targets=targets,transform=list(tf)[:6],crs=w['crs']))
  if reasons:excluded.append(dict(parent=w['scene'],reasons=reasons))
  else:rows.extend(local)
 (out/'excluded.json').write_text(json.dumps(excluded,indent=2))
 return write_manifest(out,rows,'four nonoverlapping crops per 1024 core; whole-mother QC')
def load_crop(path,model_size=1008):
 """Raw uint8 RGB for the model's own normalization; no height input channel."""
 import cv2
 with np.load(path,allow_pickle=False)as z:a={k:z[k].copy()for k in ['rgb','instance_ids','valid_pixels','reliable_background','height','height_valid','rgb_valid']}
 ids=a['instance_ids'];keys=np.unique(ids[(ids>0)&a['valid_pixels']])
 if len(keys)>200:raise ValueError('Query capacity exceeded')
 assert a['rgb'].shape==(512,512,3)and ids.shape==(512,512)
 a['rgb_model']=cv2.resize(a['rgb'],(model_size,model_size),interpolation=cv2.INTER_LINEAR)
 a['masks']=np.asarray([ids==k for k in keys],bool).reshape(-1,512,512)
 a['object_ids']=keys.tolist()
 return a
