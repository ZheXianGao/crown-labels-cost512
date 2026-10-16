import numpy as np
from scipy import ndimage as nd

def refine(ids,domain):
 out=ids.copy();records=[]
 for ident,sl in enumerate(nd.find_objects(ids),1):
  if sl is None:continue
  sl=tuple(slice(max(0,s.start-2),min(ids.shape[k],s.stop+2)) for k,s in enumerate(sl))
  src=ids[sl];mask=src==ident;holes=nd.binary_fill_holes(mask)&~mask
  labs,n=nd.label(holes);sizes=np.bincount(labs.ravel());dist=nd.distance_transform_edt(~mask)*.1
  proposed=np.zeros(mask.shape,bool);blocked=0
  for j in range(1,n+1):
   if sizes[j]>100:continue
   hole=labs==j
   # Reject whole holes crossing forbidden/nodata/another instance, not just
   # individual pixels: otherwise protected gaps could grow back piecemeal.
   if np.any(src[hole]>0) or not domain[sl][hole].all() or dist[hole].max()>.4+1e-8:
    blocked+=int(sizes[j]);continue
   proposed|=hole
  target=out[sl];target[proposed]=ident
  # Remove truly tiny detached fragments; keep largest and frame-cut parts.
  components,nc=nd.label(target==ident,np.ones((3,3),bool));areas=np.bincount(components.ravel());removed=0
  if nc>1:
   largest=1+np.argmax(areas[1:])
   for j in range(1,nc+1):
    if j==largest or areas[j]>20:continue
    part=components==j;ys,xs=np.where(part)
    at_edge=((sl[0].start==0 and ys.min()==0) or (sl[1].start==0 and xs.min()==0) or (sl[0].stop==ids.shape[0] and ys.max()==part.shape[0]-1) or (sl[1].stop==ids.shape[1] and xs.max()==part.shape[1]-1))
    if not at_edge:target[part]=0;removed+=int(areas[j])
  records.append(dict(id=ident,filled_pixels=int(proposed.sum()),removed_fragment_pixels=removed,protected_hole_pixels=blocked))
 assert set(np.unique(out))==set(np.unique(ids))
 assert not np.any(out[~domain])
 return out,dict(version='surface_v2_enclosed_holes',hole_max_m2=1.,hole_max_distance_m=.4,fragment_max_m2=.2,new_IDs=False,energy_changed=False,filled_pixels=sum(r['filled_pixels'] for r in records),removed_pixels=sum(r['removed_fragment_pixels'] for r in records),records=records)
