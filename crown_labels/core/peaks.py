import numpy as np
from scipy.spatial import cKDTree

def detect(pts,rawids):
 x=pts[:,:2];h=pts[:,2];tree=cKDTree(x)
 suppressed=np.zeros(len(x),bool);short=[]
 for i in np.argsort(-h,kind='stable'):
  if suppressed[i]:continue
  short.append(int(i));suppressed[tree.query_ball_point(x[i],.5)]=True
 peaks=[];nlocal=0;nsupport=0
 for i in short:
  near=np.asarray(tree.query_ball_point(x[i],2.),int)
  if np.any(h[near]>h[i]+1e-5):continue
  nlocal+=1
  dist=np.linalg.norm(x[near]-x[i],axis=1)
  support=int(((dist<=.35)&(h[near]>=h[i]-.5)).sum())
  if support<3:continue
  nsupport+=1
  ring=(dist>=1.08)&(dist<=1.8);delta=x[near[ring]]-x[i];rh=h[near[ring]]
  if not len(rh):continue
  sectors=np.floor((np.arctan2(delta[:,1],delta[:,0])+np.pi)/(2*np.pi)*12).astype(int)%12
  tops=[float(np.quantile(rh[sectors==s],.95)) for s in range(12) if (sectors==s).sum()>=3]
  if len(tops)<6:continue
  prom=float(h[i]-np.percentile(tops,35))
  if prom<.3:continue
  peaks.append(dict(raw_index=int(rawids[i]),x=float(x[i,0]),y=float(x[i,1]),height=float(h[i]),prominence=prom,support=support))
 accepted=[]
 for p in sorted(peaks,key=lambda p:p['height'],reverse=True):
  if any(np.hypot(p['x']-q['x'],p['y']-q['y'])<2. for q in accepted):continue
  accepted.append(p)
 return accepted,dict(shortlist=len(short),local_maxima=nlocal,support_passed=nsupport,prominence_passed=len(peaks))
