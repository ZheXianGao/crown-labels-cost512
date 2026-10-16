"""Cost512 production geometry. RGB and manual labels do not enter the solver."""
from pathlib import Path
from types import SimpleNamespace
import os,json,time,hashlib
import numpy as np
from scipy.special import logsumexp
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import maximum_filter
from .core import algorithm as alg, joint_model as jm, builder, unary, family_solver
from .core import conditional_cache, background as bg, gap_repair, cleanup
from .core.cleanup_config import COMPACT
from .core.proposals import get_groups
from .core.admission import admit_peaks as _admit_peaks
from .core.peaks import detect
from . import policy as reviewed
P=Path(os.environ.get('CROWN_RUN_ROOT','.')).resolve()
v=SimpleNamespace(alg=alg,sm=builder,jm=jm,method=unary,moves=family_solver,
 make_cache=conditional_cache.make,repair=gap_repair.repair,
 make_fast_cleanup=lambda:cleanup.clean_partition,CleanupConfig=cleanup.CleanupConfig,
 CONFIGS={'compact_medium':COMPACT})
def detector():
 return detect
def check_sources():
 root=Path(__file__).resolve().parents[1]
 lock=root/'SOURCE_LOCK.json'
 if lock.exists():
  for rel,expected in json.loads(lock.read_text()).items():
   assert sha(root/rel)==expected, 'Source changed: '+rel
 return True

def admit_peaks(peaks, highest_points):
 accepted, rows = _admit_peaks(peaks, highest_points, radius_m=2., min_vertical_ratio=.7)
 for row in rows: row.update(min_vertical_ratio=.7, radius_m=2.)
 return accepted, rows

def write(path,obj):
 path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build(data,peaks,out):
 graph=v.alg.make_graph(data['nodes'],.8,12);assert (graph-graph.T).nnz==0
 records=[dict(id=i+1,col=p['x']/.1-.5,row=p['y']/.1-.5,height=p['height']) for i,p in enumerate(peaks)]
 ns=dict(v.sm.__dict__);ns['Config']=lambda:v.alg.Config(voxel_m=.1,raster_m=.1);ns['make_graph']=lambda *a,**kw:graph
 ns['stable_seeds']=lambda chm,*a:(records,np.zeros(chm.shape,np.int32),dict(method='fixed_direct_2m_peak',fallback=False))
 def directed(g,*a,**kw):kw['directed']=True;return dijkstra(g,*a,**kw)
 ns['dijkstra']=directed;ns['maximum_filter']=maximum_filter;ns['context_footprint']=np.ones((2,2),bool)
 source=(Path(__file__).parent/"core/builder_kernel.py.txt").read_text()
 # Same dynamically compiled builder as the accepted box2 small-ROI launcher.
 assert hashlib.sha256(source.encode()).hexdigest()=='8043e66eaa53e29ec292cd2d10afeb60ed4964115075671256aea21826a288c8'
 exec(compile(source,'<locked_v11_batch_builder>','exec'),ns)
 cfg=v.jm.JointConfig(fit_radius_m=2.,max_joint_candidates=300,max_mahalanobis2=16.,graph_strength=1.5,label_cost_voxels=512.)
 model=ns['build_geometry_model'](dict(voxel_xyz=data['nodes'],voxel_height=data['heights'],local_support=data['local_support']),dict(raw_upper_height=data['upper'],raw_return_count=data['counts'],valid=data['valid']),{},cfg)
 # A no-peak region is background, never a reason to inject fallback seeds.
 if not model['pool']:return model,dict(leaf=0,pair=0,hierarchy=0,empty=True)
 groups,ga=get_groups(model)
 original_bg=bg.background
 def parallel_background(upper,counts,cache_dir=None,workers=12):
  return original_bg(upper,counts,P/'cache'/'background',workers=2)
 bg.background=parallel_background
 try:background=v.method.prepare_unary(model,data)
 finally:bg.background=original_bg
 lookup={p['source_id']:k for k,p in enumerate(model['pool'],1) if p['source'].startswith('local_')}
 seen={tuple(sorted(p['source_members'])) for p in model['pool']};cols=[];added=[];oldnum=len(model['pool'])
 for group in groups:
  members=tuple(sorted(group['source_members']))
  if members in seen or len(members)<3:continue
  seen.add(members);children=[lookup[s] for s in members];parts=[model['pool'][k-1] for k in children]
  p=dict(source='joint_CHM',source_id=200000+len(added),ff_id=0,source_members=list(members),center=np.mean([x['center'] for x in parts],axis=0),core=np.unique(np.concatenate([x['core'] for x in parts])),fit=np.unique(np.concatenate([x['fit'] for x in parts])))
  center,cov,inverse,logdet=v.sm.gaussian_model(model['nodes'][p['fit']],cfg)
  ix=np.flatnonzero(np.isfinite(model['unary'][:,children]).any(1));col=np.full(len(model['nodes']),np.inf,np.float32)
  col[ix]=(-logsumexp(-8*model['unary'][np.ix_(ix,children)],axis=1)+np.log(len(children)))/8
  ident=oldnum+len(added)+1;rec=dict(id=ident,source=p['source'],source_id=p['source_id'],ff_id=0,source_members=list(members),center_xyz_m=center.tolist(),center_xy_m=center[:2].tolist(),covariance=cov.tolist(),logdet=float(logdet),core_nodes=len(p['core']),fit_nodes=len(p['fit']),feasible_nodes=len(ix),accepted=True,unary_model='equal_original_leaf_mixture_beta8',hierarchy_evidence=group)
  model['pool'].append(p);model['candidates'].append(rec);cols.append(col);added.append(rec)
 if cols:model['unary']=np.column_stack([model['unary'],*cols])
 model['costs']=np.full(model['unary'].shape[1],512.);model['costs'][0]=0
 audit=dict(active_instance_cost=512,candidate_admission_min_vertical_ratio=.7,graph_strength=1.5,nodes=len(model['nodes']),leaf=len(lookup),pair=oldnum-len(lookup),hierarchy=len(added),builder_sha256=hashlib.sha256(source.encode()).hexdigest(),background=background,hierarchy_audit=ga,all_three_energy_terms=True,initialization='fresh_leaf_only',peaks_from_raw_points=True,old_generator_called=False)
 write(out/'model.json',dict(records=model['candidates'],audit=audit));return model,audit

def solve(model,out,initial=None):
 if model['pool']:
  assert model['costs'][0]==0 and np.all(model['costs'][1:]==512), 'Foreground activation costs must be 512'
 if not model['pool']:return np.zeros(len(model['nodes']),np.int32),dict(converged_full_pass=True,inner_unconverged=0,empty_candidate_pool=True,trace=[])
 labels=v.method.initialize(model) if initial is None else initial.copy();rounds=[]
 original=v.moves.optimize;v.moves.optimize=v.make_cache(original)
 def checkpoint(x,log):
  tmp=out/'optimizer_labels.tmp'
  with tmp.open('wb') as f:np.save(f,x)
  tmp.replace(out/'optimizer_labels.npy');write(out/'progress.json',log)
 try:
  for repeat in range(30):
   labels,log=v.moves.solve(model,model['costs'],labels,time_limit=3600,callback=checkpoint)
   write(out/f'solver_round_{repeat+1:02d}.json',log);rounds.append(dict(round=repeat+1,converged=log['converged_full_pass'],inner_unconverged=log['inner_unconverged'],seconds=log['seconds'],final_energy=log['trace'][-1]))
   if log['converged_full_pass'] and not log['inner_unconverged']:break
  else:raise RuntimeError('Full-pass convergence not attained; export withheld')
 finally:v.moves.optimize=original
 log['continuations']=rounds;write(out/'solver.json',log);np.save(out/'optimizer_labels.npy',labels);return labels,log

def clean3d(labels,model):
 if not model['pool']:return labels.copy(),[]
 seedids=np.zeros(len(labels),np.int32)
 for i,p in enumerate(model['pool'],1):seedids[p['core'][labels[p['core']]==i]]=i
 original=v.alg.dijkstra
 def directed(g,*a,**kw):kw['directed']=True;return dijkstra(g,*a,**kw)
 v.alg.dijkstra=directed
 try:return v.alg.clean_3d(labels,model['graph'],seedids,model['nodes'],model['heights'],model['base'])
 finally:v.alg.dijkstra=original

def postprocess(cleaned,data,w):
 pv=data['projection_voxel'];observed=pv>=0;raw=np.zeros(pv.shape,np.int32);raw[observed]=cleaned[pv[observed]];raw[~data['valid']]=0
 from affine import Affine
 policy=json.load(open(P/'review_policy.json'))
 hard,whole,policylog=reviewed.policy_masks(policy,w['uav'],Affine(*w['transform']),w['crs'],raw.shape)
 arrays,log=reviewed.finish(raw,data,data['rgb_valid'],hard,whole,v)
 # Same independent observed-height contract, evaluated directly on 10 cm cells.
 upper=data['upper'];hv=data['rgb_valid']&data['ground_support']&(data['counts']>=3)&np.isfinite(upper)&(upper>=-2)&(upper<=60)&~arrays['geographic_forbidden']
 height=np.full(raw.shape,np.nan,np.float32);height[hv]=np.maximum(upper[hv],0)
 bins=np.full(raw.shape,255,np.uint8);bins[hv]=np.digitize(height[hv],[.5,2.5,5.,10.],right=False)
 arrays.update(height=height,height_valid=hv,height_bin=bins,height_ignore_pixels=~hv,height_observed_raw=upper.copy())
 log.update(geographic_policy=policylog,height_contract='raw 0.1m upper return; >=3 returns; observed ground cell; -2..60m; prior negative-ground clamp; no morphological height filling')
 return raw,arrays,log
