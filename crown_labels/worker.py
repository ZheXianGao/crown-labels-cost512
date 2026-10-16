from pathlib import Path
import os,sys,time,json,resource,traceback,fcntl,hashlib
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from . import science,prepare,exporter as export,surface
P=Path(os.environ.get('CROWN_RUN_ROOT','.')).resolve()

def run(name):
 science.check_sources();plan=json.load(open(P/'plan.json'));item=next(x for x in plan['selected'] if x['window']['scene']==name);w=item['window'];out=P/'runs'/name;out.mkdir(exist_ok=True,parents=True)
 assert item['candidate_admission_min_vertical_ratio']==.7 and item['active_instance_cost']==512
 with (out/'worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (out/'status.json').exists() and json.load(open(out/'status.json'))['stage']=='complete':return json.load(open(out/'result.json'))
  start=time.monotonic();timings={}
  def stage(s,**more):science.write(out/'status.json',dict(name=name,stage=s,elapsed_seconds=time.monotonic()-start,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,**more));print('STAGE',name,s,round(time.monotonic()-start,1),flush=True)
  try:
   stage('prepare_raw');t=time.monotonic();data,peaks,prep=prepare.prepare(item,out);timings['prepare']=time.monotonic()-t
   stage('build_new_method',nodes=len(data['nodes']),peaks=len(peaks));t=time.monotonic()
   if len(data['nodes']):model,build=science.build(data,peaks,out)
   else:model=dict(pool=[],nodes=data['nodes']);build=dict(empty=True,leaf=0,pair=0,hierarchy=0)
   timings['build']=time.monotonic()-t;stage('optimize',nodes=len(data['nodes']),candidates=len(model['pool']));t=time.monotonic();labels,solver=science.solve(model,out);timings['solve']=time.monotonic()-t
   assert solver['converged_full_pass'] and solver['inner_unconverged']==0
   stage('clean_3d');t=time.monotonic();cleaned,objects=science.clean3d(labels,model);timings['clean3d']=time.monotonic()-t
   np.save(out/'cleaned_voxel_ids.npy',cleaned);science.write(out/'clean3d.json',objects);del model
   stage('cleanup_2d_and_policy');t=time.monotonic();raw,arrays,post=science.postprocess(cleaned,data,w)
   domain=data['valid']&data['rgb_valid']&~arrays['geographic_forbidden']
   final,surface_audit=surface.refine(arrays['candidate_ids'],domain)
   export.save_npz(out/'projection_and_cleaned.npz',dict(raw_projection=raw,pre_surface_ids=arrays['candidate_ids'],cleaned_ids=final))
   arrays['candidate_ids']=final;science.write(out/'surface_audit.json',surface_audit);timings['postprocess']=time.monotonic()-t
   stage('export_native_DOM');t=time.monotonic()
   result=export.export(arrays,w,solver,dict(preparation=prep,model=build,postprocessing=post,surface=surface_audit,stage_seconds=timings),out);timings['export']=time.monotonic()-t
   stage('verify_export');t=time.monotonic();checks=[]
   with np.load(result['native_path']) as core:
    for c in result['crops']:
     with np.load(c['native_path']) as crop:
      r,col=c['origin'];lut=np.r_[0,np.asarray(c['core_instance_ids'],np.int32)]
      assert np.array_equal(lut[crop['instance_ids']],core['instance_ids'][r:r+512,col:col+512])
      assert np.array_equal(crop['reliable_background'],crop['valid_pixels']&(crop['instance_ids']==0))
      checks.append(dict(name=c['name'],core_ID_mapping_exact=True,targets=c['targets'],query_capacity_ok=c['query_capacity_ok']))
   science.write(out/'file_audit.json',dict(crops=checks,passed=True))
   timings['verify']=time.monotonic()-t;result.update(stage_seconds=timings,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,seconds=time.monotonic()-start,final_surface_cleaned=True,file_audit_passed=True)
   science.write(out/'result.json',result);science.write(export.O/'review'/name/'summary.json',result)
   stage('complete',IDs=result['core_instances'],seconds=result['seconds']);return result
  except BaseException:
   science.write(out/'error.json',dict(name=name,traceback=traceback.format_exc()));stage('failed');raise
