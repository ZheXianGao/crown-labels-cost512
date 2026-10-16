import argparse,os,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed

def bind(root):
 os.environ['CROWN_RUN_ROOT']=str(root)
 from . import science,prepare,exporter,extract,worker
 for m in [science,prepare,exporter,extract,worker]:m.P=root
 exporter.O=root/'release'
 return extract,worker
def work(job):
 root,name=job;_,worker=bind(Path(root));return worker.run(name)
def main():
 parser=argparse.ArgumentParser(description='LiDAR crown labels: cost512 / 0.1 m / dense export')
 sub=parser.add_subparsers(dest='command',required=True)
 p=sub.add_parser('generate');p.add_argument('--plan',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--scene');p.add_argument('--skip-extract',action='store_true')
 p=sub.add_parser('build-crops');p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--paired-template');p.add_argument('--edge-ignore',type=int,default=8)
 sub.add_parser('check')
 a=parser.parse_args()
 if a.command=='check':
  from .science import check_sources
  check_sources();print('SOURCE_CHECK_OK');return
 if a.command=='build-crops':
  from . import datasets
  result=datasets.paired(a.run,a.paired_template,a.out)if a.paired_template else datasets.build(a.run,a.out,a.edge_ignore)
  print(json.dumps({'train_crops':result['train_crops']}));return
 if a.workers<1:parser.error('--workers must be positive')
 root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True);plan=Path(a.plan).resolve();target=root/'plan.json'
 if target.exists():assert target.read_bytes()==plan.read_bytes(),'Existing run has another plan'
 elif target!=plan:shutil.copyfile(plan,target)
 data=json.loads(target.read_text());items=data['selected']
 for item in items:
  assert item['active_instance_cost']==512 and item['candidate_admission_min_vertical_ratio']==.7
  assert item['window']['size']==item['window']['core_offset']*2+1024
 if a.scene:items=[x for x in items if x['window']['scene']==a.scene];assert items,'Scene not found'
 policy=root/'review_policy.json'
 if not policy.exists():policy.write_text('{}')
 extract,_=bind(root)
 if not a.skip_extract:
  groups={}
  for x in items:groups.setdefault(x['window']['uav'],[]).append(x)
  for group in groups.items():extract.flight(group)
 with ProcessPoolExecutor(max_workers=a.workers)as pool:
  futures={pool.submit(work,(str(root),x['window']['scene'])):x['window']['scene']for x in items}
  for f in as_completed(futures):f.result();print('COMPLETE',futures[f],flush=True)
if __name__=='__main__':main()
