import unittest
import numpy as np
from crown_labels.core.admission import admit_peaks
from crown_labels.core.gap_repair import repair
from crown_labels.surface import refine
from crown_labels.core.joint_solver import energy,optimize
from crown_labels.core.joint_model import JointConfig

class Contracts(unittest.TestCase):
 def test_config(self):
  self.assertEqual(JointConfig().label_cost_voxels,512.)
  self.assertEqual(JointConfig().background_cost,.8)
 def test_no_new_ids(self):
  ids=np.zeros((32,32),np.int32);ids[6:22,6:22]=1;ids[12,12]=0
  out,log=refine(ids,np.ones(ids.shape,bool))
  self.assertEqual(set(np.unique(out)),{0,1});self.assertEqual(out[12,12],1)
 def test_dense_domains(self):
  ids=np.array([[0,1],[2,0]],np.int32);d=np.array([[1,1],[1,0]],bool)
  fg=(ids>0)&d;bg=(ids==0)&d
  self.assertTrue(np.array_equal(fg|bg,d));self.assertFalse((fg&bg).any())
 def test_energy(self):
  unary=np.array([[.8,0],[.8,0]],np.float32);a=np.array([0]);b=np.array([1]);w=np.array([1.]);costs=np.array([0.,.1])
  out,log=optimize(unary,a,b,w,costs,initial=np.zeros(2,np.int32))
  self.assertTrue(log['converged_full_pass']);self.assertTrue((out==1).all())

class DatasetContract(unittest.TestCase):
 def test_gdal_dense_export_and_model_resize(self):
  import tempfile,json
  from pathlib import Path
  import rasterio
  from rasterio.transform import Affine
  from rasterio.enums import ColorInterp
  from crown_labels.datasets import build,load_crop
  with tempfile.TemporaryDirectory()as td:
   root=Path(td);(root/'runs/example').mkdir(parents=True)
   tf=Affine(.1,0,0,0,-.1,102.4);dom=root/'rgb.tif'
   rgb=np.full((4,1024,1024),120,np.uint8);rgb[3]=255;rgb[3,100,100]=0
   with rasterio.open(dom,'w',driver='GTiff',height=1024,width=1024,count=4,dtype='uint8',transform=tf,crs='EPSG:32650')as ds:
    ds.write(rgb);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
   ids=np.zeros((1024,1024),np.int32);ids[20:70,20:70]=1;ids[490:540,490:540]=2
   valid=np.ones(ids.shape,bool);valid[100,100]=False;height=np.full(ids.shape,7,np.float32)
   source=root/'mother.npz';np.savez_compressed(source,instance_ids=ids,valid_pixels=valid,height=height,height_valid=valid,height_bin=np.full(ids.shape,3,np.uint8))
   plan={'selected':[{'window':{'scene':'example','uav':1,'dom':str(dom),'crs':'EPSG:32650'}}]};(root/'plan.json').write_text(json.dumps(plan))
   result={'native_path':str(source),'converged':True,'file_audit_passed':True,'active_instance_cost':512,'transform':list(tf)[:6]};(root/'runs/example/result.json').write_text(json.dumps(result))
   manifest=build(root,root/'dataset');self.assertEqual(manifest['train_crops'],4)
   a=load_crop(manifest['samples'][0]['native_path'])
   self.assertEqual(a['rgb_model'].shape,(1008,1008,3));self.assertEqual(a['rgb'].shape,(512,512,3))
   self.assertFalse(a['valid_pixels'][100,100]);self.assertFalse(a['reliable_background'][100,100])
   self.assertTrue(np.array_equal(a['instance_ids'],ids[:512,:512]))
   self.assertTrue(np.array_equal(a['reliable_background'],a['valid_pixels']&(a['instance_ids']==0)))
   self.assertEqual(a['rgb'][30,30].tolist(),[120,120,120])
   self.assertEqual(set(a['object_ids']),{1,2})

class PublicContextContract(unittest.TestCase):
 def test_production_context_is_explicit_in_fresh_process(self):
  import os,sys,subprocess
  env=os.environ.copy();env.pop('GRID01_CONTEXT',None)
  result=subprocess.run([sys.executable,'-B','-c',"import os;import crown_labels.science;assert os.environ['GRID01_CONTEXT']=='box2'"],env=env,capture_output=True,text=True,timeout=30)
  self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()

