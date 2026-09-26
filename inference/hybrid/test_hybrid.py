import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import hybrid_runtime as h

class HybridTests(unittest.TestCase):
    def setUp(self):
        self.sample=pd.DataFrame([[u]+[.5]*12 for u in ['study-a','study-b']],columns=[h.ID,*h.TARGETS])
        self.cnn=self.sample.copy();self.cnn[h.TARGETS]=.8
        self.dino=self.sample.copy();self.dino[h.TARGETS]=.2

    def test_alignment_and_blend(self):
        self.dino.loc[0,'ACL']=.6
        frame=self.dino.iloc[::-1][[h.ID,*reversed(h.TARGETS)]]
        result=h.blend_predictions({'resnet34':self.cnn,'dinov2':frame},self.sample,{'resnet34':.8,'dinov2':.2})
        self.assertEqual(result[h.ID].tolist(),['study-a','study-b'])
        self.assertAlmostEqual(result.loc[0,'ACL'],.76)
        self.assertAlmostEqual(result.loc[1,'ACL'],.68)

    def test_bad_predictions(self):
        for kind in ['duplicate','missing','nan','range','label']:
            f=self.cnn.copy()
            if kind=='duplicate':f.loc[1,h.ID]='study-a'
            elif kind=='missing':f=f.iloc[:1]
            elif kind=='nan':f.loc[0,'ACL']=np.nan
            elif kind=='range':f.loc[0,'ACL']=1.1
            else:f=f.rename(columns={'ACL':'Wrong'})
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.align_predictions(f,self.sample)

    def test_weights_and_zero_component(self):
        for weights in [{'resnet34':.8,'dinov2':.3},{'resnet34':-1,'dinov2':2},{'resnet34':float('nan'),'dinov2':.2}]:
            with self.assertRaises(ValueError):h.check_weights(weights)
        r=h.blend_predictions({'resnet34':self.cnn},self.sample,{'resnet34':1,'dinov2':0})
        np.testing.assert_allclose(r[h.TARGETS],self.cnn[h.TARGETS])

    def exercise(self,fail=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);competition=root/'competition';competition.mkdir()
            self.sample.to_csv(competition/'sample_submission.csv',index=False)
            self.sample[[h.ID]].to_csv(competition/'test.csv',index=False)
            packages={}
            for n in h.HANDLES:
                p=root/n;p.mkdir();(p/'manifest.json').write_text('{}');packages[n]=p
            output=root/'output';output.mkdir();(output/'submission.csv').write_text('stale')
            calls=[]
            def fake_run(command,**kwargs):
                c=json.loads(Path(command[-1]).read_text());calls.append(c['name'])
                self.assertTrue(kwargs['check']);self.assertGreater(kwargs['timeout'],0)
                self.assertIn('/kaggle/working/_dicom_deps',kwargs['env']['PYTHONPATH'])
                if fail and c['name']=='dinov2':raise subprocess.CalledProcessError(1,command)
                dest=Path(c['output']);(self.cnn if c['name']=='resnet34' else self.dino).to_csv(dest/'submission.csv',index=False)
                (dest/'run_manifest.json').write_text(json.dumps({'status':'complete','skipped_series':0}))
            with patch.object(h.subprocess,'run',side_effect=fake_run):
                if fail:
                    with self.assertRaises(subprocess.CalledProcessError):h.run_hybrid(root,root,output,packages=packages,competition=competition)
                    self.assertFalse((output/'submission.csv').exists())
                    self.assertEqual(json.loads((output/'hybrid_run_manifest.json').read_text())['status'],'failed')
                else:
                    frame,receipt=h.run_hybrid(root,root,output,packages=packages,competition=competition)
                    self.assertEqual(receipt['status'],'complete')
                    np.testing.assert_allclose(frame[h.TARGETS],.68)
                    self.assertTrue((output/'submission.csv').exists())
            self.assertEqual(calls,['resnet34','dinov2'])
            self.assertEqual(len(list(output.glob('submission.previous-*.csv'))),1)

    def test_orchestration(self):self.exercise()
    def test_failure_no_stale_submission(self):self.exercise(fail=True)

if __name__=='__main__':unittest.main()
