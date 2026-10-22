import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import common
from verify_publication import bounds, flag


class PublicationTests(unittest.TestCase):
    def test_signed_changes_keep_losses_negative(self):
        ci,_=bounds(np.array([[-1],[-1],[0]],dtype=np.int8),100,42)
        self.assertLess(ci[0,0],0)
        self.assertLessEqual(ci[1,0],0)
        with self.assertRaises(ValueError):
            flag('0')

    def test_runtime_paths_are_relative_to_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            common.write_json(root/'runtime.json',{'cot_root':'evidence/cot','direct_answer_root':None,'analysis_python':None,'gpu_lock':'.lock'})
            with patch.object(common,'ROOT',root):
                cfg=common.runtime()
                self.assertEqual(cfg['cot_root'],str(root/'evidence/cot'))
                self.assertEqual(cfg['gpu_lock'],str(root/'.lock'))
                self.assertIsNone(cfg['analysis_python'])
                self.assertEqual(common.local_path('models/base'),root/'models/base')
