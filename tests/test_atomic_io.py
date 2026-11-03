"""Exercise temporary and persistent file locks without training a model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pinn_pca.atomic_io import write_json


class AtomicIOTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'status.json';self.path.write_text('{"step":100}')

    def test_transient_lock_is_retried_and_old_status_is_never_truncated(self):
        replace=Path.replace;calls=[]
        def locked(source,target):
            calls.append(source)
            if len(calls)<3:
                self.assertEqual(json.loads(self.path.read_text()),{'step':100})
                raise PermissionError('temporary sharing lock')
            return replace(source,target)
        with patch.object(Path,'replace',locked),patch('pinn_pca.atomic_io.time.sleep') as sleep:
            self.assertTrue(write_json(self.path,{'step':250}))
        self.assertEqual(len(calls),3);self.assertEqual(sleep.call_count,2)
        self.assertEqual(json.loads(self.path.read_text()),{'step':250})
        self.assertEqual(list(self.path.parent.glob('*.tmp')),[])

    def test_persistent_progress_lock_warns_instead_of_stopping_training(self):
        with patch.object(Path,'replace',side_effect=PermissionError('locked')) as replace,patch('pinn_pca.atomic_io.time.sleep'),self.assertWarns(RuntimeWarning):
            self.assertFalse(write_json(self.path,{'step':250},critical=False))
        self.assertEqual(replace.call_count,8)
        self.assertEqual(json.loads(self.path.read_text()),{'step':100})

    def test_completion_markers_remain_critical(self):
        with patch.object(Path,'replace',side_effect=PermissionError('locked')),patch('pinn_pca.atomic_io.time.sleep'):
            with self.assertRaises(PermissionError):write_json(self.path,{'passed':True})
        self.assertEqual(json.loads(self.path.read_text()),{'step':100})

    def test_existing_stale_temporary_does_not_collide(self):
        stale=self.path.with_suffix('.tmp');stale.write_text('stale')
        self.assertTrue(write_json(self.path,{'step':500}))
        self.assertEqual(stale.read_text(),'stale')
        self.assertEqual(json.loads(self.path.read_text()),{'step':500})

    def test_invalid_payload_is_not_silently_ignored(self):
        with self.assertRaises(TypeError):write_json(self.path,{'not_json':object()},critical=False)


if __name__=='__main__':unittest.main()
