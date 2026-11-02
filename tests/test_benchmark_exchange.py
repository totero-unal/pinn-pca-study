"""Archive integrity and merge tests using synthetic files, never PINN training."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/exchange_benchmark_results.py'
spec=importlib.util.spec_from_file_location('exchange_test',SCRIPT)
exchange=importlib.util.module_from_spec(spec);spec.loader.exec_module(exchange)


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.sender=self.base/'sender';self.receiver=self.base/'receiver'
        self.config=dict(run_id='fixture',problems=['burgers','wave','allen_cahn'])
        for root in (self.sender,self.receiver):
            p=root/'pca_collocation/configs/benchmark_full.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(self.config))
        self.out=self.sender/'pca_collocation/results/benchmark/fixture'
        self.out.mkdir(parents=True);(self.out/'config.json').write_text(json.dumps(self.config))
        ref=self.out/'references';ref.mkdir();(ref/'burgers.npz').write_bytes(b'fixture-reference')
        (ref/'burgers_audit.json').write_text('{"passed": true}')
        problem=self.out/'burgers';problem.mkdir()
        (problem/'environment.json').write_text('{"code_sha256":"fixture-code"}')
        (problem/'report_complete.json').write_text('{"passed":true,"completed_runs":50}')
        for method in range(5):
            for seed in range(10):
                p=problem/f'method_{method}'/f'seed_{seed}';p.mkdir(parents=True)
                (p/'completed.json').write_text('{"passed":true}')
        self.archive=self.base/'burgers.zip'

    def call(self,root,args):
        with patch.object(exchange,'ROOT',root),patch.object(exchange,'CONFIG',root/'pca_collocation/configs/benchmark_full.json'),patch.object(exchange,'code_digest',return_value='fixture-code'),patch.object(sys,'argv',['exchange',*args]),contextlib.redirect_stdout(io.StringIO()):
            exchange.main()

    def export(self):self.call(self.sender,['export','--problem','burgers','--output',str(self.archive)])

    def test_roundtrip_and_idempotent_import(self):
        self.export()
        for _ in range(2):self.call(self.receiver,['import',str(self.archive)])
        result=self.receiver/'pca_collocation/results/benchmark/fixture'
        self.assertEqual((result/'references/burgers.npz').read_bytes(),b'fixture-reference')
        self.assertEqual(len(list(result.glob('burgers/*/seed_*/completed.json'))),50)

    def test_conflict_is_detected_before_any_import(self):
        self.export()
        result=self.receiver/'pca_collocation/results/benchmark/fixture'
        conflict=result/'burgers/environment.json';conflict.parent.mkdir(parents=True);conflict.write_text('conflict')
        with self.assertRaises(FileExistsError):self.call(self.receiver,['import',str(self.archive)])
        self.assertFalse((result/'config.json').exists())

    def test_semantically_identical_configuration_formats_can_merge(self):
        self.export()
        target=self.receiver/'pca_collocation/results/benchmark/fixture/config.json'
        target.parent.mkdir(parents=True);target.write_text(json.dumps(self.config,indent=4))
        self.call(self.receiver,['import',str(self.archive)])
        self.assertEqual(json.loads(target.read_text()),self.config)

    def test_corruption_is_rejected_before_any_import(self):
        self.export();bad=self.base/'bad.zip'
        with zipfile.ZipFile(self.archive) as original,zipfile.ZipFile(bad,'w') as corrupted:
            for name in original.namelist():
                data=original.read(name)
                corrupted.writestr(name,b'corrupt' if name=='references/burgers.npz' else data)
        with self.assertRaisesRegex(ValueError,'Checksum failed'):self.call(self.receiver,['import',str(bad)])
        self.assertFalse((self.receiver/'pca_collocation/results/benchmark/fixture').exists())


if __name__=='__main__':unittest.main()
