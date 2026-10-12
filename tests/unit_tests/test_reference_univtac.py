"""Offline checks for environment fidelity and native output routing."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.univtac.reference_runtime import prepare_runtime, reference_root, verify_reference
from scripts.univtac.eval_policy_native import adapt_source
from scripts.univtac import eval_simpletac_managed as launcher


class ReferenceRuntimeTest(unittest.TestCase):
    def test_all_reference_hashes_match(self):
        manifest = verify_reference(reference_root())
        self.assertEqual(manifest['evaluation_environment']['livestream'], 2)
        self.assertEqual(manifest['asset_file_count'], 67)

    def test_native_action_and_seed_loop_are_unchanged(self):
        source = (reference_root() / 'scripts/eval_policy.py').read_text()
        def function(text, name):
            return ast.dump(next(n for n in ast.parse(text).body
                                 if isinstance(n, ast.FunctionDef) and n.name == name))
        adapted = adapt_source(source)
        self.assertEqual(function(source, 'eval_policy'), function(adapted, 'eval_policy'))
        self.assertIn('env_cfg.worker_name = _managed_worker_name', adapted)
        self.assertIn('run_root / "workers" / _managed_worker_name', adapted)
        for bad in [source.replace('worker_name(0)', 'worker_name(1)'),
                    source.replace('run_root / "log.log"', 'other / "log.log"')]:
            with self.assertRaises(RuntimeError):
                adapt_source(bad)

    def test_prepare_rejects_changed_assets_before_writing_target(self):
        with tempfile.TemporaryDirectory() as temp:
            base, target = Path(temp) / 'base', Path(temp) / 'runtime'
            base.mkdir()
            asset = next(iter(verify_reference(reference_root())['checked_assets']))
            (base / asset).parent.mkdir(parents=True)
            (base / asset).write_bytes(b'wrong')
            with self.assertRaisesRegex(RuntimeError, 'asset differs'):
                prepare_runtime(base, target)
            self.assertFalse(target.exists())

    def test_prepared_runtime_is_private_and_detects_later_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            base, ref, target = temp / 'base', temp / 'reference', temp / 'runtime'
            for folder in ['assets', 'third_party', 'encoder', 'policy/SimpleTac']:
                (base / folder).mkdir(parents=True)
            (base / 'assets/object.usd').write_bytes(b'asset')
            files = {'envs/task.py': b'reference environment',
                     'scripts/eval_policy.py': b'args_cli.livestream = int(os.environ.get("UNIVTAC_LIVESTREAM", "2"))\n'}
            for name, value in files.items():
                (ref / name).parent.mkdir(parents=True, exist_ok=True)
                (ref / name).write_bytes(value)
            manifest = {'files': {n: hashlib.sha256(v).hexdigest() for n, v in files.items()},
                        'checked_assets': {'assets/object.usd': hashlib.sha256(b'asset').hexdigest()},
                        'evaluation_environment': {'livestream': 2}}
            (ref / 'manifest.json').write_text(json.dumps(manifest))
            prepare_runtime(base, target, temp / 'kit', reference=ref)
            self.assertFalse((target / 'assets').is_symlink())
            self.assertFalse((target / 'policy/SimpleTac').exists())
            self.assertEqual((target / 'envs/task.py').read_bytes(), files['envs/task.py'])
            prepare_runtime(base, target, temp / 'kit', reference=ref)
            (target / 'envs/task.py').write_bytes(b'unexpected mutation')
            with self.assertRaisesRegex(RuntimeError, 'runtime source changed'):
                prepare_runtime(base, target, reference=ref)
            self.assertEqual((base / 'assets/object.usd').read_bytes(), b'asset')


class ManagedReferenceContractTest(unittest.TestCase):
    def test_plan_records_actual_reference_source_and_rejects_environment_change(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            weights, upstream = temp / 'weights', temp / 'public'
            weights.mkdir()
            for name in ['config.yaml', '_SUCCESS']:
                (weights / name).write_text('checkpoint identity')
            (upstream / 'scripts').mkdir(parents=True)
            (upstream / 'scripts/eval_policy.py').write_text('public source')
            for name in ['envs/_base_task.py', 'envs/lift_bottle.py', 'task_config/demo.yml']:
                (upstream / name).parent.mkdir(parents=True, exist_ok=True)
                (upstream / name).write_text('public configuration')
            args = launcher.parser().parse_args([
                '--checkpoint', str(weights), '--task', 'lift_bottle',
                '--result-dir', str(temp / 'result')])
            with mock.patch.object(launcher, 'UNIVTAC', upstream), mock.patch.dict(
                os.environ, {'UNIVTAC_ENVIRONMENT': 'reference'}):
                root, contract, _, _ = launcher.resolve_plan(args)
                self.assertEqual(contract['environment_profile'], 'reference')
                self.assertIn(str(reference_root() / 'envs/_base_task.py'),
                              contract['evaluation_code_sha256'])
                self.assertNotIn(str(upstream / 'scripts/eval_policy.py'),
                                 contract['evaluation_code_sha256'])
                self.assertIn('reference_manifest_sha256', contract)
                root.mkdir()
                (root / '.simpletac_eval.json').write_text(json.dumps({'contract': contract}))
                resume = launcher.parser().parse_args([
                    '--resume-dir', str(root), '--resume-from-episode', '1000000'])
                self.assertEqual(launcher.resolve_plan(resume)[1], contract)
                os.environ['UNIVTAC_ENVIRONMENT'] = 'public'
                with self.assertRaisesRegex(ValueError, 'contract'):
                    launcher.resolve_plan(resume)
                os.environ['UNIVTAC_ENVIRONMENT'] = 'reference'
                os.environ['UNIVTAC_LIVESTREAM'] = '0'
                try:
                    with self.assertRaisesRegex(ValueError, 'LIVESTREAM=2'):
                        launcher.resolve_plan(resume)
                finally:
                    os.environ.pop('UNIVTAC_LIVESTREAM', None)

    def test_worker_clears_stale_driver_and_keeps_reference_contract(self):
        contract = {'checkpoint': '/weights', 'task': 'lift_bottle', 'task_config': 'demo',
                    'horizon': 50, 'cpu_threads': 16, 'use_tactile': 'auto',
                    'livestream': '2', 'environment_profile': 'reference'}
        worker = {'gpu': 1, 'log_dir': '/logs', 'seeds': [1000000]}
        with mock.patch.dict(os.environ, {'UNIVTAC_REFERENCE_READY': 'true',
                                          'UNIVTAC_EVAL_DRIVER': 'native'}):
            env = launcher.worker_env(contract, worker, Path('/plan.json'), 7)
        self.assertEqual(env['UNIVTAC_ENVIRONMENT'], 'reference')
        self.assertEqual(env['UNIVTAC_EVAL_DRIVER'], 'simpletouch')
        self.assertNotIn('UNIVTAC_REFERENCE_READY', env)


if __name__ == '__main__':
    unittest.main()
