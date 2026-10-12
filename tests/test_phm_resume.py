"""Resume recovery uses completed telemetry and readable artifacts, never ML training."""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/phm_exp'))
import phm_resume as resume


class ResumeTests(unittest.TestCase):
    def test_historical_partial_fold_quantile_and_truncated_artifact(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            root = Path(tmp)
            os.environ.update(PHM_MONITOR_DIR=tmp, PHM_RESUME='1')
            roots = [root / name for name in ('models', 'outputs', 'metrics')]
            for fold in (1, 2):
                for tau in (0.25, 0.75):
                    for base in roots:
                        directory = base / f'fold_{fold}' / f'quantile_{tau}'
                        directory.mkdir(parents=True)
                        value = pd.DataFrame({'Eval Loss': [tau]}) if base == roots[2] else {'y_pred': [1], 'y_true': [1]}
                        (directory / 'saved.pickle').write_bytes(pickle.dumps(value))
            events = [{'kind': 'fold_completed', 'fold': 1, 'validation_loss': 0.3},
                      {'kind': 'epoch', 'fold': 2, 'tau': 0.25, 'epoch': 5, 'epochs': 5, 'best_val_loss': 0.2}]
            (root / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)) + '\n{"kind":')
            result = resume.completed_quantiles(*roots, [0.25, 0.75])
            self.assertEqual(set(result), {(1, 0.25), (1, 0.75), (2, 0.25)})
            self.assertEqual(result[(2, 0.25)]['validation_loss'], 0.2)
            (roots[1] / 'fold_2/quantile_0.25/saved.pickle').write_bytes(b'broken')
            self.assertNotIn((2, 0.25), resume.completed_quantiles(*roots, [0.25, 0.75]))
            os.environ['PHM_RESUME'] = '0'
            self.assertEqual(resume.completed_quantiles(*roots, [0.25, 0.75]), {})

    def test_crash_then_resume_preserves_quantiles_and_summary(self):
        import numpy as np
        import pandas as pd
        import cv_training as cv
        from exp_config import ExperimentConfig
        from types import SimpleNamespace

        class Dataset:
            def __len__(self): return 4
            def __getitem__(self, indices): return self

        for fail_at in (2, 3):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
                root = Path(tmp)
                os.environ.update(PHM_MONITOR_DIR=tmp, PHM_RESUME='0')
                paths = {name: str(root / name) for name in ('best_model_path', 'outputs_path', 'combined_outputs_path', 'metrics_path')}
                for folder in paths.values(): Path(folder).mkdir()
                config = ExperimentConfig(n_folds=2, stop_fold_id=2, quantiles=[0.25, 0.75])
                config.compute_metrics = config.save_outputs = True
                config.save_combined_outputs = config.return_outputs = False
                calls = []
                crash = [True]
                def trainer(**kwargs):
                    def run(**kw):
                        calls.append(kwargs['tau'])
                        if crash[0] and len(calls) == fail_at:
                            raise RuntimeError('disk full')
                        (Path(kwargs['best_model_path']) / 'model.pth').write_bytes(b'checkpoint')
                    return SimpleNamespace(best_val_loss=kwargs['tau'], run=run)
                def predictions(**kwargs):
                    (Path(kwargs['outputs_path']) / 'outputs.pickle').write_bytes(pickle.dumps({'y_pred': [1], 'y_true': [1]}))
                def metrics(**kwargs):
                    frame = pd.DataFrame({'Eval Loss': [kwargs['tau'] * 10]}, index=['Life_7'])
                    (Path(kwargs['metrics_path']) / 'metrics.pickle').write_bytes(pickle.dumps(frame))
                    return frame
                dataset = Dataset()
                with patch.multiple(cv, get_raw_phm_data=lambda **kw: (dataset, dataset),
                                    MergeData=lambda **kw: dataset,
                                    load_phm_data=lambda **kw: {'test_idx': np.array([7])},
                                    load_cv_data=lambda **kw: {},
                                    wandb_cv_data=lambda **kw: (None, None, None, object(), None, None, None, None, config),
                                    init_wandb=lambda **kw: None, get_trainer=trainer,
                                    best_model_perf=predictions, lifes_metrics=metrics), patch.object(cv.wandb, 'finish'):
                    with self.assertRaisesRegex(RuntimeError, 'disk full'):
                        cv.train_k_fold(config, SimpleNamespace(), **paths)
                    crash[0] = False
                    os.environ['PHM_RESUME'] = '1'
                    frames, _ = cv.train_k_fold(config, SimpleNamespace(), **paths)
                self.assertEqual(len(calls), 5)  # Four successful runs plus the interrupted attempt.
                self.assertEqual(len(frames), 4)
                events = resume.read_events(root)
                self.assertEqual(events[-1]['table']['values'], [[2.5, 7.5]])
                self.assertEqual(events[-1]['completed_folds'], 2)

    def test_completed_stage_is_skipped_until_training_is_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = [{'kind': 'quantile_started'}, {'kind': 'stage_completed', 'stage': 'quantile_metrics'}]
            path = Path(tmp) / 'events.jsonl'
            path.write_text('\n'.join(map(json.dumps, events)))
            command = [sys.executable, str(ROOT / 'src/phm_resume.py'), 'quantile_metrics']
            env = {**os.environ, 'PHM_MONITOR_DIR': tmp, 'PHM_RESUME': '1'}
            self.assertEqual(subprocess.run(command, env=env).returncode, 0)
            events.append({'kind': 'quantile_started'})
            path.write_text('\n'.join(map(json.dumps, events)))
            self.assertEqual(subprocess.run(command, env=env).returncode, 1)

    def test_new_attempt_separates_a_partial_event_tail(self):
        from phm_monitor import emit
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, PHM_MONITOR_DIR=tmp):
            path = Path(tmp) / 'events.jsonl'
            path.write_text('{"kind":"fold_completed","fold":1}\n{"kind":')
            emit('started', status='running')
            self.assertEqual([e['kind'] for e in resume.read_events(tmp)], ['fold_completed', 'started'])

    def test_campaign_results_paths_honor_overrides_and_relocation(self):
        from phm_paths import campaign_results_root
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            os.environ.pop('PHM_RESULTS_DIR', None)
            directory = Path(tmp)
            legacy = directory / 'legacy'
            self.assertEqual(campaign_results_root(directory, {}, legacy), legacy)
            (directory / 'results').mkdir()
            self.assertEqual(campaign_results_root(directory, {}, legacy), directory / 'results')
            for key in ('results_root', 'environment'):
                recorded = '/old/home/campaign/results'
                manifest = {'environment': {'PHM_MONITOR_DIR': '/old/home/campaign'}}
                if key == 'results_root': manifest[key] = recorded
                else: manifest[key]['PHM_RESULTS_DIR'] = recorded
                self.assertEqual(campaign_results_root(directory, manifest, legacy), directory / 'results')
            manifest = {'results_root': str(directory / 'external')}
            self.assertEqual(campaign_results_root(directory, manifest, legacy), directory / 'external')
            os.environ['PHM_RESULTS_DIR'] = str(directory / 'override')
            self.assertEqual(campaign_results_root(directory, manifest, legacy), directory / 'override')

    def test_completed_standard_campaign_resume_only_regenerates_report(self):
        import phm_campaign as campaign
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            directory = Path(tmp)
            manifest = {'campaign_id': 'finished', 'environment': {'PHM_MONITOR_DIR': tmp}}
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            (directory / 'status.json').write_text('{"status":"completed"}')
            for name in ('experiment.yaml', 'model.yaml', 'shell_config'):
                (directory / name).write_text('{}')
            with patch.object(campaign.signal, 'signal'), \
                 patch.object(campaign.subprocess, 'Popen', return_value=SimpleNamespace(wait=lambda: 0)) as spawn, \
                 patch('phm_dashboard.snapshot_html', return_value='refreshed dashboard'):
                self.assertEqual(campaign.worker(directory, resume=True), 0)
            self.assertEqual(spawn.call_args.args[0][1], str(campaign.HERE / 'phm_campaign_report.py'))
            self.assertEqual((directory / 'dashboard_snapshot.html').read_text(), 'refreshed dashboard')
            self.assertEqual(json.loads((directory / 'status.json').read_text())['status'], 'completed')
            self.assertTrue(any(e['kind'] == 'stage_completed' for e in resume.read_events(directory)))

    def test_worker_lock_excludes_duplicate_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = resume.worker_lock(tmp)
            try:
                with self.assertRaisesRegex(ValueError, 'already running'):
                    resume.worker_lock(tmp)
            finally:
                lock.close()
            resume.worker_lock(tmp).close()

    def test_both_launchers_resume_dry_run_reads_snapshots_without_rebuilding(self):
        for script in ('phm_campaign.py', 'ablation_campaign.py'):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / 'saved'
                directory.mkdir()
                manifest = {'campaign_id': 'saved', 'session': 'test', 'environment': {'PHM_MONITOR_DIR': str(directory)}}
                (directory / 'manifest.json').write_text(json.dumps(manifest))
                for name in ('experiment.yaml', 'model.yaml', 'shell_config'):
                    (directory / name).write_text('snapshot')
                before = {p.name: p.read_bytes() for p in directory.iterdir()}
                result = subprocess.run([sys.executable, str(ROOT / 'experiments/phm_exp' / script),
                                         '--campaign-id=saved', '--artifact-root', tmp, '--resume', '--dry-run'],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_moved_campaign_relocates_runtime_paths_preserving_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            old = '/old/full/home/campaign'
            manifest = {'environment': {'PHM_MONITOR_DIR': old, 'PHM_EXP_CONFIG': old + '/experiment.yaml'},
                        'command': ['python', 'runner', '--config', old + '/experiment.yaml']}
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            (directory / 'experiment.yaml').write_text('model_config_path: ' + old + '/model.yaml\n')
            (directory / 'model.yaml').write_text('test: true')
            (directory / 'shell_config').write_text('snapshot')
            (directory / 'results').mkdir()
            original = (directory / 'experiment.yaml').read_bytes()
            resolved = resume.load_campaign(directory)
            self.assertEqual(resolved['environment']['PHM_RESULTS_DIR'], str(directory / 'results'))
            self.assertEqual(resolved['command'][-1], str(directory / 'resume_experiment.yaml'))
            self.assertFalse((directory / 'resume_experiment.yaml').exists())
            resume.prepare_resume_config(directory)
            self.assertIn(str(directory / 'model.yaml'), (directory / 'resume_experiment.yaml').read_text())
            self.assertEqual((directory / 'experiment.yaml').read_bytes(), original)
            self.assertEqual(json.loads((directory / 'manifest.json').read_text()), manifest)

    def test_real_tmux_resume_keeps_dashboard_and_appends_worker_outcome(self):
        import shutil
        import socket
        import time
        import ablation_campaign as campaign
        from test_ablation_campaign import fixture
        if not shutil.which('tmux'): self.skipTest('tmux unavailable')
        with tempfile.TemporaryDirectory(prefix="resume space '") as tmp:
            directory = Path(tmp)
            manifest = fixture(directory, 'raise SystemExit(7)')
            for name in ('experiment.yaml', 'model.yaml', 'shell_config'):
                (directory / name).write_text('{}')
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            def wait_failures(count):
                for _ in range(100):
                    if sum(e['kind'] == 'failed' for e in resume.read_events(directory)) >= count:
                        return
                    time.sleep(0.05)
                self.fail('Worker did not finish')
            try:
                campaign.launch(directory, manifest, port)
                wait_failures(1)
                # Let the shell/logging pipeline finish after the worker emits failure.
                for _ in range(100):
                    pane = subprocess.run(['tmux', 'display-message', '-p', '-t', manifest['session'] + ':pipeline', '#{pane_dead}'], capture_output=True, text=True)
                    if pane.stdout.strip() == '1': break
                    time.sleep(0.05)
                before = subprocess.check_output(['tmux', 'display-message', '-p', '-t', manifest['session'] + ':dashboard', '#{pane_pid}'])
                campaign.launch(directory, manifest, port, resume=True)
                wait_failures(2)
                after = subprocess.check_output(['tmux', 'display-message', '-p', '-t', manifest['session'] + ':dashboard', '#{pane_pid}'])
                self.assertEqual(before, after)
                self.assertEqual(json.loads((directory / 'manifest.json').read_text()), manifest)
            finally:
                subprocess.run(['tmux', 'kill-session', '-t', manifest['session']], capture_output=True)

    def test_resuming_live_tmux_pipeline_is_rejected(self):
        with patch.object(resume.subprocess, 'run', side_effect=[subprocess.CompletedProcess([], 0),
                                                                  subprocess.CompletedProcess([], 0, stdout='0\n')]):
            with self.assertRaisesRegex(ValueError, 'still active'):
                resume.resume_existing_pipeline('campaign', 'worker command')

    def test_exited_pipeline_is_respawned_without_restarting_monitor(self):
        with patch.object(resume.subprocess, 'run', side_effect=[subprocess.CompletedProcess([], 0),
                                                                  subprocess.CompletedProcess([], 0, stdout='1\n'),
                                                                  subprocess.CompletedProcess([], 0)]) as run:
            self.assertTrue(resume.resume_existing_pipeline('campaign', 'worker command'))
            self.assertEqual(run.call_args.args[0], ['tmux', 'respawn-pane', '-t', 'campaign:pipeline', 'worker command'])


if __name__ == '__main__':
    unittest.main()
