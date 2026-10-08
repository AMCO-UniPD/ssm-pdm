"""Cheap lifecycle and saved-artifact tests, without starting ML training."""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/phm_exp'))
import ablation_campaign as campaign
import ablation_dashboard as dashboard

PNG = __import__('base64').b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1kAAAAASUVORK5CYII=')


def fixture(directory, code='pass'):
    (directory / 'ablation_results').mkdir()
    manifest = {'campaign_id': 'fixture_' + uuid.uuid4().hex[:8], 'created_at': '2026-10-08T10:00:00Z',
                'session': 'ablation-test-' + uuid.uuid4().hex[:8], 'port': 0, 'config': {'n_folds': 2, 'epochs': 2},
                'model_config': {}, 'settings': {}, 'planned_work_units': 4, 'ratios': ['unweighted', '1'],
                'environment': {'PHM_MONITOR_DIR': str(directory)}, 'wandb_enabled': False,
                'command': [sys.executable, '-c', code], 'cwd': str(directory)}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    (directory / 'status.json').write_text(json.dumps({'status': 'waiting'}))
    return manifest


class DashboardTests(unittest.TestCase):
    def test_partial_results_nonfinite_and_offline_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixture(directory)
            (directory / 'events.jsonl').write_text(json.dumps({'kind': 'epoch', 'setting': 'ratio_1', 'fold': 1, 'epoch': 1, 'train_loss': 0.2})+'\n{"kind":')
            (directory / 'ablation_results/summary_rmse.csv').write_text('setting,mean_overall_rmse,std_overall_rmse\nratio_1,2,nan\n')
            (directory / 'ablation_results/overall_rmse_by_weight.png').write_bytes(PNG)
            (directory / 'runner.log').write_text('</script><script>unsafe</script>')
            data = dashboard.payload(directory)
            self.assertEqual(len(data['events']), 1)
            self.assertEqual(data['tables']['summary_rmse'][0]['mean_overall_rmse'], 2)
            self.assertIsNone(data['tables']['summary_rmse'][0]['std_overall_rmse'])
            target = dashboard.export_snapshot(directory)
            html = target.read_text()
            saved = html.split('<script id="saved-ablation" type="application/json">')[1].split('</script>')[0]
            self.assertNotIn('</script>', saved)
            embedded = json.loads(saved)
            self.assertEqual(embedded['log'], '</script><script>unsafe</script>')
            self.assertIn('data:image/png;base64,', embedded['plot_assets']['overall_rmse_by_weight.png'])
            self.assertIn('href="data:image/svg+xml;base64,', html)
            self.assertIn('if(offline)render(offline);else refresh()', html)

    def test_stale_is_observation_not_terminal_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); fixture(directory)
            (directory / 'status.json').write_text('{"status":"running"}')
            self.assertIn('stale', dashboard.payload(directory)['observed_status'])
            (directory / 'heartbeat.json').write_text(json.dumps({'unix_time': time.time()}))
            self.assertEqual(dashboard.payload(directory)['observed_status'], 'running')
            (directory / 'status.json').write_text('{"status":"completed"}')
            self.assertEqual(dashboard.payload(directory)['observed_status'], 'completed')

    def test_read_only_routes_favicon_and_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); fixture(directory); dashboard.export_snapshot(directory)
            (directory / 'secret.json').write_text('private')
            (directory / 'ablation_results/fold_rmse.csv').write_text('setting,fold\nratio_1,1\n')
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(dashboard.Handler, directory=directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                base = f'http://127.0.0.1:{server.server_port}'
                for path, mime in [('/api/state', 'application/json'), ('/favicon.svg', 'image/svg+xml'), ('/snapshot.html', 'text/html'), ('/results/fold_rmse.csv', 'text/csv')]:
                    with urlopen(base + path) as response:
                        self.assertEqual(response.headers.get_content_type(), mime)
                        self.assertGreater(len(response.read()), 0)
                for path in ('/secret.json', '/results/%2e%2e/secret.json', '/results/../../manifest.json'):
                    with self.assertRaises(HTTPError) as error: urlopen(base + path)
                    self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown(); server.server_close(); thread.join()

    def test_worker_success_and_failure_are_recorded_with_final_snapshot(self):
        for code, expected in [('pass', 0), ('raise SystemExit(7)', 7)]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
                directory = Path(tmp); fixture(directory, code)
                self.assertEqual(campaign.worker(directory), expected)
                status = json.loads((directory / 'status.json').read_text())
                self.assertEqual(status['status'], 'completed' if expected == 0 else 'failed')
                self.assertEqual(status['exit_code'], expected)
                self.assertTrue((directory / 'dashboard_snapshot.html').exists())

    def test_interrupt_stops_own_worker_and_records_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); fixture(directory, 'import time; time.sleep(60)')
            process = subprocess.Popen([sys.executable, str(ROOT / 'experiments/phm_exp/ablation_campaign.py'), '--worker', str(directory)])
            try:
                for _ in range(100):
                    if json.loads((directory / 'status.json').read_text()).get('status') == 'running': break
                    time.sleep(0.05)
                process.terminate()
                self.assertEqual(process.wait(timeout=15), 130)
                self.assertEqual(json.loads((directory / 'status.json').read_text())['status'], 'interrupted')
            finally:
                if process.poll() is None: process.kill(); process.wait()

    def test_tmux_lifecycle_spaces_failure_collision_and_monitor_restart(self):
        import shutil
        if not shutil.which('tmux'): self.skipTest('tmux unavailable')
        with tempfile.TemporaryDirectory(prefix="ablation space '") as tmp:
            directory = Path(tmp); manifest = fixture(directory, 'raise SystemExit(7)')
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            try:
                campaign.launch(directory, manifest, port)
                with self.assertRaises(ValueError): campaign.launch(directory, manifest, port)
                for _ in range(100):
                    state = json.loads((directory / 'status.json').read_text())
                    if state.get('status') == 'failed': break
                    time.sleep(0.05)
                self.assertEqual(state['exit_code'], 7)
                with urlopen(f'http://127.0.0.1:{port}/api/state') as response:
                    self.assertEqual(json.load(response)['state']['status'], 'failed')
            finally:
                subprocess.run(['tmux', 'kill-session', '-t', manifest['session']], capture_output=True)
            for _ in range(50):
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        probe.bind(('127.0.0.1', port)); break
                    except OSError:
                        time.sleep(0.05)
            try:
                campaign.launch(directory, manifest, port, dashboard_only=True)
                self.assertEqual(json.loads((directory / 'status.json').read_text())['exit_code'], 7)
            finally:
                subprocess.run(['tmux', 'kill-session', '-t', manifest['session']+'-monitor'], capture_output=True)


if __name__ == '__main__':
    unittest.main()
