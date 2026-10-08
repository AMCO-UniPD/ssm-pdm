"""Launch an isolated window-weight ablation with a persistent tmux dashboard."""
import argparse
import json
import hashlib
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from phm_monitor import emit
from phm_campaign import settings
from ablation_dashboard import export_snapshot


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def worker(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    env = {**os.environ, **manifest['environment']}
    os.environ['PHM_MONITOR_DIR'] = str(directory)
    process = None
    cancelled = False
    stopped = threading.Event()
    def heartbeat():
        while not stopped.is_set():
            atomic_json(directory / 'heartbeat.json', {'unix_time': time.time()})
            stopped.wait(15)
    def interrupted(signum, frame):
        nonlocal cancelled
        cancelled = True
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
    beat = threading.Thread(target=heartbeat, daemon=True)
    try:
        emit('started', status='running', stage='starting', pid=os.getpid())
        beat.start()
        process = subprocess.Popen(manifest['command'], cwd=manifest['cwd'], env=env, start_new_session=True)
        while True:
            if cancelled:
                raise KeyboardInterrupt
            try:
                code = process.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
        if cancelled:
            raise KeyboardInterrupt
        emit('completed' if code == 0 else 'failed', stage='finished' if code == 0 else 'failed', exit_code=code)
        return code
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        emit('interrupted', stage='interrupted', exit_code=130)
        return 130
    except Exception as error:
        emit('failed', stage='failed', error=str(error), exit_code=1)
        return 1
    finally:
        stopped.set()
        if beat.ident is not None:
            beat.join(timeout=2)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # The dashboard also exports periodically; use a distinct temporary file.
        try:
            target = directory / 'final_snapshot.html'
            export_snapshot(directory, target)
            target.replace(directory / 'dashboard_snapshot.html')
        except Exception as error:
            print(f'Final snapshot export failed; existing snapshot retained: {error}', flush=True)


def build_plan(args):
    import yaml
    import torch
    from exp_config import ExperimentConfig, ModelConfig, check_arguments
    shell = settings(args.shell_config.resolve())
    raw = yaml.safe_load(args.config.read_text())
    model_path = Path(raw.get('model_config_path', 'config/ssm_config.yaml'))
    if not model_path.is_absolute():
        model_path = HERE / model_path
    model_raw = yaml.safe_load(model_path.read_text())
    if args.head == 'scale':
        raw['quantile_scale'] = True
        model_raw.update(tau_feat=False, tau_mult=False)
    elif args.head == 'feature':
        raw['quantile_scale'] = False
        model_raw.update(tau_feat=True, tau_mult=False)
    config = ExperimentConfig.from_dict(raw)
    model = ModelConfig.from_dict(model_raw)
    for key in ('model_name', 'failure_type', 'train_phm_tools', 'test_phm_tools'):
        setattr(config, key, shell[key])
    config.data_name = 'PHM'
    config.quantiles = [0.5]
    config.quantile_run = 0.5
    config.use_wandb = args.wandb
    config.resume_training = False
    for key in ('save_best_model', 'save_outputs', 'compute_metrics', 'save_metrics_df'):
        setattr(config, key, True)
    for key in ('save_combined_outputs', 'return_outputs', 'get_test_idx'):
        setattr(config, key, False)
    check_arguments(config)
    if not config.cv or config.n_folds < 2 or config.start_fold_id != 0 or config.stop_fold_id != config.n_folds:
        raise ValueError('Ablation campaigns require all configured CV folds, with at least two folds')
    if config.loss != 'window_quantile_reg' or not config.quantile_reg or config.monotonic:
        raise ValueError('Ablation requires non-monotonic windowed SQR training')
    if config.transformer_type not in (1, 2, 5) or config.max_rul != 500:
        raise ValueError('Regional metrics require the PHM RUL target clipped at 500')
    if config.quantile_scale and (model.tau_feat or model.tau_mult):
        raise ValueError('Scale head requires tau_feat=false and tau_mult=false; use --head scale')
    if not config.quantile_scale and not (model.tau_feat or model.tau_mult):
        raise ValueError('QuantileHead needs tau_feat or tau_mult; use --head feature')
    ratios = []
    for item in args.ratios:
        ratio = None if item == 'unweighted' else float(item)
        if ratio is not None and (not math.isfinite(ratio) or ratio <= 0):
            raise ValueError('Weight ratios must be finite and positive')
        label = 'unweighted' if ratio is None else f'{ratio:g}'
        if label not in ratios:
            ratios.append(label)
    directory = (args.artifact_root.expanduser() / args.campaign_id).resolve()
    config.model_config_path = str(directory / 'model.yaml')
    # Each ratio overrides this during the run; do not misrepresent a fixed ratio.
    config.window_weight_ratio = None
    model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    environment = {'PHM_CAMPAIGN_ID': args.campaign_id, 'PHM_MONITOR_DIR': str(directory),
                   'PHM_RESULTS_DIR': str(directory / 'results'), 'WANDB_DIR': str(directory / 'wandb'),
                   'CUDA_VISIBLE_DEVICES': str(shell['device_num']), 'PYTHONUNBUFFERED': '1',
                   'MPLBACKEND': 'Agg', 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']}
    for key in ('WANDB_ENTITY', 'WANDB_BASE_URL'):
        if key in os.environ:
            environment[key] = os.environ[key]
    command = [sys.executable, str(HERE / 'phm_weight_ablation.py'), '--config', str(directory / 'experiment.yaml'),
               '--model-name', shell['model_name'], '--failure-type', shell['failure_type'],
               '--train-phm-tools', *shell['train_phm_tools'], '--test-phm-tools', *shell['test_phm_tools'],
               '--device-num', '0', '--seed', str(args.seed), '--ratios', *ratios,
               '--results-dir', str(directory / 'ablation_results')]
    if args.wandb:
        command.append('--wandb')
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    manifest = {'campaign_id': args.campaign_id, 'created_at': datetime.now(timezone.utc).isoformat(),
                'session': 'phm-ablation-' + args.campaign_id, 'port': args.port, 'cwd': str(HERE),
                'repository': str(ROOT), 'revision': revision, 'dirty_checkout': bool(dirty),
                'config': vars(config), 'model_config': vars(model), 'model_snapshot': model_raw, 'settings': shell,
                'ratios': ratios, 'seed': args.seed, 'evaluation_quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
                'planned_work_units': len(ratios) * config.n_folds, 'wandb_enabled': args.wandb,
                'environment': environment, 'command': command, 'python': sys.executable,
                'selection': 'Minimum validation loss at tau=0.5 within each ratio/fold; all folds evaluated',
                'aggregation': 'Equal-weight mean across lives, then mean and sample SD across evaluated folds'}
    manifest["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), HERE / "ablation_dashboard.py", HERE / "phm_weight_ablation.py", ROOT / "src/model_classes.py", ROOT / "src/cv_training.py", ROOT / "src/trainer.py"]}
    return directory, json.loads(json.dumps(manifest, default=str))


def launch(directory, manifest, port, tunnel=False, dashboard_only=False):
    session = manifest['session'] + ('-monitor' if dashboard_only else '')
    if not shutil.which('tmux'):
        raise ValueError('tmux is required')
    cloudflared = shutil.which(os.environ.get('CLOUDFLARED_BIN', 'cloudflared')) if tunnel else None
    if tunnel and not cloudflared:
        raise ValueError('--tunnel requires cloudflared')
    if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
        raise ValueError(f'tmux session already exists: {session}')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    def tmux(*argv):
        return subprocess.run(['tmux', *argv], check=True, capture_output=True, text=True)
    def window(name, argv, log):
        pane = tmux('new-window', '-d', '-P', '-F', '#{window_id}', '-t', session, '-n', name, 'sleep 120').stdout.strip()
        tmux('set-window-option', '-t', pane, 'remain-on-exit', 'on')
        command = shlex.join(argv) + ' 2>&1 | tee -a ' + shlex.quote(str(log))
        tmux('respawn-window', '-k', '-t', pane, shlex.join(['bash', '-o', 'pipefail', '-c', command]))
    created = False
    try:
        tmux('new-session', '-d', '-s', session, '-n', 'control', 'sleep 120')
        created = True
        window('dashboard', [sys.executable, str(HERE / 'ablation_dashboard.py'), '--directory', str(directory), '--port', str(port)], directory / 'dashboard.log')
        for _ in range(50):
            try:
                with urlopen(f'http://127.0.0.1:{port}/api/state', timeout=1) as response:
                    if json.load(response)['manifest']['campaign_id'] == manifest['campaign_id']:
                        break
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError('Dashboard readiness failed; inspect dashboard.log')
        url = f'http://127.0.0.1:{port}'
        if tunnel:
            window('tunnel', [cloudflared, 'tunnel', '--no-autoupdate', '--url', url], directory / 'cloudflared.log')
            for _ in range(100):
                log = directory / 'cloudflared.log'
                contents = log.read_text(errors='replace') if log.exists() else ''
                matches = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', contents)
                if matches and 'Registered tunnel connection' in contents:
                    url = matches[-1]; break
                time.sleep(0.5)
            else:
                raise RuntimeError('Tunnel readiness failed; inspect cloudflared.log')
        (directory / 'dashboard_url.txt').write_text(url + '\n')
        if not dashboard_only:
            window('pipeline', [sys.executable, str(Path(__file__).resolve()), '--worker', str(directory)], directory / 'runner.log')
        window('logs', ['tail', '-n', '80', '-F', str(directory / 'runner.log')], directory / 'log-tail.log')
        tmux('kill-window', '-t', session + ':control')
    except BaseException:
        if created:
            subprocess.run(['tmux', 'kill-session', '-t', session], capture_output=True)
        raise
    print(f'Dashboard: {url}\nArtifacts: {directory}\nLogs: {directory / "runner.log"}\nAttach: tmux attach -t {session}\nStop all services (also stops training): tmux kill-session -t {session}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-id', default='window_weights_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--artifact-root', type=Path, default=Path(os.environ.get('PHM_ARTIFACT_ROOT', '/mnt/disk1/davide_frizzo/experiments/ssm-pdm')))
    parser.add_argument('--config', type=Path, default=HERE / 'config/ssm_exp_config.yaml')
    parser.add_argument('--shell-config', type=Path, default=HERE / 'phm_exp_config')
    parser.add_argument('--head', choices=['configured', 'scale', 'feature'], default='configured', help='Use current config, or override the campaign snapshot only')
    parser.add_argument('--ratios', nargs='+', default=['unweighted', '0.25', '0.5', '1', '2', '4'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true', help='Enable W&B; disabled by default as in the ablation runner')
    parser.add_argument('--port', type=int, default=8879)
    parser.add_argument('--tunnel', action='store_true', help='Explicitly expose the dashboard through a public Cloudflare tunnel')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved plan without creating files, sessions, or processes')
    parser.add_argument('--dashboard-only', type=Path, help='Restart monitoring from an existing campaign; does not train')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.campaign_id) or not 1024 <= args.port <= 65535:
        parser.error('Invalid campaign ID or port (expected 1024–65535)')
    try:
        if args.dashboard_only:
            directory = args.dashboard_only.resolve()
            manifest = json.loads((directory / 'manifest.json').read_text())
            if args.dry_run:
                print(json.dumps({'directory': str(directory), 'port': args.port, 'mode': 'dashboard only'}, indent=2)); return 0
            launch(directory, manifest, args.port, args.tunnel, dashboard_only=True); return 0
        directory, manifest = build_plan(args)
        print(json.dumps(manifest, indent=2), flush=True)
        if args.dry_run:
            return 0
        if directory.exists():
            raise ValueError(f'Campaign directory already exists: {directory}')
        # Validate session, executables and bind before creating campaign artifacts.
        if not shutil.which('tmux'):
            raise ValueError('tmux is required')
        if args.tunnel and not shutil.which(os.environ.get('CLOUDFLARED_BIN', 'cloudflared')):
            raise ValueError('--tunnel requires cloudflared')
        if subprocess.run(['tmux', 'has-session', '-t', manifest['session']], capture_output=True).returncode == 0:
            raise ValueError('tmux session already exists: ' + manifest['session'])
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', args.port))
        import yaml
        directory.mkdir(parents=True)
        for name in ('results', 'ablation_results', 'wandb'):
            (directory / name).mkdir()
        atomic_json(directory / 'manifest.json', manifest)
        atomic_json(directory / 'status.json', {'status': 'waiting'})
        (directory / 'experiment.yaml').write_text(yaml.safe_dump(manifest['config']))
        (directory / 'model.yaml').write_text(yaml.safe_dump(manifest['model_snapshot']))
        (directory / 'shell_config').write_text(args.shell_config.read_text())
        patch = subprocess.check_output(['git', '-C', str(ROOT), 'diff', 'HEAD'], text=True)
        (directory / 'source_changes.patch').write_text(patch)
        export_snapshot(directory)
        try:
            launch(directory, manifest, args.port, args.tunnel)
        except BaseException as error:
            os.environ['PHM_MONITOR_DIR'] = str(directory)
            emit('failed', stage='startup', error=str(error), exit_code=1)
            export_snapshot(directory)
            raise
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    sys.exit(main())
