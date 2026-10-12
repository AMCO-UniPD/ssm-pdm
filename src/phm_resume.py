"""Saved work-unit recovery shared by PHM campaign runners."""
import fcntl
import json
import os
import pickle
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def read_events(directory):
    path = Path(directory) / 'events.jsonl'
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A disk-full interruption may leave an incomplete record.
    return events


def worker_lock(directory):
    stream = (Path(directory) / 'worker.lock').open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise ValueError('A campaign worker is already running')
    return stream  # Keep open for the worker lifetime.


def load_campaign(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name in ('experiment.yaml', 'model.yaml', 'shell_config'):
        if not (directory / name).is_file():
            raise ValueError(f'Missing campaign snapshot: {directory / name}')
    recorded = manifest['environment'].get('PHM_MONITOR_DIR')
    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and recorded and (value == recorded or value.startswith(recorded + '/')):
            return str(directory) + value[len(recorded):]
        return value
    manifest = relocate(manifest)
    environment = manifest['environment']
    environment['PHM_MONITOR_DIR'] = str(directory)
    environment['PHM_SHELL_CONFIG'] = str(directory / 'shell_config')
    environment['PHM_EXP_CONFIG'] = str(directory / 'resume_experiment.yaml')
    environment.setdefault('WANDB_DIR', str(directory))
    # Older migrated campaigns have results/ but predate PHM_RESULTS_DIR.
    if (directory / 'results').is_dir():
        environment.setdefault('PHM_RESULTS_DIR', str(directory / 'results'))
    if 'command' in manifest:
        manifest['command'] = [str(directory / 'resume_experiment.yaml')
                               if arg == str(directory / 'experiment.yaml') else arg
                               for arg in manifest['command']]
    return manifest


def prepare_resume_config(directory):
    """Relocate the model reference without modifying the original snapshots."""
    import yaml
    directory = Path(directory)
    config = yaml.safe_load((directory / 'experiment.yaml').read_text())
    config['model_config_path'] = str(directory / 'model.yaml')
    temporary = directory / 'resume_experiment.tmp'
    temporary.write_text(yaml.safe_dump(config))
    temporary.replace(directory / 'resume_experiment.yaml')


def resume_existing_pipeline(session, command):
    """Reuse monitoring when an exited pipeline remains in tmux."""
    if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode:
        return False
    pane = subprocess.run(['tmux', 'display-message', '-p', '-t', session + ':pipeline',
                           '#{pane_dead}'], capture_output=True, text=True)
    if pane.returncode == 0:
        if pane.stdout.strip() != '1':
            raise ValueError('Campaign pipeline is still active; stop it before resuming')
        subprocess.run(['tmux', 'respawn-pane', '-t', session + ':pipeline', command], check=True)
    else:
        subprocess.run(['tmux', 'new-window', '-d', '-t', session, '-n', 'pipeline', command], check=True)
    return True


def completed_quantiles(best_model_path, outputs_path, metrics_path, quantiles,
                        evaluation_quantiles=None):
    """Return verified completed units, including campaigns predating resume."""
    if os.environ.get('PHM_RESUME') != '1' or not os.environ.get('PHM_MONITOR_DIR'):
        return {}
    events = read_events(os.environ['PHM_MONITOR_DIR'])
    setting = next((e.get('setting') for e in reversed(events) if e['kind'] == 'setting_started'), None)
    records = {}
    for event in events:
        if event.get('setting') != setting:
            continue
        if event['kind'] == 'quantile_started':
            records.pop((event['fold'], event['tau']), None)
        elif event['kind'] == 'epoch' and event.get('epoch') == event.get('epochs'):
            records[(event['fold'], event['tau'])] = {**event, 'validation_loss': event['best_val_loss']}
        elif event['kind'] == 'quantile_completed':
            records[(event['fold'], event['tau'])] = event
        elif event['kind'] == 'fold_completed':
            for tau in quantiles:
                records.setdefault((event['fold'], tau), {**event, "historical_fold": True})
    verified = {}
    for (fold, tau), event in records.items():
        folders = [Path(best_model_path) / f'fold_{fold}' / f'quantile_{tau}',
                   Path(metrics_path) / f'fold_{fold}' / f'quantile_{tau}']
        folders.extend(Path(outputs_path) / f'fold_{fold}' / f'quantile_{level}'
                       for level in dict.fromkeys([tau, *(evaluation_quantiles or [])]))
        if all(folder.is_dir() and any(p.is_file() and p.suffix in ('.pickle', '.pth')
                                     and p.stat().st_size > 0 for p in folder.iterdir())
               for folder in folders):
            try:
                if event['kind'] == 'epoch' and event.get('time'):
                    finished = datetime.fromisoformat(event['time']).timestamp()
                    if any(max(p.stat().st_mtime for p in folder.iterdir() if p.is_file()) < finished
                           for folder in folders):
                        continue
                # Detect truncated artifacts left by a disk-full write.
                for folder in folders[1:]:
                    latest = max(folder.glob('*.pickle'), key=lambda p: p.stat().st_mtime)
                    with latest.open('rb') as stream:
                        value = pickle.load(stream)
                    if folder == folders[1]:
                        if 'Eval Loss' not in value.columns:
                            raise ValueError('Missing saved metrics')
                    elif not {'y_pred', 'y_true'} <= value.keys():
                        raise ValueError('Missing saved predictions')
            except (OSError, EOFError, pickle.UnpicklingError, ValueError, AttributeError):
                continue
            verified[(fold, tau)] = event
    # A historical fold mean cannot stand in for an individual quantile loss
    # when only part of that fold survives. Retry those unknown units together.
    for (fold, tau), event in list(verified.items()):
        if event.get('historical_fold') and any((fold, level) not in verified for level in quantiles):
            verified.pop((fold, tau))
    return verified


if __name__ == '__main__':
    from phm_monitor import emit
    stage = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == '--complete':
        emit('stage_completed', stage=stage)
    else:
        events = read_events(os.environ['PHM_MONITOR_DIR']) if os.environ.get('PHM_MONITOR_DIR') else []
        last_training = max((i for i, e in enumerate(events) if e['kind'] == 'quantile_started'), default=-1)
        done = os.environ.get('PHM_RESUME') == '1' and any(
            i > last_training and e['kind'] == 'stage_completed' and e.get('stage') == stage
            for i, e in enumerate(events))
        sys.exit(0 if done else 1)
