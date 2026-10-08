"""Launch run_phm_exp and its read-only dashboard in persistent tmux windows."""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from phm_monitor import emit


def worker(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    env = {**os.environ, **manifest["environment"]}
    os.environ["PHM_MONITOR_DIR"] = str(directory)
    process = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    emit("started", status="running", stage="starting", pid=os.getpid())
    try:
        process = subprocess.Popen(["bash", str(HERE / "run_phm_exp")], cwd=HERE,
                                   env=env, start_new_session=True)
        code = process.wait()
        emit("completed" if code == 0 else "failed", exit_code=code)
        return code
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        emit("interrupted", exit_code=130)
        return 130
    except Exception as error:
        emit("failed", error=str(error), exit_code=1)
        raise


def settings(shell_config):
    # Source the same trusted shell configuration used by the pipeline launchers.
    code = "import json,sys; names=['project_name','data_name','model_name','failure_type','device_num','wandb_entity']; d=dict(zip(names,sys.argv[1:7])); i=sys.argv.index('--'); d['train_phm_tools']=sys.argv[7:i]; d['test_phm_tools']=sys.argv[i+1:]; print(json.dumps(d))"
    command = 'source "$1"; "$2" -c "$3" "$project_name" "$data_name" "$model_name" "$failure_type" "$device_num" "$wandb_entity" "${train_phm_tools[@]}" -- "${test_phm_tools[@]}"'
    result = subprocess.run(["bash", "-eu", "-c", command, "phm-settings", str(shell_config),
                             sys.executable, code], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="S4D_tau_fixed_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--artifact-root", type=Path, default=HERE / "campaigns")
    parser.add_argument("--port", type=int, default=8878)
    parser.add_argument("--exp-config", type=Path, default=HERE / "config/ssm_exp_config.yaml")
    parser.add_argument("--shell-config", type=Path, default=HERE / "phm_exp_config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.campaign_id):
        parser.error("campaign ID must contain only letters, numbers, underscores and hyphens")
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    import yaml
    from exp_config import ExperimentConfig, ModelConfig
    shell = settings(args.shell_config.resolve())
    raw = yaml.safe_load(args.exp_config.read_text())
    model_path = Path(raw.get("model_config_path", "config/ssm_config.yaml"))
    if not model_path.is_absolute():
        model_path = HERE / model_path
    model_raw = yaml.safe_load(model_path.read_text())
    config = ExperimentConfig.from_dict(raw)
    if not config.cv or config.start_fold_id != 0 or config.stop_fold_id != config.n_folds:
        parser.error("dashboard campaigns require a complete CV run from fold 1")
    if config.n_folds < 2 or not config.quantiles or not config.quantile_reg:
        parser.error("campaign requires at least two folds and quantile regression")
    model = ModelConfig.from_dict(model_raw)
    import torch
    config.use_wandb = True
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.quantile_scale and (model.tau_mult or model.tau_feat):
        parser.error("scale head requires tau_mult=false and tau_feat=false")
    directory = (args.artifact_root / args.campaign_id).resolve()
    session = "phm-" + args.campaign_id
    environment = {"PHM_CAMPAIGN_ID": args.campaign_id, "PHM_MONITOR_DIR": str(directory),
                   "PHM_EXP_CONFIG": str(directory / "experiment.yaml"),
                   "PHM_SHELL_CONFIG": str(directory / "shell_config"),
                   "CUDA_VISIBLE_DEVICES": str(shell["device_num"]), "PYTHONUNBUFFERED": "1",
                   "MPLBACKEND": "Agg", "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    # Preserve the caller's W&B routing choices without storing credentials.
    for key in ("WANDB_ENTITY", "WANDB_BASE_URL"):
        if key in os.environ:
            environment[key] = os.environ[key]
    manifest = {"campaign_id": args.campaign_id, "session": session, "port": args.port,
                "created_at": datetime.now().astimezone().isoformat(),
                "config": vars(config), "model_config": vars(model), "settings": shell,
                "environment": environment, "python": sys.executable,
                "selection": "Mean minimum validation loss across quantile checkpoints; lower is better"}
    manifest = json.loads(json.dumps(manifest, default=str))
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return 0
    cloudflared = shutil.which(os.environ.get("CLOUDFLARED_BIN", "cloudflared"))
    if not shutil.which("tmux") or not cloudflared:
        parser.error("tmux and cloudflared are required")
    if directory.exists():
        parser.error(f"campaign directory already exists: {directory}")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        parser.error(f"tmux session already exists: {session}")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (directory / "status.json").write_text(json.dumps({"status": "waiting"}))
    (directory / "shell_config").write_text(args.shell_config.read_text())
    (directory / "model.yaml").write_text(yaml.safe_dump(model_raw))
    raw["model_config_path"] = str(directory / "model.yaml")
    (directory / "experiment.yaml").write_text(yaml.safe_dump(raw))

    def tmux(*values):
        return subprocess.run(["tmux", *values], check=True, capture_output=True, text=True)

    def shell_command(argv, log):
        command = shlex.join(argv) + " 2>&1 | tee -a " + shlex.quote(str(log))
        return shlex.join(["bash", "-o", "pipefail", "-c", command])

    def window(name, command):
        identifier = tmux("new-window", "-d", "-P", "-F", "#{window_id}", "-t", session,
                          "-n", name, "sleep 120").stdout.strip()
        tmux("set-window-option", "-t", identifier, "remain-on-exit", "on")
        tmux("respawn-window", "-k", "-t", identifier, command)

    created = False
    try:
        tmux("new-session", "-d", "-s", session, "-n", "dashboard", shell_command(
            [sys.executable, str(HERE / "phm_dashboard.py"), "--directory", str(directory), "--port", str(args.port)], directory / "dashboard.log"))
        created = True
        tmux("set-option", "-t", session, "remain-on-exit", "on")
        for _ in range(50):
            try:
                with urlopen(f"http://127.0.0.1:{args.port}/api/state", timeout=1) as response:
                    response.read()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("Dashboard failed to start")
        tunnel_log = directory / "cloudflared.log"
        window("tunnel", shell_command([cloudflared, "tunnel", "--no-autoupdate", "--url",
                                         f"http://127.0.0.1:{args.port}"], tunnel_log))
        url = None
        for _ in range(100):
            contents = tunnel_log.read_text(errors="replace") if tunnel_log.exists() else ""
            matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", contents)
            if matches and "Registered tunnel connection" in contents:
                url = matches[-1]
                break
            time.sleep(0.5)
        if not url:
            raise RuntimeError(f"Tunnel failed to start: {tunnel_log}")
        (directory / "dashboard_url.txt").write_text(url + "\n")
        window("pipeline", shell_command([sys.executable, str(Path(__file__).resolve()), "--worker", str(directory)], directory / "runner.log"))
        tmux("select-window", "-t", session + ":pipeline")
    except BaseException:
        if created:
            subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        raise
    print(f"Dashboard: {url}\nAttach: tmux attach -t {session}\nArtifacts: {directory}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
