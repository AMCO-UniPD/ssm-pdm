"""Campaign telemetry, validation selection, serving boundaries and final plots."""

import json
import os
import pickle
import sys
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/phm_exp"))
import phm_monitor
import phm_dashboard


class MonitorTests(unittest.TestCase):
    def test_result_paths_respect_override_and_legacy_default(self):
        from phm_paths import results_root
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(results_root(tmp), Path(tmp).resolve())
            with patch.dict(os.environ, {"PHM_RESULTS_DIR": str(Path(tmp) / "external results")}):
                self.assertEqual(results_root("/unused"), Path(tmp) / "external results")

    def test_cached_ceruleo_transformer_has_sklearn_estimator_tags(self):
        import utils
        from sklearn.utils.validation import check_is_fitted
        # Cached instances keep their existing class and fitted state.
        transformer = utils.Transformer.__new__(utils.Transformer)
        transformer.fitted_ = True
        restored = pickle.loads(pickle.dumps(transformer))
        check_is_fitted(restored)

    def test_inactive_without_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            phm_monitor.emit("epoch", val_loss=1.0)

    def test_state_events_context_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PHM_MONITOR_DIR": tmp}):
            phm_monitor.emit("fold_started", fold=2, status="running")
            phm_monitor.emit("quantile_started", tau=0.25)
            phm_monitor.emit("epoch", epoch=1, val_loss=float("nan"))
            phm_monitor.emit("failed", exit_code=1)
            state = json.loads((Path(tmp) / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["fold"], 2)
            self.assertEqual(state["tau"], 0.25)
            self.assertIsNone(state["val_loss"])
            self.assertEqual(len(phm_dashboard.read_events(Path(tmp))), 4)

    def test_best_fold_uses_validation_and_ignores_missing_scores(self):
        events = [
            {"kind": "fold_completed", "fold": 1, "validation_loss": 2.0, "test_loss": 0.1},
            {"kind": "fold_completed", "fold": 2, "validation_loss": 1.0, "test_loss": 10.0},
            {"kind": "fold_completed", "fold": 3, "validation_loss": None},
        ]
        self.assertEqual(phm_monitor.best_fold(events)["fold"], 2)
        with self.assertRaises(ValueError):
            phm_monitor.best_fold([])

    def test_dashboard_partial_event_and_first_wandb_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = {"campaign_id": "test", "config": {}, "model_config": {}, "settings": {}}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            events = [{"kind": "wandb_run", "url": f"https://wandb.ai/team/project/runs/{i}"} for i in (1, 2)]
            (directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + '\n{"kind":')
            data = phm_dashboard.payload(directory)
            self.assertEqual(len(data["events"]), 2)
            self.assertEqual(data["wandb"]["url"], events[0]["url"])

    def test_offline_snapshot_embeds_data_images_and_escapes_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "plots").mkdir()
            (directory / "plots/result.png").write_bytes(b"saved plot bytes")
            manifest = {"campaign_id": "snapshot", "config": {}, "model_config": {}, "settings": {}}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            (directory / "runner.log").write_text('</script><script>alert("unsafe")</script>')
            events = [{"kind": "plot", "filename": "result.png", "title": "Result"},
                      {"kind": "plot", "filename": "../outside.png", "title": "Outside"}]
            (directory / "outside.png").write_bytes(b"must not embed")
            (directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            html = phm_dashboard.snapshot_html(directory)
            saved = html.split('<script id="saved-dashboard" type="application/json">', 1)[1].split('</script>', 1)[0]
            data = json.loads(saved)
            self.assertEqual(data["log"], (directory / "runner.log").read_text())
            self.assertNotIn("</script>", saved)
            self.assertEqual(data["plot_assets"], {"result.png": "data:image/png;base64,c2F2ZWQgcGxvdCBieXRlcw=="})
            self.assertIn('href="data:image/svg+xml;base64,', html)
            self.assertIn('if(offlineData){render(offlineData)}else{refresh()}', html)

    def test_snapshot_download_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "manifest.json").write_text(json.dumps(
                {"campaign_id": "offline_test", "config": {}, "model_config": {}, "settings": {}}))
            server = ThreadingHTTPServer(("127.0.0.1", 0), partial(phm_dashboard.Handler, directory=directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/snapshot.html") as response:
                    self.assertEqual(response.headers.get_content_type(), "text/html")
                    self.assertEqual(response.headers["Content-Disposition"], 'attachment; filename="phm-dashboard-offline_test.html"')
                    self.assertIn(b'id="saved-dashboard"', response.read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_http_does_not_expose_arbitrary_campaign_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "plots").mkdir()
            (directory / "secret.json").write_text("private")
            (directory / "plots/allowed.png").write_bytes(b"image")
            server = ThreadingHTTPServer(("127.0.0.1", 0), partial(phm_dashboard.Handler, directory=directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/assets/allowed.png") as response:
                    self.assertEqual(response.read(), b"image")
                for path in ("/assets/../secret.json", "/assets/%2e%2e/secret.json", "/secret.json"):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(base + path)
                    self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_best_fold_report_generates_all_requested_plots(self):
        import numpy as np
        import phm_campaign_report as report
        from exp_config import ExperimentConfig

        for migrated in (False, True):
            with self.subTest(migrated=migrated):
                with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PHM_MONITOR_DIR": tmp}):
                    os.environ.pop("PHM_RESULTS_DIR", None)
                    directory = Path(tmp)
                    config = vars(ExperimentConfig(n_folds=2, nrows=1, ncols=1, plot_life_idx=[0], life_idx=[0], n_last_samples=50))
                    manifest = {"campaign_id": "fixture", "config": config,
                                "results_root": str(directory / "external results"),
                                "settings": {"model_name": "S4D", "failure_type": "flow_low"}}
                    output_root = Path(manifest["results_root"])
                    if migrated:
                        manifest.pop("results_root")
                        manifest["environment"] = {"PHM_MONITOR_DIR": "/old/home/campaign"}
                        output_root = directory / "results"
                    (directory / "manifest.json").write_text(json.dumps(manifest))
                    phm_monitor.emit("fold_completed", fold=1, validation_loss=2.0, test_idx=[0])
                    phm_monitor.emit("fold_completed", fold=2, validation_loss=1.0, test_idx=[0])
                    outputs = output_root / "combined_outputs/S4D/flow_low" / config["approach"] / "fixture/fold_2"
                    outputs.mkdir(parents=True)
                    true = np.concatenate([np.linspace(100, 0, 1501), np.zeros(1200)])
                    combined = {"y_true": [true]}
                    for tau in config["quantiles"]:
                        pred = true + (tau - 0.5) * 10
                        combined[f"pred_quantile_{tau}"] = [pred]
                        path = outputs / f"quantile_{tau}"
                        path.mkdir()
                        with (path / "outputs.pickle").open("wb") as stream:
                            pickle.dump({"y_true": [true], "y_pred": [pred]}, stream)
                    with (outputs / "combined.pickle").open("wb") as stream:
                        pickle.dump(combined, stream)
                    from matplotlib.axes import Axes
                    original_plot = Axes.plot
                    lengths = []
                    def tracked_plot(ax, values, *args, **kwargs):
                        if kwargs.get("label") == "True RUL":
                            lengths.append(len(values))
                        return original_plot(ax, values, *args, **kwargs)
                    with patch.object(report, "ROOT", directory), patch.object(Axes, "plot", tracked_plot):
                        report.generate(directory)
                    self.assertEqual(lengths, [1500, 1000, 50])
                    plots = list((directory / "plots").glob("*.png"))
                    self.assertEqual(len(plots), 6)
                    self.assertTrue(all(path.name.startswith("fold_2_") for path in plots))
                    events = phm_dashboard.read_events(directory)
                    self.assertEqual([e for e in events if e["kind"] == "best_fold_selected"][0]["best_fold"], 2)
                    self.assertEqual(len([e for e in events if e["kind"] == "plot"]), 6)
                    self.assertTrue((directory / "plots/fold_2_quantiles_last_1000.png").is_file())

    def test_cv_publishes_fold_tables_and_resets_quantile_models(self):
        import numpy as np
        import pandas as pd
        import cv_training as cv
        from exp_config import ExperimentConfig
        from types import SimpleNamespace

        class Dataset:
            def __len__(self):
                return 4

            def __getitem__(self, indices):
                return self

        config = ExperimentConfig(n_folds=2, stop_fold_id=2, quantiles=[0.25, 0.75])
        config.compute_metrics = config.save_outputs = config.save_combined_outputs = True
        config.return_outputs = False
        dataset = Dataset()
        events = []
        models = []

        def ingredients(**kwargs):
            model = object()
            models.append(model)
            config.test_idx = [99]  # CV split IDs must not label the held-out metrics.
            return (None, None, None, model, None, None, None, None, config)

        def trainer(**kwargs):
            return SimpleNamespace(best_val_loss=kwargs["tau"], run=lambda **kw: None)

        def metrics(**kwargs):
            self.assertEqual(list(config.test_idx), [7])
            return pd.DataFrame({"Eval Loss": [kwargs["tau"] * 10]}, index=["Life_7"])

        with patch.multiple(cv,
                            get_raw_phm_data=lambda **kw: (dataset, dataset),
                            MergeData=lambda **kw: dataset,
                            load_phm_data=lambda **kw: {"test_idx": np.array([7])},
                            load_cv_data=lambda **kw: {},
                            wandb_cv_data=ingredients,
                            init_wandb=lambda **kw: None,
                            get_trainer=trainer,
                            best_model_perf=lambda **kw: None,
                            lifes_metrics=metrics,
                            generate_path=lambda **kw: "/unused",
                            emit=lambda kind, **values: events.append({"kind": kind, **values})), \
                patch.object(cv.wandb, "finish"):
            frames, _ = cv.train_k_fold(config, SimpleNamespace(), "cpu")
        self.assertEqual(len(models), 4)
        self.assertEqual(len(frames), 4)
        folds = [event for event in events if event["kind"] == "fold_completed"]
        self.assertEqual([event["fold"] for event in folds], [1, 2])
        self.assertEqual([event["validation_loss"] for event in folds], [0.5, 0.5])
        self.assertEqual(folds[0]["table"]["values"], [[2.5, 7.5]])
        self.assertEqual(events[-1]["kind"], "summary")
        self.assertEqual(events[-1]["table"]["values"], [[2.5, 7.5]])


if __name__ == "__main__":
    unittest.main()
