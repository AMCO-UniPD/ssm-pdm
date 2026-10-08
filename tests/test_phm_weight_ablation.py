"""Uncertainty aggregation and inference-only quantile evaluation checks."""

import ast
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phm_exp"))

import cv_training
from exp_config import ExperimentConfig
from phm_weight_ablation import (
    EVALUATION_QUANTILES,
    save_uncertainty_results,
    uncertainty_rows,
)


def test_ablation_post_training_config_attributes_are_available():
    config = ExperimentConfig.from_dict({})
    runner = ast.parse((ROOT / "experiments/phm_exp/phm_weight_ablation.py").read_text())
    # The runner sets dataset identities and output controls before training.
    supplied = {
        node.attr
        for node in ast.walk(runner)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
    }
    for filename, functions in {
        "models.py": {"load_best_model", "load_ssm_model", "best_model_perf"},
        "perf.py": {"lifes_metrics"},
        "utils.py": {"get_phm_feature_names", "get_mono_mask", "get_transformer"},
        "evaluator.py": {"get_life_evaluator", "QuantileLifeEvaluator", "LifeEvaluator"},
    }.items():
        module = ast.parse((ROOT / "src" / filename).read_text())
        for function in module.body:
            if getattr(function, "name", None) not in functions:
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
                    continue
                owner = ast.unparse(node.value)
                if owner not in {"config", "exp_config", "self.config"}:
                    continue
                # CMAPSS-only branches and the explicitly guarded optional scaler.
                if node.attr in {"cmapss_models", "rul_scaler"}:
                    continue
                assert node.attr in supplied or hasattr(config, node.attr), (
                    f"{filename}:{node.lineno} requires missing config.{node.attr}"
                )
    assert config.save_summary_dict is False
    assert config.model_summary is False


def make_outputs(targets):
    return {
        tau: {
            "y_true": targets,
            "y_pred": [np.asarray(t) + (tau - 0.5) * 20 for t in targets],
        }
        for tau in EVALUATION_QUANTILES
    }


def test_uncertainty_aggregation_weights_lives_equally():
    outputs = make_outputs([np.array([500, 400, 300, 0]), np.array([200])])
    outputs[0.1]["y_pred"][0][1] = 410  # Miss one of three valid samples.
    outputs[0.1]["y_pred"][1][0] = 210  # Second life misses its only sample.

    pinball, interval = uncertainty_rows(outputs, 500, "ratio_1", 1)

    assert len(pinball) == 5
    assert interval["overall_coverage"] == pytest.approx((2 / 3 + 0) / 2)
    assert interval["constant_coverage"] == 1
    assert interval["decreasing_coverage"] == pytest.approx((0.5 + 0) / 2)
    assert interval["nominal_coverage"] == 0.8


def test_intervals_reject_misaligned_lives():
    outputs = make_outputs([np.array([500, 400])])
    outputs[0.1]["y_true"] = [np.array([400, 500])]
    with pytest.raises(ValueError, match="ordering"):
        uncertainty_rows(outputs, 500, "ratio_1", 1)
    outputs[0.1]["y_true"] = []
    with pytest.raises(ValueError, match="counts"):
        uncertainty_rows(outputs, 500, "ratio_1", 1)


def test_exports_include_uncertainty_and_handle_absent_plateau(tmp_path):
    pinball_rows, interval_rows = [], []
    for fold in (1, 2):
        pinball, interval = uncertainty_rows(
            make_outputs([np.array([200, 100, 0])]), 500, "ratio_1", fold
        )
        pinball_rows.extend(pinball)
        interval_rows.append(interval)
    save_uncertainty_results(tmp_path, pinball_rows, interval_rows)

    with (tmp_path / "summary_intervals.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]["mean_overall_coverage"]) == 1
    assert float(rows[0]["mean_overall_width"]) == 16
    assert float(rows[0]["std_overall_width"]) == 0
    assert rows[0]["mean_constant_coverage"] == ""
    with (tmp_path / "summary_pinball.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 5
    assert len(list(tmp_path.glob("*.png"))) == 24


@pytest.mark.parametrize(
    "extra_quantiles, expected_evaluations",
    [(None, 2), (list(EVALUATION_QUANTILES), 10)],
)
def test_cv_evaluates_one_checkpoint_per_fold_without_retraining(
    monkeypatch,
    tmp_path,
    extra_quantiles,
    expected_evaluations,
):
    config = SimpleNamespace(
        quantile_reg=True,
        quantiles=[0.5],
        n_folds=2,
        start_fold_id=0,
        stop_fold_id=2,
        return_outputs=False,
        save_outputs=True,
        save_combined_outputs=False,
        compute_metrics=True,
    )
    trained, evaluated, completed = [], [], []
    monkeypatch.setattr(
        cv_training, "get_raw_phm_data", lambda **kwargs: (np.arange(4), [])
    )
    monkeypatch.setattr(cv_training, "MergeData", lambda data_list: data_list[0])
    monkeypatch.setattr(cv_training, "load_phm_data", lambda **kwargs: {"test_idx": [7]})
    monkeypatch.setattr(cv_training, "load_cv_data", lambda **kwargs: {})
    monkeypatch.setattr(
        cv_training, "wandb_cv_data", lambda **kwargs: (None,) * 8 + (config,)
    )
    monkeypatch.setattr(
        cv_training,
        "get_trainer",
        lambda **kwargs: SimpleNamespace(best_val_loss=0.1, run=lambda runWB: trained.append(kwargs)),
    )
    monkeypatch.setattr(
        cv_training, "best_model_perf", lambda **kwargs: evaluated.append(kwargs)
    )
    monkeypatch.setattr(cv_training, "lifes_metrics", lambda **kwargs: pd.DataFrame({"Eval Loss": [1.0]}, index=["Life_7"]))
    monkeypatch.setattr(cv_training, "init_wandb", lambda **kwargs: None)
    monkeypatch.setattr(cv_training.wandb, "finish", lambda: None)
    monkeypatch.setattr(cv_training.setproctitle, "setproctitle", lambda name: None)

    paths = {
        key: str(tmp_path / name)
        for key, name in (
            ("best_model_path", "models"),
            ("outputs_path", "outputs"),
            ("combined_outputs_path", "combined"),
            ("metrics_path", "metrics"),
        )
    }
    for path in paths.values():
        Path(path).mkdir()
    metrics, _ = cv_training.train_k_fold(
        config,
        SimpleNamespace(),
        **paths,
        evaluation_quantiles=extra_quantiles,
        fold_complete_callback=lambda fold: completed.append((fold, len(evaluated))),
    )

    assert len(trained) == 2
    assert len(evaluated) == expected_evaluations
    assert len(metrics) == 2
    assert all(frame["Eval Loss"].tolist() == [1.0] for frame in metrics)
    assert completed == [(1, expected_evaluations // 2), (2, expected_evaluations)]
    for fold in (1, 2):
        calls = [
            call for call in evaluated if f"fold_{fold}" in call["best_model_path"]
        ]
        assert len({call["best_model_path"] for call in calls}) == 1
        assert "quantile_0.5" in calls[0]["best_model_path"]
        assert {call["tau"] for call in calls} == set(extra_quantiles or [0.5])
        for call in calls:
            assert f"quantile_{call['tau']}" in call["outputs_path"]


def test_ablation_collects_and_exports_each_fold_before_next_training(monkeypatch, tmp_path):
    import pickle
    import yaml
    import phm_weight_ablation as runner

    model = tmp_path / 'model.yaml'
    model.write_text(yaml.safe_dump({'tau_feat': False, 'tau_mult': False}))
    config = tmp_path / 'experiment.yaml'
    config.write_text(yaml.safe_dump({'model_config_path': str(model), 'cv': True,
        'n_folds': 2, 'start_fold_id': 0, 'stop_fold_id': 2, 'quantile_reg': True,
        'quantile_scale': True, 'transformer_type': 5, 'max_rul': 500,
        'loss': 'window_quantile_reg', 'eval_loss': 'window_pinball'}))
    results = tmp_path / 'summaries'
    artifacts = tmp_path / 'isolated results'
    monkeypatch.setenv('PHM_RESULTS_DIR', str(artifacts))
    monkeypatch.setattr(sys, 'argv', ['ablation', '--config', str(config),
        '--train-phm-tools', '01M01', '--test-phm-tools', '01M02',
        '--ratios', 'unweighted', '1', '--results-dir', str(results)])
    monkeypatch.setattr(runner.torch.cuda, 'is_available', lambda: False)
    plotted = []
    monkeypatch.setattr(runner, 'save_plot', lambda path, *a, **kw: plotted.append(path.name))
    partial_counts = []
    def train(**kwargs):
        assert str(artifacts) in kwargs['outputs_path']
        assert kwargs['exp_config'].quantiles == [0.5]
        assert kwargs['evaluation_quantiles'] == list(EVALUATION_QUANTILES)
        for fold in (1, 2):
            for tau in EVALUATION_QUANTILES:
                path = Path(kwargs['outputs_path']) / f'fold_{fold}' / f'quantile_{tau}'
                path.mkdir(parents=True)
                with (path / 'outputs.pickle').open('wb') as stream:
                    pickle.dump(make_outputs([np.array([500., 400., 100., 0.])])[tau], stream)
            kwargs['fold_complete_callback'](fold)
            with (results / 'summary_rmse.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            partial_counts.append([int(row['completed_folds']) for row in rows])
        return [pd.DataFrame(), pd.DataFrame()], 'fixture'
    monkeypatch.setattr(runner, 'train_k_fold', train)
    runner.main()
    assert partial_counts == [[1], [2], [2, 1], [2, 2]]
    with (results / 'fold_intervals.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(float(row['overall_width']) == 16 for row in rows)
    assert len(set(plotted)) == 27
