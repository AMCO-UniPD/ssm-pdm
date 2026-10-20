# Instructiono for `chronos_exp` experiments

In this README some instructions on the experiments runnable from this directory are reported.

## CMAPSS benchmark parameter and model-size table

From the repository root, using the project Python environment:

```bash
python experiments/chronos_exp/profile_benchmark_models.py \
  --output experiments/chronos_exp/profile_outputs/benchmark_model_profile.md
```

The script prints Markdown to stdout and optionally saves the same report with
`--output`. Diagnostics go to stderr, so redirecting stdout to a `.md` file also
works. It profiles S4, S5, S4D, LSTM, RULTransformer, and RULInformer from
`experiments/chronos_exp/config/ssm_exp_config.yaml` and its model configuration.
No dataset, checkpoint, training, or W&B connection is needed. Paths work from
any working directory; relative `model_config_path` entries resolve against the
experiment directory (the parent of the config directory).

The default input is one 170-step window with 14 selected CMAPSS sensors. The
quantile feature is added inside the model according to `tau_feat`. Each row
includes the input projector, feature extractor, and regression head, for one
quantile prediction. Use `--models`, `--exp-config`, `--model-config`,
`--sequence-length`, or `--input-features` to override the defaults. For the
benchmark variant without the tau input feature, add `--no-tau-feat`; to disable
output multiplication by tau, add `--no-tau-mult`.

Parameter counts describe the current implementations initialized with these
configs, rather than historical checkpoints. Model size is the serialized
`state_dict` size from `torch.save`, reported in MiB (1 MiB = 1,048,576 bytes).
It includes parameters, persistent buffers, and serialization overhead, retaining
native tensor dtypes, including complex SSM weights. Optimizer state, gradients,
and activations are excluded. This is a weights-only checkpoint size, not runtime
memory usage; serialization overhead can vary across PyTorch versions. The
generated report records the effective input/configuration and size definition.

The table also reports `torchinfo.summary(...).total_params` in a separate
column, using the old `model_summary` / `summary_func: torchinfo` computation
with batch size one and the sensor-only input shape (tau is appended internally).
If torchinfo fails, the original manual-count fallback is used and explicitly
labeled; the report includes the exception. Both columns contain raw parameter
counts. This matters for S5, whose summary currently fails under torchinfo 1.8.0
in the project environment.

For the older S5 architecture used before the extractor refactor:

```bash
python experiments/chronos_exp/profile_benchmark_models.py --models S5 --legacy-s5 \
  --output experiments/chronos_exp/profile_outputs/legacy_s5_profile.md
```

`--legacy-s5` reproduces `S5Model` from `src/ssm_models.py` at commit `85969ec`,
using the installed S5 dependency and the selected configs. It only changes S5.
The old blocks operate on sensor features before projection, with
`state_dim=d_model` and no extra LayerNorms. The current extractor projects to
`d_model` first and uses `state_dim=d_state`. With the default configs, the old
model has **57,808** parameters versus **397,738** for the current implementation.
This is consistent with S5 appearing near 58 K in thesis Figure 5.2, but the
original summary artifacts and exact dependency versions are needed to verify
an exact reproduction.

There is also a unit issue in the old blob-plot pipeline: `extract_number`
discards K/M suffixes from calflops strings, while torchinfo integers are kept
as raw counts; the axis is labeled `Parameters (K)` without normalization.
For example, `57.808 K` becomes `57.808`, whereas an integer `57808` stays
`57808`. Mixed profiler results can therefore appear 1,000 times apart.
`blob_plot.py` explicitly selects torchinfo for LSTM; the other models follow
`config.summary_func`, with no S5-specific selection in that code. The old
summary branch (commit `85969ec`) falls back to manual counting on exceptions.
The thesis plot also uses existing pickled checkpoint file sizes, whereas this
script measures freshly initialized state dictionaries serialized by `torch.save`.

