---
id: old_code
aliases: []
tags: []
---

# Old Code `chronos_pdm` Project

In this note I keep some chuncks of old code.

## `perf.py`

This old code is actually not that old. This is the code I used to try to substitute the `Mult Adds` metric in the blob plot (which is producing strange results because of the strange behavior of `calflops`) with the time to perform a `forward` pass on one of the models I compared in the paper. This time execution metric should have been used for the radius of the blobs in the blob plot. The code to implement the experiment is contained in the `time_exp` function. The code is being removed because the execution time seems to be the same for all models.

```python
print("#" * 50)
print(f"Performing time experiment for model: {model_name}")
print("#" * 50)
dict_time = time_exp(config=config, model_config=model_config)
plot_dict["test_time"] = dict_time["avg_time"]
```

## `plots.py`

Some old comments inside the `blob_plot` function used to produce the blob plot:

```python
# Add jitter to text positions
jitter_strength = 0.01  # Adjust as needed

df['text_x'] = df['Parameters (K)'] + np.random.normal(0, jitter_strength * df['Parameters (K)'].mean(), len(df))
df['text_y'] = df['Test Loss'] + np.random.normal(0, jitter_strength * df['Test Loss'].mean(), len(df))
```

## `blob_plot.py`

Some old comments from `blob_plot.py`. This one was a first try to convert the `params` key of the `plot_dict` dictionary into a `float` values:

```python
plot_dict["params_float"] = [extract_number(param) for param in plot_dict["params"]] if all(isinstance(item,str) for item in plot_dict["params"]) else plot_dict["params"]
```

Old code to define the path from where to extract the metrics dataframe needed to add the `test_metric` key to `plot_dict`. In here we are using `get_most_recent_dir` to get the `exp_name` folder. However after some experiment is difficult to use this function because the `file_pos` argument would change for each different model type, so the current implementation simply hard codes the experiment names we need in the `exp_names` variable defined in `launch_blob_plot.sh`

```python
metrics_exp_dirpath_model = get_most_recent_dir(
    metrics_dirpath_model, file_pos=config.file_pos
)
plot_dict["metrics_dirpath"].append(os.path.basename(metrics_exp_dirpath_model))
```
