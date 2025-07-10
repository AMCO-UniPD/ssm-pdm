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
