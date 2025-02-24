---
id: old_code
aliases: []
tags:
  - code
---

# Old code `CHRONOS` project

In this note I will keep `python` code blocks containing old pieces of code (or pieces of code I inserted to do some temporary stuff) that I do not want to leave all commented inside the codebase.

## `update_metrics` function

Here I insert the code I used to update the `index` of the `metrics_df` `pd.DataFrame`s after I realized that the original way I used to compute the index was wrong (in fact I used `f"Life_{i}` instead of `f"Life_{i+exp_config.test_idx[0]}`).

```python
if exp_config.update_metrics:
    new_index=[f"Life_{i+exp_config.test_idx[0]}" for i in range(metrics_df.shape[0]-1)]
    new_index.append("Life_mean")
    # Update the index of metrics_df with new_index
    metrics_df.index=new_index

    save_element(
        element=metrics_df,
        dirpath=metrics_path,
        filename=f"{get_current_time()}_lifes_metrics_{exp_config.model_name}_{exp_config.cmapss_models}_exp_{n_files-i}.pkl"
    )
```
