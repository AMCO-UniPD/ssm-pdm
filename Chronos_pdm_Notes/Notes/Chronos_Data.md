---
id: chronos-data
aliases: []
tags:
  - data
---

# `CHRONOS` Data

In this note I will make some consideration on the possible datasets that can be used for fine tuning `CHRONOS` ⏰ on the `RUL` estimation task.

## `CMAPSS` Dataset

Here we are another note on `C-MAPSS`. I have already used and seen this dataset several times and every time I have to take a not again on it because I always forget something. 

In any case in this note I will point point the things I need in order to work with this dataset in the `chronos-pdm` project, I have to consider it in a different way than I did in the `SSM_PDM` project. In fact, as explained [[chronos-for-pdm|here]], we have to adapt the [[chronos-learning-the-language-of-time-series|`CHRONOS`]] model to work with the `CMAPSS` dataset and in general with the `RUL` task.

Moreover I also have to understand how to properly load the dataset using the 🤗 `Datasets` library. 

>[!important]
> Check this function in [`CeRULeO`](https://lucianolorenti.github.io/ceruleo/dataset/catalog/#ceruleo.dataset.catalog.CMAPSS.obtain_raw_files) to load and unzip the raw files of the `CMAPSS` dataset.

Moreover in `CeRULeO` we have also the `CMAPSS-2` and the `PHM08` datasets we can try to use for benchmarking the `CHRONOS` model.
