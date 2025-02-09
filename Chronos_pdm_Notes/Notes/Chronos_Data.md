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

Moreover in `CeRULeO` we have also the `CMAPSS-2` and the `PHM08` datasets we can try to use for benchmarking the `CHRONOS` model.

### Loading `CMAPSS` Dataset

As we have seen [[chronos_exp#forecasting-with-cmapss|here]] the `CMAPSS` dataset is automatically stored in `~/.ceruleo/data` once we call the `CMAPSSDataset` class the first time. So from now on we can easily load it and obtain `pd.DataFrame` objects. 

These objects may be useful to do a first exploratory data analysis on the dataset but then in order to perform the fine tuning maybe it's better to follow the hints contained in [here](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts) and use the Arrow data format which is more efficient.

## Exploratory Data Analysis

In this section we will perform a first exploratory data analysis on the `CMAPSS` data to see how it is structured and the pre processing steps we could perform on it to structure it for the fine tuning task.

>[!note]
> Look at the notes of the `SSM_PDM` project (or maybe also the ones on the `CERULEO` inside the `PHD` vault) to see weather there are already some useful informations on how to preprocess the `CMAPSS` dataset.


## How to use `CMAPSS` with the model

Here I have to understand how to use the `CMAPSS` dataset with the `CHRONOS` model. In fact here it is not like with the forecasting task where we pass an input sequence and the model predicts the next values of it.

In this case we pass an input sequence and the model should predict the `RUL` of the engine. So here I should decide how to structure the `(input,target)` pairs for the fine tuning task. In fact in the `SMM_PDM` project we used those two approaches where all the `RUL` values over time were predicted for an entire input signal or for an entire piece of signal.

Here now some approaches come into my mind:

- If we do as I test up to now in `chronos_reg_test.py` with `AutoModelForSequenceClassification(checkpoint,num_labels=1)` we will obtain a single output, so if we pass a time series in input we should obtain a single `RUL` value which will be the `RUL` of the last time step of the input sequence. This is a simple approach but maybe it is not the best one because we are not exploiting the fact that the model can predict the `RUL` at each time step of the input sequence.
- In order to exploit the fact that we have a model that can predict the `RUL` at each time step of the input sequence we can use `AutoModelForSequenceClassification(checkpoint,num_labels=sequence_length)` and pass as target the `RUL` values of the entire input sequence. Even though in the model architecture it writes down `classification_head` and the model predictions are called `logits` it should work also for regression because we just need to pass the `RUL` labels and then use a regression loss function (i.e. `MAE,MSE` or some weigthed versions of those like the `Pinball Loss`).
