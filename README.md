# `CHRONO-RUL`: Exploiting a Time Series Language Model for Remaining Useful Life Estimation

This is the repository containing the codebase for the `CHRONO-RUL` project. The aim is to try to exploit the capabilities of the recently introduced [*Chronos: Learning the Language of Time Series*](https://arxiv.org/abs/2403.07815) model as a time forecaster in the Remaining Useful Life (`RUL`) estimation task in the field of Predictive Maintenance.

## `CHRONOS` Model

`CHRONOS` applies simple and minimal transformations (i.e. scaling, binning and quantization) to time series in order to convert them into sequences of tokens which can be used as inputs to a Large Language Model. In this way the `LLM` can be used without any modification. 

The `LLM` used in the `CHRONOS` paper is [`T5`](https://arxiv.org/abs/1910.10683) but in principle any `LLM` can be used. The reason under the choice of `T5` as the backbone was driven by the fact that it is available on `HuggingFace` 🤗 in different sizes so that it does not require a huge computational power to be used.

`CHRONOS` showed excellent performance in Zero Shot Inference surpassing the performances of Deep Learning model trained on the benchmark datasets. 

## `CHRONOS` for `RUL` Estimation

The idea for employing `CHRONOS` in `RUL` estimation is the following:

- The starting point are the raw sensor historical data typically available in any Predictive Maintenance task (e.g. `CMAPSS`,`N-CMAPSS` datasets)
- From the raw sequence data, if needed, Feature Extraction techniques are applied to produce Health Indexes: features that are directly correlated with the health of the system (i.e. features showind the degradation of the system over time).
- Once the Health Indexes are available, the forecasting abilities of `CHRONOS` will be used to predict their future behavior
- The maintenance point (i.e. time instant were maintenance should be performed) can be computed on the base of a threshold value of the Health Indexes or on the base of the predicted future values of the Health Indexes.
