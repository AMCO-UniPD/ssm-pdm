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

### Data Structure

In the note `RESULTS  `XAI_PDM.md` there are some useful information on some preprocessing steps to perform on the `CMAPSS` dataset. Here I will report them:

- First of all we have to use the `sensor_indices` list (
`from ceruleo.dataset.catalog.CMAPSS import sensor_indices`
) that should contain the list of the sensor indices to include in the data. Some of the indeces can in fact be removed since they are constant values and thus are pretty useless.
- See some of the code snippets to apply the Feature Extraction and Feature Selection steps using the `CeRULeO` library.
- Successively there is the problem that a `CeRULeO` dataset is actually a list of `pd.DataFrame`s each one representing a single life (i.e. run to failure cycle) of the machine. Probably it makes sense to create a `DataLoader` with mini batches of size 1 each one containing the data of a single life, as I am doing in the `ad_mg` project with the data of the different acquisitions.

>[!note]
> I think that if we want to divide the lifes into single mini batches in a `DataLoader` is easier to design the fine tuning code using `PyTorch` following the `Full pytorch training` note in the 🤗 `NLP` course notes.

## How to use `CMAPSS` with the model

Here I have to understand how to use the `CMAPSS` dataset with the `CHRONOS` model. In fact here it is not like with the forecasting task where we pass an input sequence and the model predicts the next values of it.

In this case we pass an input sequence and the model should predict the `RUL` of the engine. So here I should decide how to structure the `(input,target)` pairs for the fine tuning task. In fact in the `SMM_PDM` project we used those two approaches where all the `RUL` values over time were predicted for an entire input signal or for an entire piece of signal.

Here now some approaches come into my mind:

- If we do as I test up to now in `chronos_reg_test.py` with `AutoModelForSequenceClassification(checkpoint,num_labels=1)` we will obtain a single output, so if we pass a time series in input we should obtain a single `RUL` value which will be the `RUL` of the last time step of the input sequence. This is a simple approach but maybe it is not the best one because we are not exploiting the fact that the model can predict the `RUL` at each time step of the input sequence.
- In order to exploit the fact that we have a model that can predict the `RUL` at each time step of the input sequence we can use `AutoModelForSequenceClassification(checkpoint,num_labels=sequence_length)` and pass as target the `RUL` values of the entire input sequence. Even though in the model architecture it writes down `classification_head` and the model predictions are called `logits` it should work also for regression because we just need to pass the `RUL` labels and then use a regression loss function (i.e. `MAE,MSE` or some weigthed versions of those like the `Pinball Loss`).

### Problem: `CHRONOS` does not work with multivariate time series

From the first experiments after having structured the `CMAPSS` dataset similarly to what I did in the `SSM_PDM` project I realized that probably the model does not work on multivariate time series, even though I was pretty sure it should have worked since in the video [[chronos-learning-the-language-of-time-series|presenting the paper]] the main author talked about multivariate time series.

In any case it makes perfect sense that it works only on single time series because any language model works with one dimensional sequences. So at this point there are two possible solutions:

- Modify the code in the repository to make it work with multivariate time series. It should not be a big deal since I just need to change the shapes and dimensions. However using multiple time series maybe the model is not an `LLM` anymore, it because just a fancier version of the transformers I used in the `SSM_PDM` project.
- Accept the fact that we will have to use univariate time series in input (i.e. which is actually the same thing I am doing in the `ad_mg` project). Also in this case I probably have to adapt in some points the code because if I want the entire `RUL` sequence in output from the model I need to have the input sequences and targets with the shape `(1,sequence_length,1)`. The main problem here is: how can I combine the `RUL` predictions returned by the different sensors? I will in fact have different predictions from different sensors → the better approach is, since we always want to be conservative and avoid underestimations in the `RUL` estimation, to take the minimum `RUL` value predicted at each time step. A simpler approach is to select the `RUL` sequence predicted by the sensor that has the lowest `RUL` value at the end (in its last time step).

I tried to use multiple time series in input to the `ChronosPipeline` and to the `predict_quantiles` method in `chronos_forecast.py` and I remembered that it works for multiple time series → we just need to pass a list of `1D` tensor to the model. What happens undert the hood is that the different time series are divided into different batches and then they are forecasted separately. Then in the output `quantiles` we have the shape `(n_time_series,sequence_length,n_quantile_levels)`.

>[!warning] Another problem
> This makes perfect sense because in times series forecasting we forecast a time series alone, it does not make sense to produce a single forecast from multiple time series. This is the problem because in `RUL` estimation we want to predict a single `RUL` value from all the sensor measurement data we have and we can't do that with this modified version of the `CHRONOS` model.

So we have to resort to this unviariate version of `RUL` where essentially we compute the `RUL` separately for each sensor and than we take:

- The worst `RUL` (i.e. the minimum one) at each time step
- Select the `RUL` sequence of the sensors closer to failure as the final prediction. In this way we may also have a sort of interpretability because we may say that the sensor we have chosen is the one that is causing the failure of the engine.

However there is a big limitation in this approach because we are not considering all the intereactions there may be between the different sensors in the equipment. In fact it is most likely that the failure of the engine is due to failure of multiple components and not just of one of them.

#### Regression approach is really to throw away?

I quickly read the paper *Lithium-ion batteries remaining useful life prediction based on a mixture of empirical mode decomposition and ARIMA model* where they exploited forecasts to predict the `RUL`. Here the job was much easier because they had a single `SOH` time series for each battery which represents the degradation pattern of the battery, I do not have that in `CMAPSS` and I don't really know how to extract that from all the sensor measurement I have. However also in the `ad_mg` project we are considering the different sensors as separate and independent entities without considering the interactions between them. At the end the first sensor that reaches an high enough damage value will trigger the alarm on the machine and successively the maintenance will be performed. So maybe it makes sense to do this independence assumption.

The only problem is: what kind of labels do I have to use for each different sensor? Here it is not like in the `ad_mg` project where I have a damage value for each different sensor, here I have a `RUL` sequence associated to the entire life, so to all the sensor together.

Also in this case I may do the assumption of a linear degradation trend for each sensor and so create the `RUL` as the decreasing sequence of integers from the last time step to the first one → in this way all the different sensors for the same life will have the exact some `RUL` sequence as their target → I don't know if that makes sense but it is a way to start. Maybe in this way I can see for each life which one is the best sensor in predicting the `RUL`.

##### Idea on how to implement the Regression approach

In `RegressionDataset` we create a dataset **on a single life** where:
    - `sequences` takes `life[sensors]` where `sensors` are the features passed in input through `exp_config` (these can be the raw sensor measurements like `SensorMeasure4,SensorMeasure5` or some extracted features like `SensorMeasure4_mean,SensorMeasure5_std`).
    - `targets` takes the `RUL` sequence of the life → this will be the shared target for all the sensors in the life.

So at the end the end we can create a `DataLoader` for each life where in each mini batch we put a sensor measurement → so the final shape of `sequences` in the `DataLoader` will be `(n_sensors,sequence_length,1)`.

Finally if we create the model using `AutoModelForSequenceClassification` and `n_labels=1` this should output a tensor with shape (`n_sensors,sequence_length,1)` where the `RUL` sequence of each sensor is predicted separately. Then we can compute the loss function using the `RUL` as the target, which will have shape `(sequence_length,1)`.

To make it easier to compute the loss I may replicate the `RUL` sequence over the batch dimension so that it has shape `(n_sensors,sequence_length,1)` and then I can compute the loss using the `MSE` or `MAE` loss functions.

#### New approach idea: Forecast the `RUL` values

I am a little bit stuck on the Regression approach because it is not possible to obtain an output of shape `(n_sensors,sequence_length,1)` because the `sequence_length` changes from life to life and so it not possible to create a Regression Head unless we fix a maximum `sequence_length` value and use padding to make all the sequences of the same length, but I do not like this option. 

This other idea I have in mind makes more sense intuitively and may also be potentially easier to implement because we do not have to change the model too much. In fact we will keep `chronos` as a forecasting model but instead of forecasting the next values of the input sequence we will forecast the `RUL` values of the engine. So this means that if I have a life of length `T` we can pass in input the first `T-k` time steps and the model should forecast the `RUL` values of the last `k` time steps. This may make sense because the `RUL` values at the beginning of a life are not very interesting usually, we are more interested in the `RUL` values when the engine is closer to failure.Using this approach we can also exploit the uncertainty quantification automatically built inside `chronos`.

The only thing is that we do not have the quantile levels or any sort of distribution for the ground truth `RUL` values, so we may just compare the predicted `RUL` values for the 0.5 quantile (i.e. the median) with the true `RUL` values and then compute the loss using the `MSE` or `MAE` loss functions.

>[!error] This approach is not possible
> I realized that this approach is not possible because the `chronos` model is a language model and so the forecast values are tokens coming from the vocabulary which depends on the range of values in the signals, which may not include the `RUL` values. Maybe it is possible to do some strange normalizations to make it work but I do not like this idea. At this point the only viable option is to use the Regression approach with padding.

#### Padding Regression Approach

In this other approach I am trying right now we pad all the sequences (with 0 padding) to have a fixed sequence length (which is an hyperparameter). Then in the `RegressionHead` of the model we will have a `nn.Linear(hidden_size,sequence_length)` layer so that we are able to output the entire `RUL` sequence for each sensor. Then we can compute the loss using the `MSE` or `MAE` loss functions only on the non padded values of the `RUL` sequence.

Obviously this approach is not very generalizable to arbitrary sequence lengths there may be in datasets different from `CMAPSS` but we can fine tune the model on a new dataset properly choosing the correct value for the `sequence_length` hyperparameter on that dataset.

With this new approach now the `logits` attribute of the `output` object returned by the `forward` method of the model has shape `(batch_size,sequence_length)` as we wanted. Now the values we get are pretty bad because we are using a randomly initialized head but fine tuning on the `CMAPSS` dataset should improve the results.


