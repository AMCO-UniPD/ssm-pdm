---
id: rnn_exp
aliases: []
tags:
  - experiments
  - ssm_pdm
---

# `RNN` `PdM` Experiments

In this note I will report the results obtained with the `RNN` base models (i.e. `RNN,LSTM,GRU`).

## `padding` approach experiments 🦜

Let's consider also here the `padding` approach first.

### `RNN` Model Experiments ♻️

Let's use the `model_summary` parameter to get the size of the `RNN` model.


```txt
=========================================================================
=================
Total params: 2,628,084
Trainable params: 2,628,084
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 1.19
=========================================================================
=================
Input size (MB): 0.03
Forward/backward pass size (MB): 2.05
Params size (MB): 10.51
Estimated Total Size (MB): 12.59
=========================================================================
=================
```
The total number of parameter it's very high, that's probably because of all the connections that are created by backpropagation through time.

#### Dataset `FD001`

##### Experiment 1 `RNN` `FDOO1` `padding` ♻️ 1️⃣ 🦜

Let's start with a configuration very similar to the one of the first experiments done with the `SSM` models.

| Parameter | Value |
|-----------|-------|
| `model_type` | `RNN` |
| `cmapss_model` | `FD001` |
| `val_idx` | `[0,50]` |
| `test_idx` | `[50,100]` |
| `transformer_type` | 1 (no feature extraction) |
| `window_size` | 20 |
| `scaler` | `MinMaxScaler(-1,1)` |
| `epochs`  | 100    |
| `lr` | 1e-3 |
| `weight_decay` | 1e-4 |
| `sequence_length` | 500 |
| `n_layers` | 5 |
| `hidden_size` | 512 |
| `loss` | `mae` |
| `eval_loss` | `mse` |

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/3wgoddwe?nw=nwuserfrizzodavide)

As in `LSTM` the loss quickly saturates at an higher value than in `SSM` so probably there will be worse results.

###### Metrics Table

As expected the mean metric value on the test lifes is higher than the ones obtained in the `SSM` models.


```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   31.58
##################################################
```

The `Life Mean` over the first 15 lifes is a bit higher than the overall `Life Mean` over all the test lifes.

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 63.78 |
| Life_51 | 22.29 |
| Life_52 | 4.81 |
| Life_53 | 23.85 |
| Life_54 | 56.97 |
| Life_55 | 42.15 |
| Life_56 | 68.52 |
| Life_57 | 18.0 |
| Life_58 | 15.03 |
| Life_59 | 52.77 |
| Life_60 | 14.48 |
| Life_61 | 84.09 |
| Life_62 | 32.68 |
| Life_63 | 2.08 |
| Life_64 | 4.84 |
| Life_mean | 33.76 |

###### Prediction plots

Even though the error is higher the `RUL` predictions are much more similar to the decreasing line of the true values. However looking at the plots it seems that the predictions are always in the same range of values, they all start around 200 and then go down following a decreasing trend. This is the same thing that happened in the first experiments with `chronos`. Even though the results may look good from the plots this is because the `RUL` ranges are similar to the constant one that is predicted by the model, if we have new test data with different ranges the performances will probably be worse.

On the other hand the `SSM` models are able to adapt to different ranges of `RUL` values, even though the predicted `RUL` is less stable.

##### Experiment 2 `RNN` `FDOO1` `padding` ♻️ 2️⃣ 🦜

As written [[transformer-pdm-experiments#Experiment 2 `Transformer` `FDOO1` `padding` 🤖 2️⃣ 🦜|here]] let's try to see what happens using the new architecture.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/2sy0wx7t?nw=nwuserfrizzodavide)

###### Metrics Table

Nope, even worse

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 96.87 |
| Life_51 | 60.01 |
| Life_52 | 48.28 |
| Life_53 | 68.56 |
| Life_54 | 100.44 |
| Life_55 | 42.21 |
| Life_56 | 96.29 |
| Life_57 | 57.3 |
| Life_58 | 68.13 |
| Life_59 | 86.17 |
| Life_60 | 45.94 |
| Life_61 | 97.96 |
| Life_62 | 67.84 |
| Life_63 | 50.34 |
| Life_64 | 68.15 |
| Life_mean | 70.3 |

###### Prediction plots

The predictions are constant over all the time steps of each life, very bad.

### `LSTM` Model Experiments 🧠

Let's use the `model_summary` parameter to get the size of the `LSTM` model.


```txt
=========================================================================
=================
Total params: 9,742,836
Trainable params: 9,742,836
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 4.74
=========================================================================
=================
Input size (MB): 0.03
Forward/backward pass size (MB): 2.05
Params size (MB): 38.97
Estimated Total Size (MB): 41.05
=========================================================================
=================
```

As expected the number of parameters here it's even higher, in fact `LSTM` is a more complex version of `RNN`.

#### Dataset `FD001`

##### Experiment 1 `LSTM` `FDOO1` `padding` 🧠 1️⃣ 🦜

Let's use the same configuration used for the `RNN` model.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/vu1n8q04?nw=nwuserfrizzodavide)

The loss plots in `wandb` go down in the first few epochs and then saturate soon to a constant value which is higher than the loss vales of the `SSM` models so there will probably be worse results in the metrics table.

###### Metrics Table

The metrics table is similar to the one of `RNN`, as we can see also from the overall `Life mean` over all the test lifes.


```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   31.30
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 63.19 |
| Life_51 | 21.49 |
| Life_52 | 5.48 |
| Life_53 | 23.26 |
| Life_54 | 56.06 |
| Life_55 | 37.41 |
| Life_56 | 70.36 |
| Life_57 | 17.5 |
| Life_58 | 13.32 |
| Life_59 | 53.14 |
| Life_60 | 15.19 |
| Life_61 | 83.35 |
| Life_62 | 38.59 |
| Life_63 | 1.91 |
| Life_64 | 4.21 |
| Life_mean | 33.63 |

###### Prediction plots

As expected the plots are also very similar to the `RNN` ones, so very stable but always predicting the same values.

##### Experiment 2 `LSTM` `FDOO1` `padding` 🧠 2️⃣ 🦜

I also performed an experiment with `LSTM` with the different type of Regression Head where we remove the step where we compute the mean over all the `L` time steps. Also in this case, as it happened with the `RNN` model, the results are pretty bad with constant predictions, so I do not report here the results.

### `GRU` Model Experiments 🏗️

Let's use the `model_summary` parameter to get the size of the `GRU` model.


```txt
=========================================================================
=================
Total params: 7,371,252
Trainable params: 7,371,252
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 3.56
=========================================================================
=================
Input size (MB): 0.03
Forward/backward pass size (MB): 2.05
Params size (MB): 29.49
Estimated Total Size (MB): 31.57
=========================================================================
=================
```

With `GRU` we are in the middle between `RNN` and `LSTM` in terms of number of parameters.

#### Dataset `FD001`

##### Experiment 1 `GRU` `FDOO1` `padding` 🏗️ 1️⃣ 🦜

Let's use the same configuration used for the `RNN` model.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/3pcg3vot?nw=nwuserfrizzodavide)

Loss behavior very similar to the one of `RNN` and `LSTM`.

###### Metrics Table

Similar to `RNN` and `LSTM`.


```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   31.66
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 63.93 |
| Life_51 | 22.4 |
| Life_52 | 4.71 |
| Life_53 | 24.0 |
| Life_54 | 56.0 |
| Life_55 | 43.02 |
| Life_56 | 68.65 |
| Life_57 | 18.12 |
| Life_58 | 14.0 |
| Life_59 | 52.91 |
| Life_60 | 14.35 |
| Life_61 | 84.19 |
| Life_62 | 32.81 |
| Life_63 | 2.19 |
| Life_64 | 5.0 |
| Life_mean | 33.75 |

###### Prediction plots

Similar to the ones obtained with `RNN,LSTM`.

## `windowed` Approach Experiments 🪟

In this section we report the results of the experiments using the `windowed` approach.

### `RNN` Model Experiments ♻️

#### Dataset `FD001`

##### Experiment 1 `RNN` `FDOO1` `windowed` ♻️ 1️⃣ 🪟

Let's use the same configuration used in the `SSM` models.

| Parameter | Value |
|-----------|-------|
| `model_type` | `RNN` |
| `cmapss_model` | `FD001` |
| `val_idx` | `[0,50]` |
| `test_idx` | `[50,100]` |
| `transformer_type` | 1 (no feature extraction) |
| `window_size` | 20 |
| `scaler` | `MinMaxScaler(-1,1)` |
| `epochs`  | 100    |
| `lr` | 1e-3 |
| `batch_size` | 100 |
| `weight_decay` | 1e-4 |
| `sequence_length` | 170 |
| `n_layers` | 5 |
| `dropout` | 0.0 |
| `activation` | `relu` |
| `final_act` | `glu` |
| `hidden_size` | 128 |
| `d_state` | 64 |
| `loss` | `mae` |
| `eval_loss` | `mse` |


>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/sup9ait6?nw=nwuserfrizzodavide)

As it happened in the `padding` approach the loss quickly saturates at an higher value than in `SSM` so probably there will be worse results.

###### Metrics Table

Actually the results are good but as usual the model predicts always the same range of `RUL` values, as it happened in the `padding` approach. So the predictions are good jsut because the test lifes we have are more or less on the range of values learned by the model.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   28.56
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 58.75 |
| Life_51 | 9.55 |
| Life_52 | 9.34 |
| Life_53 | 18.81 |
| Life_54 | 50.82 |
| Life_55 | 48.23 |
| Life_56 | 63.68 |
| Life_57 | 10.66 |
| Life_58 | 8.85 |
| Life_59 | 47.74 |
| Life_60 | 19.32 |
| Life_61 | 57.25 |
| Life_62 | 27.7 |
| Life_63 | 3.37 |
| Life_64 | 0.16 |
| Life_mean | 28.95 |

###### Prediction plots

The plots look good but we can clearly see that the predicted `RUL` signal is essentially the same across all the lifes.

#### Dataset `FD002`

##### Experiment 1 `RNN` `FDOO2` `windowed` ♻️ 1️⃣ 🪟

Let's use the same configuration used in the `SSM` experiments on `FD002`.

| Parameter | Value |
|-----------|-------|
| `model_type` | `RNN` |
| `cmapss_model` | `FD002` |
| `val_idx` | `[0,131]` |
| `test_idx` | `[131,259]` |
| `transformer_type` | 1 (no feature extraction) |
| `window_size` | 20 |
| `scaler` | `MinMaxScaler(-1,1)` |
| `epochs`  | 100    |
| `lr` | 1e-3 |
| `batch_size` | 100 |
| `weight_decay` | 1e-4 |
| `sequence_length` | 170 |
| `n_layers` | 5 |
| `dropout` | 0.0 |
| `activation` | `relu` |
| `final_act` | `glu` |
| `hidden_size` | 128 |
| `d_state` | 64 |
| `loss` | `mae` |
| `eval_loss` | `mse` |

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/su3jle9w?nw=nwuserfrizzodavide)

Usual behavior of an `RNN` based model, the loss quickly saturates at an higher value than in `SSM` so probably there will be worse results.

###### Metrics Table

The results are worse than the ones of the `SSM` based models and I also checked that the predictions are always in the same range of values.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   35.08
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_131 | 40.0 |
| Life_132 | 15.13 |
| Life_133 | 14.14 |
| Life_134 | 86.0 |
| Life_135 | 37.0 |
| Life_136 | 85.78 |
| Life_137 | 10.23 |
| Life_138 | 34.0 |
| Life_139 | 99.24 |
| Life_140 | 23.0 |
| Life_141 | 9.0 |
| Life_142 | 78.85 |
| Life_143 | 15.0 |
| Life_144 | 39.0 |
| Life_145 | 47.0 |
| Life_mean | 42.22 |

### `LSTM` Model Experiments 🧠

#### Dataset `FD001`

##### Experiment 1 `LSTM` `FDOO1` `windowed` 🧠 1️⃣ 🪟

Let's use the same configuration used in `RNN`.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/fmzf6ry0?nw=nwuserfrizzodavide)

The loss plots are essentially overlapped to the ones of the `RNN` experiment.

###### Metrics Table

The metriccs are almost exactly equal to the `RNN` ones.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   28.56
Name: Life mean Loss, dtype: float64
##################################################
Std eval loss over all the test lifes: Eval Loss   23.16
Name: Life_std, dtype: float64
##################################################
```
| Life  | Eval Loss |
| --- | --- |
| Life_50 | 58.74 |
| Life_51 | 9.54 |
| Life_52 | 9.35 |
| Life_53 | 18.79 |
| Life_54 | 50.81 |
| Life_55 | 48.24 |
| Life_56 | 63.67 |
| Life_57 | 10.65 |
| Life_58 | 8.84 |
| Life_59 | 47.72 |
| Life_60 | 19.33 |
| Life_61 | 57.23 |
| Life_62 | 27.69 |
| Life_63 | 3.38 |
| Life_64 | 0.15 |
| Life_mean | 28.94 |
| Life_std | 22.02 |

Here in terms of `Life_std` we are close to `S4`, so pretty stable but the `Life_mean` is sligthly higher.

###### Prediction plots

Very similar to the `RNN` ones.

#### Dataset `FD002`

##### Experiment 1 `LSTM` `FDOO2` `windowed` 🧠 1️⃣ 🪟

Let's use the same configuration used in `RNN`.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/2lzfp7j8?nw=nwuserfrizzodavide)

The loss plots are essentially overlapped to the ones of the `RNN` experiment.

###### Metrics Table

The result it's exaclty equal to the one obtained with the `RNN` model, this is pretty strange, I have to investigate. It is pretty strange: in fact I tried to execute the script with `ipdb` with the `test_script` and `save_outputs` parameters set to `True` and I saw that the model indeeded changes its feature extraction backbone if I use `RNN` or `LSTM` in the `model_name`. I also checked the `state_dict` of the model and they are different. Now let's see what happens with the `GRU` model.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   35.08
##################################################
Std eval loss over all the test lifes: Eval Loss   24.03
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_131 | 40.0 |
| Life_132 | 15.13 |
| Life_133 | 14.14 |
| Life_134 | 86.0 |
| Life_135 | 37.0 |
| Life_136 | 85.78 |
| Life_137 | 10.23 |
| Life_138 | 34.0 |
| Life_139 | 99.24 |
| Life_140 | 23.0 |
| Life_141 | 9.0 |
| Life_142 | 78.85 |
| Life_143 | 15.0 |
| Life_144 | 39.0 |
| Life_145 | 47.0 |
| Life_mean | 42.22 |
| Life_std | 29.08 |


### `GRU` Model Experiments 🏗️

#### Dataset `FD001`

##### Experiment 1 `GRU` `FDOO1` `windowed` 🏗️ 1️⃣ 🪟

Let's use the same configuration used in `RNN`.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/koe9e5xb?nw=nwuserfrizzodavide)

Also in this case the loss plots are essentially overlapped with the ones obtained with `RNN,LSTM`.

###### Metrics Table

As expected also the metrics are almost the same as the ones obtained with `RNN,LSTM`.
 
```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   28.56
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 58.74 |
| Life_51 | 9.55 |
| Life_52 | 9.35 |
| Life_53 | 18.8 |
| Life_54 | 50.82 |
| Life_55 | 48.23 |
| Life_56 | 63.67 |
| Life_57 | 10.66 |
| Life_58 | 8.85 |
| Life_59 | 47.73 |
| Life_60 | 19.32 |
| Life_61 | 57.25 |
| Life_62 | 27.7 |
| Life_63 | 3.37 |
| Life_64 | 0.16 |
| Life_mean | 28.95 |

###### Prediction plots

Very similar to the `RNN,LSTM` ones.

#### Dataset `FD002`

##### Experiment 1 `GRU` `FDOO2` `windowed` 🏗️ 1️⃣ 🪟

>[!note]
> [Link to the `wandb` run]()

Also here the same exact results as the other `RNN` models and `Transformer`.

## `windowed` Approach Experiments with `Pinball Loss` 🪟 🎈

Let's group here the experiments combining the `windowed` approach with the `Pinball Loss`.

### `RNN` Model Experiments ♻️

#### Dataset `FD001`

##### Experiment 1 `RNN` `FDOO1` `windowed` `Pinball Loss` ♻️ 1️⃣ 🪟 🎈

Let's use the same configuration of the `LSTM` experiment and let's see if also in this case we get exactly the same results or not.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/la2su4y3?nw=nwuserfrizzodavide)

As expected, exactly equal results 😠

### `LSTM` Model Experiments 🧠

#### Dataset `FD001`

##### Experiment 1 `LSTM` `FDOO1` `windowed` `Pinball Loss` 🧠 1️⃣ 🪟 🎈

Let's use the configuration that is bringing the best results in the `SSM` models. I am trying this approach also to see weather I obtain the same exact results on all the `RNN` and `Transformer` models also in this case.


| Parameter | Value |
|-----------|-------|
| `model_type` | `LSTM` |
| `cmapss_model` | `FD001` |
| `val_idx` | `[0,50]` |
| `test_idx` | `[50,100]` |
| `transformer_type` | 1 (no feature extraction) |
| `window_size` | 20 |
| `scaler` | `MinMaxScaler(-1,1)` |
| `epochs`  | 100    |
| `lr` | 1e-3 |
| `batch_size` | 100 |
| `weight_decay` | 1e-4 |
| `sequence_length` | 170 |
| `n_layers` | 5 |
| `dropout` | 0.0 |
| `activation` | `relu` |
| `final_act` | `glu` |
| `hidden_size` | 128 |
| `d_state` | 64 |
| `loss` | `pinball` |
| `tau` | 0.7 |
| `eval_loss` | `mse` |


>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/n33god7n?nw=nwuserfrizzodavide)

The `eval_loss` plots have the same shape as the ones obtained in the previous experiments but with a sligthly lower set of values.

###### Metrics Table

The results are better than in the previous experiment: the average `RMSE` loss drops from 42.22 to

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   31.42
##################################################
Std eval loss over all the test lifes: Eval Loss   20.10
##################################################
```

## `windowed` Approach + Quantile Regression Experiments 🪟 🌗

Here we will list the results of the `windowed` approach experiments using the models in the Quantile Regression mode. The experiments will be structured as explained [[ssm_experiments#`windowed` Approach + Quantile Regression Experiments 🪟 🌗|here]].

Considering the fact that the results of the experiments performed on the `RNN` based models are always very similar we will do the experiments on `LSTM` for the moment.

### `LSTM` Model Experiments 🪟 🌗

Let's use the same configuration used in the `SSM` experiments and let's start directly with the multi run experiments.

| Parameter | Value |
|-----------|-------|
| `model_type` | `LSTM` |
| `cmapss_model` | `FD001` |
| `val_idx` | `[0,50]` |
| `test_idx` | `[50,100]` |
| `transformer_type` | 1 (no feature extraction) |
| `window_size` | 20 |
| `scaler` | `MinMaxScaler(-1,1)` |
| `epochs`  | 100    |
| `lr` | 1e-3 |
| `batch_size` | 100 |
| `weight_decay` | 1e-4 |
| `sequence_length` | 170 |
| `n_layers` | 5 |
| `dropout` | 0.0 |
| `activation` | `relu` |
| `final_act` | `glu` |
| `hidden_size` | 128 |
| `d_state` | 64 |
| `loss` | `quantile_reg` |
| `eval_loss` | `mse` |
| `quantile_dist` | `uniform` |
| `bounds` | `[0.1,0.9]` |
| `quantiles` | `[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]` |

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/ufv8pp47?nw=nwuserfrizzodavide)

###### Metrics Table

Worse than `S4,S5`, better than `S4D` but only on `Life_mean`, `Life_median` is lower for `S4D`. Here the differences in the metrics between the different quantiles are minimal and that's probably because, as it always happens to the `RNN` based models, the predictions are always in the same range of values independently on the life, and also on the quantile in this case.


```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1    28.60
quantile_0.5    28.63
quantile_0.9    28.60
##################################################
Median eval loss over all the test lifes:
quantile_0.1    23.46
quantile_0.5    23.08
quantile_0.9    23.50
##################################################
Std eval loss over all the test lifes:
quantile_0.1    22.95
quantile_0.5    23.05
quantile_0.9    22.94
##################################################
```

| Life  | quantile_0.1 | quantile_0.5 | quantile_0.9 |
| --- | --- | --- | --- |
| Life_51 | 58.9 | 59.3 | 58.89 |
| Life_52 | 9.86 | 10.24 | 9.85 |
| Life_53 | 9.19 | 8.79 | 9.2 |
| Life_54 | 19.0 | 19.41 | 18.98 |
| Life_55 | 51.04 | 51.44 | 51.01 |
| Life_56 | 48.08 | 47.67 | 48.09 |
| Life_57 | 63.85 | 64.25 | 63.84 |
| Life_58 | 10.88 | 11.28 | 10.87 |
| Life_59 | 9.18 | 9.57 | 9.14 |
| Life_60 | 47.89 | 48.3 | 47.88 |
| Life_61 | 19.16 | 18.76 | 19.17 |
| Life_62 | 57.62 | 58.0 | 57.61 |
| Life_63 | 27.87 | 28.27 | 27.86 |
| Life_64 | 3.24 | 2.84 | 3.24 |
| Life_65 | 0.71 | 0.98 | 0.75 |
| Life_mean | 29.1 | 29.27 | 29.09 |
| Life_median | 23.52 | 23.84 | 23.52 |
| Life_std | 21.37 | 21.46 | 21.36 |

###### Prediction plots


