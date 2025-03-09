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

