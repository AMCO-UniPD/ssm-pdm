---
id: transformer_exp
aliases: []
tags:
  - experiments
  - ssm_pdm
---

# `Transformer` `PdM` Experiments

In this note I will report the results obtained with the `Transformer` based models (i.e. `Transformer`, `Informer`).

## `padding` approach experiments 🦜

Let's consider also here the `padding` approach first.

### `Transformer` Model Experiments 🤖

Let's use the `model_summary` parameter to get the size of the `Transformer` model.

>[!note]
> Here I reduced the `d_model` parameter from 512 to 128 because otherwise the model was too big (more than `19M` parameters) and since I know that `Transformer` models tend to fall into a `CUDAOutOfMemory` error I decided to reduce the size of the model. Maybe I will do the same also on the `RNN` based models.

```txt
=========================================================================
==========================================
Total params: 3,624,692
Trainable params: 3,624,692
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 3.03
=========================================================================
==========================================
Input size (MB): 0.03
Forward/backward pass size (MB): 59.91
Params size (MB): 12.13
Estimated Total Size (MB): 72.06
=========================================================================
==========================================
```

The number of parameters it's really high. To have an idead of how high it is let's consider that `S4` has `3.3M` parameters with `d_model=512` and here we are already higher than that with `d_model=128`.

#### Dataset `FD001`

##### Experiment 1 `Transformer` `FDOO1` `padding` 🤖 1️⃣ 🦜

A part from the `d_model` parameter we have more or less the same configuration of the `S4` model.

| Parameter | Value |
|-----------|-------|
| `model_type` | `Transformer` |
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
| `activation` | `relu` |
| `final_act` | `glu` |
| `hidden_size` | 128 |
| `d_ff` | 2048 |
| `n_heads` | 8 |
| `loss` | `mae` |
| `eval_loss` | `mse` |

>[!warning]
> Remember to change `d_model` from 512 to 128 in `ssm_config.yaml` before starting the experiment.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/fmvh1hdx?nw=nwuserfrizzodavide)

In the `wandb` loss plots the behavior is very similar to the one observed on the `RNN` based models but than after 30 epochs it starts to overfit.

###### Metrics table

The metrics table is very bad, the worse seen up to now.

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 105.29 |
| Life_51 | 23.27 |
| Life_52 | 17.41 |
| Life_53 | 86.23 |
| Life_54 | 122.75 |
| Life_55 | 12.87 |
| Life_56 | 95.86 |
| Life_57 | 30.47 |
| Life_58 | 87.67 |
| Life_59 | 96.58 |
| Life_60 | 15.46 |
| Life_61 | 50.32 |
| Life_62 | 62.07 |
| Life_63 | 17.54 |
| Life_64 | 85.12 |
| Life_mean | 60.59 |

###### Prediction plots

The prediction plots are similar to the ones observe in the `RNN` models (so with a stable decreasing line) but it can be seen that the error is higher → the prediction lines are futher from the real ones.

Differently from the `RNN` based models though here the prediction are not all in the same `RUL` range, the problem is that the range predicted are very far from the real ones. Here I think that the main problem is that the model overfits and so I think I need to try to add some `dropout` layers to the model.

##### Experiment 2 `Transformer` `FDOO1` `padding` 🤖 2️⃣ 🦜

I realized that I was using a slightly different way of obtaining the predictions with the `fc` layers in the `forward` pass in the `Transformers` and `RNN` based models. I used a sort of Global Average Pooling step. Now I removed it using a single `nn.Linear` layer that brings us form `(B,L,H`) to `(B,L,1)` as I did for the `SSM` models. Let's try to do an experiment with this new architecture and let's see what happens.

In this experiment let's also try to use `d_model=512`, maybe `Transformer` was bad in the first experiment because `d_model` was just 128.


>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/botmirtm?nw=nwuserfrizzodavide)

###### Metrics table

Nope, even worse

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 107.44 |
| Life_51 | 59.19 |
| Life_52 | 47.98 |
| Life_53 | 84.39 |
| Life_54 | 118.52 |
| Life_55 | 32.99 |
| Life_56 | 105.11 |
| Life_57 | 55.29 |
| Life_58 | 87.82 |
| Life_59 | 98.27 |
| Life_60 | 41.44 |
| Life_61 | 82.03 |
| Life_62 | 75.13 |
| Life_63 | 44.14 |
| Life_64 | 90.41 |
| Life_mean | 75.34 |

###### Prediction plots

With this approach the plots are very bad, the predictions are essentially constant, a part from the final part of the life where they start to oscillate. Maybe the reason why this Regression Head architecture works better in the `SSM` model and worse on `Transformer` model it's because of the different way the `Transformer` model is able to capture the temporal dependencies in the data, since it uses the Attention mechanism while `SSM` models are more similar to `RNN` models.

##### Experiment 3 `Transformer` `FDOO1` `padding` 🤖 3️⃣ 🦜

Let's try to use the original configuration of the `Transformer` model but with `d_model=512`.


>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/a5870vt7?nw=nwuserfrizzodavide)

Here it started immediately to overfit.

###### Metrics table

Interestingly though the average metrics are sligthly better then the previous experiments.


```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   60.22
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 105.95 |
| Life_51 | 23.31 |
| Life_52 | 18.34 |
| Life_53 | 85.64 |
| Life_54 | 121.98 |
| Life_55 | 11.79 |
| Life_56 | 97.73 |
| Life_57 | 31.4 |
| Life_58 | 87.47 |
| Life_59 | 96.73 |
| Life_60 | 15.19 |
| Life_61 | 50.04 |
| Life_62 | 63.17 |
| Life_63 | 18.46 |
| Life_64 | 86.52 |
| Life_mean | 60.91 |

###### Prediction plots

The plots look more similar to the ones of the first experiment.

##### Experiment 4 `Transformer` `FDOO1` `padding` 🤖 4️⃣ 🦜

Let's try to set `dropout=0.2` and reduce `d_ff` to 128.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/cchwq1ol?nw=nwuserfrizzodavide)

In the loss plots the minimum `val_loss` was reached faster and it's smaller than the minimum one obtained in the previous experiment but then the `val_loss` still increases and then saturates after some epochs.

###### Metrics table

The results are still not very good.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   65.63
Name: Life mean Loss, dtype: float64
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 106.0 |
| Life_51 | 23.42 |
| Life_52 | 21.13 |
| Life_53 | 89.86 |
| Life_54 | 128.41 |
| Life_55 | 12.75 |
| Life_56 | 97.62 |
| Life_57 | 31.21 |
| Life_58 | 98.27 |
| Life_59 | 97.67 |
| Life_60 | 17.16 |
| Life_61 | 49.25 |
| Life_62 | 63.98 |
| Life_63 | 20.83 |
| Life_64 | 102.84 |
| Life_mean | 64.03 |

###### Prediction plots

The plors are similar to the ones of the previous experiments.

### `Informer` Model Experiments 🧙‍♂️

Let's use the `model_summary` parameter to get the size of the `Informer` model. Here we had to use the manual computation of `model_summary` because `torchinfo` was not able to compute the number of parameters of the model.

```txt
##################################################
Total parameters: 3233524
Trainable parameters: 3233524
##################################################
```

The number of parameters is similar to the one of the `Transformer` model with `d_model=128`.

#### Dataset `FD001`

##### Experiment 1 `Informer` `FDOO1` `padding` 🧙‍♂️ 1️⃣ 🦜

We will us the same configuration of the `Transformer` model.

>[!warning]
> Remember to change `d_model` from 512 to 128 in `ssm_config.yaml` before starting the experiment.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/6ywl9v1a?nw=nwuserfrizzodavide)

The trend of the losses on `wandb` is more similar to the ones we observed in the `RNN` based models: the loss quickly reaches is minimum value in the first epochs and then saturates. The saturation value is sligthly higher than the ones reached in the `RNN` based models.

###### Metrics table

The metrics are slightly worse than `SSM` and `RNN` based models but at least they are much better than the `Transformer` model.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   32.54
Name: Life mean Loss, dtype: float64
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 73.47 |
| Life_51 | 21.24 |
| Life_52 | 3.2 |
| Life_53 | 27.79 |
| Life_54 | 75.55 |
| Life_55 | 11.39 |
| Life_56 | 70.07 |
| Life_57 | 12.48 |
| Life_58 | 36.15 |
| Life_59 | 58.98 |
| Life_60 | 5.79 |
| Life_61 | 67.79 |
| Life_62 | 41.62 |
| Life_63 | 7.45 |
| Life_64 | 32.55 |
| Life_mean | 36.37 |

###### Prediction plots

The plots are similar to the ones produced by `RNN` based models but the predictions are more stable. Moreover the models also predicts different `RUL` ranges for the different lifes.

##### Experiment 2 `Informer` `FDOO1` `padding` 🧙‍♂️ 2️⃣ 🦜

Like with `Transformer` let's add `dropout=0.2` and reduce `d_ff` to 128.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/ia6urinl?nw=nwuserfrizzodavide)

###### Metrics table

Even thought the loss seemed pretty equal to the previous experiment the results are actually improving a little bit.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   31.31
Name: Life mean Loss, dtype: float64
##################################################
```

|  | Eval Loss |
| --- | --- |
| Life_50 | 62.09 |
| Life_51 | 20.83 |
| Life_52 | 6.3 |
| Life_53 | 21.99 |
| Life_54 | 53.93 |
| Life_55 | 44.92 |
| Life_56 | 66.93 |
| Life_57 | 16.48 |
| Life_58 | 11.79 |
| Life_59 | 51.09 |
| Life_60 | 16.07 |
| Life_61 | 82.81 |
| Life_62 | 31.06 |
| Life_63 | 1.48 |
| Life_64 | 2.62 |
| Life_mean | 32.69 |

###### Prediction plots

The plots are similar to the previous experiment, although we can see that the predictions have improved a bit in some lifes.

## `windowed` approach experiments 🪟

In this section we will report the results obtained in the `windowed` approach 🪟

### `Transformer` Model Experiments 🤖

#### Dataset `FD001`

##### Experiment 1 `Transformer` `FDOO1` `windowed` 🤖 1️⃣ 🪟

Let's use the configuration used in the `SSM` and `RNN` experiments.

| Parameter | Value |
|-----------|-------|
| `model_type` | `RULTransformer` |
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
| `d_ff` | 128 |
| `n_heads` | 8 |
| `loss` | `mae` |
| `eval_loss` | `mse` |

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/6agf0faz?nw=nwuserfrizzodavide)

The loss plots have a similar trend to the one of the `RNN` based models but the loss value at which it saturates is slightly lower.

###### Metrics table

The results are not bad at all but still slightly worse than the best run obtained with `S4`.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   25.70
Name: Life mean Loss, dtype: float64
##################################################
Std eval loss over all the test lifes: Eval Loss   20.66
Name: Life_std, dtype: float64
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 44.99 |
| Life_51 | 12.31 |
| Life_52 | 16.82 |
| Life_53 | 40.33 |
| Life_54 | 70.19 |
| Life_55 | 28.73 |
| Life_56 | 10.6 |
| Life_57 | 6.86 |
| Life_58 | 30.37 |
| Life_59 | 57.98 |
| Life_60 | 7.34 |
| Life_61 | 21.2 |
| Life_62 | 7.24 |
| Life_63 | 15.15 |
| Life_64 | 24.3 |
| Life_mean | 26.29 |
| Life_std | 18.77 |

###### Prediction plots

The plots are similar to the ones we have already seen in the other `windowed` experiments, smooth decreasing lines that are overlapped with the real `RUL` signals in some lifes and in some others they are a bit far from the real ones.

#### Dataset `FD002`

##### Experiment 1 `Transformer` `FDOO2` `windowed` 🤖 1️⃣ 🪟

Let's use the configuration used in the `SSM` and `RNN` experiments.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/go2r5g2o?nw=nwuserfrizzodavide)

The loss plots are quite overlapped with the ones of `LSTM,RNN`, I hope that we do not get the exact same results because that would make no sense at all.

Ok we are getting the same results as the `RNN` experiments on `FD002` which does not make sense.

### `Informer` Model Experiments 🧙‍♂️

#### Dataset `FD001`

##### Experiment 1 `Informer` `FDOO1` `windowed` 🧙‍♂️ 1️⃣ 🪟

Let's use the configuration used in the `SSM` and `RNN` experiments.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/zj9566td?nw=nwuserfrizzodavide)

This model is the slowest of all in terms of training times. The loss plots are not great either, the loss goes down in the first epochs and than it settles at a value close to the one to which the `RULTransformer` saturates but then it starts oscillating a lot.

###### Metrics table

The results are not great, they are the worst ones after `S5,S4D` (however the situation with `S5,S4D` is a bit peculiar as we have seen).


```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   33.31
##################################################
Std eval loss over all the test lifes: Eval Loss   26.97
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 82.4 |
| Life_51 | 8.7 |
| Life_52 | 1.35 |
| Life_53 | 57.56 |
| Life_54 | 89.13 |
| Life_55 | 33.17 |
| Life_56 | 68.75 |
| Life_57 | 10.1 |
| Life_58 | 55.99 |
| Life_59 | 55.12 |
| Life_60 | 2.56 |
| Life_61 | 30.11 |
| Life_62 | 35.27 |
| Life_63 | 3.28 |
| Life_64 | 51.17 |
| Life_mean | 38.98 |
| Life_std | 28.55 |

###### Prediction plots

The plots are similar to the `Transformer` ones but with higher errors.

#### Dataset `FD002`

##### Experiment 1 `Informer` `FDOO2` `windowed` 🧙‍♂️ 1️⃣ 🪟

Let's try to use the same configuration as the `RNN` based models.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/sjzdzz9v?nw=nwuserfrizzodavide)

This is the last try: if we get the same results as the last `RNN,LSTM` and `RULTransformer` experiments then there is something stange going on.

Here it seems that it is following the same trend as the `RULTransformer` model. In the first epochs the loss goes down differently than the `RNN` models but then it starts to exactly overlap with the `RNN` models.

As expected same exact results as the `RNN` models.

## `windowed` Approach + Pinball Loss Experiments 🪟 🎈

In this section we will report the results obtained in the `windowed` approach with the `Pinball Loss` 🪟 🎈

### `Transformer` Model Experiments 🤖

#### Dataset `FD001`

##### Experiment 1 `Transformer` `FDOO1` `windowed` `Pinball` 🤖 1️⃣ 🪟 🎈

Same configuration used in the `RN` experiments. I am doing the experiment also here on `Transformer` to veriufy weather we are gettin the exact same results.

| Parameter | Value |
|-----------|-------|
| `model_type` | `Transformer` |
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
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/lubq7qo2?nw=nwuserfrizzodavide)


###### Metrics table

Ok now at least we are able to obtain a different  result from the `RNN` models. Also in this case the performances are improving.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   26.89
##################################################
Std eval loss over all the test lifes: Eval Loss   16.67
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 26.56 |
| Life_51 | 60.69 |
| Life_52 | 30.25 |
| Life_53 | 30.84 |
| Life_54 | 58.57 |
| Life_55 | 34.93 |
| Life_56 | 6.5 |
| Life_57 | 19.95 |
| Life_58 | 22.18 |
| Life_59 | 39.73 |
| Life_60 | 20.54 |
| Life_61 | 13.32 |
| Life_62 | 19.77 |
| Life_63 | 28.71 |
| Life_64 | 18.53 |
| Life_mean | 28.74 |
| Life_std | 14.57 |


###### Prediction plots

Looking at the plots the results are not bads at all , in most of the lifes the predicted `RUL` signals are very close to the true ones, however the problem is that there are several overestimation errors that may be quite dangerous.

### `Informer` Model Experiments 🧙‍♂️

#### Dataset `FD001`

##### Experiment 1 `Informer` `FDOO1` `windowed` `Pinball` 🧙‍♂️ 1️⃣ 🪟 🎈

Since with `Transformer` we obtain different results from `RNN` models, let's see what happens with the `Informer`.


>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/jhglagaq?nw=nwuserfrizzodavide)


## `windowed` Approach Experiments + Quantile Regression 🪟 🌗

In this section we group the results obtained in the `windowed` approach in the Quantile Regression mode,  the experiments will be performed [[ssm_experiments#`windowed` Approach + Quantile Regression Experiments 🪟 🌗|as explained here]].

### `Transformer` Model Experiments 🤖 🌗

#### Dataset `FD001`

##### Experiment 1 `Transformer`` `FDOO1` `windowed` 🪟 🌗

Let's use the same configuration used in the `SSM` and `RNN` experiments and multi run experiments.


| Parameter | Value |
|-----------|-------|
| `model_type` | `Transformer` |
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
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/kf0b2sr7?nw=nwuserfrizzodavide)


###### Metrics table

Similar performances to `S5` but with a slightly higher `std` and in particular the best quantile here is 0.1 and 0.9 is the worse. So probably here the model tends more to overestimate the signal when it is wrong and so with some underestimation (as the one brought by quantile 0.1) we obtain better results.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1    29.95
quantile_0.5    30.61
quantile_0.9    37.66
Name: Life_mean, dtype: float64
##################################################
Median eval loss over all the test lifes:
quantile_0.1    21.48
quantile_0.5    28.09
quantile_0.9    33.72
Name: Life_median, dtype: float64
##################################################
Std eval loss over all the test lifes:
quantile_0.1    25.33
quantile_0.5    20.15
quantile_0.9    21.94
Name: Life_std, dtype: float64
##################################################
```

| Life  | quantile_0.1 | quantile_0.5 | quantile_0.9 |
| --- | --- | --- | --- |
| Life_51 | 26.55 | 27.0 | 26.25 |
| Life_52 | 8.01 | 15.92 | 20.91 |
| Life_53 | 4.97 | 39.95 | 41.56 |
| Life_54 | 14.96 | 12.25 | 10.11 |
| Life_55 | 25.73 | 24.2 | 17.88 |
| Life_56 | 20.42 | 30.87 | 80.88 |
| Life_57 | 69.45 | 42.76 | 31.44 |
| Life_58 | 27.41 | 19.72 | 21.53 |
| Life_59 | 8.44 | 8.06 | 24.26 |
| Life_60 | 37.88 | 16.02 | 15.31 |
| Life_61 | 1.01 | 29.11 | 51.87 |
| Life_62 | 34.39 | 30.58 | 26.82 |
| Life_63 | 42.94 | 19.5 | 4.81 |
| Life_64 | 16.76 | 7.8 | 35.52 |
| Life_65 | 14.56 | 10.15 | 33.72 |
| Life_mean | 23.57 | 22.26 | 29.52 |
| Life_median | 22.0 | 20.99 | 26.54 |
| Life_std | 16.53 | 10.31 | 17.42 |

###### Prediction plots

These plots are a bit different from the ones observed up to now, which is expected considering that we are using a completely different architecture.

- In `Life_53,Life_58,Life_61,Life_64` we have quantile 0.1 underestimating (or almost overlapping) the real signal and the other quantiles overestimating it.
- In `Life_62` the predictions do not form a straight line but rather a zig zagging line
- There is not a life where the predictions are very good from all the quantiles, probably the life with the best metrics is `Life_52` followed by `Life_59`.

##### Experiment 1 bis `Transformer` `FDOO1` `windowed` 🤖 2️⃣ 🌗

Let's continue the experiment adding `quantiles=[0.25,0.75]` as done in the other models.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/2jb3hzrx?nw=nwuserfrizzodavide)

###### Metrics table

Following the results obtained in the first experiment now the best quantile is quantile 0.25 and the best one in terms of `Life_median` is quantile 0.1.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     29.95
quantile_0.25    28.14
quantile_0.5     30.61
quantile_0.75    36.10
quantile_0.9     37.66
##################################################
Median eval loss over all the test lifes:
quantile_0.1     21.48
quantile_0.25    22.75
quantile_0.5     28.09
quantile_0.75    32.59
quantile_0.9     33.72
##################################################
Std eval loss over all the test lifes:
quantile_0.1     25.33
quantile_0.25    24.52
quantile_0.5     20.15
quantile_0.75    22.59
quantile_0.9     21.94
##################################################
```

| Life  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_51 | 26.55 | 26.6 | 27.0 | 26.77 | 26.25 |
| Life_52 | 8.01 | 8.38 | 15.92 | 10.6 | 20.91 |
| Life_53 | 4.97 | 3.54 | 39.95 | 40.99 | 41.56 |
| Life_54 | 14.96 | 14.96 | 12.25 | 5.2 | 10.11 |
| Life_55 | 25.73 | 25.73 | 24.2 | 18.62 | 17.88 |
| Life_56 | 20.42 | 23.55 | 30.87 | 80.33 | 80.88 |
| Life_57 | 69.45 | 46.82 | 42.76 | 31.91 | 31.44 |
| Life_58 | 27.41 | 15.9 | 19.72 | 21.17 | 21.53 |
| Life_59 | 8.44 | 8.44 | 8.06 | 23.43 | 24.26 |
| Life_60 | 37.88 | 33.93 | 16.02 | 15.81 | 15.31 |
| Life_61 | 1.01 | 2.48 | 29.11 | 50.65 | 51.87 |
| Life_62 | 34.39 | 35.12 | 30.58 | 27.23 | 26.82 |
| Life_63 | 42.94 | 30.33 | 19.5 | 4.28 | 4.81 |
| Life_64 | 16.76 | 12.31 | 7.8 | 34.84 | 35.52 |
| Life_65 | 14.56 | 14.56 | 10.15 | 14.78 | 33.72 |
| Life_mean | 23.57 | 20.18 | 22.26 | 27.11 | 29.52 |
| Life_median | 22.0 | 18.04 | 20.99 | 25.1 | 26.54 |
| Life_std | 16.53 | 12.04 | 10.31 | 18.27 | 17.42 |

###### Metrics table Pinball Loss

Here there is a significant difference between the best quantiles (i.e. 0.1,0.25) and the others. This is probably because already using the `RMSE` we had better errors with quantiles 0.1 and 0.25 and now with Pinball Loss this difference is enlarged.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     14.28
quantile_0.25    14.08
quantile_0.5     17.60
quantile_0.75    22.46
quantile_0.9     23.64
##################################################
Median eval loss over all the test lifes:
quantile_0.1      9.99
quantile_0.25     9.06
quantile_0.5     13.00
quantile_0.75    21.48
quantile_0.9     23.23
##################################################
Std eval loss over all the test lifes:
quantile_0.1     13.72
quantile_0.25    14.22
quantile_0.5     13.58
quantile_0.75    16.23
quantile_0.9     15.93
##################################################
```

###### Prediction plots

See [[ssm_experiments#Prediction Plots bis|here]]

#### Dataset `FD002`

##### Experiment 1 `Transformer` `FDOO2` `windowed` 🤖 1️⃣ 🌗

Let's use the configuration used in the `SSM` and `RNN` experiments and multi run experiments.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/3r8nwzrx?nw=nwuserfrizzodavide)

This was faster than the `Informer`, it took about 4 hours and 20 minutes to complete all the 25 experiments.

###### Metrics Table

Not much difference between the different quantiles. The best ones in terms of `Life_mean` are quantile 0.25 and 0.5, while the best one in terms of `Life_median` is quantile 0.1. In terms of the `RMSE` values they are close to the `Informer` ones, so similar to `S4` (and better than `S5,S4D`) but, as for the `Informer`, this model is way less efficient than the `SSM` models.


```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     35.09
quantile_0.25    35.08
quantile_0.5     35.08
quantile_0.75    35.09
quantile_0.9     36.55
##################################################
Median eval loss over all the test lifes:
quantile_0.1     30.83
quantile_0.25    31.25
quantile_0.5     31.10
quantile_0.75    30.89
quantile_0.9     31.19
##################################################
Std eval loss over all the test lifes:
quantile_0.1     23.98
quantile_0.25    23.86
quantile_0.5     23.90
quantile_0.75    23.97
quantile_0.9     24.92
##################################################
```

| Life  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_132 | 39.82 | 40.26 | 40.1 | 39.88 | 40.17 |
| Life_133 | 15.28 | 14.85 | 15.01 | 15.24 | 14.93 |
| Life_134 | 14.01 | 14.44 | 14.28 | 14.05 | 14.35 |
| Life_135 | 86.18 | 85.74 | 85.89 | 86.12 | 85.83 |
| Life_136 | 36.82 | 37.25 | 37.1 | 36.87 | 37.16 |
| Life_137 | 85.92 | 85.49 | 85.65 | 85.88 | 85.58 |
| Life_138 | 10.13 | 10.53 | 10.37 | 10.17 | 10.44 |
| Life_139 | 33.82 | 34.26 | 34.1 | 33.87 | 34.17 |
| Life_140 | 99.39 | 98.99 | 99.14 | 99.35 | 99.07 |
| Life_141 | 23.17 | 22.76 | 22.9 | 23.14 | 22.85 |
| Life_142 | 9.16 | 8.72 | 8.89 | 9.11 | 8.8 |
| Life_143 | 78.99 | 78.56 | 78.72 | 78.95 | 78.65 |
| Life_144 | 14.83 | 15.26 | 15.1 | 14.88 | 15.18 |
| Life_145 | 39.17 | 38.74 | 38.9 | 39.12 | 38.82 |
| Life_146 | 46.83 | 47.26 | 47.1 | 46.88 | 47.18 |
| Life_mean | 42.23 | 42.21 | 42.22 | 42.23 | 42.21 |
| Life_median | 38.0 | 38.0 | 38.0 | 37.99 | 37.99 |
| Life_std | 28.91 | 28.73 | 28.8 | 28.89 | 28.77 |

###### Metrics Table Pinball Loss

Here there were similar results among the different quantiles in `RMSE` and also here they remain quite similar with a slight advantage for 0.1 and 0.75.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     15.76
quantile_0.25    15.85
quantile_0.5     15.82
quantile_0.75    15.77
quantile_0.9     16.15
##################################################
Median eval loss over all the test lifes:
quantile_0.1     13.09
quantile_0.25    12.96
quantile_0.5     13.01
quantile_0.75    13.08
quantile_0.9     13.15
##################################################
Std eval loss over all the test lifes:
quantile_0.1     10.60
quantile_0.25    10.66
quantile_0.5     10.64
quantile_0.75    10.60
quantile_0.9     10.90
##################################################
```

###### Prediction plots

See [[ssm_experiments#Prediction Plots `FD002`|here]]. As for the `Informer` the worst lifes are `Life_135,Life_140`, also here `Life_141` is not bad as in the `SSM` models.

### `Informer` Model Experiments 🧙‍♂️ 🌗

#### Dataset `FD001`

##### Experiment 1 `Informer` `FDOO1` `windowed` 🧙‍♂️ 🌗

We will do a multi run experiment with the same configuration used in the other models.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/gzwxyzir?nw=nwuserfrizzodavide)


###### Metrics table

The results are better than the ones of the `Transformer` which is quite surprising considering the results of the previous experiments. On the other hand it's also a good result because the `Informer` is supposed to be an improvement of the `Transformer` for time series data so theoretically it should work better in this task. With the respect to the other models it is better than `S5,S4D` but still not as good as `S4`. In any case a point against the `Informer` is that it is much slower than the `SSM` models. As in most of the other models the best quantile is still quantile 0.5.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1    28.91
quantile_0.5    25.86
quantile_0.9    31.99
Name: Life_mean, dtype: float64
##################################################
Median eval loss over all the test lifes:
quantile_0.1    25.52
quantile_0.5    19.08
quantile_0.9    29.16
Name: Life_median, dtype: float64
##################################################
Std eval loss over all the test lifes:
quantile_0.1    18.71
quantile_0.5    18.03
quantile_0.9    16.87
Name: Life_std, dtype: float64
##################################################
```


| Life        | quantile_0.1 | quantile_0.5 | quantile_0.9 |
| ----------- | ------------ | ------------ | ------------ |
| Life_51     | 63.46        | 56.27        | 48.48        |
| Life_52     | 10.39        | 16.99        | 21.51        |
| Life_53     | 8.16         | 12.91        | 17.79        |
| Life_54     | 25.43        | 28.42        | 18.71        |
| Life_55     | 51.88        | 58.01        | 39.96        |
| Life_56     | 38.05        | 36.55        | 51.32        |
| Life_57     | 66.52        | 58.72        | 51.64        |
| Life_58     | 7.77         | 15.86        | 13.15        |
| Life_59     | 26.38        | 17.83        | 24.11        |
| Life_60     | 55.15        | 51.05        | 42.67        |
| Life_61     | 9.2          | 12.91        | 23.35        |
| Life_62     | 29.5         | 27.93        | 28.47        |
| Life_63     | 31.51        | 26.29        | 24.42        |
| Life_64     | 7.07         | 7.7          | 20.7         |
| Life_65     | 21.07        | 15.34        | 27.12        |
| Life_mean   | 30.1         | 29.52        | 30.23        |
| Life_median | 27.94        | 27.11        | 25.77        |
| Life_std    | 19.47        | 17.0         | 12.2         |

###### Prediction plots

Let's start with some general comments on the plots from `run_1` and then we will see if there is also something to note on the other runs.

- In `Life_51,Life_54,Life_55,Life_57,Life_60` similarly  to `S4,S5` we have predictions a bit far from the true `RUL` signal with quantile 0.9 being the closest.
- In `Life_53,Life_61,Life_64` differently from `Tranformer` we have a situation similar to `Life_51,Life_54,Life_55,Life_57,Life_60` but with quantile 0.1 being the closest one to the true `RUL` values. 
- In `Life_58` instead it's quantile 0.5 to be almost overlapped to the true `RUL`, however the other two are not too far behind.  Another interesting thing to not in this life is that quantile 0.5 has a peculiar little zig zag shape at its very beginning and end. In `run_2` the shape of the `RUL` predicted signal is even weirder for quantile 0.9. 
- As in `Transformer`, `Life_62` has a not perfectly smooth line prediction also here. 
- We do not have lifes with all signals overlapped to the true values, probably the lifes with the best predictions are `Life_58,Life_64`.

Things to note in other runs:

- In `run_2` `Life_65` (which is known to contain overestimation errors normally) has all overestimation errors except for quantile 0.5
- In `run_5` quantile 0.9 predictions are a bit noisy.

##### Experiment 1 bis `Informer` `FDOO1` `windowed` 🧙‍♂️ 2️⃣ 🌗

Let's continue the experiment adding `quantiles=[0.25,0.75]` as done in the other models.


>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/3etxg34y?nw=nwuserfrizzodavide)

###### Metrics table

In the `Informer` the best quantile remains 0.5 which is also significantly better than the others in the `Life_median` metric. 


```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     28.91
quantile_0.25    29.94
quantile_0.5     25.86
quantile_0.75    27.16
quantile_0.9     31.99
##################################################
Median eval loss over all the test lifes:
quantile_0.1     25.52
quantile_0.25    25.73
quantile_0.5     19.08
quantile_0.75    20.94
quantile_0.9     29.16
##################################################
Std eval loss over all the test lifes:
quantile_0.1     18.71
quantile_0.25    18.30
quantile_0.5     18.03
quantile_0.75    16.54
quantile_0.9     16.87
##################################################
```


| Life  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_51 | 63.46 | 62.9 | 56.27 | 54.01 | 48.48 |
| Life_52 | 10.39 | 11.89 | 16.99 | 17.49 | 21.51 |
| Life_53 | 8.16 | 10.98 | 12.91 | 15.78 | 17.79 |
| Life_54 | 25.43 | 23.23 | 28.42 | 25.29 | 18.71 |
| Life_55 | 51.88 | 50.93 | 58.01 | 46.34 | 39.96 |
| Life_56 | 38.05 | 45.03 | 36.55 | 38.82 | 51.32 |
| Life_57 | 66.52 | 65.9 | 58.72 | 54.77 | 51.64 |
| Life_58 | 7.77 | 10.02 | 15.86 | 14.03 | 13.15 |
| Life_59 | 26.38 | 25.73 | 17.83 | 17.34 | 24.11 |
| Life_60 | 55.15 | 54.41 | 51.05 | 38.94 | 42.67 |
| Life_61 | 9.2 | 11.26 | 12.91 | 17.48 | 23.35 |
| Life_62 | 29.5 | 28.62 | 27.93 | 27.93 | 28.47 |
| Life_63 | 31.51 | 30.43 | 26.29 | 24.08 | 24.42 |
| Life_64 | 7.07 | 10.35 | 7.7 | 5.47 | 20.7 |
| Life_65 | 21.07 | 20.83 | 15.34 | 20.57 | 27.12 |
| Life_mean | 30.1 | 30.83 | 29.52 | 27.89 | 30.23 |
| Life_median | 27.94 | 27.18 | 27.11 | 24.68 | 25.77 |
| Life_std | 19.47 | 18.72 | 17.0 | 14.23 | 12.2 |

###### Prediction plots

See [[ssm_experiments#Prediction Plots bis|here]]

#### Dataset `FD002`

##### Experiment 1 `Informer` `FDOO2` `windowed` 🧙‍♂️ 1️⃣ 🌗

Let's use the same configuration of `Transformer`.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/gyt1q3th?nw=nwuserfrizzodavide)

After almost 20 hours 😱 (19h and 54 min according to the terminal) all the `RULInformer` experiments finished.

###### Metrics table

The metric values are similar across the different quantiles. The best one in terms of `Life_mean` is quantile 0.1 and 0.75 in terms of `Life_median`. Interestingly comparing the `RMSE` values with the ones produced by the other models they are pretty close to the ones of `S4` and this model is better than `S5,S4D`, its drawback its obviously the training time and parameter sizxe which are much higher than the ones of the two `SSM` models.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     34.83
quantile_0.25    34.95
quantile_0.5     35.08
quantile_0.75    34.96
quantile_0.9     36.23
##################################################
Median eval loss over all the test lifes:
quantile_0.1     30.56
quantile_0.25    31.17
quantile_0.5     31.08
quantile_0.75    30.67
quantile_0.9     31.54
##################################################
Std eval loss over all the test lifes:
quantile_0.1     22.76
quantile_0.25    22.40
quantile_0.5     23.90
quantile_0.75    24.10
quantile_0.9     25.04
##################################################
```

|  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_132 | 43.52 | 44.52 | 40.09 | 39.68 | 31.54 |
| Life_133 | 13.1 | 13.39 | 15.01 | 14.89 | 19.73 |
| Life_134 | 17.23 | 18.4 | 14.27 | 14.35 | 11.54 |
| Life_135 | 82.51 | 81.8 | 85.85 | 85.93 | 94.72 |
| Life_136 | 41.01 | 36.23 | 37.15 | 36.84 | 27.55 |
| Life_137 | 82.65 | 81.37 | 85.65 | 85.74 | 88.92 |
| Life_138 | 13.12 | 14.33 | 10.37 | 10.39 | 11.69 |
| Life_139 | 37.58 | 38.21 | 34.09 | 32.68 | 25.46 |
| Life_140 | 96.03 | 95.25 | 99.13 | 99.9 | 102.82 |
| Life_141 | 19.1 | 24.26 | 22.85 | 24.36 | 32.75 |
| Life_142 | 8.99 | 9.43 | 8.9 | 8.75 | 16.91 |
| Life_143 | 75.72 | 74.69 | 78.72 | 79.22 | 82.0 |
| Life_144 | 18.34 | 19.18 | 15.09 | 13.21 | 12.23 |
| Life_145 | 35.79 | 34.82 | 38.91 | 39.65 | 42.87 |
| Life_146 | 50.31 | 51.47 | 47.15 | 47.26 | 39.05 |
| Life_mean | 42.33 | 42.49 | 42.22 | 42.19 | 42.65 |
| Life_median | 39.3 | 37.22 | 38.03 | 38.25 | 32.14 |
| Life_std | 27.32 | 26.57 | 28.79 | 29.04 | 30.54 |

###### Metrics Table Pinball Loss

The best quantile is 0.5 followed by 0.75 and 0.1, so here it not exactly following the expectations.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     12.81
quantile_0.25    13.83
quantile_0.5     10.95
quantile_0.75    12.69
quantile_0.9     16.77
##################################################
Median eval loss over all the test lifes:
quantile_0.1     11.03
quantile_0.25    11.22
quantile_0.5      8.69
quantile_0.75    10.94
quantile_0.9     14.52
##################################################
Std eval loss over all the test lifes:
quantile_0.1      8.06
quantile_0.25     8.59
quantile_0.5      6.20
quantile_0.75     6.97
quantile_0.9     10.05
##################################################
```

###### Metrics table Pinball Loss

With `Informer` is always a bit strange: similar results across quantiles in `RMSE` and here (with an evaluation loss that should improve the underestimating quantiles) the best quantile is 0.9, however on `Life_median` the best one is 0.1.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     16.15
quantile_0.25    16.27
quantile_0.5     15.81
quantile_0.75    15.60
quantile_0.9     14.74
##################################################
Median eval loss over all the test lifes:
quantile_0.1     12.80
quantile_0.25    13.44
quantile_0.5     13.00
quantile_0.75    12.75
quantile_0.9     12.86
##################################################
Std eval loss over all the test lifes:
quantile_0.1     10.74
quantile_0.25    10.59
quantile_0.5     10.64
quantile_0.75    10.49
quantile_0.9      8.88
##################################################
```

###### Prediction plots

See [[ssm_experiments#Prediction Plots `FD002`|here]] . Also here very high loss values for `Life_141` while `Life_135` is much better than in the `SSM` models but we have `Life_140` which has significantly highe loss values now.
