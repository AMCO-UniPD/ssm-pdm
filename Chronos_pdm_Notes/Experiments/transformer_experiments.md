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

