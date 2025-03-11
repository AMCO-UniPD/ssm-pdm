---
id: ssm_exp
aliases: []
tags:
  - experiments
  - ssm_pdm
---

# `SSM` `PdM` Experiments

In this note I will report the results of the experiments performed with the `SSM` based models (i.e. `S4,S5,S4D`). 

We will start considering the same approach we used in [[chronos_reg_exp|the `chronos` experiments]], which will be called the `padding` approach.

## `padding` approach experiments 🦜

In this section the experiments using the `padding` approach will be presented.

>[!info]
> For the `padding` experiments we will use the 🦜 emoji, since the word `parrot` is similar to `padding`.

### `S4` Model Experiments 4️⃣ 🦜

Using the `model_summary` configuration argument we can get information on the size of the model in terms of number of parameters and total number of operations performed:

Parameters computed using `torchinfo summary`:

```txt
==========================================================================================
Total params: 3,300,353
Trainable params: 3,300,353
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 2.64
==========================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 43.01
Params size (MB): 13.19
Estimated Total Size (MB): 56.23
==========================================================================================
```

Parameters computed manually:

```txt
##################################################
Total parameters: 3300353
Trainable parameters: 3300353
Non-trainable parameters: 0
##################################################
```
The number is equal to the `torchinfo` one.

The number of parameters it's much higher here with the respect to the previous experiments did in the Deep Learning project because here I used `hidden_size=512` while in those experiments we used `hidden_size=90`.

#### Dataset `FD001`

##### Experiment 1 `S4` `FDOO1` `padding` 4️⃣ 1️⃣ 🦜

Let's start with this initial configuration:

| Parameter | Value |
|-----------|-------|
| `model_type` | `S4` |
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
| `hidden_size` | 512 |
| `d_state` | 64 |
| `loss` | `mae` |
| `eval_loss` | `mse` |

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/id2ektks?nw=nwuserfrizzodavide)

Looking at the `wandb` plots we have an inital decreasing phase of the loss and then it more or less saturates at a value that is slightly lower than the best `val_loss` values we obtained in the `chronos` experiments.

###### Metrics Table

Comparing with the best experiment done with `chronos` (Experiment 2 of `choronos-t5-small`) we are better in terms of mean performances, 27.68 against 30.48. Looking at the metrics values as usual we have some variability across different lifes.


```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   28.58
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 71.37 |
| Life_51 | 12.83 |
| Life_52 | 7.02 |
| Life_53 | 9.93 |
| Life_54 | 40.28 |
| Life_55 | 22.12 |
| Life_56 | 65.58 |
| Life_57 | 9.47 |
| Life_58 | 18.36 |
| Life_59 | 47.72 |
| Life_60 | 18.34 |
| Life_61 | 41.85 |
| Life_62 | 27.5 |
| Life_63 | 5.69 |
| Life_64 | 17.12 |
| Life_mean | 27.68 |

In the windowed approach used for the Elements of Deep Learning exam the `RMSE` was 19.77, so smaller than here but I don't know weather these two kind of evaluations are comparable. In that case the `RMSE` should be the mean over all the test lifes but I have to check trying to reproduce that approach.

###### Prediction plots

The `RUL` grid prediction plots do not look bad at all, in some lifes there is some overestimation/underestimation at the beginning but they converge very closely to the true `RUL` towards the end of the life, which is a good sign.

Comparing these plots to the plots we produced in the Deep Learning project here the `RUL` is more oscillating and noisy, in the projec it was almost a perfectly straight line (as the true one) so maybe that's the reason why the `RMSE` is higher here.

##### Experiment 2 `S4` `FD001` `padding` 4️⃣ 1️⃣ 🦜

Let's try to reduce `d_model` from 512 to 128.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/ftm1ag59?nw=nwuserfrizzodavide)

Looking at the `wandb` loss plots the trend is very similar to the one of the previous experiment but the loss values are a bit lower.

###### Metrics Table

Unfortunately the results are a bit worse.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   30.41
Name: Life mean Loss, dtype: float64
##################################################
```

|  | Eval Loss |
| --- | --- |
| Life_50 | 56.33 |
| Life_51 | 24.12 |
| Life_52 | 15.12 |
| Life_53 | 19.43 |
| Life_54 | 67.1 |
| Life_55 | 17.2 |
| Life_56 | 45.28 |
| Life_57 | 15.44 |
| Life_58 | 27.89 |
| Life_59 | 50.39 |
| Life_60 | 19.59 |
| Life_61 | 37.52 |
| Life_62 | 22.16 |
| Life_63 | 11.24 |
| Life_64 | 12.76 |
| Life_mean | 29.44 |

###### Prediction plots

In the prediction plots we can see the fact that we have reduced `d_model`, the predicted `RUL` signals are clearly worse than the ones of the previous experiment.

##### Experiment 3 `S4` `FD001` `padding` 4️⃣ 1️⃣ 🦜

Let's try to come back to `d_model=512` but introduce `dropout=0.2`.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/8xe7msiq?nw=nwuserfrizzodavide)

The loss plots are a bit worse than the previous experiment.

###### Metrics Table

Results a little bit worse, probably the first configuration was the best one.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   32.87
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 55.51 |
| Life_51 | 20.73 |
| Life_52 | 23.43 |
| Life_53 | 16.05 |
| Life_54 | 61.06 |
| Life_55 | 27.78 |
| Life_56 | 54.28 |
| Life_57 | 15.1 |
| Life_58 | 36.57 |
| Life_59 | 38.21 |
| Life_60 | 20.37 |
| Life_61 | 23.4 |
| Life_62 | 28.2 |
| Life_63 | 6.77 |
| Life_64 | 43.2 |
| Life_mean | 31.38 |

###### Prediction plots

### `S5` Model Experiments 5️⃣

Unfortunately for `S5` the `summary` method of the `torchinfo` library does not work because it gives some problems with the `torch.vmap` function which is used inside the implementation of the `S5Block` model (which I imported from the `s5-pytorch` library). Since it may be not very worth it to change the implementation of `S5` just to make `torchinfo` work let's compute manually the number of parameters using the `state_dict` of the `torch` objects.

```txt
Total parameters: 124123
Trainable parameters: 124123
Non-trainable parameters: 0
```

It seems a little low to me this number of parameters. `S5` had less parameters than `S4,S4D` also in the Deep Learning project but here the difference seems too high to me. Maybe it's better to compute the number of parameters for `S4,S4D` manually too.

Considering that the number of parameters computed manually for `S4,S4D` is the same as the one computed with `torchinfo` than this number should be correct. Maybe it's also becaus of this very low number of parameters that `S5` is the one with the worse performances.

#### Dataset `FD001`

##### Experiment 1 `S5` `FDOO1` `padding` 5 1️⃣ 🦜

Let's try to use the same configuration used in the `S4` experiment.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/nthexjt7?nw=nwuserfrizzodavide)

The trend of the loss plots is similar to the one of `S4` but with a slightly higher loss values.

###### Metrics Table

The metrics are a little bit higher than in `S4`. Moreover there is no life in which the loss goes below 10.


```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   30.47
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 49.05 |
| Life_51 | 22.78 |
| Life_52 | 26.91 |
| Life_53 | 23.96 |
| Life_54 | 43.42 |
| Life_55 | 25.55 |
| Life_56 | 58.38 |
| Life_57 | 15.07 |
| Life_58 | 10.16 |
| Life_59 | 47.17 |
| Life_60 | 20.25 |
| Life_61 | 47.6 |
| Life_62 | 37.62 |
| Life_63 | 18.13 |
| Life_64 | 10.64 |
| Life_mean | 30.45 |

###### Prediction plots

Comparing the `RUL` grid plots with the `S4` ones these are clearly more noisy. In `S4` there are multiple lifes in which the predictions are almost straight lines which are almost overlapped with the true `RUL` values, while here we have more oscillations. Oscillations are not very intuitive because they represent an increase in the `RUL`, as if the sensor regenerates its life which physically is not possible.

##### Experiment 2 `S5` `FD001` `padding` 5 1️⃣ 🦜

Let's try to increase `d_model` to 1028, we also use `dropout=0.2`.


>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/20735991?nw=nwuserfrizzodavide)

###### Metrics Table

The results are slightly better than the previous experiment.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   30.19
Name: Life mean Loss, dtype: float64
##################################################
```

|  | Eval Loss |
| --- | --- |
| Life_50 | 60.63 |
| Life_51 | 25.85 |
| Life_52 | 10.55 |
| Life_53 | 16.07 |
| Life_54 | 47.05 |
| Life_55 | 25.12 |
| Life_56 | 65.74 |
| Life_57 | 25.41 |
| Life_58 | 13.0 |
| Life_59 | 48.92 |
| Life_60 | 16.07 |
| Life_61 | 49.09 |
| Life_62 | 36.41 |
| Life_63 | 14.22 |
| Life_64 | 10.6 |
| Life_mean | 30.98 |

###### Prediction plots

The plots are very similar to the ones of the previous experiment.


### `S4D` Model Experiments 4️⃣D

Slightly smaller number of parameters for `S4D` with the respect to `S4`.

```txt
==========================================================================================
Total params: 2,972,673
Trainable params: 2,972,673
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 1.31
==========================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 32.77
Params size (MB): 11.86
Estimated Total Size (MB): 44.66
==========================================================================================
```

Parameters computed manually:

```txt
##################################################
Total parameters: 2972673
Trainable parameters: 2972673
Non-trainable parameters: 0
##################################################
```

Also in this case the number is equal.

#### Dataset `FD001`

##### Experiment 1 `S4D` `FDOO1` `padding` 4️⃣D 1️⃣ 🦜

Let's try to use the same configuration used in the `S4` experiment.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/yyy8fq83?nw=nwuserfrizzodavide)

Here the behavior of the loss functions is a bit peculiar because it starts very well in the first epochs (even lower than `S4`) but then it starts to go up and down. Since we take the minimum `val_loss` probably the results may be better than the `S4` ones, but maybe in the future I can try to add some `dropout` to remove this oscillating effect.

###### Metrics Table

At the end the results are in the middle between `S4` and `S5`. Differently from `S5` here we have at least some loss values going below 10. 


```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   29.24
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 45.51 |
| Life_51 | 15.63 |
| Life_52 | 24.48 |
| Life_53 | 8.75 |
| Life_54 | 64.56 |
| Life_55 | 21.9 |
| Life_56 | 58.08 |
| Life_57 | 17.39 |
| Life_58 | 16.17 |
| Life_59 | 44.19 |
| Life_60 | 23.58 |
| Life_61 | 47.38 |
| Life_62 | 27.34 |
| Life_63 | 14.18 |
| Life_64 | 8.94 |
| Life_mean | 29.21 |

###### Prediction plots

The plots are more similar to the `S4` ones, not so oscillating like the `S5` ones.

##### Experiment 2 `S4D` `FD001` `padding` 4️⃣D 1️⃣ 🦜

Let's try to add `dropout=0.2`.

>[!note]
> [Link to the `wandb` run]()

## `windowed` Approach Experiments 🪟

In this section we will report the results of the experiments using the `windowed` approach.

>[!info]
> Obviously for the `windowed` approach we will us the 🪟 emoji.

### `S4` Model Experiments 4️⃣ 🪟

Using `model_summary` we can get the number of parameters of the model on this new approach. This is the parameter count for `d_model=128`.

```txt
=========================================================================
=================
Total params: 333,569
Trainable params: 333,569
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 0.17
=========================================================================
=================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.65
Params size (MB): 1.33
Estimated Total Size (MB): 1.98
=========================================================================
```

With `d_model=512`:

```txt
==========================================================================================
Total params: 3,300,353
Trainable params: 3,300,353
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 2.64
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 14.62
Params size (MB): 13.19
Estimated Total Size (MB): 27.82
==========================================================================================

```

The parameter count is the exact same as the one of the `padding` approach (in fact the model architecture did not change), buut the input size changes and thus the memory usage for the forward/backward pass are changes. In particular they are lower since now we are passing shorter sequences in input to the model.


#### Dataset `FD001`

##### Experiment 1 `S4` `FDOO1` `windowed` 4️⃣ 1️⃣ 🪟

Let's start with the following configuration:

| Parameter | Value |
|-----------|-------|
| `model_type` | `S4` |
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
| `sequence_length` | 30 |
| `n_layers` | 5 |
| `dropout` | 0.2 |
| `activation` | `relu` |
| `final_act` | `glu` |
| `hidden_size` | 128 |
| `d_state` | 64 |
| `loss` | `mae` |
| `eval_loss` | `mse` |

>[!note]
> [Link to the `wandb` run]()

The loss trend is similar to thew one we have seen in the `RNN` based models. After a few epochs the loss saturates at a constant value, which is higher than the values we obtained in the `padding` approach.

###### Metrics Table

In fact the metrics are not very good.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   68.56
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 103.57 |
| Life_51 | 61.66 |
| Life_52 | 46.56 |
| Life_53 | 74.96 |
| Life_54 | 107.4 |
| Life_55 | 25.18 |
| Life_56 | 102.94 |
| Life_57 | 59.78 |
| Life_58 | 74.91 |
| Life_59 | 92.71 |
| Life_60 | 42.44 |
| Life_61 | 103.19 |
| Life_62 | 73.65 |
| Life_63 | 49.83 |
| Life_64 | 75.17 |
| Life_mean | 72.93 |

###### Prediction plots

As expected the plots are not good at all, they are actually very similar to the ones we obtained in the test run I did this morning with just 3 epochs (just to see weather the code worked). The predicted `RUL` signal is constant and slightly decreases in the last time steps.

##### Experiment 2 `S4` `FD001` `windowed` 4️⃣ 1️⃣ 🪟

Now I realized why the results are so bad. Looking back at the last `wandb` experiments I did on the `SSM_PDM` project ([`SSM_PDM` `wandb` project](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/SSM_PDM_final_tests?nw=nwuserfrizzodavide)), I realized that I was using `sequence_length=170`. 

Let's try then a new configuration with `sequence_length=170` and let's also remove the dropout.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/vkd4ojfv?nw=nwuserfrizzodavide)

Now in the `wandb` loss plots we can see a clearer decreasing trend and in the `test_loss` the model saturates at a lower loss value than the previous experiment.

###### Metrics Table

Much better results in the metrics table. We are more or less on the same level reached in the `padding` approach.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   31.71
Name: Life mean Loss, dtype: float64
##################################################
```
| Life  | Eval Loss |
| --- | --- |
| Life_50 | 45.66 |
| Life_51 | 27.31 |
| Life_52 | 10.11 |
| Life_53 | 38.2 |
| Life_54 | 48.83 |
| Life_55 | 30.79 |
| Life_56 | 53.27 |
| Life_57 | 19.72 |
| Life_58 | 26.81 |
| Life_59 | 28.45 |
| Life_60 | 30.2 |
| Life_61 | 36.42 |
| Life_62 | 43.7 |
| Life_63 | 31.98 |
| Life_64 | 14.11 |
| Life_mean | 32.37 |

###### Prediction plots

The prediction plots are surely better than the previous ones, since they are not constant and in same lifes they are almost overlapped to the true values but in some other (like `Life_55`) the predictions are quite bad.

##### Experiment 3 `S4` `FD001` `windowed` 4️⃣ 1️⃣ 🪟

Let's try to increase `d_model` to 512.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/0who1612?nw=nwuserfrizzodavide)

In the `wandb` loss plots the trend is the usual one, however the model reaches new minimum values in `val_loss` and `test_loss` but the different in any case it's minimal so I do not expect huge improvements.

###### Metrics Table

In fact the results are worse.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   36.60
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 45.8 |
| Life_51 | 21.22 |
| Life_52 | 39.3 |
| Life_53 | 28.61 |
| Life_54 | 44.34 |
| Life_55 | 34.83 |
| Life_56 | 54.94 |
| Life_57 | 35.93 |
| Life_58 | 32.32 |
| Life_59 | 46.28 |
| Life_60 | 35.2 |
| Life_61 | 33.91 |
| Life_62 | 40.96 |
| Life_63 | 39.82 |
| Life_64 | 23.98 |
| Life_mean | 37.16 |

###### Prediction plots

Also from the plots we can see that the results are worse than the previous experiment.

##### Experiment 4 `S4` `FD001` `windowed` 4️⃣ 1️⃣ 🪟

Looking back at the `SSM_PDM` project code I realized that in the `windowed` approach that I used back at the time in the Regression Head I used some sort of Global Average Pooling. We have already tried it out in the `padding` approach and it did not work out, let's try it here and see what happens.

The difference in the code stays in the `forward` method of the models in the `decoder` part, so after the `SSM` block has extracted the features and we have a tensor of shape `(B,L,d_model)`. In the `padding` approach we simply decode the tensor with a `nn.Linear` layer to pass from `(B,L,d_model)` to `(B,L)`:

```python
x = self.decoder(x).squeeze(-1)  # (B,L,d_model) -> (B,L)
```

In the `gap` Regression Head instead we first remove the `L` dimension averaging over it (this is the Global Average Pooling step) and then we decoded it with a `fc` layer to pass from `(B,d_model)` to `(B,L)` 

```python
x = x.mean(dim=1) # (B, L, d_model) -> (B, d_model)
x = self.decoder(x)  # (B, d_model) -> (B, d_output)
```
>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/2bwpcjkw?nw=nwuserfrizzodavide)

In the `wandb` loss plots the loss goes down at significantly lower values than all the `SSM` experiments. If previously the minimum values were around 24-25 now we are around 15.

###### Metrics Table

We have a new best run in terms of metrics.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   23.47
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 25.11 |
| Life_51 | 6.03 |
| Life_52 | 7.89 |
| Life_53 | 19.71 |
| Life_54 | 56.55 |
| Life_55 | 9.67 |
| Life_56 | 17.9 |
| Life_57 | 2.05 |
| Life_58 | 40.73 |
| Life_59 | 22.99 |
| Life_60 | 4.37 |
| Life_61 | 12.59 |
| Life_62 | 5.46 |
| Life_63 | 6.98 |
| Life_64 | 67.18 |
| Life_mean | 20.35 |

###### Prediction plots

Now I think that we have got back to the results we obtained in the Deep Learning exam project. In that case the metric values were a little bit smaller but I think that is just becuase now we are using a different train,val,test split. We have the `RUL` signal that is a smooth decreasing line like the true `RUL` signal and, I checked, the predicted `RUL` ranges are not always the same as it happens in the `RNN` based models.

### `S5` Model Experiments 5️⃣ 🪟

Let's use `model_summary` to get the parameter count.

#### Dataset `FD001`

##### Experiment 1 `S5` `FDOO1` `windowed` 5️⃣ 1️⃣ 🪟

Let's start with the same configuration used for `S4`, so `sequence_length=170,dropout=0` and let's start with `d_model=128`.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/ya5skbj3?nw=nwuserfrizzodavide)

As it happened also in the `padding` approach the loss plots are very similar to the ones of the `S4` model, but the loss values are higher.

###### Metrics Table

However the metrics are very good 💪, the best ones seen so far.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   26.41
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 42.16 |
| Life_51 | 12.15 |
| Life_52 | 15.72 |
| Life_53 | 12.99 |
| Life_54 | 36.79 |
| Life_55 | 32.6 |
| Life_56 | 47.75 |
| Life_57 | 8.5 |
| Life_58 | 7.01 |
| Life_59 | 34.92 |
| Life_60 | 22.93 |
| Life_61 | 42.64 |
| Life_62 | 28.18 |
| Life_63 | 19.27 |
| Life_64 | 8.24 |
| Life_mean | 24.79 |

###### Prediction plots

The loss plots are much better than the ones seen up to now, the `RUL` signals have a clear decreasing trend and not that oscillating trend we can see in the `S4` 🪟 experiments did up to now.

##### Experiment 2 `S5` `FD001` `windowed` 5️⃣ 1️⃣ 🪟

Let's use the same configuration used in `S4` for the `gap` approach.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/nlf0baho?nw=nwuserfrizzodavide)


###### Metrics Table

Disaster for `S5` with the `gap` layer.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   50.03
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 41.29 |
| Life_51 | 10.97 |
| Life_52 | 9.12 |
| Life_53 | 36.84 |
| Life_54 | 82.33 |
| Life_55 | 44.34 |
| Life_56 | 23.07 |
| Life_57 | 1.53 |
| Life_58 | 75.62 |
| Life_59 | 29.58 |
| Life_60 | 6.3 |
| Life_61 | 11.82 |
| Life_62 | 10.07 |
| Life_63 | 12.13 |
| Life_64 | 138.89 |
| Life_mean | 35.59 |

Actually comparing with the metrics table of `S4` we have several lifes where `S5` is significantly better: `Life_52,Life_57,Life_60`, and others (like `Life_64`) where it is extremely worse and this obviously make the `Life_mean` metric to go up.

###### Prediction plots

Following the observation did above we have some plots were the predicted `RUL` signal is essentially overlapped to the true one (i.e.`Life_52,Life_53,Life_58,Life_61,Life_62`) while there are others were the prediction is completely missed. The good thing is that at least it is an underestimation error, so the machine would be stopped before the actual failure.


### `S4D` Model Experiments 4 D 🪟

Let's use `model_summary` to get the parameter count.

#### Dataset `FD001`

##### Experiment 1 `S4D` `FDOO1` `windowed` 5️⃣ 1️⃣ 🪟

Let's start with the same configuration used for `S4`, so `sequence_length=170,dropout=0` and let's start with `d_model=128`.


>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/cxqmdjom?nw=nwuserfrizzodavide)

The training was really fast (in fact looking at the time plots in `wanbd` the `S4D` one is significantly lower than the other two) and in `val_loss` the values are lower than the ones of `S4,S5`.

###### Metrics Table

Also the metrics values are good, not as good as the ones of `S5` but still good.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   29.22
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 34.47 |
| Life_51 | 18.75 |
| Life_52 | 18.95 |
| Life_53 | 24.4 |
| Life_54 | 30.7 |
| Life_55 | 29.94 |
| Life_56 | 39.71 |
| Life_57 | 22.52 |
| Life_58 | 20.59 |
| Life_59 | 28.57 |
| Life_60 | 22.33 |
| Life_61 | 35.75 |
| Life_62 | 20.84 |
| Life_63 | 42.98 |
| Life_64 | 9.43 |
| Life_mean | 26.66 |

###### Prediction plots

The plots are surely better than the `S4` ones but not as good as the `S5` ones.

##### Experiment 2 `S4D` `FD001` `windowed` 5️⃣ 1️⃣ 🪟

Let's use the `gap` configuration.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/asmjar78?nw=nwuserfrizzodavide)

The loss plots are more oscillating than the ones of the other models but we reach lower minimum values in `val_loss,test_loss`.

###### Metrics Table

Like in `S5` the metrics are higher on average with the respect to `S4` because we have some very good lifes but also some very bad ones.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   41.91
Name: Life mean Loss, dtype: float64
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 57.65 |
| Life_51 | 3.72 |
| Life_52 | 5.08 |
| Life_53 | 70.16 |
| Life_54 | 101.47 |
| Life_55 | 23.06 |
| Life_56 | 19.68 |
| Life_57 | 4.01 |
| Life_58 | 91.07 |
| Life_59 | 38.44 |
| Life_60 | 1.89 |
| Life_61 | 12.29 |
| Life_62 | 17.98 |
| Life_63 | 6.94 |
| Life_64 | 123.05 |
| Life_mean | 38.43 |

###### Prediction plots

The plots are like the `S5` ones, some of them are overlapped with the true `RUL` signals, some other are really far from it.
