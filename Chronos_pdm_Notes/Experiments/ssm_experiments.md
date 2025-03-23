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
Std eval loss over all the test lifes: Eval Loss   20.99
Name: Life_std, dtype: float64
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
| Life_std | 19.11 |

###### Prediction plots

Now I think that we have got back to the results we obtained in the Deep Learning exam project. In that case the metric values were a little bit smaller but I think that is just becuase now we are using a different train,val,test split. We have the `RUL` signal that is a smooth decreasing line like the true `RUL` signal and, I checked, the predicted `RUL` ranges are not always the same as it happens in the `RNN` based models.

#### Dataset `FD002`

In this section we will report the results obtained on the `FD002` dataset. This dataset is more challenging then the `FD001` one because the engine is tested on 6 different operating conditions. There are 260 training trajectories and 259 test trajectories, so the experiments will take some more time to run. We will use the same kind of train,val,test split we used for the `FD001` dataset. So we will use all the training trajectories for training and then we will use the first 130 test trajectories for validation and the last 129 for testing.

##### Experiment 1 `S4` `FD002` `windowed` 4️⃣ 1️⃣ 🪟

Let's use the following configuration:

| Parameter | Value |
|-----------|-------|
| `model_type` | `S4` |
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
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/hr2x9l6y?nw=nwuserfrizzodavide)

The behavior of the loss plots it's peculiar. It starts decreasing and then it seems to saturate to a constant value, however at epoch 40 it start to go down again and it oscillates around more or less an error of 24-25.

###### Metrics Table

The results are not as bad as I thought considering that higher difficulty of the task with respect to the `FD001` dataset.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   31.51
##################################################
Std eval loss over all the test lifes: Eval Loss   30.79
##################################################
```
| Life  | Eval Loss |
| --- | --- |
| Life_131 | 10.32 |
| Life_132 | 1.86 |
| Life_133 | 16.69 |
| Life_134 | 106.01 |
| Life_135 | 0.95 |
| Life_136 | 27.07 |
| Life_137 | 4.67 |
| Life_138 | 9.82 |
| Life_139 | 57.29 |
| Life_140 | 69.61 |
| Life_141 | 8.45 |
| Life_142 | 24.06 |
| Life_143 | 12.58 |
| Life_144 | 19.64 |
| Life_145 | 29.01 |
| Life_mean | 26.54 |
| Life_std | 28.38|

Here the `RMSE` values are quite low, the `Life_mean` is ruined by the very high values of `Life_134,Life_139` and `Life_140`.

###### Prediction plots

The plots are good, there are several lifes in which the predicted `RUL` is almost overlapped to the true one.

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
Std eval loss over all the test lifes: Eval Loss   51.56
Name: Life_std, dtype: float64
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
| Life_std | 36.38 |

Actually comparing with the metrics table of `S4` we have several lifes where `S5` is significantly better: `Life_52,Life_57,Life_60`, and others (like `Life_64`) where it is extremely worse (in fact `Life_std` is very high with the respect to `S4`) and this obviously make the `Life_mean` metric to go up.

###### Prediction plots

Following the observation did above we have some plots were the predicted `RUL` signal is essentially overlapped to the true one (i.e.`Life_52,Life_53,Life_58,Life_61,Life_62`) while there are others were the prediction is completely missed. The good thing is that at least it is an underestimation error, so the machine would be stopped before the actual failure.

#### Dataset `FD002`

##### Experiment 1 `S5` `FD002` `windowed` 5️⃣ 1️⃣ 🪟

Let's use the same configuration used for `S4` on the first experiment on `FD002`.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/dlxnvw3w?nw=nwuserfrizzodavide)

In the loss plots the situation seems very good since the plots are almost always below the `S4` ones.

###### Metrics Table

The average results on the metrics are still worse than the ones obtained with `S4` always for the fact that there is an high variability in the `RMSE` across different lifes. We have also to consider that now we have 128 test lifes with the respect to the 50 of the `FD001` dataset, so the variability is more pronounced. However looking at the `std` value over all the test lifes below I expected it to be higher than 33.42 → it is still higher than the one observable in `S4` but not as high as I expected.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   40.52
##################################################
Std eval loss over all the test lifes: Eval Loss   33.42
##################################################
```
| Life  | Eval Loss |
| --- | --- |
| Life_131 | 12.41 |
| Life_132 | 42.41 |
| Life_133 | 30.1 |
| Life_134 | 72.15 |
| Life_135 | 27.38 |
| Life_136 | 20.69 |
| Life_137 | 5.08 |
| Life_138 | 3.99 |
| Life_139 | 42.24 |
| Life_140 | 87.81 |
| Life_141 | 2.36 |
| Life_142 | 27.35 |
| Life_143 | 9.93 |
| Life_144 | 30.72 |
| Life_145 | 17.29 |
| Life_mean | 28.79 |
| Life_std | 23.72 |

As in `S4` `Life_134,Life_139,Life_140` are the ones that ruin the `Life_mean` metric.

###### Prediction plots

The plots are similar to the ones of `S4`.

##### Experiment 2 `S5` `FD002` `windowed` 5️⃣ 1️⃣ 🪟

Since in the `FDOO1` dataset the best results in terms of metrics for `S5` were obtained without using the `gap` layer in the Regression Head let's try to use the same configuration here.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/ml99d1fv?nw=nwuserfrizzodavide)

###### Metrics Table

Ok the results are worse than the ones obtained with the `gap` layer but at least they are different from all the other experiments with the `RNN` based models.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   34.72
##################################################
```
| Life  | Eval Loss |
| --- | --- |
| Life_131 | 33.03 |
| Life_132 | 36.15 |
| Life_133 | 35.3 |
| Life_134 | 74.91 |
| Life_135 | 23.04 |
| Life_136 | 67.99 |
| Life_137 | 24.89 |
| Life_138 | 32.53 |
| Life_139 | 81.91 |
| Life_140 | 19.56 |
| Life_141 | 13.02 |
| Life_142 | 55.09 |
| Life_143 | 26.73 |
| Life_144 | 32.78 |
| Life_145 | 46.62 |
| Life_mean | 40.24 |

###### Prediction plots

The plots have the usual oscillating behavior typical of the non `gap` layers experiments.

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
Std eval loss over all the test lifes: Eval Loss   15.56
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
| Life_std | 8.72 |

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
Std eval loss over all the test lifes: Eval Loss   39.71
Name: Life_std, dtype: float64
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
| Life_std | 38.99 |

As in `S5` we can see that there is a lot more variability in the `RMSE` values over the different test lifes. In fact the `Life_std` is passing from 19.11 to 38.99.

###### Prediction plots

The plots are like the `S5` ones, some of them are overlapped with the true `RUL` signals, some other are really far from it.

#### Dataset `FD002`

##### Experiment 1 `S4D` `FD002` `windowed` 5️⃣ 1️⃣ 🪟

Let's use the same configuration used for `S4,S5` on the first experiment on `FD002`.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/cnaqyzg3?nw=nwuserfrizzodavide)

The trend here is simiar to the one of `S5` but the loss is always lower than the one of both `S4,S5`.

###### Metrics Table

Howewer there is still a high variability in the `RMSE` values across different lifes. That's probably why the `Life_mean` metric is so high if we consider all the test lifes.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   58.02
##################################################
Std eval loss over all the test lifes: Eval Loss   53.27
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_131 | 1.82 |
| Life_132 | 6.45 |
| Life_133 | 26.2 |
| Life_134 | 117.74 |
| Life_135 | 53.49 |
| Life_136 | 19.87 |
| Life_137 | 2.16 |
| Life_138 | 6.66 |
| Life_139 | 47.49 |
| Life_140 | 131.91 |
| Life_141 | 4.05 |
| Life_142 | 15.17 |
| Life_143 | 4.49 |
| Life_144 | 39.08 |
| Life_145 | 8.33 |
| Life_mean | 32.33 |
| Life_std | 39.78 |

Here the main problem are `Life_134,Life_140` which have really high values. We can also see how the `Life_std` over all the test lifes it's much higher than the one obtained with the `S4` model.

###### Prediction plots

Similar to `S4,S5`.

##### Experiment 2 `S4D` `FD002` `windowed` 5️⃣ 1️⃣ 🪟

As for `S5` let's try to do an experiment also without the `gap` layer.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/nbnpauhk?nw=nwuserfrizzodavide)

###### Metrics Table


A bit better than `S5` but still worse than the results obtained with the `gap` layer.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   35.28
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_131 | 33.3 |
| Life_132 | 19.49 |
| Life_133 | 14.31 |
| Life_134 | 72.15 |
| Life_135 | 31.22 |
| Life_136 | 69.08 |
| Life_137 | 13.96 |
| Life_138 | 51.72 |
| Life_139 | 86.46 |
| Life_140 | 30.66 |
| Life_141 | 12.92 |
| Life_142 | 59.51 |
| Life_143 | 10.93 |
| Life_144 | 35.52 |
| Life_145 | 33.56 |
| Life_mean | 38.32 |


###### Prediction plots

Plots similar to the ones produced with the `S5` model.

## `windowed` Approach + Pinball Loss Experiments 🪟 🎈

In this section I will group the results on the `windowed` approach using the `pinball` loss as the training loss function. I do not want to do a full Quantile Regression for the moment, I want just to use the Pinball loss with an $τ$ value that makes the model perfer underestimation over overestimation. For how I have implemented the Pinball loss we need to use $τ < 0.5$ to achieve that.

We will continue to use the `RMSE` as the evaluation metric.

>[!info]
> For these kind of experiments we will add the 🎈 emoji to the 🪟 one.

### `S4` Model Experiments 4️⃣ 🪟 🎈

#### Dataset `FD001`

##### Experiment 1 `S4` `FDOO1` `windowed` 4️⃣ 1️⃣ 🪟 🎈

We will start by weighting the `pinball` loss with $\tau=0.2$. So the initial configuration will be the following:

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
| `sequence_length` | 170 |
| `n_layers` | 5 |
| `dropout` | 0.0 |
| `activation` | `relu` |
| `final_act` | `glu` |
| `hidden_size` | 128 |
| `d_state` | 64 |
| `loss` | `pinball` |
| `tau` | 0.2 |
| `eval_loss` | `mse` |

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/nwn97ubo?nw=nwuserfrizzodavide)

In the `wandb` loss plots the loss is always lower than the one we had in the last `FD001` `S4` experiment but we do not have to judge the experiment by that since now the loss function is different. In fact if we look at the `eval_val_loss` and `eval_test_loss` plots we are sligthly above.

###### Metrics Table

As expected the results are worse now:

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   30.64
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 41.31 |
| Life_51 | 3.91 |
| Life_52 | 3.71 |
| Life_53 | 38.5 |
| Life_54 | 74.92 |
| Life_55 | 4.61 |
| Life_56 | 21.71 |
| Life_57 | 2.43 |
| Life_58 | 64.76 |
| Life_59 | 33.46 |
| Life_60 | 0.77 |
| Life_61 | 10.82 |
| Life_62 | 9.65 |
| Life_63 | 6.1 |
| Life_64 | 84.7 |
| Life_mean | 26.76 |


###### Prediction plots

The plots are quite similar to the one of the best `S4` run on `FD001`. However in `Life_55`, where there was a small overestimation of the `RUL`, now the plots shows how the model has learned to tend to underestimate and in fact the error here has dropped from 9.67 to 4.61 and in the plots the predicted and true `RUL` signals seems almost overlapped.

##### Experiment 2 `S4` `FDOO1` `windowed` 4️⃣ 1️⃣ 🪟 🎈

I think that the reason why the prediction were worse using the Pinball loss is because in all the experiments we have done up to now when the model makes mistakes in general it underestimates the `RUL` , so maybe to solve those errors we should make the model overestimate the `RUL` a little bit. So I want to try to use $\tau=0.7$ to see what happens.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/ptb1zbag?nw=nwuserfrizzodavide)

Now in the `wandb` loss plots the `eval_val_loss` and `eval_test_loss` charts show sligthly lower values in this experiment than in the one with $\tau=0.2$ and the one using simply the `mae` loss for training.

###### Metrics Table

In fact we have now a new best run in terms of performances 💪.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   19.87
##################################################
```

| Life      | Eval Loss |
| --------- | --------- |
| Life_50   | 18.65     |
| Life_51   | 2.17      |
| Life_52   | 3.54      |
| Life_53   | 21.27     |
| Life_54   | 50.17     |
| Life_55   | 13.01     |
| Life_56   | 11.74     |
| Life_57   | 3.63      |
| Life_58   | 31.35     |
| Life_59   | 8.38      |
| Life_60   | 5.98      |
| Life_61   | 11.85     |
| Life_62   | 2.18      |
| Life_63   | 8.74      |
| Life_64   | 52.03     |
| Life_mean | 16.31     |

###### Prediction plots

The plots seems still very similar to the ones obtained without using the Pinball loss (here there is also to say that I am comparing the plots looking them at the little preview window of `ghostty` and moreover they are all in little subplots inside the main plot so small differences are not very easy to spot). However looking at the `metrics_df` . However for example the `Life_55` I was considering in the [[ssm_experiments#Experiment 1 `S4` `FDOO1` `windowed` 4️⃣ 1️⃣ 🪟 🎈|previous experiment]] now has an higher loss → that's one drawback of this approach which is a bit risky because if we have close cases the model may still create an *unexpected break*.

### `S5` Model Experiments 5️⃣ 🪟 🎈

#### Dataset `FD001`

##### Experiment 1 `S5` `FDOO1` `windowed` 5️⃣ 1️⃣ 🪟 🎈

Let's use the last configuration used for `S4`, so the Pinball loss with $\tau=0.7$.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/e39bjmns?nw=nwuserfrizzodavide)

The `eval_val_loss` and `eval_test_loss` plots  in `wandb` become smaller than the ones of the previous `S5` experiment in the last 40 epochs more or less.

###### Metrics Table

The results improved a lot in this case, the `Life_mean` dropped from 50.03 to 34.31. Moreover the `Life_std` is not so high as in `S4D`.

```txt
##################################################
Mean eval loss over all the test lifes: Eval Loss   34.31
Name: Life_mean, dtype: float64
##################################################
Std eval loss over all the test lifes: Eval Loss   29.70
Name: Life_std, dtype: float64
##################################################
```

| Life  | Eval Loss |
| --- | --- |
| Life_50 | 31.93 |
| Life_51 | 17.01 |
| Life_52 | 26.1 |
| Life_53 | 13.55 |
| Life_54 | 57.59 |
| Life_55 | 35.72 |
| Life_56 | 14.13 |
| Life_57 | 9.59 |
| Life_58 | 48.01 |
| Life_59 | 18.15 |
| Life_60 | 19.98 |
| Life_61 | 12.14 |
| Life_62 | 3.13 |
| Life_63 | 17.39 |
| Life_64 | 83.32 |
| Life_mean | 27.18 |
| Life_std | 20.67 |



###### Prediction plots

Similar to the ones of the previous run, we can see how there is some overestimation in `Life_52`, where previously the predicted and true `RUL` signals were almost overlapped, and in fact the error is higher (passed from 9.12 to 26.1).

### `S4D` Model Experiments 4 D 🪟 🎈

#### Dataset `FD001`

##### Experiment 1 `S4D` `FDOO1` `windowed` 5️⃣ 1️⃣ 🪟 🎈

Let's use the last configuration used for `S4`, so the Pinball loss with $\tau=0.7$.

>[!note]
> [Link to the `wandb` run]()

In the `eval_val_loss,eval_test_loss` `wandb` charts we are getting a sligthly lower loss values. Moreover now that I am comparing just the plots of the two `S4D` run (with and without the Pinball loss) it's easier to realize how much the loss it's osicllating, when the `wandb` charts are considered together to the ones of the other models this is not so evident.

###### Metrics Table

The results does not change a lot with respect to the experiment without the Pinball Loss, we pass in fact from 41.91 to 42.86.

```txt

##################################################
Mean eval loss over all the test lifes: Eval Loss   42.86
##################################################
Std eval loss over all the test lifes: Eval Loss   40.05
##################################################
```


| Life  | Eval Loss |
| --- | --- |
| Life_50 | 52.5 |
| Life_51 | 1.57 |
| Life_52 | 8.11 |
| Life_53 | 69.82 |
| Life_54 | 103.66 |
| Life_55 | 26.04 |
| Life_56 | 15.35 |
| Life_57 | 4.65 |
| Life_58 | 94.98 |
| Life_59 | 32.39 |
| Life_60 | 8.31 |
| Life_61 | 8.48 |
| Life_62 | 10.25 |
| Life_63 | 8.93 |
| Life_64 | 126.9 |
| Life_mean | 38.13 |
| Life_std | 40.06 |

###### Prediction plots

Similar to the ones of the previous experiment without the Pinball loss.

## `windowed` Approach + Quantile Regression Experiments 🪟 🌗

In this section we will report the results of the experiments performed with the `windowed` approach and with the Quantile Regression mode. The model is trained to learn all the quantiles of the distribution of the `RUL` signal. The model is tested on different pre decided quantile levels (for each quantile level we have a different `wandb` experiment, and thus a different best model). In the evaluation part we will compare the performances on the different quantile levels to see which one is the best, taking also into account that we prefer underestimations of the `RUL` rather than overestimations.

>[!info]
> For the Quantile Regression mode we will use the 🌗 emoji which represents a quarter of the moon, the word `quarter` somehow recalls the concept of quantile.

### Experiments Names

Here I report all the `exp_name` strings that contain the name of the folders associated to the different experiments performed.

```txt
# Experiment names - FD001 dataset
exp_name: multi_run_14-03-2025_08-20-13_S4_FD001_windowed_quantile_reg
exp_name: multi_run_14-03-2025_09-43-44_S5_FD001_windowed_quantile_reg
exp_name: multi_run_14-03-2025_10-56-41_S4D_FD001_windowed_quantile_reg
exp_name: multi_run_14-03-2025_11-52-58_LSTM_FD001_windowed_quantile_reg
exp_name: multi_run_14-03-2025_15-09-39_RULTransformer_FD001_windowed_quantile_reg
exp_name: multi_run_14-03-2025_16-37-39_RULInformer_FD001_windowed_quantile_reg

# S4 bounds=[0.1,0.5]
exp_name: multi_run_20-03-2025_12-00-49_S4_FD001_windowed_quantile_reg
# S4 bounds=[0.5,0.99]
exp_name: multi_run_20-03-2025_14-05-21_S4_FD001_windowed_quantile_reg
# S4 bounds=[0.75,0.01] normal distribution
exp_name: multi_run_20-03-2025_15-56-11_S4_FD001_windowed_quantile_reg
# S4 bounds=[0.25,0.01] normal distribution
exp_name: multi_run_20-03-2025_18-22-48_S4_FD001_windowed_quantile_reg
# S4 with tau multiplication at the end
exp_name: multi_run_22-03-2025_15-41-44_S4_FD001_windowed_quantile_reg
# S5 with tau multiplication at the end
exp_name: multi_run_23-03-2025_09-14-06_S5_FD001_windowed_quantile_reg

# Experiment names - FD002 dataset
exp_name: multi_run_18-03-2025_07-58-34_S4_FD002_windowed_quantile_reg
exp_name: multi_run_18-03-2025_10-58-34_S5_FD002_windowed_quantile_reg
exp_name: multi_run_18-03-2025_15-32-30_S4D_FD002_windowed_quantile_reg
exp_name: multi_run_18-03-2025_18-13-05_RULInformer_FD002_windowed_quantile_reg
exp_name: multi_run_19-03-2025_15-00-46_RULTransformer_FD002_windowed_quantile_reg
exp_name: multi_run_19-03-2025_21-27-39_LSTM_FD002_windowed_quantile_reg

# Experiment names - FD001 Feature Extraction
exp_name: multi_run_20-03-2025_10-24-54_S4_FD001_windowed_quantile_reg_feat_extraction 
```

### `S4` Model Experiments 4‍⃣ 🪟 🌗

#### Dataset `FD001`

##### Experiment 1 `S4` `FD001` 4‍⃣ 🪟 🌗 

Let's start with the following configuration:

| Parameter          | Value                     |
| ------------------ | ------------------------- |
| `model_type`       | `S4`                      |
| `cmapss_model`     | `FD001`                   |
| `val_idx`          | `[0,50]`                  |
| `test_idx`         | `[50,100]`                |
| `transformer_type` | 1 (no feature extraction) |
| `window_size`      | 20                        |
| `scaler`           | `MinMaxScaler(-1,1)`      |
| `epochs`           | 100                       |
| `lr`               | 1e-3                      |
| `batch_size`       | 100                       |
| `weight_decay`     | 1e-4                      |
| `sequence_length`  | 170                       |
| `n_layers`         | 5                         |
| `dropout`          | 0.0                       |
| `activation`       | `relu`                    |
| `final_act`        | `glu`                     |
| `hidden_size`      | 128                       |
| `d_state`          | 64                        |
| `loss`             | `quantile_reg`            |
| `eval_loss`        | `mse`                     |
| `quantile_dist`    | `uniform`                 |
| `bounds`           | `[0.1,0.9]`               |
| `quantiles`        | `[0.1,0.5,0.9]`           |


>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/3x0z2rmk?nw=nwuserfrizzodavide)

The loss plots in `wandb` seems to be going down quite well with `quantile_0.1` being the one with the lowest errors, followed by `quantile_0.5` and `quantile_0.9`.

###### Metrics Table

Wow man these `Life_mean` values are very good 💪. In particular  `quantile_0.9` seems to be the best one while `quantile_0.1` and `quantile_0.5` are similar. This result somehow confirms what we have observed in the [[ssm_experiments#`windowed` Approach + Pinball Loss Experiments 🪟 🎈|experiments with the Pinball Loss]]: since normally the model commits underestimation errors if we use an high quantile level (so we force the model to overestimate) the metrics are better. The only risk is that in the lifes where the model was making good predictions (i.e. predicted and true `RUL` signal almost overlapped) we may now pass to have an overestimation error which is not good.

|  | quantile_0.1 | quantile_0.5 | quantile_0.9 |
| --- | --- | --- | --- |
| Life_50 | 35.69 | 24.86 | 19.02 |
| Life_51 | 4.47 | 0.46 | 1.19 |
| Life_52 | 1.76 | 2.38 | 8.08 |
| Life_53 | 25.79 | 27.29 | 20.05 |
| Life_54 | 55.79 | 51.95 | 42.64 |
| Life_55 | 7.92 | 10.27 | 4.52 |
| Life_56 | 20.4 | 15.14 | 7.49 |
| Life_57 | 4.15 | 0.91 | 3.55 |
| Life_58 | 42.94 | 35.17 | 17.84 |
| Life_59 | 22.12 | 17.75 | 20.13 |
| Life_60 | 3.43 | 5.56 | 3.83 |
| Life_61 | 12.84 | 7.96 | 10.3 |
| Life_62 | 4.68 | 1.33 | 3.74 |
| Life_63 | 5.42 | 9.06 | 9.88 |
| Life_64 | 43.22 | 56.68 | 9.7 |
| Life_mean | 19.37 | 17.78 | 12.13 |
| Life_std | 17.07 | 17.51 | 10.26 |
A pretty  clear example of how the predictions change when we change quantile is `Life_64` where the errors in the 0.1 and 0.5 quantiles are all over 40 while in quantile 0.9 the error is lower than 10 😱.

###### Prediction Plots

Ok the plot is not very clear from `yazi` (maybe I have to save it locally on my pc and visualize it full screen with a `PDF` viewer). In any case what we can see is the following:

- In many lifes, where the predictions are good, all the 4 lines (i.e. the true `RUL` and the predicted `RUL` on the three quantiles) are essentially overlapped one over the other
- In some lifes where the model normally commits high errors (e.g `Life_64` in particular) we can see how the closest prediction is the one obtained with quantile 0.9

##### Experiment 1 bis `S4` `FD001` 4‍⃣ 🪟 🌗

Since the single experiments take about 2 mins to run let's try to complete this experiment doing the training and evaluation also for quantiles: `[0.2,0.3,0.4,0.6,0.7,0.8]`. I want to see weather there is a gradual improvement in the performances as we go towards overestimation.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/b8o1yvx5?nw=nwuserfrizzodavide)

The `wandb` loss plots seems to follow the same trend as in the previous experiment.

###### Metrics Table

|  | quantile_0.1 | quantile_0.2 | quantile_0.3 | quantile_0.4 | quantile_0.5 | quantile_0.6 | quantile_0.7 | quantile_0.8 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Life_50 | 35.69 | 31.94 | 21.63 | 34.39 | 24.86 | 26.79 | 29.94 | 31.49 | 19.02 |
| Life_51 | 4.47 | 0.75 | 1.25 | 2.36 | 0.46 | 1.7 | 1.93 | 1.6 | 1.19 |
| Life_52 | 1.76 | 3.2 | 1.35 | 1.77 | 2.38 | 0.88 | 6.25 | 1.97 | 8.08 |
| Life_53 | 25.79 | 25.08 | 31.11 | 32.16 | 27.29 | 22.35 | 29.38 | 34.95 | 20.05 |
| Life_54 | 55.79 | 55.34 | 57.97 | 55.4 | 51.95 | 51.73 | 46.68 | 45.4 | 42.64 |
| Life_55 | 7.92 | 9.82 | 10.13 | 10.44 | 10.27 | 8.09 | 13.3 | 10.39 | 4.52 |
| Life_56 | 20.4 | 16.38 | 13.98 | 14.72 | 15.14 | 16.19 | 9.92 | 19.25 | 7.49 |
| Life_57 | 4.15 | 1.24 | 3.46 | 2.96 | 0.91 | 2.57 | 9.52 | 6.56 | 3.55 |
| Life_58 | 42.94 | 40.13 | 40.48 | 38.81 | 35.17 | 41.21 | 32.44 | 10.56 | 17.84 |
| Life_59 | 22.12 | 24.42 | 20.14 | 22.3 | 17.75 | 25.42 | 19.04 | 24.56 | 20.13 |
| Life_60 | 3.43 | 5.15 | 1.86 | 8.87 | 5.56 | 4.82 | 8.04 | 4.05 | 3.83 |
| Life_61 | 12.84 | 17.55 | 9.04 | 15.84 | 7.96 | 12.65 | 5.89 | 5.54 | 10.3 |
| Life_62 | 4.68 | 7.68 | 2.45 | 10.54 | 1.33 | 6.84 | 4.04 | 3.6 | 3.74 |
| Life_63 | 5.42 | 6.71 | 9.55 | 8.29 | 9.06 | 8.7 | 10.47 | 15.12 | 9.88 |
| Life_64 | 43.22 | 43.31 | 40.32 | 44.6 | 56.68 | 41.01 | 45.21 | 2.19 | 9.7 |
| Life_mean | 19.37 | 19.25 | 17.65 | 20.23 | 17.78 | 18.06 | 18.14 | 14.48 | 12.13 |
| Life_std | 17.07 | 16.44 | 16.91 | 16.31 | 17.51 | 15.64 | 14.37 | 13.33 | 10.26 |

As expected we can see how the `Life_mean` metric decreases as we go towards the highest quantiles.

In `Life_63` we can see how the overestimation is probably too much and in fact on high quantile levels the error is increasing. Actually in this life the best predictions are obtained with low quantile levels.

###### Prediction Plots

Ok now it is even more difficult to understand something in the plot, I surely have to visualize it full screen.

Now that I have a clear view of the plot from a `PDF` viewer we can make the following observations:

- There are several lifes (e.g. `Life_51,Life_54,Life_55,Life_59`) in which all the predictions from the quantile levels are one next to the other and they are close to the true `RUL` signal but not overlapped to it. In all these cases the closest line to the true `RUL` is the light blue one of the 0.9 quantile. This confirms the results obtained in the `metrics_df`.
- In `Life_59,Life_65` we have a similar effect but here there is a significant different between quantiles 0.8 and 0.9 which are much closer to the true `RUL` signal than the other quantiles.
- In `Life_64` all the quantiles are overestimating the true `RUL` signal with quantile 0.8 and 0.9 being the furthest from it.

From what I could see we have a similar effect to the previous experiment in `Life_64` as we can see also on then `metrics_df`.

##### Experiment 2 `S4` `FD001` 4‍⃣ 🪟 🌗

Considering the [[ssm_experiments#^7aa41a|small error committed here]], I now implemented the multi run approach, so let's execute an experiment with the same configuration but now 5 runs will be executed (each one with a different seed for the generation of the quantile levels in the `train_loop`) and then the `metrics_df` will average the `RMSE` metrics over all the runs in order to obtain more robust results and reduce the bias of lucky or unlucky runs.

Since we have now added also the multiple runs we will reduce the number of evaluation quantiles to 3: `[0.1,0.5,0.9]`.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/cqu95cq5?nw=nwuserfrizzodavide)

Now the time to execute all the experiments is multiplied by  `n_runs`, so , considering about 2 minutes per single experiment, this should take about 30 mins.


There is an interesting thing to note regarding the `wandb` loss plots, that now I also noted in the experiments on the other models so I will insert it just here. If we consider the experiment over the different quantiles for a single run in `val_loss,test_loss` we can see a clear difference in the plots between the different quantiles → in particular 0.9 is the highest, 0.5 in the middle and 0.1 the lowest. On the other hand in `eval_val_loss,val_test_loss` the plots are almost overlapped one to the other. This is probably due to the different nature of the two loss that are used here, in fact in `val_loss,test_loss` we are using the `QuantileLoss` while in `eval_val_loss,eval_test_loss` we are using the `RMSELoss`. In any case the loss logged on the `wandb` runs should not be fully trusted because they are computed on the mini batches that, for how they are constructed, contain sampled from different lifes evaluated together. The metrics we have to look at are the ones contained in the `metrics_df` tables.

###### Metrics Table

The results are following what we observed in the previous single run experiments. Quantile 0.9 is still the one with the best metric values. However now we do not have such big differences in the `Life_mean,Life_median` and `Life_std` metrics since we have averaged the results over all the runs.

Actually now looking at the results over all the test lifes the best quantile is 0.5, followed by 0.9 and 0.1.

Another thing we can notice is that comparing the results of quantile 0.9 to the ones of quantile 0.5 (which corresponds to no Quantile Regression, in fact with $\tau=0.5$ the `QuantileLoss` coincides with the `MAELoss`) we have an advantage when the loss in 0.5 is higher than 10-15 while quantile 0.9 is worse when the loss is lower than 10-15. This means that when the model is good in predicting the `RUL` (i.e. predicted and true `RUL` signals very close/overlapped in the plots) the overestimation brought by the high quantile levels is not good.

More or less the opposite effect can be observed on quantile 0.1: when the loss in 0.5 is high it increases in 0.1 and it instead increses when the loss is quite low because of the underestimation effect brought by 0.1.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1    19.49
quantile_0.5    17.03
quantile_0.9    18.22
##################################################
Median eval loss over all the test lifes:
quantile_0.1    16.48
quantile_0.5    11.30
quantile_0.9    13.99
##################################################
Std eval loss over all the test lifes:
quantile_0.1    16.33
quantile_0.5    13.83
quantile_0.9    13.24
##################################################
```

| Life  | quantile_0.1 | quantile_0.5 | quantile_0.9 |
| --- | --- | --- | --- |
| Life_51 | 29.15 | 26.44 | 23.94 |
| Life_52 | 1.59 | 2.51 | 3.31 |
| Life_53 | 1.75 | 3.95 | 5.55 |
| Life_54 | 29.13 | 24.92 | 23.9 |
| Life_55 | 54.37 | 48.88 | 47.15 |
| Life_56 | 5.46 | 9.78 | 11.99 |
| Life_57 | 15.21 | 14.24 | 6.89 |
| Life_58 | 3.47 | 2.31 | 3.34 |
| Life_59 | 36.74 | 30.87 | 28.23 |
| Life_60 | 19.7 | 17.6 | 11.96 |
| Life_61 | 4.62 | 4.44 | 8.33 |
| Life_62 | 10.16 | 9.17 | 7.34 |
| Life_63 | 9.37 | 8.22 | 3.62 |
| Life_64 | 5.51 | 8.8 | 10.42 |
| Life_65 | 39.78 | 28.35 | 30.02 |
| Life_mean | 17.73 | 16.03 | 15.07 |
| Life_median | 12.68 | 12.01 | 11.19 |
| Life_std | 15.42 | 12.53 | 12.01 |

###### Prediction Plots

In this new kind of multi run experiments we have to consider the fact that we have multiple runs, the plots from the different runs are more or less similar to each other but there are some small differences. In any case now I am looking at the plot of `run_1` and we can see a similar behavior seen in the previous experiment.

- We have some lifes in which there is some difference between the predicted and true signals and here the closest predictions are the ones of quantile 0.5 and quantile 0.9 → this happens in: `Life_51,Life_54,Life_55,Life_57,Life_60`
- Peculiar is the situation of `Life_65` where there is some difference between the predictions and the true values but the best prediction it's the one of quantile 0.5, while quantile 0.9 is the worst one.
- In `Life_56,Life_64` all the predictions are overestimating the true `RUL` signal and here quantile 0.9 and 0.5 have the highest errors as it can be observed also in the `metrics_df` above.
- In some lifes the predictions are quite good with predicted and true `RUL` signals almost overlapped one to the other. I am referring to `Life_52,Life_53,Life_58,Life_61,Life_62`. If we look at the `metrics_df` in fact these are the lifes with the smallest errors.

Looking at the plots obtained in the other runs they are quite similar except for the very peculiar `Life_65` where different things happen in all the runs:
- `run_2`: Here quantile 0.1 is overlapped to the true signal, quantile 0.9 is the second one and quantile 0.5 the last one.
- `run_3`: Now the best one is quantile 0.5 while the other two have similar performances.
- `run_4`: Here all the quantile are quite far from the true value with 0.9 being the closest one.
- `run_5`: Here quantile 0.9 is the best one but it is overestimating, while the other two are worse.

##### Experiment 2  bis `S4` `FD001` 4‍⃣ 🪟 🌗

Now that we have performed the multi run experiment on all the models it was possible to notice how the best quantile was always 0.5 (so the one that is used when no Quantile Regression is employed and the `train_loss` coincides with the `MAELoss`) which is not very good if we want to exploit Quantile Regression to push the model to prefer overestimation or underestimation. One possible interpretation under these results is that the other two quantile levels used in the evaluation phase (i.e. quantile 0.1 and 0.9) are too high/extreme and thus bring too much underestimation/overestimation to produce an advantage in the metrics with the respect to quantile 0.5. So we can continue the experiment using other evaluation quantiles like for example 0.25 and 0.75. We will use the same configuration as in the previous experiment, we will just add 5 runs of training and evaluation with `quantiles=[0.25,0.75]`.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/g3p06psz?nw=nwuserfrizzodavide)

###### Metrics Table

Now we have a new best quantile, quantile 0.75 even though in `Life_median` quantile 0.5 is still the best one. Symmetrically quantile 0.25 is the second worst one after quantile 0.1.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     19.49
quantile_0.25    18.85
quantile_0.5     17.03
quantile_0.75    16.26
quantile_0.9     18.22
##################################################
Median eval loss over all the test lifes:
quantile_0.1     16.48
quantile_0.25    15.46
quantile_0.5     11.30
quantile_0.75    11.90
quantile_0.9     13.99
##################################################
Std eval loss over all the test lifes:
quantile_0.1     16.33
quantile_0.25    15.66
quantile_0.5     13.83
quantile_0.75    12.78
quantile_0.9     13.24
##################################################
```


| Life  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_51 | 29.15 | 27.32 | 26.44 | 25.07 | 23.94 |
| Life_52 | 1.59 | 1.5 | 2.51 | 2.69 | 3.31 |
| Life_53 | 1.75 | 2.4 | 3.95 | 5.12 | 5.55 |
| Life_54 | 29.13 | 28.45 | 24.92 | 23.1 | 23.9 |
| Life_55 | 54.37 | 53.27 | 48.88 | 46.75 | 47.15 |
| Life_56 | 5.46 | 6.72 | 9.78 | 11.35 | 11.99 |
| Life_57 | 15.21 | 13.85 | 14.24 | 11.9 | 6.89 |
| Life_58 | 3.47 | 3.19 | 2.31 | 2.17 | 3.34 |
| Life_59 | 36.74 | 35.17 | 30.87 | 27.31 | 28.23 |
| Life_60 | 19.7 | 17.42 | 17.6 | 14.94 | 11.96 |
| Life_61 | 4.62 | 5.22 | 4.44 | 5.92 | 8.33 |
| Life_62 | 10.16 | 9.49 | 9.17 | 7.36 | 7.34 |
| Life_63 | 9.37 | 8.49 | 8.22 | 6.31 | 3.62 |
| Life_64 | 5.51 | 6.45 | 8.8 | 9.98 | 10.42 |
| Life_65 | 39.78 | 38.04 | 28.35 | 22.81 | 30.02 |
| Life_mean | 17.73 | 17.13 | 16.03 | 14.85 | 15.07 |
| Life_median | 12.68 | 11.67 | 12.01 | 11.62 | 11.19 |
| Life_std | 15.42 | 14.77 | 12.53 | 11.39 | 12.01 |


###### Metrics Table Pinball Loss

I realized that probably the `RMSE` is not the ideal metric for evaluating the models since we want a model that prefers underestimation over overestimation. So here I will produce the usual `metrics_df` but using the `Pinball` loss with $\tau=0.3$ instead of the `RMSE` loss.

Now quantile 0.5 is the better one but we can start to see how the quantile 0.1 and 0.25 are the second and third best ones differently from the previous evaluation.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     6.68
quantile_0.25    6.68
quantile_0.5     6.35
quantile_0.75    6.48
quantile_0.9     7.98
##################################################
Median eval loss over all the test lifes:
quantile_0.1     5.21
quantile_0.25    5.42
quantile_0.5     5.21
quantile_0.75    5.43
quantile_0.9     7.08
##################################################
Std eval loss over all the test lifes:
quantile_0.1     4.86
quantile_0.25    4.83
quantile_0.5     4.52
quantile_0.75    4.63
quantile_0.9     6.17
##################################################
```

###### Statistical test results

Since the `RMSE` metrics are quite similar among the different quantiles in several models I want to try to perform some simple statistical tests to see weather, when there are some differences, these differences are statistically significant or are just due to random noise. In other words in this way we can see weather these different quantiles have an impact or not on the model predictions or not, otherwise these experiments are a bit useless → in this case we may just return back to the experiments with the Pinball Loss so that at least we push the model to prefer underestimation over overestimation.

We start with the comparison between quantile 0.1 and 0.9, which are the ones with the highest difference in `Life_mean`.

```txt
##################################################
Independent T-test results between quantiles 0.1 and 0.9
t-statistic: -2.3302
p-value: 0.0481
##################################################
```

Here the `p_value` is lower than 0.05 so the difference is statistically significant.

Quantile 0.25 and 0.75:

```txt
##################################################
Independent T-test results between quantiles 0.25 and 0.75
t-statistic: 0.7102
p-value: 0.4977
##################################################
```

Here there is not more statistical significance, no good 😢.

Quantile 0.5 and 0.25:

```txt
##################################################
Independent T-test results between quantiles 0.25 and 0.5
t-statistic: 1.6434
p-value: 0.1389
##################################################
```

Now we are still not significant but with a smaller `p_value`.

Quantile 0.1 and 0.25. Here the `Life_mean` values are equal (at least if we consider the first 2 decimal digits) so I do not have high hopes:

```txt
##################################################
Independent T-test results between quantiles 0.1 and 0.25
t-statistic: -0.0093
p-value: 0.9928
##################################################
```

As expected the `p_value` is almost 1, so no statistical significance.

###### Prediction Plots bis

After looking a little bit at the plots it does not make a lot of sense to provide a detailed explanation/comment on them since the situation is similar to what I already described in the previous experiment. Probably it's better if I produce a plot with the quantiles 0.25,0.5 and 0.75 since they are better than 0.1 and 0.9 respectively → the difference however it's not enormous so the order in which the line will appear will be the same. So for example in `Life_51,Life_54,Life_55,Life_57,Life_60` if in the first set of plots we had quantile 0.5 and 0.9 as the closest ones to the true `RUL` if I produce the plot with quantile 0.25,0.5 and 0.75 the closest line to the true `RUL` will be the one of quantile 0.5 and quantile 0.75.

##### Experiment 2 tris `S4` `FD001` 4‍⃣ 🪟 🌗

Considering the not very exciting results obtained in the statistical tests of the previous experiment let's try to add 5 more runs to it, these will be run 6 to 10, having an higher sample size may help to obtain more statistically significant results.


>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/ex5fz7vn?nw=nwuserfrizzodavide)

###### Metrics Table

Adding more runs the trend showed in the previous experiment is confirmed. Quantile 0.75 is still the best quantile followed by 0.25. This is due as usual by the fact that the `RMSE` metric care about having the closest possible prediction to the target (independently on weather we have an under or overestimation) and thus overestimation are better because the biggest errors committed by the model are underestimation errors.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     20.49
quantile_0.25    19.14
quantile_0.5     17.54
quantile_0.75    16.82
quantile_0.9     18.20
##################################################
Median eval loss over all the test lifes:
quantile_0.1     16.81
quantile_0.25    16.05
quantile_0.5     13.38
quantile_0.75    12.32
quantile_0.9     13.68
##################################################
Std eval loss over all the test lifes:
quantile_0.1     16.62
quantile_0.25    16.05
quantile_0.5     13.88
quantile_0.75    13.19
quantile_0.9     13.41
##################################################
```

###### Metrics Table Pinball Loss

Looking at the Pinball Loss we expect to have better performances in the lower quantiles. In terms of `Life_mean` the best quantile is still 0.75 but with a small advantage on quantile 0.25, while in `Life_median` the best quantile is clearly 0.25. However here I don't know weather these differences are statistically significant.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     6.99
quantile_0.25    6.63
quantile_0.5     6.53
quantile_0.75    6.57
quantile_0.9     8.06
##################################################
Median eval loss over all the test lifes:
quantile_0.1     5.65
quantile_0.25    5.44
quantile_0.5     5.55
quantile_0.75    5.61
quantile_0.9     6.92
##################################################
Std eval loss over all the test lifes:
quantile_0.1     4.88
quantile_0.25    4.81
quantile_0.5     4.49
quantile_0.75    4.65
quantile_0.9     6.46
##################################################
```
##### Experiment 3 `S4` `FD001` 4‍⃣ 🪟 🌗

Let's try to reduce the bounds of the quantile sampling distribution to `bounds=[0.1,0.5]`. In this way we should force the model to prefer underestimation rather than overestimation. We will use the usual evaluation quantiles `[0.1,0.25,0.5,0.75,0.9]`. Theoretically now we should see a clear difference in the evaluations with quantiles 0.1,0.25 and 0.5 → then we will see how that affects the `RUL` predictions. The fear I have is that, since when the model commits big errors these are always underestimation errors, the model will be pushed to commit even bigger errors.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/oc9of0gt?nw=nwuserfrizzodavide)

###### Metrics Table

Apparently restricting the bounds still makes the quantile 0.75 as the best one. Probably because, as written above, the errors are mainly in underestimation and thus the model needs to overestimate a little bit to get fairly close to the true `RUL` signal.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     24.92
quantile_0.25    21.23
quantile_0.5     22.44
quantile_0.75    18.55
quantile_0.9     22.80
##################################################
Median eval loss over all the test lifes:
quantile_0.1     16.42
quantile_0.25    15.84
quantile_0.5     17.31
quantile_0.75    13.03
quantile_0.9     15.79
##################################################
Std eval loss over all the test lifes:
quantile_0.1     21.95
quantile_0.25    18.59
quantile_0.5     19.64
quantile_0.75    15.53
quantile_0.9     18.51
##################################################
```

###### Metrics Table Pinball Loss

Now using the Pinball Loss with $\tau=0.3$ should produce an improvement in the test metrics with the respect to Experiment 2.

Now the best quantile is 0.25 followed by 0.75. This makes sense because since we are using a Pinball loss with $\tau=0.3$ the underestimations brought by quantile 0.1 are too extreme. However comparing the magnitude of the errors these are higher than the ones obtained in Experiment 2.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     7.76
quantile_0.25    6.80
quantile_0.5     7.41
quantile_0.75    7.11
quantile_0.9     8.32
##################################################
Median eval loss over all the test lifes:
quantile_0.1     5.70
quantile_0.25    6.00
quantile_0.5     5.28
quantile_0.75    6.20
quantile_0.9     7.44
##################################################
Std eval loss over all the test lifes:
quantile_0.1     6.33
quantile_0.25    5.32
quantile_0.5     5.61
quantile_0.75    5.74
quantile_0.9     5.38
##################################################
```

##### Experiment 4 `S4` `FD001` 4‍⃣ 🪟 🌗

Considering the results of the previous experiment let's try to see what happens with `bounds=[0.5,0.99]` → in this way we should force the model to prefer overestimation rather than underestimation and this may hopefully balance the high underestimation errors in the most challenging lifes → for example I want to see what happens with `Life_55,Life_65` that normally have high errors.

>[!note]
> Up to now we never used quantiles lower than 0.1 or higher than 0.9. Now in this experiment with 0.99 let's see what happens.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/5t997kv0?nw=nwuserfrizzodavide)
###### Metrics Table

As I expected the results are now better → as usual the best quantile is 0.75 in both `Life_mean` and `Life_median`.


```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     18.60
quantile_0.25    17.98
quantile_0.5     17.57
quantile_0.75    15.91
quantile_0.9     16.11
##################################################
Median eval loss over all the test lifes:
quantile_0.1     16.95
quantile_0.25    13.65
quantile_0.5     14.17
quantile_0.75    12.91
quantile_0.9     14.70
##################################################
Std eval loss over all the test lifes:
quantile_0.1     14.38
quantile_0.25    14.04
quantile_0.5     13.28
quantile_0.75    11.41
quantile_0.9     10.06
##################################################
```


| Life  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_51 | 29.29 | 24.23 | 23.17 | 15.77 | 16.43 |
| Life_52 | 2.65 | 3.06 | 2.58 | 3.13 | 5.72 |
| Life_53 | 2.5 | 3.38 | 3.9 | 6.39 | 8.51 |
| Life_54 | 24.48 | 22.97 | 24.27 | 22.1 | 18.13 |
| Life_55 | 49.85 | 49.01 | 46.14 | 42.17 | 38.77 |
| Life_56 | 9.87 | 10.81 | 10.61 | 16.92 | 15.37 |
| Life_57 | 17.03 | 12.84 | 8.3 | 4.71 | 2.78 |
| Life_58 | 5.12 | 2.97 | 2.94 | 6.28 | 4.4 |
| Life_59 | 30.87 | 31.93 | 30.4 | 26.58 | 19.79 |
| Life_60 | 21.25 | 17.47 | 13.39 | 9.68 | 10.33 |
| Life_61 | 5.97 | 5.23 | 7.19 | 9.32 | 10.01 |
| Life_62 | 15.2 | 12.88 | 14.72 | 9.56 | 7.13 |
| Life_63 | 8.58 | 6.12 | 3.46 | 4.69 | 7.25 |
| Life_64 | 6.2 | 7.53 | 10.63 | 14.22 | 17.1 |
| Life_65 | 37.54 | 38.67 | 39.75 | 24.13 | 31.49 |
| Life_mean | 17.76 | 16.61 | 16.1 | 14.38 | 14.21 |
| Life_median | 16.12 | 12.86 | 12.01 | 11.95 | 12.27 |
| Life_std | 13.31 | 13.26 | 12.92 | 10.01 | 9.45 |

>[!todo] Things to look at in the results
> - [x] Compare the performances of `Life_55,Life_59,Life_65` → these are lifes that in Experiment 2 had high underestimation errors. I want to see here if the errors decrease in the high quantiles (i.e. 0.75,0.9)

- `Life_55` → In quantiles 0.75 and 0.9 of Experiment 2 we had `RMSE` values of 46.75 and 47.15 respectively. Now we have 42.17 and 38.77, so we have surely improved.
- `Life_59` → In quantiles 0.75 and 0.9 of Experiment 2 we had `RMSE` values of 27.31 and 28.23 respectively. Now we have 26.58 and 19.79, so we have improved.
- `Life_65` → In quantiles 0.75 and 0.9 of Experiment 2 we had `RMSE` values of 22.81 and 30.02 respectively. Now we have 24.13 and 31.49, so we have worsened. On the other hand the performances have improved in quantile 0.1 and 0.25.

>[!todo]
> - [x] Compare the performances in `Life_56,Life_64`: these are lifes that in Experiment 2 are overestimated → in this experiment where we are pushing overestimation there is the risk of increase the error in that direction.

- `Life_56` → In Experiment 2 quantile 0.75 and 0.9 had `RMSE` values of 11.35 and 11.99 respectively. Now we have 16.92 and 15.37, so we have worsened.
- `Life_64` → In Experiment 2 quantile 0.75 and 0.9 had `RMSE` values of 9.98 and 10.42 respectively. Now we have 14.22 and 17.1, so we have worsened.

So in this point we have verified the risk to prefer overestimation → the errors in which the model already overestimates the damage the error increases.

>[!todo]
> - [x] See what happens in `Life_52,Life_53,Life_58,Life_61,Life_62` → these lifes had predictions almost overlapped to the true `RUL` in Experiment 2, now there may be the risk of a small overestimation.

- `Life_52` → In Experiment 2 quantile 0.75 and 0.9 had `RMSE` values of 2.69 and 3.31 respectively. Now we have 3.13 and 5.72, so we have not worsened.
- `Life_53` → In Experiment 2 quantile 0.75 and 0.9 had `RMSE` values of 5.12 and 5.55 respectively. Now we have 6.39 and 8.51, so we have worsened.
- `Life_58` → In Experiment 2 quantile 0.75 and 0.9 had `RMSE` values of 2.17 and 3.34 respectively. Now we have 6.28 and 4.4, so we have worsened.
- `Life_61` → In Experiment 2 quantile 0.75 and 0.9 had `RMSE` values of 5.92 and 8.33 respectively. Now we have 9.32 and 10.01, so we have worsened.
- `Life_62` → In Experiment 2 quantile 0.75 and 0.9 had `RMSE` values of 7.36 and 7.34 respectively. Now we have 9.56 and 7.13, so we have worsened.

The main conclusion after a careful analysis of the results of these experiments:

>[!note] We have to use another evaluation metric other than `RMSE`
I can use the Pinball Loss with a low `tau` so that underestimatinos are more penalized than overestimations, or one of the famous business metrics typically used in `PdM` (that in any case are similar to the Pinball Loss with the difference that the costs for overestimations and underestimations are normally defined according to some business rules which depend on the specific application and data). Moreover probably the costs for overestimations and underestimations are not symmetric as it is in the Pinball Loss

###### Metrics Table Pinball Loss

Let's evaluate the model using the Pinball Loss with $\tau=0.3$.

Now quantiles 0.25 and 0.1 are the best ones and with smaller errors than the ones obtained in Experiment 2. This result is a bit strange.


```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     6.29
quantile_0.25    6.25
quantile_0.5     6.42
quantile_0.75    6.99
quantile_0.9     6.92
##################################################
Median eval loss over all the test lifes:
quantile_0.1     5.87
quantile_0.25    5.63
quantile_0.5     6.10
quantile_0.75    6.28
quantile_0.9     6.55
##################################################
Std eval loss over all the test lifes:
quantile_0.1     4.11
quantile_0.25    3.98
quantile_0.5     3.77
quantile_0.75    4.93
quantile_0.9     3.80
##################################################
```

##### Experiment 5 `S4` `FD001` 4‍⃣ 🪟 🌗

Up to now we have sampled the quantiles from a uniform distribution. Now let's try to sample them from a normal distribution: in this way we are more likely to sample quantiles close to the mean of the distribution and we can sample from a small interval of values. Since the best quantile in the previous experiments was 0.75 I want to try to sample around this value.

Doing some computation exploiting the properties of the Gaussian distribution we will use a `N(mean=0.75,std=0.0048)` that will produce values in `[0.6,0.9]` with a probability close to 100% (i.e. this is the $3\sigma$ interval) but most likely the values will be in `[0.7,0.8]` which is the $σ$ interval.

Since looking at the sampled quantiles, that are logged in the terminal during training, it seemed to me that the model wa always sampling values too close to 0.75 I changed the distribution to `N(mean=0.75,std=0.01)`.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/ltspysjd?nw=nwuserfrizzodavide)

###### Metrics Table

The metrics are a bit worse than the previous experiment, only quantile 0.5 slighlty improved. The metrics are better than the ones of Experiment 2 expect for quantile 0.75 but the difference is minimal (16.63 versus 16.26).

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     18.80
quantile_0.25    18.08
quantile_0.5     16.86
quantile_0.75    16.63
quantile_0.9     17.94
##################################################
Median eval loss over all the test lifes:
quantile_0.1     16.08
quantile_0.25    15.81
quantile_0.5     12.90
quantile_0.75    13.67
quantile_0.9     14.60
##################################################
Std eval loss over all the test lifes:
quantile_0.1     14.09
quantile_0.25    13.30
quantile_0.5     12.28
quantile_0.75    12.13
quantile_0.9     12.97
##################################################
```

###### Metrics Table Pinball Loss

Let's see what happens here with the Pinball Loss as the evaluation metric. Now this one makes slightly more sense, in fact the best quantile is not more one of the two small quantiles 0.25 or 0.1 but the one in the middle 0.5. This is probably due to the fact that in this experiment we have sampled quantile values around 0.75, which is a quantile that is more focused on overestimation than underestimation. The metric values are in any case slightly smaller than the ones obtained in Experiment 2.


```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     6.33
quantile_0.25    6.48
quantile_0.5     6.29
quantile_0.75    6.61
quantile_0.9     6.76
##################################################
Median eval loss over all the test lifes:
quantile_0.1     6.06
quantile_0.25    5.53
quantile_0.5     5.93
quantile_0.75    6.46
quantile_0.9     6.76
##################################################
Std eval loss over all the test lifes:
quantile_0.1     3.90
quantile_0.25    3.83
quantile_0.5     3.46
quantile_0.75    3.99
quantile_0.9     3.60
##################################################
```

##### Experiment 6 `S4` `FD001` 4‍⃣ 🪟 🌗

Let's try to do an experiment which is the opposite of experiment 5, so we will use a normal distribution sampling quantiles around 0.25. We will use `N(mean=0.25,std=0.01)`.


>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/z2klzvl3?nw=nwuserfrizzodavide)

###### Metrics Table Pinball Loss

In this experiment I used directly the Pinball Loss to evaluate the models. 

Ok the results start to be strange again, since the best quantile is quantile 0.75, in any case there is a slight difference with the respect to quantile 0.5 and 0.25. The worst quantile is 0.1 which makes sense because 0.1 is a quantile associated with a too high underestimation with the respect to what we want.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     9.76
quantile_0.25    7.92
quantile_0.5     7.77
quantile_0.75    7.62
quantile_0.9     8.42
##################################################
Median eval loss over all the test lifes:
quantile_0.1     6.40
quantile_0.25    6.13
quantile_0.5     5.90
quantile_0.75    5.67
quantile_0.9     6.44
##################################################
Std eval loss over all the test lifes:
quantile_0.1     8.53
quantile_0.25    6.26
quantile_0.5     6.07
quantile_0.75    5.83
quantile_0.9     6.08
##################################################
```

##### Experiment 7 `S4` `FD001` 4‍⃣ 🪟 🌗

Let's add a multiplication by $\tau$ at the end of the `forward` method of `S4Model` as an additional effect of the quantile other than the concatenation in the input signal.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/2ceixitk?nw=nwuserfrizzodavide)

###### Metrics Table Pinball Loss

Now the best quantile is 0.25 (as it should be) both in `Life_mean` and `Life_median`. However probably the difference with the respect to the other quantiles is still not very significant, so probably it's worth trying to extend the experiment with an additional set of 5 runs to see if the results are confirmed.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     6.68
quantile_0.25    6.30
quantile_0.5     7.26
quantile_0.75    6.84
quantile_0.9     7.50
Name: Life_mean, dtype: float64
##################################################
Median eval loss over all the test lifes:
quantile_0.1     5.21
quantile_0.25    5.19
quantile_0.5     6.43
quantile_0.75    6.24
quantile_0.9     7.18
Name: Life_median, dtype: float64
##################################################
Std eval loss over all the test lifes:
quantile_0.1     4.86
quantile_0.25    4.58
quantile_0.5     5.37
quantile_0.75    5.46
quantile_0.9     5.08
Name: Life_std, dtype: float64
##################################################
```

##### Experiment 7 bis `S4` `FD001` 4‍⃣ 🪟 🌗

Let's extend experiment 7 with 5 additional runs to see weather the results are confirmed.


>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/kjfjt9mz?nw=nwuserfrizzodavide)


###### Metrics Table Pinball Loss

By mistake I set `n_runs: 10` so I added 10 additional runs, which is better actually. Looking at the results however we have a confirmation of the order of the quantiles in the evaluation metrics, but the differences in performances remain more or less the same. In any case quantile 0.25 is still the best one in `Life_mean` while in terms of `Life_median` the best one is quantile 0.1.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     6.93
quantile_0.25    6.51
quantile_0.5     7.08
quantile_0.75    7.07
quantile_0.9     7.52
##################################################
Median eval loss over all the test lifes:
quantile_0.1     5.10
quantile_0.25    5.61
quantile_0.5     6.34
quantile_0.75    6.36
quantile_0.9     6.95
##################################################
Std eval loss over all the test lifes:
quantile_0.1     4.90
quantile_0.25    4.68
quantile_0.5     4.99
quantile_0.75    5.16
quantile_0.9     5.43
##################################################
```

###### Statistical Test Results

Now the difference in `Life_mean` between quantile 0.25 and 0.75 is also statistically significant 💪.


```txt
##################################################
Independent T-test results between quantiles 0.25 and 0.75
t-statistic: -2.2610
p-value: 0.0317
##################################################
```

Also the other pairs of quantiles have statistically significant differences except for 0.1 and 0.75 and 0.5 and 0.75.


#### Dataset `FD002`

##### Experiment 1 `S4` `FD002` 4‍⃣ 🪟 🌗

For the `FD002` dataset we will start with the configuration used in the last experiments and we will use `quantiles=[0.1,0.25,0.5,0.75,0.9]` as the evaluation quantiles. We will have 25 `wandb` experiments launched so it will take some time to finish, considering also the fact that we have much more lifes in this dataset.

| Parameter | Value |
|-----------|-------|
| `model_type` | `S4` |
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
| `n_runs` | 5 |
| `loss` | `quantile_reg` |
| `eval_loss` | `rmse` |
| `quantile_dist` | `uniform` |
| `bounds` | `[0.1,0.9]` |
| `quantiles` | `[0.1,0.25,0.5,0.75,0.9]` |

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/7npxxhga?nw=nwuserfrizzodavide)

The first run took 5 minutes, with 25 runs it will take 2 hours to finish 😱. At the end it took about 2h and 20 min.

###### Metrics Table

The metric values are higher than in `FD001` because of the increased difficulty of the `RUL` prediction task on the `FD002` dataset. For what concerns the comparison between the different quantiles, the best one in `Life_mean` is quantile 0.9 while in `Life_median` quantile 0.75 it's on top. Moreover here the improvement in perfomance of quantiles 0.75 and 0.9 with the respect to quantile 0.5 is more significant.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     41.68
quantile_0.25    34.56
quantile_0.5     35.34
quantile_0.75    32.77
quantile_0.9     30.51
##################################################
Median eval loss over all the test lifes:
quantile_0.1     32.97
quantile_0.25    22.47
quantile_0.5     23.04
quantile_0.75    18.70
quantile_0.9     21.72
##################################################
Std eval loss over all the test lifes:
quantile_0.1     37.49
quantile_0.25    31.93
quantile_0.5     31.07
quantile_0.75    31.40
quantile_0.9     25.58
##################################################
```


| Life  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_132 | 11.18 | 8.85 | 7.21 | 9.2 | 13.86 |
| Life_133 | 6.65 | 4.41 | 5.5 | 5.68 | 8.65 |
| Life_134 | 9.46 | 17.21 | 15.44 | 18.78 | 22.7 |
| Life_135 | 107.96 | 98.46 | 95.07 | 88.2 | 84.67 |
| Life_136 | 34.7 | 18.39 | 27.4 | 21.34 | 21.54 |
| Life_137 | 21.79 | 21.78 | 22.29 | 15.3 | 11.86 |
| Life_138 | 6.8 | 4.55 | 5.39 | 3.61 | 4.24 |
| Life_139 | 23.66 | 11.08 | 7.8 | 11.23 | 10.03 |
| Life_140 | 58.05 | 62.23 | 55.9 | 55.11 | 58.32 |
| Life_141 | 98.25 | 78.44 | 80.97 | 74.39 | 64.14 |
| Life_142 | 6.17 | 8.07 | 8.7 | 10.68 | 20.87 |
| Life_143 | 15.25 | 16.86 | 14.49 | 10.55 | 7.74 |
| Life_144 | 5.62 | 8.03 | 8.43 | 13.38 | 20.72 |
| Life_145 | 27.26 | 18.67 | 19.98 | 10.22 | 15.87 |
| Life_146 | 14.09 | 17.44 | 18.66 | 23.01 | 26.66 |
| Life_mean | 29.79 | 26.3 | 26.22 | 24.71 | 26.12 |
| Life_median | 18.52 | 17.33 | 17.05 | 14.34 | 20.8 |
| Life_std | 30.89 | 27.2 | 26.52 | 24.53 | 22.17 |

###### Metrics Table Pinball Loss

The best quantiles are 0.25 and 0.75 with a minimal differences. Comparing the results with the `metrics_df` computed using the `RMSE` I think that this result is due to the fact that with `RMSE` there was a significant gap between quantiles 0.75 and 0.9 so the Pinball Loss with $\tau=0.3$ as not able to completely invert the trend. This is probably happening because in this more difficult dataset we have higher underestimation errors (so the overestimation brought by 0.75 and 0.9 is beneficial if we want to be as close as possible to the true `RUL`).

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     13.26
quantile_0.25    11.69
quantile_0.5     12.46
quantile_0.75    11.70
quantile_0.9     12.31
##################################################
Median eval loss over all the test lifes:
quantile_0.1     10.74
quantile_0.25     8.55
quantile_0.5     10.50
quantile_0.75     9.20
quantile_0.9      9.76
##################################################
Std eval loss over all the test lifes:
quantile_0.1     10.90
quantile_0.25     9.27
quantile_0.5      8.96
quantile_0.75     9.03
quantile_0.9      8.11
##################################################
```

###### Prediction Plots `FD002`

I will provide a detailed description of the plots only for `S4` since it is probably the model whose results will be inserted in the paper. In any case, as we have seen in the `FD001` experiments, the plots are following more or less the same trend on all the models (with the little exceptions of `LSTM` and `Transformer`). Let's start with some first comments on the `run_1` plot:

- `Life_132` has an interesting set of predictions. Quantiles 0.25 and 0.75 are intersecting the true `RUL` signal: so we have underestimation at the beginning and overestimation at the end. On the other hand quantile 0.5 is always overestimating the target.
- `Life_133` → in this life we have quantile 0.25 and 0.75 overlapped to the target while 0.5 is sligthly overestimating the `RUL`.
- Also in this case we have some lifes where the models are significantly underestimating the `RUL` signal. These are `Life_135,Life_141,Life_140`. In particular `Life_135,Life_141` can be noticed also looking at the `metrics_df` above as the ones with the highest `RMSE` values.
- We have then some lifes in which the predictions are good for all quantiles: `Life_138,Life_142,Life_145`.
- `Life_134,Life_144,Life_146` → all overestimating the target, which 0.75 with the highest error obviously.
- `Life_140,Life_143` → here the predicted `RUL` signals are not properly straight across all the time samples.

### `S5` Model Experiments 5️⃣ 🪟 🌗

#### Dataset `FD001`

##### Experiment 1 `S5` `FD001` 5️⃣ 🪟 🌗

Let's use the same configuration used in the `S4` model. This time however we will start directly using all the 10 quantile levels from 0.1 to 0.9, so `quantiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]`.


>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/wd5tzu05?nw=nwuserfrizzodavide)

The trend of the loss plots on `wandb` is similar to the one of the `S4` model, maybe the loss lines are less smooth than with `S4` but this is aligned with what we saw in the experiments on the other approaches where `S4D,S5` produce predictions with more variable errors across the different lifes.

###### Metrics Table

The metric values are a bit higher in this case with respect to the `S4` model. 

|  | quantile_0.1 | quantile_0.2 | quantile_0.3 | quantile_0.4 | quantile_0.5 | quantile_0.6 | quantile_0.7 | quantile_0.8 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Life_50 | 48.44 | 49.33 | 32.55 | 23.66 | 33.79 | 31.49 | 15.87 | 15.56 | 17.96 |
| Life_51 | 2.47 | 2.46 | 5.07 | 9.67 | 4.31 | 5.49 | 6.72 | 7.35 | 18.96 |
| Life_52 | 5.21 | 6.01 | 15.08 | 10.5 | 9.05 | 11.53 | 20.75 | 14.64 | 21.23 |
| Life_53 | 34.31 | 50.92 | 16.09 | 8.98 | 2.54 | 31.74 | 8.23 | 16.86 | 2.23 |
| Life_54 | 68.19 | 81.57 | 39.0 | 22.0 | 34.41 | 49.04 | 8.43 | 49.63 | 13.13 |
| Life_55 | 18.6 | 30.56 | 67.33 | 57.0 | 78.43 | 78.0 | 84.39 | 63.03 | 79.11 |
| Life_56 | 27.36 | 34.01 | 0.75 | 22.2 | 3.07 | 13.21 | 15.81 | 12.14 | 17.32 |
| Life_57 | 6.73 | 1.37 | 1.59 | 2.46 | 2.83 | 6.53 | 3.19 | 1.76 | 7.73 |
| Life_58 | 64.62 | 86.43 | 17.58 | 14.91 | 36.15 | 27.84 | 43.79 | 43.04 | 17.05 |
| Life_59 | 27.5 | 29.23 | 10.73 | 20.73 | 8.8 | 10.94 | 15.05 | 8.29 | 15.47 |
| Life_60 | 4.38 | 8.68 | 23.13 | 18.95 | 24.94 | 18.36 | 32.28 | 17.65 | 30.77 |
| Life_61 | 18.3 | 19.32 | 14.86 | 13.74 | 2.27 | 4.76 | 12.26 | 10.3 | 21.2 |
| Life_62 | 9.35 | 11.3 | 16.83 | 9.55 | 24.16 | 23.01 | 35.4 | 4.84 | 36.37 |
| Life_63 | 0.99 | 5.36 | 3.82 | 7.53 | 12.0 | 6.33 | 18.16 | 6.36 | 24.34 |
| Life_64 | 90.14 | 137.17 | 28.29 | 55.31 | 52.16 | 7.79 | 28.87 | 61.63 | 23.73 |
| Life_mean | 28.44 | 36.91 | 19.51 | 19.81 | 21.93 | 21.74 | 23.28 | 22.21 | 23.11 |
| Life_std | 26.73 | 37.52 | 16.72 | 15.48 | 21.37 | 19.43 | 19.78 | 20.31 | 16.98 |

The best quantile in terms of the `Life_mean` metric is `quantile_0.3` while `quantile_0.2` is the worst one. The `Life_std` is in general higher than the ones obtained with the `S4` model. This result is a bit surprising considering the considerations we made on the `S4` results and I think that I know the reason why.

One thing to note is the incredible difference in performances there is in `Life_57` between the different quantiles. In `quantile_0.3` we have an almost null error of 0.75, a small error of 3.07 for `quantile_0.5` and then all the other quantiles have an error higher than 10 😱. This may also be the reason why at the end the `Life_mean` of `quantile_0.3` is so low, I need to also introduce a `Life_median` metric to see if this is the case.

>[!error]
> In these first two experiments with the Quantile Regression Approach I committed a small error, that however can be solved. In the training loop we are sampling at random the quantile level `tau` for each sample and this has an effect on the performances of the models in the evaluations. So in this case it may be that quantile levels 0.2 and 0.3 obtained better results just becuase during the training loop there was a majority of samples with those quantile levels. We are using the uniform distribution to extract the quantile levels so it's unlikely that in a certain run there is a majority of samples with a certain quantile level. However to be robust to the possibility of a skewed sampling we should perform multiple runs of this experiment with a different seed for each run and then average the metrics over all the runs.

^7aa41a

###### Prediction Plots

I will give a more detailed analysis of the plots when I will transfer them locally on my pc where I can see tham better, but probably they will be quite similar to the ones produced in the `S4` experiment.

##### Experiment 2 `S5` `FD001` 5️⃣ 🪟 🌗

Let's perform a multi run experiment with the same configuration used in `S4`.


>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/204g887h?nw=nwuserfrizzodavide)

###### Metrics Table

The metrics values, as expected, are sligthly higher than the ones of `S4`. What is interesting is that in this case the best quantile is 0.5,however the difference with the respect to 0.9 is not so huge.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1    32.44
quantile_0.5    30.00
quantile_0.9    31.28
##################################################
Median eval loss over all the test lifes:
quantile_0.1    26.87
quantile_0.5    23.76
quantile_0.9    25.10
##################################################
Std eval loss over all the test lifes:
quantile_0.1    20.64
quantile_0.5    23.85
quantile_0.9    20.67
##################################################
```

| Life  | quantile_0.1 | quantile_0.5 | quantile_0.9 |
| --- | --- | --- | --- |
| Life_51 | 39.3 | 30.4 | 18.13 |
| Life_52 | 4.97 | 4.95 | 12.91 |
| Life_53 | 8.44 | 12.69 | 17.96 |
| Life_54 | 33.07 | 24.94 | 10.6 |
| Life_55 | 58.71 | 56.17 | 41.36 |
| Life_56 | 35.1 | 31.78 | 55.15 |
| Life_57 | 30.0 | 15.98 | 12.72 |
| Life_58 | 11.84 | 4.16 | 7.45 |
| Life_59 | 43.18 | 48.37 | 36.46 |
| Life_60 | 26.61 | 10.17 | 7.67 |
| Life_61 | 12.76 | 13.19 | 19.88 |
| Life_62 | 18.67 | 7.2 | 10.95 |
| Life_63 | 8.57 | 7.44 | 12.91 |
| Life_64 | 6.41 | 9.63 | 20.74 |
| Life_65 | 76.97 | 81.05 | 75.64 |
| Life_mean | 27.64 | 23.87 | 24.04 |
| Life_median | 27.12 | 14.58 | 18.05 |
| Life_std | 19.55 | 21.05 | 18.52 |

###### Prediction Plots

The general shape of the plots is similar to the one of the `S4` model.

Below some general comments based on the plot of `run_1`:

- In `Life_51,Life_54,Life_55,Life_59,Life_65` we have the prediction significantly far from the true values
- `LIfe_56` is the only life with all overestimation errors. Interestingly in `run_2,run_3` quantile 0.1 is intersecting the true `RUL` signal, something rarely seen in these experiments, normally the predicted and true signals are always almost parallel lines.
- The lifes where the predictions are good are: `Life_52,Life_53,Life_58,Life_61,Life_62`, like the ones observed in the `S4` model.
- Differently from `S4` in some lifes we can see some small oscillations in some of the predicted `RUL` signals which are not perfectly smooth lines. I am referring to `Life_57,Life_59,Life_60,Life_63,Life_65`.

Interesting things from the plots of other runs:

- In `run_3` in `Life_55,Life_56` and `Life_59` all the 4 signals (the predictions from the three quantile levels and the true `RUL` signal) are all clearly separated one from each other, with quantile 0.9 being the closest to the true signal.

##### Experiment 2 bis `S5` `FD001` 5️⃣ 🪟 🌗

Continue the experiment adding `quantiles=[0.25,0.75]` as evaluation quantiles.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/xjwfg67m?nw=nwuserfrizzodavide)

###### Metrics Table

Also here now the best quantile is 0.75 and here it is also the best on `Life_median`. The `Life_std` metrics are higher than in `S4`.


```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     32.44
quantile_0.25    30.78
quantile_0.5     30.00
quantile_0.75    29.01
quantile_0.9     31.28
##################################################
Median eval loss over all the test lifes:
quantile_0.1     26.87
quantile_0.25    29.05
quantile_0.5     23.76
quantile_0.75    21.84
quantile_0.9     25.10
##################################################
Std eval loss over all the test lifes:
quantile_0.1     20.64
quantile_0.25    19.88
quantile_0.5     23.85
quantile_0.75    21.14
quantile_0.9     20.67
##################################################
```


| Life  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_51 | 39.3 | 33.98 | 30.4 | 21.84 | 18.13 |
| Life_52 | 4.97 | 5.26 | 4.95 | 6.71 | 12.91 |
| Life_53 | 8.44 | 9.94 | 12.69 | 17.81 | 17.96 |
| Life_54 | 33.07 | 29.16 | 24.94 | 17.71 | 10.6 |
| Life_55 | 58.71 | 53.45 | 56.17 | 48.49 | 41.36 |
| Life_56 | 35.1 | 37.64 | 31.78 | 36.21 | 55.15 |
| Life_57 | 30.0 | 25.83 | 15.98 | 10.57 | 12.72 |
| Life_58 | 11.84 | 9.63 | 4.16 | 3.44 | 7.45 |
| Life_59 | 43.18 | 40.36 | 48.37 | 42.15 | 36.46 |
| Life_60 | 26.61 | 21.79 | 10.17 | 4.47 | 7.67 |
| Life_61 | 12.76 | 13.89 | 13.19 | 17.6 | 19.88 |
| Life_62 | 18.67 | 15.65 | 7.2 | 7.77 | 10.95 |
| Life_63 | 8.57 | 8.52 | 7.44 | 12.35 | 12.91 |
| Life_64 | 6.41 | 7.25 | 9.63 | 16.0 | 20.74 |
| Life_65 | 76.97 | 74.92 | 81.05 | 75.63 | 75.64 |
| Life_mean | 27.64 | 25.82 | 23.87 | 22.58 | 24.04 |
| Life_median | 27.12 | 23.8 | 14.58 | 17.66 | 18.05 |
| Life_std | 19.55 | 18.46 | 21.05 | 18.72 | 18.52 |

###### Metrics Table Pinball Loss

The metric values are much higher than the `S4` ones but also in this case we can see better metrics in quantile 0.25 and 0.5 than in the others, so we prefer underestimating models.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     11.59
quantile_0.25    11.49
quantile_0.5     11.16
quantile_0.75    11.90
quantile_0.9     13.75
##################################################
Median eval loss over all the test lifes:
quantile_0.1     11.59
quantile_0.25    11.47
quantile_0.5      9.35
quantile_0.75    12.22
quantile_0.9     12.22
##################################################
Std eval loss over all the test lifes:
quantile_0.1     6.70
quantile_0.25    6.91
quantile_0.5     7.12
quantile_0.75    7.04
quantile_0.9     8.72
##################################################
```

###### Prediction Plots

See [[ssm_experiments#Prediction Plots bis|here]]

##### Experiment 3 `S5` `FD001` 5️⃣ 🪟 🌗

Seen the good results of `S4` on [[ssm_experiments#Experiment 7 bis `S4` `FD001` 4‍⃣ 🪟 🌗|experiment 7]] with the addition of $\tau$ as a multiplicative factor on the final model output I added it also in `S5` and let's see what happens here. We will start with a 5 run experiment just to see what kind of metrics we get, then we may decide to extend the experiment to 15 runs as it was done for `S4`.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/m2qzw29x?nw=nwuserfrizzodavide)

Looking at the `metrics_df` after 5 runs the results were quite promising: quantile 0.1 is the best one with a significant margin on 0.1 and 0.75. The closer quantile to 0.25 is 0.5. Given these results I decided to expand the experiment with 10 additional runs.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/as9vfo4a?nw=nwuserfrizzodavide)

###### Metrics Table Pinball Loss

The results are confirming the ones obtained with the first 5 runs. The best quantile is 0.25 followed by 0.1.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     11.90
quantile_0.25    11.09
quantile_0.5     12.34
quantile_0.75    12.88
quantile_0.9     13.12
##################################################
Median eval loss over all the test lifes:
quantile_0.1     11.12
quantile_0.25    11.03
quantile_0.5     11.14
quantile_0.75    12.30
quantile_0.9     12.62
##################################################
Std eval loss over all the test lifes:
quantile_0.1     7.89
quantile_0.25    6.85
quantile_0.5     7.73
quantile_0.75    7.49
quantile_0.9     7.15
##################################################
```

Doing the T-test on quantile 0.25 and 0.75 we have a super small p-value, lower than $10^{-4}$ meaning that there is a super statistical significance in this difference.

```txt
##################################################
Independent T-test results between quantiles 0.25 and 0.75
t-statistic: -5.3361
p-value: 0.0000
##################################################
```

#### Dataset `FD002`

##### Experiment 1 `S5` `FD002` 5️⃣ 🪟 🌗

Let's use the same configuration as in the first `FD002` experiment for `S4`.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/efvz46dd?nw=nwuserfrizzodavide)

The first run took about 8 minutes so here probably the whole experiment will take some more time than `S4`. At the end it finished in about 4h and 20 minutes 😱.

###### Metrics Table

The results are confirming what we saw with `S4`. In `Life_Mean` the best quantile is still 0.9 but quantile 0.75 is very close and it's the best quantile in terms of `Life_median`. The `RMSE` metrics value are higher than the ones obtained with `S4`, in particular the ones in quantiles 0.1 and 0.25.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     51.84
quantile_0.25    53.86
quantile_0.5     45.85
quantile_0.75    41.62
quantile_0.9     40.66
##################################################
Median eval loss over all the test lifes:
quantile_0.1     40.37
quantile_0.25    44.07
quantile_0.5     45.85
quantile_0.75    33.91
quantile_0.9     34.19
##################################################
Std eval loss over all the test lifes:
quantile_0.1     38.12
quantile_0.25    38.60
quantile_0.5     28.11
quantile_0.75    28.86
quantile_0.9     25.01
##################################################
```


| Life  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_132 | 24.84 | 35.84 | 52.46 | 28.69 | 38.12 |
| Life_133 | 5.42 | 3.47 | 8.94 | 11.01 | 10.56 |
| Life_134 | 28.67 | 28.29 | 28.75 | 26.23 | 37.64 |
| Life_135 | 77.09 | 67.16 | 65.95 | 73.12 | 60.65 |
| Life_136 | 66.36 | 84.1 | 78.42 | 95.52 | 72.45 |
| Life_137 | 40.37 | 29.69 | 25.8 | 14.39 | 13.62 |
| Life_138 | 9.03 | 12.2 | 6.24 | 4.58 | 4.08 |
| Life_139 | 35.65 | 44.61 | 51.32 | 33.02 | 47.62 |
| Life_140 | 67.35 | 62.27 | 56.4 | 53.51 | 51.81 |
| Life_141 | 97.66 | 98.21 | 81.21 | 58.64 | 60.13 |
| Life_142 | 6.11 | 7.0 | 11.63 | 10.69 | 11.71 |
| Life_143 | 26.18 | 35.84 | 38.43 | 33.11 | 23.68 |
| Life_144 | 18.55 | 19.75 | 26.95 | 13.14 | 24.32 |
| Life_145 | 26.42 | 30.17 | 18.31 | 8.34 | 13.86 |
| Life_146 | 13.78 | 23.04 | 36.24 | 25.54 | 33.77 |
| Life_mean | 36.23 | 38.78 | 39.14 | 32.64 | 33.6 |
| Life_median | 27.54 | 33.01 | 37.34 | 27.46 | 33.68 |
| Life_std | 26.46 | 26.06 | 22.76 | 24.85 | 19.94 |

###### Metrics Table Pinball Loss

Here with `RMSE` the gap between 0.75-0.9 is even higher than the one seen in `S4` so here still 0.75 and 0.9 are the best quantiles but the gap is reduced.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     21.07
quantile_0.25    24.12
quantile_0.5     24.11
quantile_0.75    20.35
quantile_0.9     20.47
##################################################
Median eval loss over all the test lifes:
quantile_0.1     19.83
quantile_0.25    23.81
quantile_0.5     24.11
quantile_0.75    17.52
quantile_0.9     19.83
##################################################
Std eval loss over all the test lifes:
quantile_0.1     13.48
quantile_0.25    16.09
quantile_0.5     15.18
quantile_0.75    15.26
quantile_0.9     12.83
##################################################
```

###### Prediction Plots

See [[ssm_experiments#Prediction Plots `FD002`|here]] . Also in this case `Life_141,Life_135` are clearly the ones where the models perform the worst.

### `S4D` Model Experiments 4 D 🪟 🌗

#### Dataset `FD001`

##### Experiment 1 `S4D` `FD001` 4 D 🪟 🌗

Let's perform a multi run experiment with the same configuration used in `S4`.


>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/r99i7gfn?nw=nwuserfrizzodavide)

###### Metrics Table

The high variability in the results that we have observed in the previous experiments for `S4D` is confirmed also in this case. The metrics values are worse than the ones obtained with `S4,S5`. The mean and median values among the different quantiles are quite similar and the best ones are quantile 0.5 and quantile 0.9.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1    41.34
quantile_0.5    39.95
quantile_0.9    40.56
##################################################
Median eval loss over all the test lifes:
quantile_0.1    26.14
quantile_0.5    28.52
quantile_0.9    28.34
##################################################
Std eval loss over all the test lifes:
quantile_0.1    38.11
quantile_0.5    36.59
quantile_0.9    35.70
##################################################
```

| Life  | quantile_0.1 | quantile_0.5 | quantile_0.9 |
| --- | --- | --- | --- |
| Life_51 | 59.33 | 50.58 | 46.72 |
| Life_52 | 8.21 | 2.89 | 2.95 |
| Life_53 | 1.38 | 4.34 | 9.11 |
| Life_54 | 62.63 | 64.94 | 63.64 |
| Life_55 | 102.6 | 103.09 | 98.27 |
| Life_56 | 18.47 | 36.51 | 46.97 |
| Life_57 | 24.95 | 13.2 | 12.06 |
| Life_58 | 1.48 | 3.67 | 10.52 |
| Life_59 | 88.23 | 89.91 | 90.69 |
| Life_60 | 42.07 | 35.07 | 30.68 |
| Life_61 | 3.59 | 6.69 | 11.61 |
| Life_62 | 18.16 | 10.23 | 4.55 |
| Life_63 | 19.61 | 12.51 | 9.9 |
| Life_64 | 1.64 | 7.03 | 10.56 |
| Life_65 | 118.37 | 116.8 | 118.71 |
| Life_mean | 38.05 | 37.16 | 37.8 |
| Life_median | 22.28 | 24.14 | 21.37 |
| Life_std | 36.93 | 36.83 | 36.14 |

###### Prediction Plots

In the plots we can see a clear difference from the other `SSM` models, which is also a confirmation of the bad performances of `S4D`. The predicted signals are not at all smooth decreasing lines but show clear oscillations.

General comments based on the plot of `run_1`:

- As usual `Life_51,Life_54,Life_55,Life_59,Life_65` are the lifes with the worst predictions. In particular in `Life_55` and `Life_65` the error is very high (i.e. higher than 100 as we can see from the `metrics_df` above).
- The lifes with the best predictions are the usual `Life_52,Life_53,Life_58,Life_61,Life_62`.
- As in the other experiments `Life_56` has all overestimation errors the higher being the one of quantile 0.9.

##### Experiment 2 bis `S4D` `FD001` 4‍⃣ D 🪟 🌗

Continue the experiment adding `quantiles=[0.25,0.75]` as evaluation quantiles.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/ldwnbeoc?nw=nwuserfrizzodavide)

###### Metrics Table

The best quantile is 0.75 but strangely in `Life_median` the best one is quantile 0.25 → this is probably due to the high variability in the metrics we have in `S4D` which is the model with the highest values on the `Life_std` metric. 

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     41.34
quantile_0.25    40.19
quantile_0.5     39.95
quantile_0.75    38.53
quantile_0.9     40.56
##################################################
Median eval loss over all the test lifes:
quantile_0.1     26.14
quantile_0.25    24.04
quantile_0.5     28.52
quantile_0.75    26.29
quantile_0.9     28.34
##################################################
Std eval loss over all the test lifes:
quantile_0.1     38.11
quantile_0.25    37.28
quantile_0.5     36.59
quantile_0.75    34.72
quantile_0.9     35.70
##################################################
```


| Life  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_51 | 59.33 | 57.0 | 50.58 | 46.32 | 46.72 |
| Life_52 | 8.21 | 6.15 | 2.89 | 1.65 | 2.95 |
| Life_53 | 1.38 | 2.02 | 4.34 | 7.61 | 9.11 |
| Life_54 | 62.63 | 61.75 | 64.94 | 63.14 | 63.64 |
| Life_55 | 102.6 | 101.24 | 103.09 | 100.13 | 98.27 |
| Life_56 | 18.47 | 23.07 | 36.51 | 43.69 | 46.97 |
| Life_57 | 24.95 | 22.17 | 13.2 | 8.59 | 12.06 |
| Life_58 | 1.48 | 1.66 | 3.67 | 7.27 | 10.52 |
| Life_59 | 88.23 | 87.31 | 89.91 | 87.81 | 90.69 |
| Life_60 | 42.07 | 39.6 | 35.07 | 30.98 | 30.68 |
| Life_61 | 3.59 | 5.24 | 6.69 | 9.86 | 11.61 |
| Life_62 | 18.16 | 15.19 | 10.23 | 5.68 | 4.55 |
| Life_63 | 19.61 | 17.35 | 12.51 | 8.92 | 9.9 |
| Life_64 | 1.64 | 2.75 | 7.03 | 10.45 | 10.56 |
| Life_65 | 118.37 | 117.3 | 116.8 | 112.65 | 118.71 |
| Life_mean | 38.05 | 37.32 | 37.16 | 36.32 | 37.8 |
| Life_median | 22.28 | 22.62 | 24.14 | 20.72 | 21.37 |
| Life_std | 36.93 | 36.41 | 36.83 | 35.69 | 36.14 |


###### Metrics Table Pinball Loss

Similarly to `S5` higher errors than `S4` but also in this case the best quantiles are 0.25 and 0.5.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     13.02
quantile_0.25    12.97
quantile_0.5     13.50
quantile_0.75    13.66
quantile_0.9     14.56
##################################################
Median eval loss over all the test lifes:
quantile_0.1     11.18
quantile_0.25    10.59
quantile_0.5     10.92
quantile_0.75    11.09
quantile_0.9     12.59
##################################################
Std eval loss over all the test lifes:
quantile_0.1     11.01
quantile_0.25    10.66
quantile_0.5     10.51
quantile_0.75    10.11
quantile_0.9     10.36
##################################################
```

###### Prediction Plots

See [[ssm_experiments#Prediction Plots bis|here]]

##### Experiment 3 `S4D` `FD001` 4 D 🪟 🌗

Let's do a 15 runs experiment with the new method of using $\tau$ as a multiplicative factor on the model output, ad done for `S4,S5`.


>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/4bqk2jzt?nw=nwuserfrizzodavide)


#### Dataset `FD002`

##### Experiment 1 `S4D` `FD002` 4 D 🪟 🌗

Let's use the same configuration used for `S4,S5`.

>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/1mynri1b?nw=nwuserfrizzodavide)

The first run took about 8 minutes like `S5` so I hope it gets it done in the same time, so if it takes 4h 20 min it should finish around 19:50/20:00. At the end it took just 2 hours.

###### Metrics Table

Here there is a clear dominance of quantile 0.75 as the best one in terms both of `Life_mean` and `Life_median`. Moreover the metric values are not so much higher than the ones obtained in `S5`. The `Life_std` values are still very high as usual though.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     53.70
quantile_0.25    50.42
quantile_0.5     49.27
quantile_0.75    42.36
quantile_0.9     48.88
##################################################
Median eval loss over all the test lifes:
quantile_0.1     42.56
quantile_0.25    37.77
quantile_0.5     35.78
quantile_0.75    28.22
quantile_0.9     33.17
##################################################
Std eval loss over all the test lifes:
quantile_0.1     45.46
quantile_0.25    44.65
quantile_0.5     42.94
quantile_0.75    37.09
quantile_0.9     42.39
##################################################
```


| Life  | quantile_0.1 | quantile_0.25 | quantile_0.5 | quantile_0.75 | quantile_0.9 |
| --- | --- | --- | --- | --- | --- |
| Life_132 | 11.0 | 5.66 | 9.64 | 10.71 | 6.42 |
| Life_133 | 4.06 | 3.34 | 4.92 | 12.89 | 7.59 |
| Life_134 | 16.59 | 12.54 | 18.09 | 21.25 | 22.65 |
| Life_135 | 122.57 | 121.52 | 116.0 | 116.65 | 115.73 |
| Life_136 | 40.61 | 35.05 | 29.66 | 27.36 | 29.71 |
| Life_137 | 28.65 | 21.09 | 18.08 | 9.69 | 13.1 |
| Life_138 | 4.04 | 9.56 | 3.25 | 3.19 | 3.8 |
| Life_139 | 12.28 | 6.0 | 6.85 | 9.82 | 4.24 |
| Life_140 | 61.99 | 60.14 | 62.02 | 50.69 | 50.01 |
| Life_141 | 120.1 | 119.66 | 118.58 | 105.2 | 116.53 |
| Life_142 | 3.73 | 3.41 | 13.59 | 13.34 | 12.42 |
| Life_143 | 25.85 | 17.93 | 17.91 | 8.19 | 10.66 |
| Life_144 | 7.28 | 2.58 | 3.25 | 8.9 | 9.14 |
| Life_145 | 50.47 | 43.14 | 36.53 | 33.71 | 34.18 |
| Life_146 | 13.08 | 15.38 | 19.02 | 25.64 | 23.42 |
| Life_mean | 34.82 | 31.8 | 31.83 | 30.48 | 30.64 |
| Life_median | 21.22 | 16.66 | 18.08 | 17.3 | 17.88 |
| Life_std | 36.9 | 37.24 | 35.59 | 32.85 | 34.74 |

###### Metrics Table Pinball Loss

The results are closer between different quantiles. The best one in `Life_mean` is quantile 0.75 but quantile 0.25 is the best one in `Life_median`.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     16.85
quantile_0.25    15.95
quantile_0.5     16.09
quantile_0.75    14.78
quantile_0.9     16.63
##################################################
Median eval loss over all the test lifes:
quantile_0.1     13.60
quantile_0.25    11.97
quantile_0.5     13.15
quantile_0.75    12.32
quantile_0.9     13.62
##################################################
Std eval loss over all the test lifes:
quantile_0.1     13.07
quantile_0.25    12.83
quantile_0.5     12.20
quantile_0.75    10.37
quantile_0.9     11.91
##################################################
```

###### Prediction Plots

See [[ssm_experiments#Prediction Plots `FD002`|here]] . Also here very high loss values for `Life_135,Life_141`.

## `windowed` Approach + Quantile Regression + Feature Extraction 🌗 🤺

In this section we will describe the experiments performed with the same exact setting as the [[ssm_experiments#`windowed` Approach + Quantile Regression Experiments 🪟 🌗|Quantile Regression experiments]] but we are using Feature Extraction as an additional pre processing step on the input lifes. The Feature Extraction is applied using the `RollingStatistics` transformation from `CeRULeO` which computes the chosen summary statistics on rolling windows of the signal. This new pre processing method increases a lot the number of input features since for each statistic computed a new feature is created. So if we extract 4 features, the number of input features will be quadrupled.

### ### `S4` Model Experiments 4‍⃣ 🪟 🌗 🤺

#### Dataset `FD001`

##### Experiment 1 `S4` `FD001` 4‍⃣ 🪟 🌗  🤺

Let's start with the following configuration:

| Parameter          | Value                        |
| ------------------ | ---------------------------- |
| `model_type`       | `S4`                         |
| `cmapss_model`     | `FD001`                      |
| `val_idx`          | `[0,50]`                     |
| `test_idx`         | `[50,100]`                   |
| `transformer_type` | 2                            |
| `window_size`      | 20                           |
| `features`         | `kurtosis,skewness,mean,rms` |
| `window_size`      | 20                           |
| `scaler`           | `MinMaxScaler(-1,1)`         |
| `epochs`           | 100                          |
| `lr`               | 1e-3                         |
| `batch_size`       | 100                          |
| `weight_decay`     | 1e-4                         |
| `sequence_length`  | 170                          |
| `n_layers`         | 5                            |
| `dropout`          | 0.0                          |
| `activation`       | `relu`                       |
| `final_act`        | `glu`                        |
| `hidden_size`      | 128                          |
| `d_state`          | 64                           |
| `loss`             | `quantile_reg`               |
| `eval_loss`        | `mse`                        |
| `quantile_dist`    | `uniform`                    |
| `bounds`           | `[0.1,0.9]`                  |
| `quantiles`        | `[0.1,0.25,0.5,0.75,0.9]`    |


>[!note]
> [Link to the first `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/nmhd7xxc?nw=nwuserfrizzodavide)

The experiments took about 1 hours and 20 minutes, so essentially the same time taken by the `S4` model without the feature extraction. Comparing the `val_loss` plots of experiments with and without feature extraction on the same run and quantile it seems that with feature extraction that loss saturates at about 80 epochs while without feature extraction after 80 epochs it still decreases a bit in the following epochs.

###### Metrics Table

The results are a bit worse than the ones obtained without feature extraction, however the order of the best quantiles is the same, so the best quantile is 0.75 in both `Life_mean` and `Life_median`. The `Life_std` values are a bit higher than the ones obtained without feature extraction.

```txt
##################################################
Mean eval loss over all the test lifes:
quantile_0.1     28.01
quantile_0.25    27.60
quantile_0.5     24.00
quantile_0.75    21.90
quantile_0.9     23.27
##################################################
Median eval loss over all the test lifes:
quantile_0.1     21.93
quantile_0.25    19.54
quantile_0.5     16.98
quantile_0.75    13.84
quantile_0.9     17.37
##################################################
Std eval loss over all the test lifes:
quantile_0.1     22.04
quantile_0.25    22.00
quantile_0.5     19.12
quantile_0.75    17.06
quantile_0.9     16.96
##################################################
```
