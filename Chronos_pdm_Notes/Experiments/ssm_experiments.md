---
id: chronos_reg_exp
aliases: []
tags:
  - experiments
  - ssm_pdm
---

# `SSM` `PdM` Experiments

In this note I will report the results of the experiments performed with the `SSM` based models (i.e. `S4,S5,S4D`). 

We will start considering the same approach we used in [[chronos_reg_exp|the `chronos` experiments]], which will be called the `padding` approach.

## `padding` approach experiments

In this section the experiments using the `padding` approach will be presented.

>[!info]
> For the `padding` experiments we will use the 🦜 emoji, since the word `parrot` is similar to `padding`.

### `S4` Model Experiments 4️⃣

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

In the windowed approach used for the Elements of Deep Learning exam the `RMSE` was 19.77, so smaller than here but I don't know weather these two kind of evaluations are comparable. In that case the `RMSE` should be the mean over all the test lifes but I have to check trying to reproduce that approach.

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

###### Prediction plots

The `RUL` grid prediction plots do not look bad at all, in some lifes there is some overestimation/underestimation at the beginning but they converge very closely to the true `RUL` towards the end of the life, which is a good sign.

Comparing these plots to the plots we produced in the Deep Learning project here the `RUL` is more oscillating and noisy, in the projec it was almost a perfectly straight line (as the true one) so maybe that's the reason why the `RMSE` is higher here.

##### Experiment 2 `S4` `FD001` `padding` 4️⃣ 1️⃣ 🦜

Let's try to add some dropout, putting `dropout=0.2`.

>[!note]
> [Link to the `wandb` run]()

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
