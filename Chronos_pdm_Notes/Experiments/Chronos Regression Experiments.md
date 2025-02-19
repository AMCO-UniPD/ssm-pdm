---
id: chronos_reg_exp
aliases: []
tags:
  - experiments
  - chronos-pdm
---

# `CHRONOS` Regression Experiments

Now that we have set up the [[chronos-data#Padding Regression Approach|padding regression approach experiment code]] we can start running experiments and see how the results are. 

## Experiment 1

This first experiment was using a small configuration (in the sense that I fine tuned the model for a small number of epochs) because it was used just to see weather the code worked or not. So the configuration is the following:

| Parameter | Value |
|-----------|-------|
| `cmapss_model` | `FD001` |
| `val_idx` | `[0,15]` |
| `test_idx` | `[15,30]` |
| `transformer_type` | 1 (no feature extraction) |
| `window_size` | 20 |
| `scaler` | `MinMaxScaler(-1,1)` |
| `epochs`  | 3    |
| `lr` | 1e-3 |
| `sequence_length` | 500 |
| `model_id` | `amazon/chronos-t5-small` |
| `hidden_size` | 512 |
| `loss` | `mae` |
| `eval_loss` | `mse` |

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/kah9klar?nw=nwuserfrizzodavide)

The loss plots do not tell us a lot because we just did 3 epochs but from a first inspection of the results they seem not bad at all. To obtain these results I used the `best_model_perf` function and I just called the `eval_loop` method on the best model found from this experiment.

The `y_pred` and  `y_true` tensors that are returned by the method have shape `(num_lifes,num_features,sequence_length)`. In this particular case `num_features=num_sensors` because I did not apply any feature extraction transformation and so I just used the raw sensor measurements. However let's consider the predictions for the first 10 time steps of the first life on the first sensor:


```python
y_true[0,0,:10]
array([196., 195., 194., 193., 192., 191., 190., 189., 188., 187.],
      dtype=float32)

y_pred[0,0,:10]
array([201.55055, 200.37863, 199.34567, 198.32872, 197.51808, 196.17805,
       195.13364, 194.4966 , 193.5073 , 192.00241], dtype=float32)
```

As we can see they are preetty good, the model has learned the decreasing trend of the `RUL` and the values are not so different from the true ones. The only potentially negative thing is that there is an small overestimation of the `RUL` values but maybe we can obtain better results using more epochs.

>[!warning]
> Differently from the `AD_MG` project now we want to **underestimate** the `RUL` to avoid unexpected breaks in the system. In fact here the smaller the `RUL` the closer we are to failure. Inversely, in the `AD_MG` project the higher the damage the closer to failure.

## Experiment 2

In [[chronos_reg_exp#experiment-1|experiment 1]] the model had the pre trained weights of `chronos` in its hidden layers (only the model head was randomly initialized since I changed it) and the predictions were not bad at all. Now I want to see what happens if I use a model with randomly initialized weights with the same configuration. If the results are worse we can confirm the utility of using the pre trained weights → it will mean that the knowledge `chronos` gained when pre trained for the time series forecasting task was transferred to the `RUL` estimation taks.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/lts3ilb9?nw=nwuserfrizzodavide)

Looking at the predictions, at the loss plots and at the loss values we can confirm that this is clearly worse than the model exploiting the pre trained weights of `chronos` 💪:


```python
y_true[0,0,:10]
array([196., 195., 194., 193., 192., 191., 190., 189., 188., 187.],
      dtype=float32)

y_pred[0,0,:10]
array([98.33607 , 98.3955  , 97.90683 , 98.1296  , 98.50427 , 97.86396 ,
       98.47129 , 98.586784, 98.05804 , 98.070755], dtype=float32)
```

Here the model predicts more or less a constant value, so it has not learned yet the decreasing trend of the `RUL`. However looking at the plots the loss was going down so maybe with more epochs it could have learned the trend. We should do another comparison with more epochs to see if this is true.
