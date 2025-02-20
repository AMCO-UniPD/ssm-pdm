---
id: chronos_reg_exp
aliases: []
tags:
  - experiments
  - chronos-pdm
---

# `CHRONOS` Regression Experiments

Now that we have set up the [[chronos-data#Padding Regression Approach|padding regression approach experiment code]] we can start running experiments and see how the results are. 

## Experiment 1 ⏰

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

### Metrics table

As written [[chronos-data#`CMAPSS` Data Split|here]] actually the model performance vary a lot across different lifes since the predictions are almost equal across all the sensors and across all the lifes (always start from 201, so in lifes where the inital `RUL` value is higher the model is not doing very well). A confirmation of this can be seen in the metrics table where I report the `RMSE` loss for all the pairs of lifes and sensors.

|  | SensorMeasure2 | SensorMeasure3 | SensorMeasure4 | SensorMeasure7 | SensorMeasure8 | SensorMeasure9 | SensorMeasure11 | SensorMeasure12 | SensorMeasure13 | SensorMeasure14 | SensorMeasure15 | SensorMeasure17 | SensorMeasure20 | SensorMeasure21 | Sensor_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Life_0 | 4.76 | 4.76 | 4.76 | 4.69 | 4.76 | 4.76 | 4.76 | 4.69 | 4.76 | 4.76 | 4.76 | 4.75 | 4.69 | 4.69 | 4.74 |
| Life_1 | 12.62 | 12.62 | 12.62 | 12.67 | 12.62 | 12.62 | 12.62 | 12.68 | 12.62 | 12.62 | 12.62 | 12.62 | 12.67 | 12.67 | 12.64 |
| Life_2 | 40.65 | 40.66 | 40.66 | 40.6 | 40.66 | 40.66 | 40.66 | 40.6 | 40.66 | 40.66 | 40.66 | 40.66 | 40.61 | 40.61 | 40.64 |
| Life_3 | 20.36 | 20.36 | 20.35 | 20.42 | 20.35 | 20.36 | 20.35 | 20.43 | 20.35 | 20.36 | 20.35 | 20.36 | 20.41 | 20.42 | 20.37 |
| Life_4 | 6.67 | 6.67 | 6.67 | 6.65 | 6.68 | 6.68 | 6.67 | 6.65 | 6.68 | 6.68 | 6.67 | 6.67 | 6.65 | 6.65 | 6.67 |
| Life_5 | 3.26 | 3.26 | 3.26 | 3.33 | 3.26 | 3.26 | 3.26 | 3.33 | 3.26 | 3.26 | 3.26 | 3.27 | 3.32 | 3.32 | 3.28 |
| Life_6 | 52.2 | 52.2 | 52.2 | 52.15 | 52.2 | 52.2 | 52.2 | 52.14 | 52.2 | 52.19 | 52.2 | 52.19 | 52.15 | 52.16 | 52.18 |
| Life_7 | 41.34 | 41.35 | 41.34 | 41.42 | 41.34 | 41.35 | 41.34 | 41.42 | 41.34 | 41.35 | 41.34 | 41.35 | 41.4 | 41.41 | 41.36 |
| Life_8 | 5.65 | 5.65 | 5.65 | 5.67 | 5.66 | 5.66 | 5.65 | 5.67 | 5.65 | 5.66 | 5.65 | 5.65 | 5.67 | 5.67 | 5.66 |
| Life_9 | 9.17 | 9.18 | 9.17 | 9.11 | 9.18 | 9.17 | 9.17 | 9.1 | 9.18 | 9.17 | 9.17 | 9.17 | 9.11 | 9.11 | 9.15 |
| Life_10 | 6.96 | 6.96 | 6.96 | 6.88 | 6.96 | 6.96 | 6.96 | 6.88 | 6.96 | 6.95 | 6.96 | 6.94 | 6.9 | 6.89 | 6.94 |
| Life_11 | 4.33 | 4.33 | 4.33 | 4.4 | 4.33 | 4.33 | 4.33 | 4.39 | 4.33 | 4.33 | 4.33 | 4.34 | 4.39 | 4.39 | 4.35 |
| Life_12 | 52.78 | 52.77 | 52.77 | 52.82 | 52.77 | 52.77 | 52.77 | 52.83 | 52.77 | 52.77 | 52.77 | 52.78 | 52.82 | 52.82 | 52.79 |
| Life_13 | 57.97 | 57.97 | 57.96 | 58.03 | 57.96 | 57.96 | 57.96 | 58.03 | 57.96 | 57.96 | 57.97 | 57.97 | 58.03 | 58.02 | 57.98 |
| Life_14 | 56.25 | 56.24 | 56.25 | 56.28 | 56.24 | 56.24 | 56.24 | 56.29 | 56.24 | 56.24 | 56.25 | 56.25 | 56.28 | 56.28 | 56.26 |
| Life_mean | 25.0 | 25.0 | 25.0 | 25.01 | 25.0 | 25.0 | 25.0 | 25.01 | 25.0 | 25.0 | 25.0 | 25.0 | 25.01 | 25.01 | 25.0 |

>[!warning]
> To have a better visualization of the table use the Markdown Preview (`leader+n+p`)

We can see that some lifes have much higher `RMSE` values than others and the errors across the different sensors are very similar.

### Prediction plots

Now I also finish to set up the `plot_predictions_grid` function that produces a subplot with the comparison between the true and predicted `RUL` values over all the test lifes for a specific sensor.

#### Truncated life plot

File `20-02-2025_11-16-26_chronos-rul_FD001_SensorMeasure2_predictions_grid.pdf`.

The first kind of plots I want to analyze are the ones I used also in the `SSM_PDM` project. Since we are using the padding regression approach our prediction are actually longer than the real length of the raw sensor measurements signals. So in these plots I used the `mask` to select the `y_pred` and `y_true` time steps where there was no padding. These are in fact the time steps used to compute the loss that was used to train the model.


These plots are actually very similar to the ones obtained in the `SSM_PDM` project, this confirms the probably non optimal format of the predictions [[chronos-data#`CMAPSS` Data Split|discussed here]]. In fact in some lifes the predictions are pretty good, in other they are much worse.

#### Full life plot

File `20-02-2025_11-05-17_chronos-rul_FD001_SensorMeasure2_predictions_grid.pdf`.

However the model predicts the `RUL` also for the padded time steps and actually also these `RUL` predictions are not so bad, they continue the decreasing trend going towards 0, this can be seen in these plots where the `mask` is not used and all the `sequence_length` time steps are plotted.

In `y_true` after the last `RUL` value everything goes to 0 (because of the padding) while in `y_pred` we have this decreasing trend that at the end start to oscillate a little bit. In fact, even though these predictions make some sort of sense, they are all predictions obtained in correspondance of padded values all equal to 0.

One interesting thing to try may be the following:

- In the test set use the forecasting version of `CHRONOS` to forecast the values of the sensor measurements
- This forecasted values will form a new dataset that will be fed to the regression version of `CHRONOS` to see what kind of `RUL` it predicts
- We do not have any label to evaluate how good the predictions are but we can see the future of the `RUL` values. In fact the test lifes of `CMAPSS` represent a real-world situation where the machine is working and we are monitoring it, so the life it's still going on and we want to predict how that life changes in the future. This is a very interesting experiment to do.

## Experiment 2 ⏰

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
