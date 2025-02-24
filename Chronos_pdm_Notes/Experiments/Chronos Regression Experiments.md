---
id: chronos_reg_exp
aliases: []
tags:
  - experiments
  - chronos-pdm
---

# `CHRONOS` Regression Experiments

Now that we have set up the [[chronos-data#Padding Regression Approach|padding regression approach experiment code]] we can start running experiments and see how the results are. 

## Pretrained model experiments 🧠 - `chronos-t5-small`

These experiments start from `CHRONOS` with the pre trained weights of `chronos-t5-small` from the time series forecasting task.

There are different version of pre trained weights for `chronos`, each one using a different version of `T5` as the backbone `LLM` model. The only difference is in the size of the model:

- `chronos-t5-small` → `40M` parameters
- `chronos-t5-base` → `200M` parameters
- `chronos-t5-large` → `710M` parameters

>[!info]
> We will represent these experiments with the 🧠 emoji because the model has some knowledge in its pre training weights so it has something in its 🧠. We will use 🔹emoji to indicate the fact that we are using the `chronos-t5-small` model.

### Experiment 1 ⏰ 🧠 🔹

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

As we can see they are pretty good, the model has learned the decreasing trend of the `RUL` and the values are not so different from the true ones. The only potentially negative thing is that there is an small overestimation of the `RUL` values but maybe we can obtain better results using more epochs.

>[!warning]
> Differently from the `AD_MG` project now we want to **underestimate** the `RUL` to avoid unexpected breaks in the system. In fact here the smaller the `RUL` the closer we are to failure. Inversely, in the `AD_MG` project the higher the damage the closer to failure.

#### Metrics table

As written [[chronos-data#`CMAPSS` Data Split|here]] actually the model performance vary a lot across different lifes since the predictions are almost equal across all the sensors and across all the lifes (always start from 201, so in lifes where the inital `RUL` value is higher the model is not doing very well). A confirmation of this can be seen in the metrics table where I report the `RMSE` loss for all the pairs of lifes and sensors.

>[!note]
> I decided to remove the full visualization of the `metrics_df` `pd.DataFrame` since it is quite unintuitive and moreover the performances are very similar across the different columns (i.e. across the different sensors). So from now on I will report the `metrics_df` in the first 15 test lifes (which are the same test lifes normally represented in the prediction plots) and only for `SensorMeasure2`.




>[!warning]
> To have a better visualization of the table use the Markdown Preview (`leader+n+p`)

We can see that some lifes have much higher `RMSE` values than others and the errors across the different sensors are very similar.

#### Prediction plots

Now I also finish to set up the `plot_predictions_grid` function that produces a subplot with the comparison between the true and predicted `RUL` values over all the test lifes for a specific sensor.

##### Truncated life plot

File `20-02-2025_11-16-26_chronos-rul_FD001_SensorMeasure2_predictions_grid.pdf`.

The first kind of plots I want to analyze are the ones I used also in the `SSM_PDM` project. Since we are using the padding regression approach our prediction are actually longer than the real length of the raw sensor measurements signals. So in these plots I used the `mask` to select the `y_pred` and `y_true` time steps where there was no padding. These are in fact the time steps used to compute the loss that was used to train the model.


These plots are actually very similar to the ones obtained in the `SSM_PDM` project, this confirms the probably non optimal format of the predictions [[chronos-data#`CMAPSS` Data Split|discussed here]]. In fact in some lifes the predictions are pretty good, in other they are much worse.

##### Full life plot

File `20-02-2025_11-05-17_chronos-rul_FD001_SensorMeasure2_predictions_grid.pdf`.

However the model predicts the `RUL` also for the padded time steps and actually also these `RUL` predictions are not so bad, they continue the decreasing trend going towards 0, this can be seen in these plots where the `mask` is not used and all the `sequence_length` time steps are plotted.

In `y_true` after the last `RUL` value everything goes to 0 (because of the padding) while in `y_pred` we have this decreasing trend that at the end start to oscillate a little bit. In fact, even though these predictions make some sort of sense, they are all predictions obtained in correspondance of padded values all equal to 0.

One interesting thing to try may be the following:

- In the test set use the forecasting version of `CHRONOS` to forecast the values of the sensor measurements
- This forecasted values will form a new dataset that will be fed to the regression version of `CHRONOS` to see what kind of `RUL` it predicts
- We do not have any label to evaluate how good the predictions are but we can see the future of the `RUL` values. In fact the test lifes of `CMAPSS` represent a real-world situation where the machine is working and we are monitoring it, so the life it's still going on and we want to predict how that life changes in the future. This is a very interesting experiment to do.


### Experiment 2 ⏰ 🧠 🔹

Let's try now to launch an experiment with more epochs and more test and validation lifes. In particular the new configuration will be:


| Parameter | Value |
|-----------|-------|
| `cmapss_model` | `FD001` |
| `val_idx` | `[0,50]` |
| `test_idx` | `[50,100]` |
| `transformer_type` | 1 (no feature extraction) |
| `window_size` | 20 |
| `scaler` | `MinMaxScaler(-1,1)` |
| `epochs`  | 10    |
| `lr` | 1e-3 |
| `sequence_length` | 500 |
| `model_id` | `amazon/chronos-t5-small` |
| `hidden_size` | 512 |
| `loss` | `mae` |
| `eval_loss` | `mse` |

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo- [ ]davide-Univeristy%20of%20Padova/chronos-rul/runs/jolend3p?nw=nwuserfrizzodavide)

Looking at the `wandb` plots unfortunately the results are not very reassuring. In fact the `test_loss` goes down in the first 2 epochs (like in experiment 1) but then settles at a sligthly higher value for the rest of the epochs, as if the model stopped the learning process. Maybe this happens because of the maybe too simple regression head I am using without non linearities.

Looking at the predictions we can see that now they all start from 200 instead of 201 and as usual they are more or less the same across all the sensors and all the lifes.

Moreover I want to see what happens with the randomly initialized model here becase looking at the loss plots of [[chronos_reg_exp#randomly-initialized-model-experiments|experiment 1 of the 👶 experiments]] the loss was going down pretty fast.

#### Metrics table

This experiment is the one with the best performances (on this set of lifes) which is a bit strange because this is the simplest model with a single `fc` layer in the regression head and no activation function. Moreover the predictions are always in the same range of values independently on the life. So probably if the model is tested on a test life with a very different range of `RUL` values (maybe a life from another `cmapss_model` or from another dataset) it will not do a good job. In the following experiments (which use multiple `fc` layers interleaved with activation functions) instead the predictions change between different lifes but they pretty bad on multiple lifes.

|  | SensorMeasure2 |
| --- | --- |
| Life_50 | 56.52 |
| Life_51 | 14.83 |
| Life_52 | 12.66 |
| Life_53 | 16.6 |
| Life_54 | 48.57 |
| Life_55 | 50.34 |
| Life_56 | 60.96 |
| Life_57 | 10.65 |
| Life_58 | 6.49 |
| Life_59 | 45.46 |
| Life_60 | 22.13 |
| Life_61 | 76.62 |
| Life_62 | 25.28 |
| Life_63 | 7.41 |
| Life_64 | 2.64 |
| Life_mean | 30.48 |

#### Prediction plots

The plots are very similar to the ones produced in experiment 1, in some of them the predictions are good (because certain life have `RUL` ranges close to the one predicted by the model) in others they are bad. Now we are looking at different lifes from the ones in experiment 1 because we changes `test_idx` in the experiment configuration. In particular we are looking at the test lifes from 51 to 65. In life 62 there is this peculiar behavior in the predictions were at the end the `RUL` increases a little bit to then decrease again.

### Experiment 3 ⏰ 🧠 🔹

After the error in [[chronos_reg_exp#experiment-3--|experiment 3]] let's use a similar configuration but with the 🧠 models. So here we will use 3 `fc` layers, with  `ReLU` activations in the middle, in the regression head of the model for 20 epochs with `lr=1e-4`.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/iesdc6h9?nw=nwuserfrizzodavide)

Ok here, since we are using the 🧠 weights, it starts to overfit after just 1 epoch. In fact it reaches much faster the good point (at least this time it is better in terms of `val_loss` than the previous `chronos-random-small` experiment) and then starts to overfit.

>[!important]
> After analyzing better the results I realized that now with the multiple `fc` layers approach at least the predictions vary from life to life. So they start from different starting values and then have a decreasing trend, this holds for this experiment and the following ones and can be easily visualized in the prediction plots. Then in some lifes the model completely misses the starting point of the `RUL` prediction but at least now the results make more sense.

#### Metrics table

Here the metrics table is not bad at all compared to the previous ones.
|  | SensorMeasure2 |
| --- | --- |
| Life_50 | 102.28 |
| Life_51 | 17.96 |
| Life_52 | 18.95 |
| Life_53 | 84.58 |
| Life_54 | 121.21 |
| Life_55 | 19.12 |
| Life_56 | 92.34 |
| Life_57 | 25.53 |
| Life_58 | 90.4 |
| Life_59 | 90.41 |
| Life_60 | 18.57 |
| Life_61 | 40.62 |
| Life_62 | 61.72 |
| Life_63 | 20.88 |
| Life_64 | 93.94 |
| Life_mean | 59.9 |


### Experiment 4 ⏰ 🧠 🔹

In order to reduce overfitting I can add some dropout layers in the new regression head (in fact I am pretty sure dropout is already applied inside the `CHRNOS` model).

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/0q4tynif?nw=nwuserfrizzodavide)

I started an experiment with `dropout_rate=0.2` and it started overfitting immediately 😢. It does not make a lot of sense considering that dropout should help to reduce overfitting, now I started a new experiment with `dropout_rate=0.3`, let's see what happens. Nope it seems that is overfitting also in this case, maybe it's a problem with the change in the `lr`.

#### Metrics table

Here the metrics are similar to the ones of the previous experiment.

| Life  | SensorMeasure2 |
| --- | --- |
| Life_50 | 107.34 |
| Life_51 | 26.4 |
| Life_52 | 19.56 |
| Life_53 | 81.88 |
| Life_54 | 118.49 |
| Life_55 | 12.37 |
| Life_56 | 93.85 |
| Life_57 | 34.34 |
| Life_58 | 81.47 |
| Life_59 | 94.05 |
| Life_60 | 18.73 |
| Life_61 | 45.77 |
| Life_62 | 62.42 |
| Life_63 | 21.99 |
| Life_64 | 78.75 |
| Life_mean | 59.83 |

### Experiment 5 ⏰ 🧠 🔹

Let's try to use the same configuration as above but removing dropout (i.e. `dropout_rate=0.0`) and coming back to `lr=1e-3`.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/cr1uj0xo?nw=nwuserfrizzodavide)

Surely better than before, the `train_loss` is going down a bit slower than in the other experiments and the `val_loss` and `test_loss` are oscillating but at least they are not just straight increasing as in the previous cases. Maybe I can try to incrase the number of epochs.

#### Metrics table

This one is very bad with the respect to the others:


|  | SensorMeasure2 |
| --- | --- |
| Life_50 | 123.27 |
| Life_51 | 35.01 |
| Life_52 | 44.36 |
| Life_53 | 102.59 |
| Life_54 | 137.72 |
| Life_55 | 32.36 |
| Life_56 | 119.57 |
| Life_57 | 51.62 |
| Life_58 | 107.67 |
| Life_59 | 112.88 |
| Life_60 | 39.8 |
| Life_61 | 62.25 |
| Life_62 | 85.39 |
| Life_63 | 40.78 |
| Life_64 | 110.22 |
| Life_mean | 80.37 |

### Experiment 6 ⏰ 🧠 🔹

Let's try to increase the number of epochs to 30.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/6cup6b2d?nw=nwuserfrizzodavide)

Not much different then before.

#### Metrics table

Still not very good.

|  | SensorMeasure2 |
| --- | --- |
| Life_50 | 107.78 |
| Life_51 | 30.06 |
| Life_52 | 30.76 |
| Life_53 | 88.03 |
| Life_54 | 123.91 |
| Life_55 | 19.64 |
| Life_56 | 101.45 |
| Life_57 | 40.95 |
| Life_58 | 95.41 |
| Life_59 | 93.81 |
| Life_60 | 27.16 |
| Life_61 | 54.02 |
| Life_62 | 70.25 |
| Life_63 | 28.38 |
| Life_64 | 100.69 |
| Life_mean | 67.49 |

### Experiment 7 ⏰ 🧠 🔹

Let's try to see what happens adding dropout with `lr=1e-3`, let's use `dropout_rate=0.2`.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/91xre3oj?nw=nwuserfrizzodavide)

Seems to behave like the others, but in the `val_loss` it reached an all time minimum value with something around 29.

#### Metrics table

Still not very good.

|  | SensorMeasure2 |
| --- | --- |
| Life_50 | 115.94 |
| Life_51 | 32.82 |
| Life_52 | 31.02 |
| Life_53 | 99.99 |
| Life_54 | 137.82 |
| Life_55 | 26.32 |
| Life_56 | 104.22 |
| Life_57 | 40.16 |
| Life_58 | 117.06 |
| Life_59 | 100.48 |
| Life_60 | 28.84 |
| Life_61 | 59.62 |
| Life_62 | 74.27 |
| Life_63 | 34.27 |
| Life_64 | 133.26 |
| Life_mean | 75.74 |

## Pretrained model experiments 🧠 ⚾ - `chronos-t5-base`

In this section I will report the details of the experiments using the `chronos-t5-base` model checkpoint which is a bit larger than `chronos-t5-small` and so hopefully it may produce better results.

>[!info]
> Here we will use the ⚾ emoji to indicate the fact that we are using the `chronos-t5-base` model.

>[!warning]
> In these experiments the models will be much bigger so I have to be careful at `CUDA OutOfMemory` errors, maybe I need to reduce the number of lifes used for the experiments. However in `acquario3` with `ollama` we are running `70B` parameter models so probably this won't be a huge problem.

### Experiment 1 ⏰ 🧠 ⚾

After all the experiments done with `chronos-t5-small` it's time to use some bigger models. Here we will use `chronos-t5-base` which has `220M` parameters with the respect to the `40M` of `chronos-t5-small`. We will start with the following configuration:

| Parameter | Value |
|-----------|-------|
| `cmapss_model` | `FD001` |
| `val_idx` | `[0,50]` |
| `test_idx` | `[50,100]` |
| `transformer_type` | 1 (no feature extraction) |
| `window_size` | 20 |
| `scaler` | `MinMaxScaler(-1,1)` |
| `epochs`  | 10    |
| `lr` | 1e-3 |
| `sequence_length` | 500 |
| `model_id` | `amazon/chronos-t5-base` |
| `hidden_size` | 512 |
| `n_fc_layers` | 3 |
| `act` | `relu` |
| `loss` | `mae` |
| `eval_loss` | `mse` |

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/6uassobn?nw=nwuserfrizzodavide)

The `val_loss` is going down pretty well, I hope it is not alreeady saturated after a few epochs, we'll see since obviously it is going slower and moreover I am using a lot of the `GPU` memory (maybe it's also because there are also other processes running in there). In any case I have to be careful in particular with `chronos-t5-large`.


#### Metrics table

As expected after having looked at the plots here the metrics are much better than the ones of the previous experiments with multiple `fc` layers. They are sligthly worse than the ones of the bugged [[chronos_reg_exp#experiment-2---|experiment 2]].

|  | SensorMeasure2 |
| --- | --- |
| Life_50 | 71.76 |
| Life_51 | 20.24 |
| Life_52 | 7.94 |
| Life_53 | 33.02 |
| Life_54 | 67.09 |
| Life_55 | 19.85 |
| Life_56 | 78.12 |
| Life_57 | 21.81 |
| Life_58 | 25.81 |
| Life_59 | 62.76 |
| Life_60 | 11.9 |
| Life_61 | 77.92 |
| Life_62 | 41.5 |
| Life_63 | 10.8 |
| Life_64 | 20.99 |
| Life_mean | 38.1 |

#### Prediction plots

Even though the loss saturate after some epochs from a quick look at the plot it seems better than the ones with `chronos-t5-small`. For example in the previous experiments on life 64 and 65 the predictions were quite out of range but now the model predictions are much closer to the true ones.

### Experiment 2 ⏰ 🧠 ⚾

Seing the good results obtained with a single `fc` layer in [[chronos_reg_exp#experiment-2---|experiment 2]] let's try to use a single `fc` layer also with `chronos-t5-base` and let's see what happens. In particular I want to see weather also in this case the model predicts always the same range of `RUL` values independently on the lifes.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/mkbro1av?nw=nwuserfrizzodavide)

From a first look to the `wandb` plots the `val_loss` and `test_loss` are almost flat so probably we have the model that predicts the otputs always in the same range?

#### Metrics table

The metrics values are a bit better than experiment 2 (and that is because we are using a pre trained model with more parameters) but the outputs are still in the same range of values, they all start from 202 and then start their decreasing trend.

|  | SensorMeasure2 |
| --- | --- |
| Life_50 | 55.02 |
| Life_51 | 13.48 |
| Life_52 | 14.16 |
| Life_53 | 15.09 |
| Life_54 | 47.04 |
| Life_55 | 51.87 |
| Life_56 | 59.47 |
| Life_57 | 9.38 |
| Life_58 | 4.91 |
| Life_59 | 43.97 |
| Life_60 | 23.61 |
| Life_61 | 75.16 |
| Life_62 | 23.81 |
| Life_63 | 8.86 |
| Life_64 | 4.21 |
| Life_mean | 30.0 |

## Pretrained model experiments 🧠 🏈 - `chronos-t5-large`

In this section I will report the details of the experiments using the `chronos-t5-large` model checkpoint which is the largest version of the `chronos` model and so hopefully it may produce better results.

>[!info]
> Here we will use the 🏈 emoji to indicate the fact that we are using the `chronos-t5-large` model.

>[!warning]
> In these experiments the models will be much bigger so I have to be careful at `CUDA OutOfMemory` errors, maybe I need to reduce the number of lifes used for the experiments.

### Experiment 1 ⏰ 🧠 🏈

Let's try an initial test experiment (in particular to see wheater all the data and the model fits on the `GPU` since it is quite big) with [[chronos_reg_exp#chronos-regression-experiments#pretrained-model-experiments-----chronos-t5-base|the same configuration of the `chronos-t5-base` experiments]].

>[!note]
> [Link to the `wandb` run]()

>[!error] `Out Of Memory` error
> I tried changing the number of test lifes, from 100, to 30 to 10 but nothing worked. I will have to try to run this script when `acquario3` is less busy.

## Randomly initialized model experiments 👶

In these experiments the model has randomly initialized weights. These are the weights that `CHRONOS` started from before the pre training stage.

>[!info]
> We will represent these experiments with the 👶 emoji because the model has no prior knowledge in its weights so it is like it is a 👶

### Experiment 1 ⏰ 👶

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

### Experiment 2 ⏰ 👶

Let's use the same configuration of [[chronos_reg_exp#experiment-2--|experiment 2 🧠]] but with the 👶 model. Let's see weather it is still worse or not.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/ya9ypvt3?nw=nwuserfrizzodavide)

As I expected also the 👶 model after some epochs is able to reach more or less the same performances of the 🧠 model. The loss goes down with the same trend and I bet the the predictions will be more or less on the same range. As I expected this is what happens: the different is that the `RUL` predictions starts from 196. It's like the model learns a starting point for the `RUL` and than from that point goes down with a more or less linearly decreasing trend. Hopefully this thing changes using a more complex regression head.

>[!important] Pretrained `CHRONOS` is not helping
> This result shows how the pretrained weights of `CHRONOS` are just helping in converging faster to the best solution but they are not giving a significant advantage in terms of performances. Let's see what happens using a different regression head but from this point onwards potentially the contribute of `CHRONOS` could be the one of [[chronos-data#Experient Idea: Forecasting + Regression `CHRONOS`|this experiment idea]] or we can simply use it to compare with the `SSM` models results combining the `SSM_PDM` and `chronos_pdm` projects.

#### Prediction plots

The plots are, as expected, very similar to the ones observed in [[chronos_reg_exp### Experiment 2 ⏰ 🧠|experiment 2 🧠]].

### Experiment 3 ⏰ 👶

>[!error]
> I thought I was running this experiment with the pre trained model 🧠 but I actually was using still the 👶 model becuase I forgot to change the `random_init` parameter from the model config `yaml` file. Now I will insert the `random_init` parameter also inside the `exp_config.yaml` file so that I can change it more easily.

Let's try now to make the regression head of the model a little bit more complex using multiple `nn.Linear` layers with `ReLU` activations in the middle. In particular we will use 3 `fc` layers with `ReLU` activations in the middle.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/96mizhf0?nw=nwuserfrizzodavide)

Unfortunately the addition of the `fc` layers in the regression head did not bring better results, the trend in the `wandb` loss plots is the usual one and here the model even saturates at a sligthly higher loss value than in the previous experiments. So I expect similar results in the predictions, `metrics_df` and plots.

#### Prediction plots

The plots are similar to the previous ones but we have this slight increase in the `RUL` more ofter towards the end of the life, in particular we can see this effect in life 58,61,62 and 64.

### Experiment 4 ⏰ 👶

>[!error]
> As in [[chronos_reg_exp#experiment-3|experiment 👶]] I thought I was using the pre trained model weights 🧠 but I was actually using the randomly initialized ones.

Let's try one last thing before passing to the bigger models. Let's try to use 20 epochs and decreasing the learning rate to `1e-4`, maybe it helps.

>[!note]
> [Link to the `wandb` run](https://wandb.ai/frizzo-davide-Univeristy%20of%20Padova/chronos-rul/runs/9ixyyo29?nw=nwuserfrizzodavide)

Looking at `wandb` while the experiment it's still going I noticed that it gets to more or less the same `val_loss` values of the other experiments after about 5-6 epochs but then it starts to overfit 😢. This is potentially not a great news if we then want to use more complex models like `chronos-t5-base` and `chronos-t5-large` which have a much highe number of parameters but it may also be due to the smaller learning rate.

On the bright side the `train_loss` got to much smaller values than the other experiments after about 5 epochs → I don't know if that is due to the `lr` or to the fact that the model is overfitting.

## Zero Shot experiments ⏰ 🚀

Here I will try some zero shot experiments, so I will pass the pretraind `chronos` model directly to `eval_loop` to evaluate its performances on the test set without any fine tuning. Probably there will be not so many experiments in this section because, considering that the regression head is randomly initialized, I am pretty sure that the results will be bad.

As expected the results are very bad and the predictions do not make any sense because of the randomly initialized head.
