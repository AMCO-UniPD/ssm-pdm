---
id: j3c_paper
aliases: []
tags:
  - paper
---
# Paper for `J3C` Conference

In this note  I want to write down the things to do to produce the paper for the `J3C` Conference. In particular I have to get back to the `SSM_PDM` project (since it's a long time I am not touching it) and see what results we can keep for the paper and what additional things we can do.

## Things to keep

Looking at the presentation me and Francesco did for the `Elements of Deep Learning` exam there are some things that can be kept:

- Results of `Windowed Anti-Causal` approach for mode `FD001,FD002` of `CMAPSS`
    - [x] Write down the table (Slide 41 for `FD001`, slide 43 for `FD002`) in `latex` format and insert it into the Overleaf project
- Prediction plots for `FD001,FD002`
    - [x] Find out where the plots are contained (probably inside `acquario3`) and insert some of them in the Overleaf project
- Table with analysis of parameter size and number of operations of each model
    - [x] Write the table of Slide 52 in `latex` format and add it to the Overleaf project
- Blob plot representing the results of the table
    - [x] Find out where the blob plot is contained and insert it in the Overleaf project. Here I have to decide weather to include the plot or the table, since they represent the same thing → maybe the plot is more intuitive.

### Images to add

All the images to add should be contained in the folder `PredMainAnomaDet/notebooks/Presentation_Img` and are:

- `CMAPSS.png` → Usual image of the `CMAPSS` dataset explaining how it is divided into 4 subsets
- `params_plot.png` → Blob plot with number of parameters and amount of operations for each model.
- `Prediction_plot_{model_name}.png` → Here we have one of these plots per `model_name` (i.e. `S4,S4D,S5,Transformer,Informer`). I can move all of them inside Overleaf but then maybe we will insert just one of them in the paper.
- `RUL_Plot_S5_FD002.png` → This is the prediction plot of `S5` for `FD002` dataset
- `RUL_Plots_S4D_PDM_Loss.png` → Prediction plot for `S4D` with `PDM_Loss` for `FD001` dataset
- `RUL_predictions_S4.png` → Prediction plot for `S4` for `FD001` dataset

## Things to remove

- `Mamba` model → The implementation we found was not the best and moreover `Mamba` 🐍 is not thought to work well with time series, and that is observable from the results.
- `Time-wise Causal` Approach? This was done by Francesco, I need to find where the code is in order to reproduce it and use it to do the experiments.
- Even if we are inside the `chronos-pdm` project folder probably it does not make sense to include it this paper since we already have the results of the `Transformer` and `Informer` architecture which are pretty similar to `chronos` and all the  `LLM` thing in `chronos` is not actually used at the end.

## Things to add

- Perform experiments on `FD003,FD004` of `CMAPSS` dataset
    - [ ] Here I need to do the Code Refactoring and I also need to be able to reproduce the experiments on `FD001` and `FD002` so that I am sure that they are done with the same settings.
- Try to use the Pinball Loss (instead of the `PDM_Loss` I wrote in the presentation) to try to deal with the underestimation/overestimation trade-off.
    - [x] Maybe if this works fine we can do all the experiments with it instead of reporting the results obtained up to now (which did not use this kind of loss).

# Results Analysis

Running the experiments that will be presented in the paper I come to some conclusions and I will report them here in this section so that I can then re elaborate them in a more formal way in the paper.

## `FD001` Experiments `padding` Approach 🦜

After the first run of experiments on all the 8 models we can already take some conclusions:

- `SSM` based models → These are the best models in terms of metrics values across the test lifes. The `RUL` predictions are not smooth linearly decreasing lines as the true `RUL`, they are sligthly oscillating and sometimes go up and down but in multiple lifes they are almost overlapped to the true `RUL` signal. In particular the predicted values are closer to the true ones near the end of the life which is the point were we want to be more precise. Looking at the trend of the loss plots on `wandb` there is a bit of overfitting (after some epochs needed to reach the minimum `val_loss`) the loss starts to slightly increase and oscillate. Maybe I can try to reduce `d_model` ( in fact here I am using 512 while on the Deep Learning exam I was using just 90). In the next set of experiments I will try to add some dropout. Moreover the `S5` model has a very small number of parameters compared to the others (this is probably due to the fact that it uses a single `SSM` block for all the signals while `S4,S4D` use a different block for each signal) and in fact it is the one with the worst performances. Maybe increase a little bit the number of parameters (i.e. increase `d_model` or `n_layers`) may lead to better performances.
- `RNN` based models  → These models are right behind the `SSM` models. In training they almost immediately reach the minimum `val_loss` value and then the loss stays almost constant in the remaining epochs. The predictions are smooth decreasing lines (parallel to the true oens) but for every life the predictions start from the same initial `RUL` value and then goes down linearly. It is as if the model has learned a common initial `RUL` value and then simply decreases it linearly. This is probably due to the fact that the `RNN` models are not able to capture the temporal dependencies in the data as the `SSM` models do. In the next set of experiments I will try to reduce the `hidden_size` parameter to see if the model is able to learn better the temporal dependencies.
- `Transformer` based models → `Transformer` is clearly the worst model of all. The predictions are similar to the ones of the `RNN` models in their shape of smooth decreasing lines but the values are very far from the true ones. Here I think that the main problem is that the model is overfitting a lot, so I will try to reduce its parameters, for example reducing `d_ff` (the value of 2048 used up to now it's pretty high). On the other hand the `Informer` model behaves similarly to the `RNN` based models but with a higher error. Also here I think that trying to reduce `d_model` or `d_ff` may help.

## `FD001` Experiments `windowed` Approach 🪟

After having performed the experiments using the `windowed` approach I am convinced that this is the best approach to use in terms of model performances and probably the one I will report in the paper. Here I will report some conclusions:

- `SSM` based models → I was able to obtain similar results to the ones obtained in the Deep Learning exam projet adding the `gap` layer to the Regression Head. Now we are able to obtain smooth decreasing lines in the predicted `RUL` signal. The best model of all is `S4` because its metrics are quite stable across the different test set lifes that we use to evaluate the model performances. On the other hand `S4D,S5` are less stable in their metrics: there are some lifes in which the models are almost perfect in predicting the `RUL`, others in which they miss completely.
- `RNN` based models → Here the situation is peculiar: for some reason the results of all these three models are almost exactly equal. Maybe it's better to insert just one of them in the paper. In any case also for this approach we have the problem that the models always predict the same range of `RUL` values across all the lifes.
- `Transformer` based models → The `Transformer` is not bad at all, we can place it right below the `SSM` models in terms of performance, the `Informer` instead has similar performances to the `Transformer` but for some reason it is much slower.

## `FD002` Experiments `windowed` Approach 🪟

I have now also performed experiments on `FD002` for the `windowed` approach. Here the results are in general slightly worse than in the `FD001` dataset because the `RUL` prediction task is harder considering the fact that we have multiple operating conditions of the engine in the different run to failure lifes contained in the dataset. Here are some conclusions:

- `SSM` based models → These models are still the top ones in terms of performances. As usual `S4` is the best one because it's more stable in its metrics across the different test lifes. `S4D,S5` are less stable, having lifes where the model is extremely good (better than `S4`) and others in which it is extremely bad.
- `RNN` and `Transformers` based models → Here it's were things start to become strange. For some reason the metrics table of all these models, even though I am pretty sure they are different models, are **exactly equal**. Obviously the result is worse than `SSM` models but the fact that the metrics are the same is strange. I will have to investigate this further.

# Result Recap

In this section I want to recap the numerical results obtained in the different experiments. I will use markdown tables that I will then convert into `latex` tables to insert them in the Overleaf project.

For the moment I evaluated the models on the test lifes just using the `RMSE` evaluation metric. Maybe in the future I may think at some other metrics to add to the evaluation.

The `RMSE` value inserted in the following tables is the mean `RMSE` over all the test lifes.

## `FD001` Experiments `padding` Approach 🦜


| Model | RMSE |
|-------|------|
| `RNN` | 31.58 |
| `LSTM` | 31.3 |
| `GRU` | 31.66 |
| `Transformer` | 60.22 |
| `Informer` | 31.31 |
| `S4` | **28.58** |
| `S4D` | 29.24 |
| `S5` | 30.19 |

## `FD001` Experiments `windowed` Approach 🪟

>[!note]
> Interestingly in these experiments `S5,S4D` performed much better without the `gap` layer in the Regression Head, differently from what happened with `S4`. For these two models I also report in `()` the `RMSE` value obtained with the `gap` layer.

| Model | Mean RMSE | Std RMSE |
|-------|------| ---------|
| `RNN` | 28.56 | 23.16 |
| `LSTM` | 28.56| 23.16 |
| `GRU` | 28.56 | 23.16 |
| `Transformer` | 25.7 | 20.66 |
| `Informer` | 33.31 | 26.97 |
| `S4` | **23.47**  | 20.99 |
| `S4D` | 29.22 (41.91) | 39.71 |
| `S5` | 26.41 (50.03)| 51.56 |


## `FD002` Experiments `windowed` Approach 🪟

>[!warning]
> In these experiments we have the strange behavior of the `RNN` and `Transformers` models that have exactly the same metrics. Hopefully I will be able to find what is going on and fix it.

| Model | Mean RMSE | Std RMSE |
|-------|------| ---------|
| `RNN` | 35.08 | 24.03 |
| `LSTM` | 35.08 | 24.03 |
| `GRU` | 35.08 | 24.03 |
| `Transformer` | 35.08 | 24.03 |
| `Informer` | 35.08 | 24.03 |
| `S4` | 31.51 | 30.79 |
| `S4D` | 58.02 | 53.27 |
| `S5` | 40.52 | 33.42 |

## Quantile Regression Experiments 🌗

This is the best approach up to now and it will be probably the one that will be presented and inserted in the paper, so that we have also the contribution of the quantile regression. In these experiments I also decided to use only the `LSTM` model among the `RNN` family since I am pretty sure that the `RNN` and `GRU` models will produce similar (if not equal) results.

### Model Parameters

Here I report a table (to be inserted in the paper) containing the model parameters for each model:

| Model | Parameters | Mult-Adds |
|-------|------------|-----------|
| `S4` | 355498 | 190378 |
| `S5` | 57808 | ??? |
| `S4D` | 273578 | 28094378 |
| `LSTM` | 624554 | 102468010 |
| `Transformer` | 621610 | 519210 |
| `Informer` | 724138 | 62937770 |

>[!note]
> For `S5` at the moment we have only the number of parameters because for some reason due to the implementation of the `S5` model (which is implemented in the `s5-pytorch` library) the `summary` method from `torchinfo` does not work.

A quick comment that we can insert in the paper is the following:

### `FD001` Experiments `windowed` Approach + Quantile Regression 🪟 🌗

>[!note]
> Here the results are obtained averaging the `RMSE` over different runs of the same model with different quantiles.

| Model         | quantile 0.25 | quantile 0.5 | quantile 0.75 |
| ------------- | ------------- | ------------ | ------------- |
| `S4`          | 18.85         | 17.03        | 16.26         |
| `S5`          | 30.78         | 30.00        | 29.01         |
| `S4D`         | 40.19         | 39.95        | 38.53         |
| `LSTM`        | 28.60         | 28.63        | 28.63         |
| `Transformer` | 28.14         | 30.61        | 36.10         |
| `Informer`    | 29.94         | 25.86        | 27.16         |

An important conclusion/comment after having done the Experiments 3 and 4 on `S4` with different `bounds` in Quantile Regression → we can say that these are ablation study experiments.

#### Pinball Loss Evaluation

Below we report the table of results obtained using the Pinball Loss as the evaluation metric.

>[!warning]
> In these results there is a bit of disparity since the `SSM` models are evaluated on 15 runs while the others are evaluated only on 5 runs.  

| Model         | quantile 0.1 | quantile 0.25 | quantile 0.5 | quantile 0.75 | quantile 0.9 |
| ------------- | ------------ | ------------- | ------------ | ------------- | ------------ |
| `S4`          | 6.93         | 6.51          | 7.08         | 7.07          | 7.52         |
| `S5`          | 11.90        | 11.09         | 12.34        | 12.88         | 13.12        |
| `S4D`         | 13.53        | 13.40         | 13.57        | 14.07         | 13.80        |
| `LSTM`        | 12.49        | 12.49         | 12.43        | 12.43         | 12.50        |
| `Transformer` | 14.28        | 14.08         | 17.60        | 22.46         | 23.64        |
| `Informer`    | 12.69        | 13.68         | 10.81        | 12.54         | 16.60        |

### `FD002` Experiments `windowed` Approach + Quantile Regression 🪟 🌗


| Model         | quantile 0.1 | quantile 0.25 | quantile 0.5 | quantile 0.75 | quantile 0.9 |
| ------------- | ------------ | ------------- | ------------ | ------------- | ------------ |
| `S4`          | ??.??        | 34.56         | 35.34        | 32.77         | ??.??        |
| `S5`          | ??.??        | 53.86         | 45.85        | 41.62         | ??.??        |
| `S4D`         | ??.??        | 50.42         | 49.27        | 42.36         | ??.??        |
| `LSTM`        | ??.??        | 35.47         | 35.18        | 35.13         | ??.??        |
| `Transformer` | ??.??        | 35.08         | 35.08        | 35.09         | ??.??        |
| `Informer`    | ??.??        | 34.95         | 35.08        | 34.96         | ??.??        |

#### Pinball Loss Evaluation

>[!warning]
> These are the results obtained without the $\tau$ multiplicative approach, that's why also the non `SSM` model metrics look so good and so similar across different quantiles.

^a2c187

| Model         | quantile 0.1 | quantile 0.25 | quantile 0.5 | quantile 0.75 | quantile 0.9 |
| ------------- | ------------ | ------------- | ------------ | ------------- | ------------ |
| `S4`          | 13.26        | 11.69         | 12.46        | 11.70         | 12.31        |
| `S5`          | 21.07        | 24.12         | 24.11        | 20.35         | 20.47        |
| `S4D`         | 16.85        | 15.95         | 16.09        | 14.78         | 16.63        |
| `LSTM`        | 15.27        | 15.60         | 15.86        | 16.02         | 15.95        |
| `Transformer` | 15.76        | 15.85         | 15.82        | 15.77         | 16.15        |
| `Informer`    | 16.15        | 16.27         | 15.81        | 15.60         | 14.74        |

### `FDOO1` Experiments `windowed` Approach + Quantile Regression 🪟 🌗 + $\tau$ multiplicative factor

This is the last set of experiments I will perform (probably). Differently from the previous Quantile Regression experiments the quantile level $\tau$ is used in the `forward` passage of the model both as a constant signal to concatenate to the input features and also as a multiplicative factor to the final model prediction → so the final sequence of predicted `RUL` values (obtained after the final `FFN` layer and the `gap` layer) is multiplied by $\tau$. In these experiments I used the usual Quantile Regression loss function to train the model and then I used the Pinball loss with $\tau=0.3$ to evaluate the different models (and the different quantiles) on the test lifes. With this new approach it seems that the quantiles we are most interesting in (quantile 0.25 and 0.1) show a visible improvement in the test metrics over the others when evaluated with the Pinball Loss.

>[!warning]
> Results affected by the [[ssm_experiments#^335e5a|gigantic error]].
> The $\tau$ multiplicative approach is used only on the non `SSM` models and it is worsening the results in terms of metrics. In any case the observations [[j3c_paper#^e78771|I did up to now]] are still valid: there is higher variability in the predictions on different quantiles made by `LSTM` and `Transformer` based models. 
> In the plots we have the high variability between diffferent quantiles also in the non $\tau$ multiplication experiments, here we have these big differences also on the metric values.

| Model         | quantile 0.1 | quantile 0.25 | quantile 0.5 | quantile 0.75 | quantile 0.9 |
| ------------- | ------------ | ------------- | ------------ | ------------- | ------------ |
| `S4`          | **31.70**      | **18.25**       | **8.36**       | **7.77**        | **10.63**      |
| `S5`          | 23.54        | 10.89         | 9.88        | 11.24         | 14.89        |
| `S4D`         | 21.77        | 12.91         | 14.12        | 14.37         | 14.54        |
| `LSTM`        | *37.28*      | *29.07*       | *16.97*      | *14.90*       | *23.58*      |
| `Transformer` | *33.98*      | *21.08*       | *11.81*      | *29.42*       | *50.63*      |
| `Informer`    | *34.69*      | *23.33*       | *11.40*      | *30.95*       | *50.91*      |

### `FDOO2` Experiments `windowed` Approach + Quantile Regression 🪟 🌗 + $\tau$ multiplicative factor

| Model         | quantile 0.1 | quantile 0.25 | quantile 0.5 | quantile 0.75 | quantile 0.9 |
| ------------- | ------------ | ------------- | ------------ | ------------- | ------------ |
| `S4`          |              |               |              |               |              |
| `S5`          |              |               |              |               |              |
| `S4D`         |              |               |              |               |              |
| `LSTM`        |              |               |              |               |              |
| `Transformer` |              |               |              |               |              |
| `Informer`    |              |               |              |               |              |

# Text to insert in the paper

In this section I will keep snippets of text containing important observations I make while analysing the experiment results that can be inserted in the paper, after having been written in a more formal way.

## Model parameters comparison


>[!todo] To insert in the paper
> The `SSM` models are the simpler ones with less parameters and the lowest number of Mult Adds, this is probably due to their linear nature, moreover the number of Multiplication and addition is reduced with the respect to the other models because of the parallel computations that are performed during training thanks to the possibility of using the convolutional mode/view of the `SSM`. In particular the model with the lowest number of parameters (and thus also the lowest number of Mult Add operation) is `S5` that simplifies the architecture of both `S4` and `S4D` thanks to the parallelization of the multiple `SSM` block into a single matrix block.
> On the other hand `LSTM,Transformer` and `Informer` are the heaviest model, in particular `LSTM,Informer` are characterized by a number of Mult Adds operations in the order of $10^{8}$.

>[!success]
> Already inserted below the model parameters table

## Why Pinball Loss as the be used as the `eval_loss`


>[!todo] To insert in the paper
> The results of experiments performed with different sampling intervals for the training quantiles lead to some conclusions. In terms of the `RMSE` metric it is preferrable to overestimate the damage because in the test lifes in which the model commits the more significant errors these are underestimation errors. As a consequence having a model that tends to overestimate the `RUL` somehow balances these underestimation errors improving the test metrics. On the other hand if a life where the model provides good or overestimated predictions is considered the model will cause an augmented test error metric. Moreover since this will most likely be an overestimation error there is the issue of a potential unexpected break in a real world application.
> Probably the reason why the metrics improve on quantiles higher than 0.5 is that the underestimation errors are higher in magnitude than the overestimation ones and so then this produces smaller errors in the test metrics in `Life_mean` and `Life_median`.
> For the reasons explained above probably the `RMSE` metric is not the ideal evaluation metric to use for the evaluation of a `RUL` prediction model. In practical scenarios the evaluation of these models is performed through business metrics (**cite paper by Luciano,DDP,Gian**) where specific costs are assigned by domain experts to underestimation and overestimation errors. In lack of the possibility of having a direct confrontation with domain experts on the `CMAPSS` dataset the Pinball Loss was employed in order to assign different weights to the underestimation and overestimation errors. Considering the higher economical cost of overestimation errors (i.e. being too optimistic in the `RUL` prediction may result in unexpected breaks of the equipment) the `tau` parameter of the Pinball Loss was set to 0.3. In this way the model is penalized more for underestimation errors than for overestimation errors while still keeping a non negliglible weight for underestimation so that there is still some penalty for extreme overestimation errors which may lead to unexploited lifetime and will result in a maintenance strategy that may result to be closer to Preventive Maintenance than to Predictive Maintenance.

>[!success]
> Added in the `Experiment Results` section


## Difference between `SSM` models and others in the plots

Important observation to insert in the paper when comment the results of the different models. In particular it is possible to notice a clear difference in the shapes of the Prediction Interval plots between the `SSM` and the other models.

>[!important] To insert in the paper
> I produced the Prediction Interval plot for `RULTransformer` obtained in the previous set of experiments (i.e. 5 runs and without $\tau$ multiplicating the output) and they are quite similar → I didn't notice this difference because I never produced the prediction plots other than for `S4` because I thought they would be all similar. Actually this is a good result for my thesis of demonstrating the `SSM` are better than the other models. In fact in this `RULTransformer` plot the predicted `RUL` signals are not exactly logarithmic (they have a shape similar to the one of a straight line) but there is much more variability on the different quantiles so for example in `Life_52` (where the model underestimates the `RUL`) the higher quantiles are close to the `RUL` in the last samples. On the other hand the advantage of `SSM` is that they are much more stable across the different quantiles and the different runs and the error stays more or less at the same distance from the true `RUL` across all the time steps → this is probably due to the better capabilities of `SSM` to model long term dependencies → so they are able to remember the linear shape of the `RUL` target all along the test life.

^e78771

This is a comment I copied from the `Prediction Interval Plots` section of experiment 1 of `Informer` model. There is a good observation on the difference with `SSM` models plots and why I think the results on `SSM` are better than the others.

>[!important] To insert in the paper
> Here we have the usual high variance in the prediction interval that distinguishes the non `SSM` based models from the `SSM` ones. Here in `Life_52` if we consider quantiles 0.9 and 0.75 they are really close to the true prediction while quantile 0.1 and 0.25 are significantly overestimating. This is a problem because the model has an high uncertainty in its predictions and thus on a general use case it is not very easy to select the correct quantile to use for the predictions since we may be lucky and select the correct one or not. On the other hand with `SSM`s the model uncertainty is much lower and so the predictions are more reliable independently on the quantile chosen.

>[!success]
> Added in the `Experiment Results` section

Comment from `Prediction Interval Plots` section of [[ssm_experiments#Experiment 8 `S4` `FD001` 4‍⃣ 🪟 🌗|the first `S4` experiment where `tau_mult` was really used]], we may have to substitute the observation done above with this one if we end up inserting the `tau_mult` approach.

>[!important] To insert in the paper
>The plot look very interesting now. We do not have anymore all the predictions close to each other but there is an estimated distribution which can be considered as *left skewed* ? In the sense that, as it is also possible to see from the metric values, we have a very high error in quantile 0.1 (whose prediction is very far from the other) while the predictions on the other quantile are much closer between one another and also closer to the true `RUL`. Moreover in `Life_56` there is not more just overestimation but we have a prediction signal (for example the 0.5 one) that intersects the true `RUL` more or less at half the life and so we have firstly an underestimation and then an overestimation. 
>Differently in the `Transformer` model (of which we have both the `tau_mult` and the non `tau_mult` plots) the predictions over the different quantiles are separated also in quantiles higher than 0.5 creating probably a wider prediction interval. In any case the single prediction signals are further from the true `RUL` than in `S4`. 

### Future predictions

>[!important] To insert in the paper
> Another point in favor of `SSM` models is the fact that the produced predicted `RUL` signals are much closer in shape to the true `RUL` signal and, since the test lifes are truncated, in case we try to extend the prediction on future time steps (for which however there is no availability of the input data on which to perform predictions) the predictions would be much more reliable than the ones produced by non `SSM` models. 

## Prediction Interval plots comparison after `tau_mult` approach

After having produced the prediction interval plots obtained with the `tau_mult` approach for `SSM` models the discussion of the results should be changed. In fact now we have some well visible differences between predictions produced by different quantiles also in the `SSM` models.

>[!important] To insert in the paper
> `SSM` based model outperform the rest of the considered architecture in the majority of the evaluation quantiles.
> In the prediction interval plots produced by the `S4` models the most significant errors are produced for quantiles lower than 0.5 which results in significant underestimations of the `RUL` target. On the other hand predictions on higher quantile values, which correspond to an overestimation of the machine's useful life, are similar across different quantiles making them more stable and makes it more unlikely to produce high overestimation errors which are very dangerous since they may cause unexpected breaks. Moreover the shape of the predicte `RUL` signal closely ensembles the one of the target `RUL` making the model potentially robust and reliable in the predictions for the remaning time steps of the test life which is truncated.
> On the other hand, predictions produced by the `Transformer` model are associated with wider prediction intervals, translating to an higher model uncertainty which may potentially produce pronounced errors in the predictions making the model less robust and reliable to future time steps predictions considering also the shape of the predicted `RUL` signals which is not so close to the true `RUL` signal.

## Blob Plot Description

After having produced a good version of the blob plot we can get some conclusions from it to insert in the paper.

From the `FD001` blob plot we may change the opinion on what is our best model → from `S4` to `S5`. iN fact `S5` is clearly the most efficient model both in terms of parameters and test metric. It is follwed by the other two `SSM` models, which have less parameters than the `Transformer` based ones (even though `S4D` has worse performances). The `Transformer` based model have acceptable performances but their size is in terms of number of parameters and `MAC`s is huge with the respect to the other models (they are in fact all to the right in the x axis ans their blobs are huge). Finally the worst model of all is `LSTM` which is placed on the top right part of the plot, however it does not have a lot of parameters but its test metric errors are too high.

>[!note] `S5` vs `S4`
> I have always considered `S4` as the best model but with the `tau_mult` approach it has lost a little bit its power producing very high errors on extreme quantiles. `S4D,S5` have more stable errors across the different quantiles. Now in this plot I used the 0.5 quantile model but `S5` is still good enough also in the other quantiles. Moreover also the Prediction Interval plots of `S5` are nice so I can also consider to insert them in the paper instead of the `S4` ones.

# Text removed from the paper

In this section I will report some pieces of text removed from the paper to make it 6 pages long.

## Model Parameters

The table with the models parameters and Mult-Adds can be sacrificed since it is not so fundamental at the moment (I think).


```tex
In Table \ref{tab:model_params} the number of parameters and Mult-Adds operations for each model are reported.
These served as useful information to understand the computational complexity of the models and thus
their applicability in real-world scenarios.

\input{tables/model_params}


State Space Models exhibit lower parameter counts and reduced computational complexity, as evidenced by their Mult-Adds values,
likely due to their inherent linear structure.  Furthermore, the convolutional mode of operation in \ac{SSM} facilitates parallel
computations during training, contributing to a reduction in the number of multiplication and addition operations 
compared to alternative models. Among the \ac{SSM} variants considered, \texttt{S5} achieves the lowest parameter count (57,808)
and, consequently, the minimal Mult-Adds value. This efficiency stems from its architectural simplification,
which consolidates multiple \ac{SSM} blocks into a single, parallelizable matrix block, streamlining the designs of both \texttt{S4} and \texttt{S4D}.
In contrast, \texttt{LSTM}, \texttt{Transformer}, and \texttt{Informer} represent more computationally intensive models, with Mult-Adds values
reaching the order of $10^8$, this stems from the quadratic complexity of the self-attention mechanism in the \texttt{Transformer} and \texttt{Informer} models,
which is one of the main limitations of attention based architectures and consequently one of the main reasons for the development of the \ac{SSM} models.
```

## Old version of Abstract

Old version of the Abstract that I had to remove to respect the limit of 100 words.

```tex
%NOTE: Original abstract → 124 words

\ac{PdM} is increasingly pivotal within Industry 4.0 and 5.0 contexts, offering a proactive strategy to enhance efficiency by
forecasting equipment \ac{RUL}. Accurate \ac{RUL} prediction optimizes maintenance scheduling,
minimizing both unexpected failures and premature interventions. This paper introduces a novel \ac{RUL} estimation approach
leveraging \ac{SSM} to efficiently capture long-term sequence dependencies. To robustly manage uncertainty inherent in \ac{RUL} estimation,
\ac{SQR} is integrated into the \ac{SSM} architecture, enabling estimation of multiple quantiles of the target distribution.
The effectiveness of the proposed methodology is evaluated against state of the art sequence modeling techniques 
(\ac{LSTM,} Transformer, and Informer) using the widely recognized \ac{C-MAPSS} benchmark dataset.
Results highlight superior predictive accuracy and improved uncertainty quantification of \ac{SSM} models,
which demonstrates significant promise for practical deployment in high-stakes industrial environments.

%NOTE: Shorter version → 116 words

\ac{PdM} is increasingly pivotal within Industry 4.0 and 5.0 contexts, offering a proactive strategy to enhance efficiency by
forecasting equipment \ac{RUL}. Accurate \ac{RUL} prediction optimizes maintenance scheduling,
minimizing unexpected failures and premature interventions. This paper introduces a novel \ac{RUL} estimation approach
leveraging \ac{SSM} to efficiently capture long-term sequence dependencies. To manage model uncertainty in \ac{RUL} estimation,
\ac{SQR} is integrated into the \ac{SSM} architecture, enabling estimation of multiple quantiles of the target distribution.
The effectiveness of the proposed methodology is evaluated against traditional sequence modeling techniques
(\ac{LSTM}, Transformer, and Informer) using the \ac{C-MAPSS} benchmark dataset.
Results highlight superior predictive accuracy and improved uncertainty quantification of \ac{SSM} models,
demonstrating significant promise for practical deployment in high-stakes industrial environments.
```

# Code removed

Code to compute the model summary with `torchinfo`:


```python
print('#'* 50)
print(f"Model summary computation with torchinfo:")
print('#'* 50)
input_size=(1, exp_config.sequence_length, d_input) if not exp_config.quantile_reg else (1, exp_config.sequence_length, d_input-1)            
model_summary=summary(
    model = model,
    input_size = input_size,
    device = model_config.device
)
params = model_summary.total_params
mult_adds = model_summary.total_mult_adds

print('#'*50)
print(f"Total params: {params}")
print(f"Total mult adds: {mult_adds}")
print('#'*50)
```

`matplotlib` version of the blob plot.

```python
# Normalize Mult-Adds to determine the radius of the blobs
max_mult_adds = np.max(plot_dict["mult_adds_float"])
blob_radii = plot_dict["mult_adds_float"] / max_mult_adds * 20  # Scale radii for better visualization

# Create the scatter plot with varying blob sizes
plt.figure(figsize=(10, 6))
plt.scatter(plot_dict["params_float"], plot_dict["test_metric"], s=np.log(blob_radii**2), alpha=0.5)  # s is area, so square radii
# plt.scatter(plot_dict["params"], plot_dict["test_metric"], s=plot_dict["mult_adds"], alpha=0.5)  # s is area, so square radii

# Add annotations for each point
for i, model_name in enumerate(config.model_names):
    plt.annotate(model_name, (plot_dict["params_float"][i], plot_dict["test_metric"][i]), textcoords="offset points", xytext=(5,5), ha='center')

```
