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
    - [ ] Write down the table (Slide 41 for `FD001`, slide 43 for `FD002`) in `latex` format and insert it into the Overleaf project
- Prediction plots for `FD001,FD002`
    - [ ] Find out where the plots are contained (probably inside `acquario3`) and insert some of them in the Overleaf project
- Table with analysis of parameter size and number of operations of each model
    - [ ] Write the table of Slide 52 in `latex` format and add it to the Overleaf project
- Blob plot representing the results of the table
    - [ ] Find out where the blob plot is contained and insert it in the Overleaf project. Here I have to decide weather to include the plot or the table, since they represent the same thing → maybe the plot is more intuitive.

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
    - [ ] Maybe if this works fine we can do all the experiments with it instead of reporting the results obtained up to now (which did not use this kind of loss).

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

>[!todo] To insert in the paper
> The `SSM` models are the simpler ones with less parameters and the lowest number of Mult Adds, this is probably due to their linear nature, moreover the number of Multiplication and addition is reduced with the respect to the other models because of the parallel computations that are performed during training thanks to the possibility of using the convolutional mode/view of the `SSM`. In particular the model with the lowest number of parameters (and thus also the lowest number of Mult Add operation) is `S5` that simplifies the architecture of both `S4` and `S4D` thanks to the parallelization of the multiple `SSM` block into a single matrix block.
> On the other hand `LSTM,Transformer` and `Informer` are the heaviest model, in particular `LSTM,Informer` are characterized by a number of Mult Adds operations in the order of $10^{8}$.

### `FD001` Experiments `windowed` Approach + Quantile Regression 🪟 🌗

>[!note]
> Here the results are obtained averaging the `RMSE` over different runs of the same model with different quantiles.

| Model | quantile 0.25 | quantile 0.5 | quantile 0.75 |
|-------|------| ---------| ---------|
| `S4` | 18.85 | 17.03 | 16.26 |
| `S5` | 30.78 | 30.00 | 29.01 |
| `S4D` | 40.19 | 39.95 | 38.53 |
| `LSTM` | 28.60 | 28.63 | 28.63 |
| `Transformer` | 28.14 | 30.61 | 36.10 |
| `Informer` | 29.94 | 25.86 | 27.16 |

### `FD002` Experiments `windowed` Approach + Quantile Regression 🪟 🌗

| Model | quantile 0.25 | quantile 0.5 | quantile 0.75 |
|-------|------| ---------| ---------|
| `S4` | 34.56 | 35.34 | 32.77 |
| `S5` | 53.86 | 45.85 | 41.62 |
| `S4D` | 50.42 | 49.27 | 42.36 |
| `LSTM` | 34.95 | 35.08 | 34.96 |
| `Transformer` | ??.?? | ??.?? | ??.?? |
| `Informer` | ??.?? | ??.?? | ??.?? |


