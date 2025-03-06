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
