---
id: ssm_pdm_integration
aliases: []
tags:
  - ssm_pdm
  - chronos-pdm
---

# `SSM_PDM` Project Integration

Considering the non properly exciting results obtained [[chronos_reg_exp|with the `chronos` model]] I think it is time to **merge the `chronos-pdm` and the `SSM_PDM` project** with the final goal of doing a paper where these two new methodologies for `RUL` estimation (i.e. `SSM` and `PdM`) are compared, together also with other more traditional architectures (e.g. `RNN,LSTM,GRU,...`).

Since the code of the `SSM_PDM` project is a bit messy and very disorganized I think it is best if I migrate it inside this codebase. So here I report some notes to help me in this process.

## Understanding the `SSM_PDM` project code

Quick note to understand the flow in the code of `SSM_PDM` project, in particular on the script `train_ssm_wandb.py`.

- Definition of `argparse` and all command line arguments
- Bunch of useless dictionary for the `wandb` configuration
- `if tune_hyperparams` block → bunch of `wandb` configurations and then definition of `sweep_train` method to perform a `wandb` sweep → we do not care about this right now.
- `else` → In the `else` we have the code for a single `wandb` run (what we care about right now):
    - We will call the `model_pipeline` function (which is similar to the `wandb_run` I use now). Inside `model_pipeline` we call:
        - `make` → This function loads all the data, model, loss, scheduler and optimizer → it's like `wandb_data` → here I want to do some `ipdb` to see the shapes of the data and the model.
        - `wandb_train_test` → Usual function that calls for each epochs the `train_loop,eval_loop` methods and logs the results on `wandb`
        - There are now some `if` blocks to decide which kind of plot function to use → here we care about `single_life_perf_plots` that calls `plot_torch_predictions_grid` that is the same function I adapted for the `chronos` project.

### Things to add

Up to now in `chronos-pdm` I have considered the lifes as a single sequence of data and I set a maximum `sequence_length` and used padding to make all the sequences with that length. In this way I can define the Regression Head of the model so that it produces in outputs a sequence of `sequence_length` `RUL` values. However in the `SSM_PDM` approach I used two different approaches to treat the lifes:

- `seq_to_seq` → In this approach the `sequence_length` parameter represents the length of the windows we use to create multiple sub sequences starting from a life. In particular we are using overlapping windows with stride 1 (e.g. so if `sequence_length=30` we will consider the first window as `time_series[0:29]`, the second one as `time_series[1:30]` and so on) and for each one of these sub sequences we have a `RUL` time series prediction of length `sequence_length`.
    - [x] I have to see how I then combined the predictions of all the subsequences in order to obtain a single `RUL` prediction for the entire life → The predictions are combined inside the function `life_predictions` (inside `models.py`) which calls the `combine_values` function.
- `time_wise` approach → In this approach the lifes were considered in their entirety and for each life we predicted a single `RUL` value (the one corresponding to the last time step).
    - [ ] Here I probably have to look at the part of the code from Francesco (he was the one who worked on this `time_wise` approach) because reviewing the code I wrote (the one where the `DataLoader`s are created using `crete_sequence_no_pad`) does not make any sense and it is completely wrong.
    - [ ] Also in this case I have to understand how we managed to still obtain a sequence of `RUL` values as prediction for each life.

### Things to remove

There are some things I probably want to remove:

- The split in training, validation and test is not done correctly in my opinion. In fact here I took the entire set of lifes of the test set and I splitted the training set into training and validation. As I wrote in [[chronos-data|this note]] I think it's better to keep all the lifes for the training set and splitting the test set into validation and test so that the model is tested on *truncated lifes* (i.e. lifes where the `RUL` does not go to 0).
- Remove the `sort_rul` approach → In order to create the `DataLoader`s I concatenated together all the different lifes and then I sorted the `RUL` columns. This is not a good approach because in doing so we are mixing together samples coming from different lifes which is not correct since normally between different lifes there are different conditions into play.
