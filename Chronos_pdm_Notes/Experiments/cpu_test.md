---
id: cpu_test
aliases: []
tags:
  - experiments
  - ssm_pdm
---

# `CPU` Consumption Tests

I am inspecting the code to find out some potential places that may be responsible for the high `CPU` consumption that my experiments are causing on the `ssm_pdm` project. In particular the main consumption is observed with the `RULInformer` model experiments where the `CPU` is more or less always at 1000% or higher on `nvtop`.

## `eval_loop` Function

The main suspect piece of code is inside `eval_loop`:

```python
batch_out = output.to("cpu").detach().numpy()
batch_target = rul.to("cpu").detach().numpy()
y_pred.append(batch_out) if config.approach=="padding" else y_pred.extend(batch_out)
y_true.append(batch_target) if config.approach=="padding" else y_true.extend(batch_target)
```

This code is used to append the prediction and true values from different batches inside a list and these two lists are returned by `eval_loop` and are used to create the `pickle` files inside the `outputs` folder which are used to create the `metrics_df` and the plots.

This piece of code may create `CPU` usage because we are moving the `output` and `rul` tensors to the `cpu` and we are appending them to a list.

The other bad thing is that this code is almost completely useless when used in the training loop 😱. In fact inside `wandb_train_test` I am using the `y_pred,y_true` list only in the last epoch to create the `model_info` dictionary that I do not even use at all.

This piece of code is useful just inside `best_model_perf` when it is used to compute the predictions and true values for the best model.

## `SSMWindowedRegressionDataset`

In this `Dataset` class there is the creation of the overlapped windows which may be another potential part of the code that requires high `CPU` load. 

The change I made in order to  try to reduce the `CPU` computations is to convert all the code used to create the overlapping windows and select the sub sequences using `torch` instead of `numpy`. 

```python
class SSMWindowRegressionDataset(Dataset):
    def __init__(
        self,
        life: pd.DataFrame,
        sequence_length: int = 500,
    ):

        life,rul = life.iloc[:,:-1],life["RUL"]
        life = torch.from_numpy(life.values).float()
        rul = torch.from_numpy(rul.values).float()

        if life.shape[0] < sequence_length:
            # Padding approach using PyTorch
            pad_len = sequence_length - life.shape[0]
            pad_arr = torch.zeros((pad_len, life.shape[1]))
            mask = torch.cat((torch.ones(life.shape[0]), torch.zeros(pad_len)))
            sequences = torch.cat((life, pad_arr))
            targets = torch.cat((rul, pad_arr[:, -1])) # Ensure target padding matches

            # Add the extra dimension to match the windowed approach
            sequences = sequences.unsqueeze(0)
            targets = targets.unsqueeze(0)
            mask = mask.unsqueeze(0)

        else:
            # Windowed approach using PyTorch
            n_windows = life.shape[0] - sequence_length + 1
            sequences = torch.stack([life[i:i + sequence_length] for i in range(n_windows)])
            targets = torch.stack([rul[i:i + sequence_length] for i in range(n_windows)])
            mask = torch.ones(n_windows, sequence_length)

        self.sequences = sequences
        self.targets = targets
        self.mask = mask

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].unsqueeze(-1)
        target = self.targets[idx].unsqueeze(-1)
        mask = self.mask[idx].unsqueeze(-1)
        return sequence, target, mask

```


# `cpu_test` Results

I created a new branch `cpu_test` to test the `CPU` usage of the code.

## `eval_loop` `cpu_test`

I modified the piece of code inside `eval_loop` to:

```python
if config.cpu_version:
    batch_out = output.to("cpu").detach().numpy()
    batch_target = rul.to("cpu").detach().numpy()
    y_pred.append(batch_out) if config.approach=="padding" else y_pred.extend(batch_out)
    y_true.append(batch_target) if config.approach=="padding" else y_true.extend(batch_target)
else:
    if epoch_number == config.epochs -1:
        y_pred.append(output) if config.approach=="padding" else y_pred.extend(output)
        y_true.append(rul) if config.approach=="padding" else y_true.extend(rul)
    else:
        y_pred,y_true=None,None
```

I tested with `cpu_version: false` and moreover with the addition of the `epoch_number` input parameter now the `append` and `extend` commands are done just in the last epoch.

However still the `CPU` usage it's very high, moreover the main problem is that even if I change the model (I tried also with `S4` and with `S5` (the lightest model) on `FD001`) the `CPU` usage remains the same → so this means that the problem is not this piece of code used to save the predictions and true values, nor the specific model, so I have to check for some other pieces of code that may create this high `CPU` load.

The fact that also with `S5` on `FD001` the `CPU` is very high means that probably the problem is due to something I did, not something in the `S5` implementation.

## `SSMWindowedRegressionDataset`

Before trying to change the code for the creation of the sequences let's try to modify `sequence_length` so that we create less overlapped windows.

Let's start trying to set `sequence_length=400` which is surely higher than all the life lengths → in this way there is no overlapping window and maybe there are less computations performed by the `CPU`?

I tried with different values for `sequence_length` but the `CPU` usage remains the same. Now I will try to modify the code using `torch.tensor` instead of `np.array` to create the windows. 

Even using the `torch` based code still the `CPU` usage does not change so apparently this piece of code is still not the one causing the high `CPU` usage.

## `ad_mg` `CPU` Test

At this point let's try to launch an experiment in the `ad_mg` project, where the structure of the code and of the experiment scripts is similar. 

Launched an experiment on the `single_sensor_downsample` approach in the `ad_mg` project with `target_fs=100` and here the `CPU` utilization is much lower → it stays always between 100 and 200 and there are some sudden peaks at 1000% or more but for some seconds only. However there is this strange thing that even if I launch the experiment on device 1 it creates two processes: one on device 1 (with high `GPU` memory and 0 `CPU`) and another in device 0 with all the `CPU` usage → it's like it divides the main process in two processes: one for the `GPU` and one for the `CPU` → very strange, I already looked at this problem some time ago and I couldn't find a solution.

Also with an experiment on `single_sensor` approach we have a similar behavior → in this case almost the entirety of the `GPU` memory (about 46 `GiB` over 47 available) is occupied and the `CPU` is not used a lot. This is probably due to the fact that we have these batches with signals of 20-30k samples each. On the other hand in the `CMAPSS` dataset we have a lot of small signal of length `sequence_length` (i.e. 170 in the setting of my experiments) which may be the reason for this high `CPU` usage.

A confirmation of this behavior is given when I performed a test with a  `rainflow_mat` approach experiment. In this case we have still very small signals/matrix (`11x2=121` which is a comparable size to the 170 samples of the sub sequences I am using here). Also in this case the `GPU` usage and memory is very low, while the `CPU` usage it's very high, always higher than 1000% for the few seconds I looked at `nvtop`. In this case it was not considered a problem because the experiments finish very fast in this approach (in fact here we have mini batches with a single `11x2` `rainflow` matrix, not mini batches of 100 `rainflow` matrix as it is the case in this project).

## Varying the `batch_size`

Some visible differences can be seen varying the `batch_size`. In fact using values smaller than the 100 I was using up to now increased the `CPU` usage to over than 1000%, while in the `cpu_test` experiments I am performing we are usually around 700-800%. So at this point I thought that the trick was to increase the `batch_size`, but no with `batch_size)=200` or 256 the `CPU` usage increases coming closer to 2000% in some instants. With `batch_size=100` or 128 we still get this values around 700. 


## Prompt for `LLM`s

I am working on a remote server to use the powerful `GPU`s and `CPU` of the server, mainly to train big deep learning models. I am now working on a codebase that has a `git` repository attached to it and it is called `ssm_pdm`. 

The server is shared with other users and some of them made me notice that I was running some processes for a long time which were using a lot of the `CPU`. In particular in `nvtop` the `CPU` was at around 1500-2000% all the time. Since this may slow down the server for everyone I am now inspecting the code to see what may trigger this high `CPU` usage and modify the code in order to use mainly the `GPU` → in fact I am using deep learning models built with `pytorch` so the aim is to exploit the `GPU` as much as possible. I tried several things: I converted all the pieces of code that were using `np.array` into using `torch.tensor` but still running new experiments it seems that the `CPU` usage does not change. I also tried with different models (with varying number of parameters and amount of Mult-Adds operations) and both with big and smaller models it seems that the `CPU` usage is still the same. 

Successively I tried to perform some experiments on the codebase of another project which is called `ad_mg`. The `ad_mg` codebase uses similar models on a similar task but on different data and I noted something interesting. In `ad_mg` I have different kind of approaches to train the models:
- `single_sensor` approach → in this approach I take the input signal (i.e. a time series containing the raw measurements of a sensor (accelerometer) placed on a machine) in the time domain (after having performed some pre processing steps on it) and use it as the input to the model. The signals used in this approach are quite big, all of them have around 20,000 to 30,000 samples. In this experiments, looking at `nvtop`, I am using almost the entire available `GPU` memory (46 `GiB` over 47 `GiB`) while the `CPU` usage is quite low (around 100-200%). Note that in this approach I use a `batch_size=1` because the different signals are obtained from different kind of acquisitions and thus they have different lengths and cannot be included together in the same mini batch.
- `rainflow_mat` approach → In this approach we apply the Rainflow Matrix Cycle Counting algorithm (an algorithm used in the field of Fatigue Analysis) which provides a summary of the input signal in the form of a matrix, which in my specific case is of shape `11x2` (also in this case each signal (and so each `rainflow` matrix) has its own mini batch). In this case, where the model's inputs are much smaller than in the `single_sensor` approach, we have a very small `GPU` usage but a very high `CPU` usage, always higher than 1000%. 

The interesting thing about this phenomenon is that also in `ssm_pdm` I am using small signals as in the `rainflow_mat` approach. In particular I have some input signals and they are converted into a list of sub sequences by applying a series of overlapping windows to the signal. Since these windows have all the same length, all the sub sequences extracted from the multiple input signals are grouped together in mini batches. In particular I selected  `sequence_length=170,batch_size=100` so I have all these mini batches with signals of 170 samples each which a shape similar to the one of the `11x2` `rainflow` matrices I have in the `rainflow_mat` approach.

So since the `CPU` consumption is similar in the `rainflow_mat` approach experiment and in the experiments in the `ssm_pdm` project apparently it seems that when I have all these small signals for some reason they are handles by the `CPU` rather than by the `GPU`? It  is a bit strange because `torch` should manage all this on the `GPU`, even if we have mini batches of `batch_size>1` it should work in parallel on all the elements of the mini batches. 

Another interesting thing I notice is that it seems that also the `batch_size` has an impact on the `CPU` usage. As I said above normally I use `batch_size=100`. Then, in this set of experiments to understand the high `CPU` usage, I tried to modify it and I saw that with `batch_size=50,10,1` the `CPU` usage increases from 700-800% to values higher than 1000%. So at this point I thought that maybe the higher the `batch_size` the lower the `CPU` usage (since decreasing `batch_size` lead to an increase `CPU` usage), so I tried to increase `batch_size` to 200,256 but now the `CPU` usage is close to 2000%. 

So at this point I don't know what to do anymore to solve this high `CPU` usage problem.

### Answer to the `LLM` suggestion

Ok I tried the last approach you proposed and I added a `torch.profiler` to my training loop, which is implemented as follows:

```python
def train_loop(
        dataloader: DataLoader,
        model: nn.Module,
        config: ExperimentConfig,
        tokenizer: MeanScaleUniformBinsSensor,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = torch.device("cpu"),
) -> float:
    """
    Train loop for one epoch

    Args:
        dataloader (DataLoader): The DataLoader object
        model (torch.nn.Module): The model object
        config (ExperimentConfig): The configuration object
        optimizer (torch.optim.Optimizer): The optimizer object
        criterion (torch.nn.Module): The loss function
        device (str): The device to use

    Returns:
        loss (float): The loss value
    """

    model.train()
    train_loss = 0.0
    num_batches = len(dataloader)
    pbar = tqdm(enumerate(dataloader))

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
    ) as prof:
        for batch_idx, (life, rul, mask) in pbar:
            life = life.to(device) if config.approach=="padding" else life.to(device).squeeze(-1)
            rul = rul.to(device).squeeze(-1)
            mask = mask.to(device) if config.approach=="padding" else mask.to(device).squeeze(-1)
            if config.quantile_reg:
                tau = sample_quantile(
                    quantile_dist=config.quantile_dist,
                    bounds=config.bounds,
                    print_quantile=False
                )

            if config.model_name.startswith("chronos"):
                input_ids, attention_mask, _ = tokenizer.context_input_transform(context=life, mask=mask)
                output = model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                life = life.permute(2,0,1) if config.approach=="padding" else life
                mask = mask.permute(1,0) if config.approach=="padding" else mask
                rul = rul.unsqueeze(0) if config.approach=="padding" else rul
                output = model(life) if not config.quantile_reg else model(life,tau=tau)

            loss = criterion(output, rul, mask) if not config.quantile_reg else criterion(output, rul, mask, tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # pbar.set_description(f"Batch Idx: {batch_idx}/{len(dataloader)} | Train Loss: {train_loss / (batch_idx + 1):.4f}")

    print('#'* 50)
    print("Profiler results:")
    print('#'* 50)
    print("CPU time train_lopp:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print('#'* 50)
    print("GPU time train_loop:")
    print('#'* 50)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print('#'* 50)

    return train_loss / num_batches
```

Moreover I also have an equivalent function to evaluate the model on the validation and test set. This is used so that at each epoch I can track the `train_loss,val_loss` and `test_loss` and I can use the `val_loss` to detect the presence of overfitting:

```python
def eval_loop(
        dataloader: DataLoader,
        model: nn.Module,
        config: ExperimentConfig,
        tokenizer: MeanScaleUniformBinsSensor,
        criterion: nn.Module,
        eval_criterion: nn.Module,
        mode: str = "Test",
        device: torch.device = torch.device("cpu"),
        use_tqdm: bool = True,
        epoch_number: int = 0,
        tau: float = 0.5,
) -> Tuple[float,float,np.ndarray,np.ndarray]:
    """
    Evaluation loop for one epoch

    Args:
        dataloader (DataLoader): The DataLoader object
        model (torch.nn.Module): The model object
        config (ExperimentConfig): The configuration object
        tokenizer (MeanScaleUniformBinsSensor): The tokenizer object
        criterion (torch.nn.Module): The loss function
        eval_criterion (torch.nn.Module): The evaluation loss function
        mode (str): The mode of evaluation
        device (str): The device to use
        use_tqdm (bool): Whether to use tqdm or not
        tau (float): The quantile level on which the model will be evaluated if the quantile regression approach is used

    Returns:
        loss (float): The loss value
    """

    model.eval()
    eval_loss, eval_rmse_loss = 0.0, 0.0
    num_batches = len(dataloader)
    pbar = tqdm(dataloader) if use_tqdm else dataloader
    y_pred,y_true = [],[]

    with torch.no_grad():
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
        ) as prof:
            for life, rul, mask in pbar:
                life = life.to(device) if config.approach=="padding" else life.to(device).squeeze(-1)
                rul = rul.to(device).squeeze(-1)
                mask = mask.to(device) if config.approach=="padding" else mask.to(device).squeeze(-1)

                if config.model_name.startswith("chronos"):
                    input_ids, attention_mask, _ = tokenizer.context_input_transform(context=life, mask=mask)
                    output = model(input_ids=input_ids, attention_mask=attention_mask).logits
                else:
                    life = life.permute(2,0,1) if config.approach=="padding" else life
                    mask = mask.permute(1,0) if config.approach=="padding" else mask
                    rul = rul.unsqueeze(0) if config.approach=="padding" else rul
                    output = model(life) if not config.quantile_reg else model(life,tau=tau)

                if config.cpu_version:
                    batch_out = output.to("cpu").detach().numpy()
                    batch_target = rul.to("cpu").detach().numpy()
                    y_pred.append(batch_out) if config.approach=="padding" else y_pred.extend(batch_out)
                    y_true.append(batch_target) if config.approach=="padding" else y_true.extend(batch_target)
                else:
                    if (epoch_number == config.epochs -1) or (use_tqdm == False):
                        y_pred.append(output) if config.approach=="padding" else y_pred.extend(output)
                        y_true.append(rul) if config.approach=="padding" else y_true.extend(rul)
                    else:
                        y_pred,y_true=None,None

                loss = criterion(output, rul, mask) if not config.quantile_reg else criterion(output, rul, mask, tau)
                rmse_loss = eval_criterion(output, rul, mask)
                eval_loss += loss.item()
                eval_rmse_loss += rmse_loss.item()

        eval_loss/=num_batches
        eval_rmse_loss/=num_batches
        # print(f"Avg {mode} Loss: {eval_loss:.4f} | \
        #         Avg {mode} eval Loss: {eval_rmse_loss:.4f}")

    print('#'* 50)
    print("Profiler results:")
    print('#'* 50)
    print("CPU time eval_loop:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print('#'* 50)
    print("GPU time eval_loop:")
    print('#'* 50)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print('#'* 50)

    if config.cpu_version:
        y_pred=np.array(y_pred)
        y_true=np.array(y_true)
        return eval_loss, eval_rmse_loss, y_pred, y_true
    else:
        if (epoch_number == config.epochs -1) or (use_tqdm == False):
            y_pred=torch.cat(y_pred).cpu().numpy()
            y_true=torch.cat(y_true).cpu().numpy()
            return eval_loss, eval_rmse_loss, y_pred, y_true
        else:
            return eval_loss, eval_rmse_loss, None, None
```

As you can see I added the `torch.profile` both to `train_loop` and to `eval_loop`.  Monitoring `nvtop` and `htop` during the code execution I noticed that the highest values in the `CPU` percentage appeared during the `eval_loop`. In particular at the end of the training script I do some post-training operations → during training in fact I keep track of the `val_loss` so that at the end of the training epochs I can select the `state_dict` of the best model (i.e. the model that across the training epochs achieved the lowest `val_loss` value). In this post-training stage I load the `state_dict` of this best model and I evaluate it on the test set calling `eval_loop`on all the test samples. It is in this stage of the execution that I noticed the higher percentages in `htop`. Now I leave you the table outputted by the `torch.profiler` during this `eval_loop` calls: 

```txt
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------
------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self 
CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------
------  ------------  ------------  ------------  
                                              aten::div        42.12%     335.327ms        42.23%     336.209ms       2.586ms      82.339us        
 0.39%      82.339us       0.633us           130  
                                             aten::ones         0.08%     665.834us        26.33%     209.621ms       1.612ms       0.000us        
 0.00%       0.000us       0.000us           130  
                                            aten::fill_        26.22%     208.710ms        26.22%     208.710ms       1.605ms       0.000us        
 0.00%       0.000us       0.000us           130  
                                            aten::copy_         2.04%      16.230ms         7.26%      57.794ms      22.229us       5.498ms        
25.85%       5.918ms       2.276us          2600  
                                               aten::to         0.21%       1.677ms         6.98%      55.571ms      30.534us       0.000us        
 0.00%       4.544ms       2.497us          1820  
                                         aten::_to_copy         0.58%       4.655ms         6.77%      53.894ms      46.064us       0.000us        
 0.00%       4.544ms       3.884us          1170  
                                            aten::index         1.08%       8.604ms         6.42%      51.088ms     130.994us       1.969ms        
 9.26%       4.583ms      11.752us           390  
                                       cudaLaunchKernel         5.66%      45.022ms         5.66%      45.022ms       8.246us       0.000us        
 0.00%       0.000us       0.000us          5460  
                                           aten::matmul         0.36%       2.884ms         3.02%      24.058ms      61.688us       0.000us        
 0.00%       3.398ms       8.713us           390  
                                          aten::nonzero         1.05%       8.323ms         2.89%      22.988ms      88.416us       2.125ms        
 9.99%       2.125ms       8.172us           260  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------
------  ------------  ------------  ------------  
Self CPU time total: 796.134ms
Self CUDA time total: 21.273ms
```

It seems that the operations bringing to the highest `CPU` load are: `aten::div,aten::ones,aten::fill_,aten::copy_`

- `aten::div` → Division
- `aten::ones` → `torch.ones` → these are all the `torch.ones` that I use in `tau_mult` and `quantile_reg` to create attach the quantile level $\tau$ as an additional feature.

## Profiling the code with `torch.profiler`

This is a suggestion that comes from Gemini 2.0 Flash as an answer to the [[cpu_test#Prompt for `LLM`s|question above]]. I can use this `torch.profile` to return information on the lines of code that are consuming most of the `CPU` or `GPU`. As in the example below this `profiler` has to be inserted in the training loop, so inside the `train_loop` function.

```python
import torch.profiler as profiler

with profiler.profile(activities=[
        profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                       record_shapes=True) as prof:
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Test Results with `torch.profiler`

Using the `tau_mult_no_tau_feat` approach and looking at the outputs of the `torch.profiler` now there is no more `aten::ones` among the most `CPU` consuming operations but there is `aten::copy` and `aten::to_copy` so that is probably due to the `.to(device)` commands to move a `device` to the `GPU` or some other copying operations.

Ok looking at the code of `train_loop` and `eval_loop` I realized that the `.to(device)` operations cannot be avoided. In fact when I access the elements of the `DataLoader` doing `for life,rul,mask in dataloader` the three tensors are by default sent to the `cpu` device so I have to use the `device` input argument that is passed to the `train_loop,eval_loop` functions to send it to the `GPU`. Now I added the `non_blocking=True` argument to the `.to(device)` operation as suggested by Gemini to move asynchronously the data. The fact is that now in the `CPU` profiler table we have the `cudaStreamSynchronize` operation as the most costly one that is needed to synchronize the data transfer between `CPU` and `GPU`. 

Other highly costly operations are `aten::index` and `aten::nonzero`:

- `aten::index` → This is related to indexing operations which may be an overhead operation that consumes `CPU` time. Here Gemini says that it is better to index through a boolean mask, which is actually what I am doing also inside the custom loss functions that I created so that should not be a problem.

```txt
                                  cudaStreamSynchronize        87.77%     729.988ms        87.77%     729.988ms       1.604ms       0.000us        
 0.00%       0.000us       0.000us           455  
                                            aten::index         0.41%       3.434ms        61.15%     508.612ms       1.956ms       1.191ms        
 6.68%       3.934ms      15.131us           260  
                                          aten::nonzero         0.59%       4.931ms        60.44%     502.628ms       1.933ms       2.743ms        

```