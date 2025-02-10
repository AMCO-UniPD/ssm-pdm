---
id: chronos_exp
aliases: []
tags:
  - experiments
  - chronos-pdm
---

# `CHRONOS` First Experiments

I want to start playing a little bit with `CHRONOS` to understand how to use the `chronos-forecasting` library, which is similar to 🤗 and it's recommended to use the model for research purposes. As suggested it is better to clone the [`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) repository since it provides a minimal configuration to use the model and potentially also the possibility of changing something in the code if necessary.

## Forecasting

The first step to do is to try to use the model to do forecasts of time series. Here I will use the example code provided in the repository to see how it works. The code is the following one:


```python
import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cuda",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
# mean is an fp32 tensor with shape [batch_size, prediction_length]
quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(df["#Passengers"]),
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9],
)
```

The main steps are the following:

- Loading the pipeline:
    - Here essentially we use `BaseChronosPipeline`, which is the correspondent of the 🤗 `pipeline` function I guess. Here we use the `from_pretrained` method to choose the specific version of `CHRONOS` we want to use (we will start with the small one), define the `cuda` device to use for inference and the type of tensors to use, here following the code we will use `torch.bfloat16`.
- Loading the data:
    - In the example considered here we use the `AirPassengers` dataset (which I think I have already used in the `BEFD` course) which is really common as an example dataset for time series forecasting. We load the data using `pandas` and the `read_csv` method. Here they pass directly the URL to the `csv` containing the dataset, so this trick of passing the `URL` apparently works also in `pandas`.
- Inference with the model:
    - In `chronos` doing inferennce means predicting the quantiles of the posterior distribution, which is the distribution of $p(x_{t+T}|x_{1:t})$ where $x_{1:t}$ is the observed data until time $t$ and $x_{t+T}$ is the data at time $t+T$ we want to predict. Here we use the `predict_quantiles` method of the pipeline to do this. Later I will take a closer look at the code to see weather there are other things we can do with this class and method but here we have three main parameters:
        - `context` → This is the input/prompt so the time series we want to forecast. In the comment above they say that this can have different formats:
            - a `1D tensor` → in case we have only one time series. I think this is the case of the example above.
            - a `list of 1D tensors` → in case we have multiple time series
            - a left padded `2D tensor` with the `batch_size` as the first dimension
            - The outputs of the method are:
                - `quantiles` → These are the quantiles predicted. It is a `torch.tensor` with shape `[batch_size,prediction_length,num_quantile_levels]`
                - `mean` → The mean of the posterior distribution at the time we want to predict. It has shape `[batch_size,prediction_length]`
        - `prediction_length` → This is the number of steps we want to forecast in the future
        - `quantile_levels` → The quantile levels we want to predict.


>[!note]
> Actually the `AirPasssengers` dataset is very simple, it just as the `Month` and the `#Passengers` columns, so here we can do forecasts on a one dimensional time series only. I will have to try with `CMAPSS` or with other multi dimensional time series data to see how it works. 

### Inference

Ok the inference works 💪.

We have the following shapes:
    - `quantiles` → `[1,12,3]`
    - `mean` → `[1,12]`

### Plots

Ok now it's time to produce some plots to have a clearer view of how the predictions look like. Here I will always follow the example code in the repository:


```python
import matplotlib.pyplot as plt  # requires: pip install matplotlib

forecast_index = range(len(df), len(df) + 12)
low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

plt.figure(figsize=(8, 4))
plt.plot(df["#Passengers"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()
```

The predictions looks very good. Obviously this is a time series with a very clear trend and seasonality behavior, I want to see how good `chronos` is on the `CMAPSS` dataset. 

## Extracting Encoder Embeddings

Another interesting example proposed in the repository let us play a little bit with the `chronos` internals. Here we can infact use the `embed` method from `ChronosPipeline` (or `ChronosBoltPipeline`) to extract the embeddings of the model encoder and the tokenizer state for a given input. Here is the code:


```python
import pandas as pd
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(df["#Passengers"])
embeddings, tokenizer_state = pipeline.embed(context)
```

Here we have the following steps:

- Loading the pipeline:
    - Here I have to understand what is the difference between the `BaseChronosPipeline` and the `ChronosPipeline`. Looking at the code it seems that `ChronosPipeline` is the a subclass of `BaseChronosPipeline` but the `embed` method is implemented inside `ChronosPipeline`
    - Loading the data as before.
    - Call the `embed` function to get the `embeddings` and the `tokenizer_state`:
        - `embedding` in this case has shape `[1,145,512]` which should be something like `[batch_size,sequence_length,embedding_size]`. Here the length of the context is 144 so this makes sense because we have the next token also in the input I guess
        - `tokenizer_state` → this is equal to 280.98 which, I checked, is exactly the mean of `context`. This makes sense because at the end the tokenizer here does a Mean Scaling and than a quantization so the mean represents the state of the tokenizer.

>[!success] Now I understand the `pipeline` classes
> Actually we have the `BaseChronosPipeline` class that is the superclass of both `ChronosPipeline` and of `ChronosBoltPipeline`, so in the `from_pretrained` method we can pass both base and `bolt` `chronos` models to `BaseChronosPipeline` while we can pass only one of the two types to `ChronosPipeline` and to `ChronosBoltPipeline`.

### Embedding of `ChronosBoltPipeline`

Let's try to use `ChronosBoltPipeline` and a `bolt` checkpoint and see how the embeddings change.

Something surely changes in the `bolt` models because the shapes of the embedding and the `tokenizer_state` are different:
- `embeddings` → `[1,10,512`] → maybe they are doing some strange optimization things to make `bolt` faster because here the sequences length drops to 10 instead of 145
- `tokenizer_state` → Now we have to two mean values → `tensor([280.2986]), tensor([119.5490])`. The first one is the same as before, I don't know what the second one is.

## Forecasting with `CMAPSS`

I finally found where the `CMAPSS` (and also `CMAPSS-2`) datasets are stored in my system. They are stored in `~/.ceruleo/data`. There is a folder `CMAPSS` and `CMAPSS-2` with all the data we need. Essentially these files where automatically created and saved on the system at that path by the `obtain_raw_files` function that is called inside the `CMAPSS` and `CMAPSS-2` classes in the `ceruleo` library, that I used to create instances of those datasets. I can actually still use them since inside them there is all the code to load correctly and with all the columns organized the datasets as `pd.DataFrame`s.

Ok from the first forecasts I can say that probably fine tuning is needed. In fact, taking the sensor `SensorMeasure4` (which has an increasing trend) the model is able to capture the increasing trend in the median and with the 80% confidence interval it is able to more or less capture the variability of the signal but it does not capture all the osicillations we have in the signal. Actually for predicting the `RUL` this may be enough in any case but it seems to be the forecast that also another less complex forecasting model would have produced.

Maybe I have to try:
- [x] Try to see what happens on `SensorMeasure1` which is constant, I expect a constant prediction
    - Ok the forecast is constant (at a sligthly higher level than the real signal but by a very small amount), there is an high variability in the 80% confidence interval though.
- [x] Use a bigger version of `chronos`
    - Ok, using `chronos-bolt-base` there is already some zig zag in the median quantile.

## `CHRONOS` Regression Model 

Let me report here the results of the first tests on the [[chronos_pdm#implementation-ideas|following idea to make `CHRONOS` a regression model]] so that we can fine tune it to directly predict the `RUL`. 

So I realized that one possible approach to change the model head into a regression head is to load `CHRONOS` as a `AutoModelForSequenceClassification` model with `num_labels=1`, this should automatically replace the next token generation head of `T5` with a regression head. First of all I tried that out with `bert-base-uncased` following the example I saw in the 🤗 course:


```python
bert=AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=1)
bert_tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

prompts=["I love you", "I hate you", "I am neutral about you"]

encoding=bert_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

outputs = bert(**encoding)
predicted_class_idx = [outputs.logits[i].argmax(-1).item() for i in range(len(prompts))]
predicted_class = [bert.config.id2label[predicted_class_id] for predicted_class_id in predicted_class_idx]
```

In this case it works, in the sense that in `ouputs.logits` we have a single prediction for each prompt. Now these are not actually logits but the regression output of the model. I tried to do the same with `chronos`:


```python
chronos=AutoModelForSequenceClassification.from_pretrained("amazon/chronos-bolt-small",num_labels=1)
```

First thing first the model works 💪. In fact if I print out `chronos` it spits out the usual `pytorch` model architecture and in the last layer we can see:

```python
  (classification_head): T5ClassificationHead(
    (dense): Linear(in_features=512, out_features=512, bias=True)
    (dropout): Dropout(p=0.0, inplace=False)
    (out_proj): Linear(in_features=512, out_features=1, bias=True)
  )
)
```

Even though this is called `classification_head` we have a `FFN` + a `dropout` layers + a final `out_proj` that returns a single output that is exactly what we want.

Now to use the tokenizer we have to use the `ChronosTokenizer` class because using `AutoTokenzier` with the `chronos` checkpoint does not work, or at least it expects the `T5` tokenizer which however is probably not what we want because the main thing that changes between `CHRONOS` and any other language model is properly the tokenization step that makes this work with time series data.

Looking at how the tokenizer is used inside the `ChronosPipeline` class there are two main methods to call:

- `context_input_transform` → This method has more or less the same effect of the traditional tokenizers we use with 🤗. Given a batch of inputs it returns the `input_ids`, the `attention_mask` and the `scale`. The only different thing here is the scale which should depend on the Mean Scaling that is done in the tokenization step of `CHRONOS`.
- `output_transform` → This method converts the predicted `input_ids` (i.e. the forecasted `input_ids`) into the forecasted values → essentially here they de quantize (so they get the center of the bin corresponding to the `input_id`) and re scale the values to the original scale. It is here that the `scale` returned by `context_input_transform` is used.

Since the model will output directly the `RUL` (so it does not predict another token) we will just need the `context_input_transform` method since at the end we won't need to convert back from token into values.

>[!warning] Use `MeanScaleUniformBins` for the tokenizer
> The class `ChronosTokenizer` is the super class which contains all the method just with the docstring and then with `raise NotImplementedError` inside them. Then in classes inheriting `ChronosTokenizer` all the methods are implemented depending on the characteristics of that specific tokenizer. Here in `chronos.py` there is only `MeanScaleUniformBins` which is the tokenizer that does Mean Scaling and Uniform Quantization. They also discussed some possible other forms of scaling or quantizing the time serie in the paper but that is not implemented in the code. So we have to use this class and not `ChronosTokenizer` directly.

The `MeanScaleUniformBins` class requires some input arguments in the constructor:

- `low_limit` and `high_limit` → These are the limits used to created the uniformed spaced bins that are used tu quantize the time series. 
- `config` → This is an instance of the `ChronosConfig` object. I have to understand how to create this object. Maybe I can use the `AutoConfig` class from `transformers` that if I pass a model checkpoint will automatically loed the model config for me.

Ok so the `ChronosConfig` class is a `dataclass` (like the `ModelConfig` class I created for the `ad_mg` project) and it has a lot of input parameters that however do not have default values so we have to pass all of them. In the `train.py` experiments they used `yaml` files to store all the configurations so probably I will have to do the same. No ok here they do a huge call with all the parameters passed one by one, I will obtian those parameters from a `dict` I can obtain from a `yaml` file and I will try to use similar values for the config parameters as the ones that they set in the config files available in the repository.

Ok now it works 💪. In the sense that using the `forward` pass of the model I just created at the end we get a single value predicted by the model on `output.logits` → then it also returns a bunch of other very long outptus but we will not care about that.

Obviously here in the test that I did the returned output makes no sense because the new regression head was randomly initialized but theoretically doing fine tuning it should then be able to predict the `RUL` directly.

Now I will have to set up all the code to perform the fine tuning, I can take inspiration from the `train.py` script in the `chronos-forecasting` repository.

Before writing down the training script however I have to understand how to structure the `CMAPSS` dataset, see all the notes [[chronos-data|here]].
