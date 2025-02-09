---
id: chronos_pdm
aliases: []
tags:
  - reasearch_idea
---

# `CHRONOS` for `PdM`

This is an idea that come up to me while I was searching papers in `PdM` for the `AIMS5` paper project.In particular I was considering the idea of adding in the literature review a subcategories about the usage of Transformer 🤖 like models in `PdM` since nowadays everyone is using them in every field.

Some time ago, I read the `CHRONOS` paper that is an easy way to create a `LLM` that works for time series forecasting. I was thinking at adding a citation to `CHRONOS` inside the paper since it is an `LLM` for time series and in `PdM` we work mainly with time series and here the idea arrived: What if we try to use `CHRONOS` for `PdM` and in particular for `RUL` estimation? 

>[!important]
> This idea may be exploited for a paper to present to the `J3C` Conference, whose General Chair is prof. Susto so we have very good chances to be accepted. Otherwise we can still go on on the `SSM_PDM` project which is already started.

## From time series forecasting to `RUL` estimation

The main obstacle to this idea at the moment is the following: `CHRONOS` is used for time series forecasting, while in `RUL` estimation we want to predict the remaining useful life of a machine, so two different things. So how to use `CHRONOS` for `RUL` estimation?

Two ideas come into my mind right now:

- In a lot of `PdM` papers they talk about using feature extraction to estimate the Health Indexes (HI) so some features that at each time step can quantitatively tell us the health state of the machine. This can be monotonic features that continues to increase (or decrease → degradation pattern) as the `RUL` decreases. If we can extract these indexes we can then perform time series forecasting on them and then estimate the `RUL` in function of a threshold that we can set. When the forecasted `HI` go over (or under if we are working with a degradation pattern) a certain value we raise the alarm signaling that maintenance has to be performed.
- Since `CHRONOS` is a language model and, through pre-training, it should have learned the language of time series I can simply fine-tune it on a `RUL` dataset (like the usual `CMAPSS`) to learn a new task, the `RUL` estimation task.
    - I asked at `DeepSeek` about this fine-tuning approach and actually it may be a little bit more complicated than I thought because it is probably necessary to modify the architecture of `CHRONOS`. Read carefully the answer [here](https://chat.deepseek.com/a/chat/s/e2672e89-8e66-4b97-998f-981da657a77b) 

## Ideas from `DeepSeek` thinking mode


I put `DeepSeek` in `DeepThink` mode and asked it about this idea and it produced a lot of interesting idea. Here are some of them:

- **Transfer learning across different equipment**: CHRONOS's ability to learn from diverse time series could be leveraged to transfer knowledge from one type of machinery to another, especially useful when data is scarce for new equipment.
    - This can be extremely important and interesting because in industrial application the problem is always the same: ok the model works on this machine, but what if we want to put it inside another machine? If the machine is somehow different we probably havew to re train from scratch.

- Wait, maybe a combination of a few ideas. For example, using CHRONOS for transfer learning across different machines, combined with explainability. Let's say, pre-training CHRONOS on a large corpus of time series data from various machines, then fine-tuning on a target machine with limited data. Additionally, using attention mechanisms to highlight which sensor signals are contributing most to the prediction, providing explanations.
    - The transfer learning thing is good and ok, explainability it seems a bit too much for the moment.

## Health Index Forecasting approach

The approach where we extract Health Indexes (`HI`) from the time series and then forecast them with `CHRONOS` seems to be the most promising at the moment. I asked for it to `DeepSeek` and it gave me some useful advices on how to extract these Health Indexes:
- It proposed the usual Feature Extraction methods 
    - Statistical features (Mean,std,skewness,kurtosis,...)
    - Frequency domain features (FFT, PSD, ...)
    - `PCA` and `ICA` for dimensionality reduction
    - An interesting thing it says is the following: Train a shallow model (e.g., linear regression, random forest) to predict RUL directly from raw sensor windows. Use the model’s output probabilities or confidence scores as an HI.

Looking more in details at some of the papers [[PdM_Survey#papers-with-feature-extraction|using feature extraction techniques]] we may also have some lucky situations in which some features are already naturally Health Indexes in the sense that they already have, also in the raw sensor measurement data, a monotonic behavior with respect to the `RUL`. So it's important to closely look at the raw data in the datasets we will use (i.e. probably `CMAPSS`) to see if there are already such kind of features.

>[!important]
> I have already worked in multiple projects with `CMAPSS` (i.e. `SSM_PDM`,`ACME_PDM`) and I have already did some Exploratory Data Analysis on it so it's better if I take look back at old notes to see if I wrote something about it. In fact I rember (from the `ACME` project) that there were some features with a clear decreasing trend (representing thus the machine's life degradation) but I do not remember if that happened because of the application of some feature extraction methods or if they were already present in the raw data.

## Implementation Ideas

- To implement the Feature Extraction part there should not be particular problems, we can use the tranformations already implemented inside `CeRULeO` 
- For what concerns `CHRONOS` it should be included in `HuggingFace` 🤗 so here I should try to look again at how to use `HuggingFace` models.

### `CHRONOS` on `HuggingFace`

Look at this [link](https://huggingface.co/amazon/chronos-t5-large) for how to use `CHRONOS` on `HuggingFace`. There is also an example on how to perform inference with `CHRONOS`. 

Interestingly there are also different sized of `CHRONOS` (in terms of number of parameters) so we do not have to use the large one if we do not have enough computational resources.


#### Idea: Change the `CHRONOS` architecture to adapt it to the `RUL` estimation task

This idea came to my mind while I was doing the 🤗 `NLP` course, in particular looking at how to [[hugging-face-🤗-tutorial-chapter-3#Training|fine-tune models]].

Here they say that if we load a model to fine tune it on a task that is different from the one it was trained on the `AutoModel*` class associated to that model will automatically change the model head to adapt it to the new task. Obviously the new head will be randomly initialized, but with the fine-tuning data we can update it to perform well on the new task.

Maybe following this reasonment we can change the head of the `CHRONOS` model (which will be the head of `T5`) to have a new head for the `PDM` task to predict the `RUL` of the machines. Here it depends if `CHRONOS` is recognized by one of the `AutoModel*` classes, maybe something like `AutoModelForTimeSeriesForecasting` or `AutoModelForTimeSeriesRegression`.

The head of `T5` is probably the typical head of a `text-generation` model, so it has a node for each token in the vocabulary and it outputs the probability of each token to be the next token. In order to do regression we need a Regression Head, which usually consists in a `FFN` layer and than a single node containing the `RUL` prediction. I don't think that there is something like that in a model like `T5` so we have to implement it by ourselves.

The alternative is to change the `CHRONOS` architecture and adapt it to the `PDM` task. We may use as backbone a model contained in 🤗 that supports some time series regression tasks. In fact, from the [[chronos|`CHRONOS` note]] we know that we can use any language model after having tokenized the time series. So I may have to find a `LLM` that works for time series regression and then use it as backbone for the `CHRONOS` model.

This would be more complicated because we have to enter inside the codebase of `CHRONOS` and work from there but may surely be more interesting and a more novel contribution than just applying the model as it is.

Another problem is that I don't know if the model is able to transfer its knowledge from a forecasting task to a regression task. This is something that has to be tested. Otherwise we have to pre train the model with `RUL` data and there are surely not enough data to do that.

>[!note]
> In fact the model checkpoints available in 🤗 are only the ones using `T5` as the backbone.

#### Update on this idea

I am now looking at the `CHRONOS` codebase on the `chronos-forecasting` repository and there is the [scripts](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts) section that is very well documented and explains well how to launch the scripts to fine tune and pre train the model. There are several interesting things:

- In the training script (they provide a bunch of shell commands so I will wrap them inside a `sh` script) it is possible to specify the `model-id` which is the identifier of the models contained in 🤗, so this means that:
    - If we pass one of the `chronos` model `ID`s and we also pass `random_init: false` in the `yaml` configuration file we will be doing fine tuning
    - If we pass a `random_init: true` we will be doing pre training
    - We can also pass the `model-id` of another 🤗 model to pre train on that model. This is very cool because it means that I can pre train on any language model I want.

Starting from this and following on the idea above I think that we may also be able to adapt `CHRONOS` to perform directly `RUL` estimation without having to perform strange passages to exploit forecasting to get the `RUL`. In fact from a theoretical point of view (as we saw in the Coursera course) **once I have a base model, I can fine tune it on a different task from its pre training task/objective** → this is in fact exactly what happens in any chatbot → these models are pre trained on next token prediciton and then they are fine tuned to do question answering.

There is a little difference (and potentially a complication) in my specific case → we **have to change the model head** from a `text-generation`/`forecasting` head to a Regression Head. Actually in this way we will also lose quantile and uncertainty estimation thing. Moreover I think I have to intervene in the code in order to perform this change of head of the model.

#### Implementation ideas

I am trying to think at exactly how to implement this. Essentially I think we have to create a new kind of model, taking inspiration from the `ChronosModel` class there is inside `chronos.py`. This class wraps a `PreTrainedModel` from 🤗 `transformers` (so here I can potentially use any model I want) and returns sample paths  for time series tokens → so essentially it should return us with the final emebedding of the model.

This thing I can keep as it is because this is what I need to load with `from_pretrained` because I need to use the pre trained model from `chronos`. What I need to change afterwards it's the model head that I add to do the predictions, which should be a simple set of `FFN` layers and then a single node as the output that will contain the `RUL` predictions. In the `chronos-forecasting` repo there is the `ChronosPipeline` that takes the `ChronosModel` and returns the forecasting, so maybe I need to add here the regression head. The problem is that I don't know weather it is actualy this pipeline that is then used in the `train.py` script to take the ouput at every epochs and compute the loss to do the backward gradient step.

>[!success] Maybe I have the solution
> Use `AutoModelForSequenceClassification` and pass `num_labels=1`. In this way the model head should be the usual `FFN` layer/s with a single node as output (since we set `num_labels=1`) and (as written in the docs) with `num_labels=1` the `forward` method returns a regression loss automatically.

#### Prompt for Gemini 2.0

Ok so since I am pretty stuck on how to do this model head change I will try to create a good prompt to pass to the new Gemini 2.0 (that is hosted for free on T3 Chat) to try to see weather it can help me. 

Here is the prompt:

I am working on a research idea I have about using the model presented in [Chronos:Learning the Language of time series](https://arxiv.org/abs/2403.07815) to the field of Predictive Maintenance, in particular in the Remaining Useful Life (`RUL`) estimation task.

Let me outline my idea:

`CHRONOS` is a very powerful time series forecasting model that essentially tokenizes time series in order to use them as inputs to a language model. It can be used with any one of the multiple language models that are coming out these days. In the paper however they pretrained it on a time series dataset they created and the used the `T5` language model as the backbone. In the paper they show how this model is very good in forecasting time series, in particular it has really high zero-shot forecasting performances. 

The `CHRONOS` checkpoints on `T5` are [available in 🤗 HuggingFace](https://huggingface.co/amazon/chronos-t5-large) in different sizes. 

So since in Predictive Maintenance and `RUL` estimation we are always using time series data (that are the raw measurements registered from sensors embedded in the machines we are studying) I thought that it may be a good idea to try to adapt `CHRONOS` to perform `RUL` estimation. 

There are however some passages to do because time series forecasting (the task in which `CHRONOS` was trained on and on which it is very good) is differenty from `RUL` estimation. In fact in `RUL` estimation we have a set of time series and we want to predict the remaining useful life of the machine, which is a scalar value → so this is a supervised learning regression task. 

What I was thinking is the following → In `CHRONOS` the model head is the same as the one of a text generation language model, so we have an output neuron for each token in the dictionary which contains the probability of that token to be the next token and then the next token is sampled from this distribution. In order to do regression we need a regression head, which is a simple feed forward neural network that takes the final embedding of the model and outputs a single scalar value. So my idea is to change the model head of `CHRONOS` to adapt it to the `RUL` estimation task. Obviously I want to keep all the pre trained weights of the model and only change the head, so that I can exploit all the knowledge embeeded in the `CHRONOS` weights throught pre training and then fine tune it on a `RUL` dataset so that it can transfer its knowledge to solve the `RUL` estimation task.

The problem is that implementing this is not very easy I think. I in fact started by cloning the [`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) repository (the official repository released by the `CHRONOS` authors).

Since `CHRONOS` is hosted in HugginFace I want to use the `transformers` API. Here from the HuggingFace documentation I know that I can use the `AutoModel*` classes to instantiate the correct model architecture just passing the model checkpoint and this should also work in case I pass a checkpoint which was not pre trained on the task I want to fine tune it on. For example I can do the following:

```python
from transformers import AutoModelForSequenceClassification
checkpoint="bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

Here I am passing the `bert-base-uncased` checkpoint (which is trained on Masked Language Modelling) to the `AutoModelForSequenceClassification` class where the task is Sequence Classification, which is a different downstream task that Masked Language Modelling and so what will happen in the background is that the model head will be changed in order to work with the new task and it will be randomly initialized. So here I also get a warning saying that the model head was changed and that I should fine tune the model on the new task in order to obtain decent results. 

So I was thinking that I can do the same with `CHRONOS` → I can load the `T5` model from the `chronos` checkpoints and then pass it to the `AutoModelForSequenceRegression` class so that it will change the `CHRONOS` head to a regression head (randomly initalized) that makes the model usable for a regression task such as `RUL` estimation.

The problem is 
