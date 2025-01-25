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

