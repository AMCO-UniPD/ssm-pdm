---
id: chronos
aliases: []
tags: []
---

# `CHRONOS` ⏰ - Learning the Language of Time Series

From a talk by Albert Gu on this differences between Transformer based approaches and `SSM` approaches he says that **Transformers 🤖 are good with discrete/tokenized data** (i.e. language where sentences are sequences of discrete elements (the tokens)). For this reason in the literature there are several approaches that simply **take different kind of data and tokenize them to bring them in a form that works well with Transformers** so that we can exploit the good performances of Transformers. The problem is that in some cases applying this tokenization process will lead to losing some potentially useful information. In `SSM`s instead, using `HiPPO` 🦛, we are able to somehow preserve the continuous time aspect of time series, as it is proven in the experiments performed. 

In any case one example of the application of this tokenization process in order to work with Transformer is the one of the `CHRONOS` ⏰ model which is a **foundation model for time series**. In very simple words they **pre process time series into sequences of tokens** and then use **pre trained Transformer based models to create a foundation model for Time Series** Forecasting. Their claim is that **language modelling and forecasting tasks should not be very different since in both cases we have sequence data** (i.e. in the case of language the tokens are part of a fixed size vocabulary while in time series the single time steps are usually coming from a continuous domain) and we want to predict the next token (i.e. in language modelling) or future tokens (i.e. in forecasting). As a conclusion, according to the authors of `CHRONOS`, a **good language model should also work on forecasting**. 

So at the end `CHRONOS` is defined as a **language modelling framework minimally adapted for time series forecasting**. 

`CHRONOS` tokenizes time series into **discrete bins using simple scaling and quantization of real values**. In this way we create a sort of *language of time series* and we can train language models on it. 

> [!important] 
> [`CHRONOS` code](https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file) 

# Notes from the ⏰ You Tube Video

> [!note] 
> The notes below are taken from this [You Tube video](https://www.youtube.com/watch?v=evf-q8s26wU&t=131s)  with the main ⏰ paper author Abdul Fatir Ansari. 

## Introduction

At the beginning Abdul classified Time Series Forecasting models into two main categories:

- **Statistical Local Model** → These are the classical Statistical learning models for time series (like the ones I saw in the course Business Economical and Social Data). These models are fitted on a single time series, so if I have a Multivariate Time Series I have to use multiple models (one for each time series/feature). Examples are `ETS` (Exponential Smoothing) and `ARIMA`. These models work well when we have limited data, like daily,monthly,quarterly time series, however they are very flexible because they impose some requirements on the kind of time series we can use. 
- **Deep Learning Models** → These are models trained on a large corpus of data and on a downstream task, like the ones I will use in this project. These models can obviously work on multiple time series at the same time. They are surely more flexible than local models (no requirements on the time series) but, as usual for Deep Learning models, they require a lot of data to work well. If we do not have a lot of data they can easily overfit.

Abdul also said that the benchmark datasets available in the literature are not very good, Deep Learning models overfit on them. So it is actually easy to develop new methods that maybe are able to sligthly improve on some benchmark but it is not a global improvement. 

`CHRONOS` is not the first attempt at leveraging `LLM`s for Time Series Forecasting, Abdul said there were several paper in `NeurIps` 2023 doing similar stuff:

- `LLM Time` → They used a `LLama 70B` or `GPT-3` model prompted with Time Series that were somehow converted into a string of numbers so that they can be processed by these `LLM`s. 
- `Time LLM, GPT4TS` → Take a backbone like `GPT-2` and use some specifically designed fine tuning scheme to adapt the weights for Time Series Forecasting. 
- Another approach, maybe not properly connected to `LLM`, trains large Transformer 🤖 based models on large corpus of time series data.

`CHRONOS` ⏰ is in the middle between these approaches in the sense that it uses a Language model but it also trains it from scratch on a lot of time series data. 

## The proposed method: `CHRONOS` ⏰

Abdul said that they wanted to do something **as simple and as lazy as possible**. In fact they **simply introduce this tokenization scheme where they transform Time Series into sequences of tokens**. They use **scaling and quantization so that the tokens**, like tokens in language models, **come from a fixed size vocabulary**. In this way the data can be passed as inputs to a `LLM`. They also **did not change anything to the Language model architecture** they used, so there are **no specific design choices on the architecture to adapt it to Time Series data**. 

Interestingly, they also did not even change the loss function (i.e. Cross Entropy). This may seem counter intuitive in fact the Cross Entropy loss is typically used in Classification tasks. Now we are using it for forecasting which is essentially Regression, but however it seems to work really well. There are other models that use a similar approach, like `WaveNet` → a generative model for audio signals. Also audio signals in fact are time series and they also discretize the audio signals in bins in order to exploit some generative models. 

### The Tokenization Scheme

Let's see how these huge time series are tokenized in order to work with Language models:

- **Scaling** → First of all the time series are put in a reasonable scale, using some normalization steps. This can be a design choice but in the paper they simply use a Mean Scaling → scale the time series by its absolute mean. Then the time series is squished into a reasonable range. In particular the scaling function is:

$$
	\tilde{x}_i = \frac{x_i}{\frac{1}{C} \ \sum_{i=1}^C |x_i|}
$$

- **Bins** → Take some bounds on the real line and construct uniformly spaced bins between these bounds that are used to quantize the time series. Then the indexes of the bins are used to represent the discrete tokens that will be passed in input to the language model. Obviously in this step we are losing some precision because obviously the bins have non 0 width → it's like when you use an histogram to represent the distribution of a continuous variable. 

More specifically we choose $B$ bin centers $c_1<\dots<c_B$ on the real line $\mathbb{R}$ and $B-1$ edges $b_i$ to separate the centers $c_i<b_i<c_{i+1}$ for $i \in \{1,\dots,B-1\}$. Given this division essentially for each point in the time series we look at which bins it belongs to and we assign to it the index of the bin. Then to de quantize the value (to obtain a time series in output when we do the forecast), given a bin index we return the center of that bin. More formally:

- Quantization function: $q: \mathbb{R} \rightarrow \{1,2,\dots,B\}$  

$$
	q(x) = 
	\begin{cases}
		1 & if \ - \infty \leq x < b_1 \\
		2 & if \ b_1 \leq x < b_2 \\
		\vdots \\
		B & if \ b_{B-1}\leq x < \infty \\
	\end{cases}
$$

- De Quantization function: $d: \{1,2,\dots,B\}  \rightarrow \mathbb{R}$ 

$$
	d(j) = c_j
$$
There are different strategies for the selection of the bin centers:

- **Uniform Binning** → The centers of the bins $c_i$ are selected uniformly in an interval $[l,r]$. 
- **Quantile Binning** → The Empirical Cumulative Distribution Function (`ECDF`) is exploited in order to choose the bin centers in such a way that each bins contains the same number of points. However in this case this is not a good choice because the data distributions in different datasets can vary a lot, so **they decided to go with the simple uniform binning**. 

However the problem of Uniform Binning is that it restricts the prediction range in the interval $[c_1,c_B]$, this makes it theoretically impossible for the model to model a time series with a strong trend (i.e. a time series that increases to $\infty$ or decreases to $-\infty$). 

Moreover they also introduced two special tokens:

- `PAD` → This is used to pad time series of different lengths to a fixed length in order to construct batches and replace missing values
- `EOS` → This is the typical `End of Sequence` token used in language models. It is not necessary for time series but it is more convenient to work with the implementations of language models. 

> [!note] 
> Normally in time series modelling it is a common practice to incorporate some time and frequency information such as `day-of-week` or `week-of-year`. In `CHRONOS` this is **ignored and the time series are treated just as sequences**.  

### Language Model 

At this point our time series have been converted into a form that can be used for any language model. In the paper they used the `T5` Encoder-Decoder mode but here **we can use any kind of language model**, so we can also use `Mamba` 🐍, `GPT` or any language model we want. 

> [!note] 
>  The only thing to change in the Language model is the size of the vocabulary $\mathcal{V}_{ts}$ created according to the [[CHRONOS ⏰ - Learning the Language of Time Series#The Tokenization Scheme|tokenization scheme]] which depends on the binning scheme used and it may be different from the original vocabulary size of the language model. If the size of the vocabulary changes as a consequence we modify the dimensions of the embeddings used by the model. 

They selected the `T5` model because it is very popular in the `NLP` literature until recently when all these Decoder only models are taking over. Moreover `T5` has a lot of different versions/configurations with different sized (in terms of number of parameters) and so it was easier for them to try with different sizes. On the other hand the more recent Decoder only models come only with very big sizes. 

### Probabilistic Forecasting

`CHRONOS` is a probabilistic model. In fact since it uses the Cross Entropy loss it means that in output it models the following predictive distribution:

$$
	p_{\theta} (z_{C+h+1} = i | z_{1:C+h})
$$

So the probability of predicting $i$ for sample $z_{C+h+1}$ given $z_{1:C+h}$ (the time series up to sample $C+h$). So to obtain the final forecast we can autoregressively sample from $p_{\theta} (z_{C+h+1} = i | z_{1:C+h})$ for $h \in \{1,2,\dots,H\}$ where $H$ is our forecasting horizon. 

- The result of this generation will be a path of token `ID`s (e.g. `IDs=[2400,2142,2282,2245,2310,...]`)
- We have now to convert this `ID`s into real values:
	- First we apply the de quantization function to all the tokens to obtain the centers of the bins 
	- Then we have to scale the centers back to the original size, multiplying all the values by the scale $s=\frac{1}{C} \sum_{i=1}^C |c_i|$ 
### Data Augmentation 

In the field of `LLM` at the end the solution to obtain high quality models is always the same and the more obvious one → **more high quality data**. The same obviously holds for Time Series Forecasting. Companies surely have high quality data but they are obviously not sharing them. In the public domain there are not so high quality data. However in the paper they tested their model on 42 datasets. It seems an high number but some of those datasets contained like 10 time series. At the end they collected 890,000 time series but there is an **unbalance between the represented domains** → some domains are over represented, some others are under represented. For example there are a lot of time series about Wikipedia page visits time series or weather forecast time series. 

In any case this is surely much less data than the datasets that are used to pre train `LLM` (which are essentially the entire internet). 

How many tokens did they use? To compare to the terminology used in `LLM` (where they say how big the vocabulary they used is) → they used 84 billion tokens. 

For the reasons explained above they had to introduce some Data Augmentation scheme to improve the quantity and quality of the data in order to improve the model performances even further. 

They considered two Augmentation schemes:

1. **Time Series Mixup** - `TSMixup` → Take some random real world Time Series from different dataset (or also from the same dataset) and generate new synthetic time series taking **convex combinations** of them. In this way we are able to diversify the patterns that the model is able to see. So if we have time series with a trend in one dataset and time series with seasonality in another dataset combining them with a convex combination we obtain a new set of time series that present a combination of these two patterns. 

More formally in `TSMixup` we randomly sample $k \sim \mathcal{U}\{1,\mathcal{K}\}$ time series of specific length $l \sim \mathcal{U}\{l_{min},l_{max}\}$ from the training set, we scale them and we take their convex combination:

$$
	\tilde{x}_{1:l}^{\text{TSMixup}} = \sum_{i=1}^k \lambda_i \tilde{x}_{1:l}^{(i)} 
$$

where $\tilde{x}_{1:l}^{(i)}$ is the $i^{th}$ scaled time series and the combination weights $[\lambda_1,\dots,\lambda_k]$ are samples from a Dirichlet distribution $\text{Dir}(\alpha)$  . Specifically in the paper they applied the `TSMixup` algorithm with:

- $K=3$
- $\alpha=1.5$
- $l_{min}=128$, here I can choose the minimum length of the 15 acquisitions I have → 11404
- $l_{max} = 2048$, here I can choose the maximum length of the 15 acquisitions I have → 30352

1. Use **Gaussian Processes to generate new synthetic data** - `KernelSynth` → They used a bank of Gaussian Process kernels representing base functions (e.g. linear kernel for linear function, periodic kernel for periodic functions, $\dots$). Starting from these kernels they combine them with addition or multiplication to obtain new more complex kernels and use them to generate new time series. 

> [!example] 
> - We can start combining two **Linear kernels with multiplication** to obtain a **quadratic kernel** 
> - Then we can **add a Periodic kernel** to obtain time series with a **quadratic behavior and also seasonality** 

There are a lot of possible combinations and this makes us able to generate a lot of different kind of time series with different complex behaviors. 

This Data Augmentation step was crucial to make the model very good in 0-shot performance, so how well the model behaves on unseen data. 

The final training dataset contained 90% of time series coming from real augmentation (point 1), the remaining 10% from Gaussian Processes synthetic augmentation (point 2). In the paper they showed that this 90/10 split is the best one. They also tried to train models with different split percentages and they also tried to train only with data coming from the synthetic generation process and the resulting model was reasonably good → obviously it was not as good as the models trained both on real and synthetic data but it was better than some base models. 

- 90 % → Time Series Mixup, so these are coming from publicly available datasets. There is a parameter `k` that tells us how many time series we want to take from each dataset. If `k=1` we are taking the original time series.
- 10 % → Purely synthetic data originated from the Gaussian Processes 

This synthetic time series generation process is easier to do in the field of time series rather in Language modelling. In fact to generate synthetic data for `LLM` we need to train a huge model and sample from it the synthetic data (like sampling answers from conversations with `ChatGPT`). Here instead we just need to use these Gaussian Processes based scheme which is surely less expensive and time consuming. 
### Evaluation 

They used some benchmark dataset for Time Series:

- Monach Benchmark 
- Electricity Datasets
- ETT datasets

For the evaluation they choose a splitting point in the time series, separating the past from the future → on the left the data seen by the model, on the right the data the model has to predict.  In datasets coming from competitions they used the same splitting point of the competition in other datasets they choose a new splitting point on their own. 

### Results 

They obtained very promising results on the Zero Shot Forecasting benchmark. In particular `CHRONOS` performed on par (i.e. similar performances) to models that were specifically trained on a specific time series dataset while `CHRONOS` has never seen those time series in its training process. So this shows very good Zero Shot forecasting capabilities. 

Moreover `CHRONOS`  is much better than pre-trained models like `Lag LLama, Time LLM`, so other models that try to exploit Language models for time series modelling. 

### How to use the model - Zero Shot vs In Domain 

Two ways of using the model:

1. **Zero Shot Forecasting** → Like an `LLM`. Here we have our pre-trained model and we just pass in input to it a segment of a time series ans we obtain the forecast of future values. 
2. **In Domain** → We pass in input a time series of a domain the model has already seen in training. 

The model will work the same weather the data are coming from the training set or not, they just made this distinction for evaluation. 

### Performances across different patterns 

Here there is not a very good information for me → the model does not perform very well on Sparse data with spikes → that are more or less the kind of time series I have to deal with in the [[Analysis Sprayer 🦅 Data|Sprayers data 🦅]] 😢. In particular Abdul talks about time series that are 0 most of them time and then suddenly present some spikes (i.e. the `Break` acquisition).  The reason why these patterns are not well captured is because of the [[CHRONOS ⏰ - Learning the Language of Time Series#The Tokenization Scheme|tokenization scheme]] used. In fact using Mean Scaling we are smoothing out the affect of sudden peaks and we lose them when dividing the time series into bins. 

>[!note]
> The Mean Scalinfg approach is not very robust against outliers (i.e. sudden spikes) because the mean is very sensitive to outliers, but we may try to use other statistical summaries like the Median which are less sensible to outliers.

However this depends on the sparsity of the spikes: 

- If we have frequently occurring spikes, like the ones we can see in the `Track` and `Chicane` 🦃 acquisitions, we are able to represent them well → this is a good news 🎊🍾 
- If the spikes are sparse, like in the `Break` acquisitions, instead it's more difficult to represent them.

On the other hand they showed how there are a lot of patterns were the model is very good also in the Zero Shot performances. In particular it works well in recognizing new seasonal patterns never seen in training. 

> [!question] Seasonality in Sprayers 🦅 Data ? 
>  I wouldn't say that there are seasonality trends in the [[Analysis Sprayer 🦅 Data|acquisition data]] I have right now but seasonality may appear when we look at the data registered by the sensors when the machine is working in the field. For example if the Sprayer 🦅 is going up and down the field it will perform an `U turn` more or less periodically to go back when the field ends. 




