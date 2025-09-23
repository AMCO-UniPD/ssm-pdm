---
id: rul_healing
aliases: []
tags:
  - ideas
---

# `RUL` Healing Problem

In this note I will report some ideas on the `RUL` Healing problem that Lucas
Brito posed me at the end of the `J3C` conference. This is a problem that
usually comes when we deal with real datasets and real-world industrial
scenarios and so I never encountered/considered it since up to now I worked
mainly with synthetic datasets like `CMAPSS` where the `RUL` or the Health
Index is considered to be a simple monotonically decreasing line.

Lucas Brito told me about a problem he is having in RUL estimation and that can
be an interesting future research direction to tackle.

This problem happens usually in real data when we have the `RUL` or the health
index (here he talked about the `RMS` so probably he is using some feature
extraction methods to find and Health Index and uses it as the `RUL` I guess
(something that I never did or considered up to now).

>[!note] Magic sensor that measures the `RUL`
> In this setting what Lucas is using as the `RUL` is like a magic sensor that
> directly spits out the `RUL` of the equipment (in reality it's a sensor (or
a feature extracted from raw sensor measurements) which is directly correlated
with the `RUL`) and we want to predict its trend because we cannot directly
> install it on the machine.

>[!question] How is this magic sensor obtained?
> How are the Health Indexes computed? I need to ask Lucas to give me
> some examples of that to see weather I understand correctly. For me
> in this approach the `RUL` may be considered as, for example, the
> oil level in a component (if it goes over or under a certain threshold
> then we have to do maintenance) or the `RMS,curtosis,skewness,...` or
> some other feature we can extract from the raw data to characterize
> the degradation process of the machine's life. Does that make sense?

>[!example] Battery Capacity Degradation
> For example in lithium-ion batteries the Health Index is represented
> by the **battery capacity degradation** which is a signal that we
> have access to and that shows an overall decreasing trend.

The problem is that maybe during the condition monitoring of the equipment
**some operators perform maintenance on other parts of the machine** to solve
some other minor issues and this creates a strange "healing" effect on the
machine (maybe is something similar to the Healing effect we have in the
batteries datasets for RUL? Maybe I can look at one of the papers that does
that and see how they deal with this problem), so **the RUL momentarily
increases** and that makes the statistical methods Lucas usually uses (Weibull
Distribution) to "go crazy" and mess up the whole prediction.

## Quick Ideas to solve the problem

Below I will list some quick ideas I have to solve or deal with this problem:

1. If we know that we have this specific issue maybe we can add an additional
   model (e.g. a time series classification model?) to the picture which is
entitled to detect when this behavior happens? However once we have detected
this strange behavior what can we do? Should we reduce the `RUL` estimation in
some way?
2. Maybe the models that Lucas is using (based on the Weibull distribution for
   example which is a typical thing that mechanical engineers do) are very
sensible to this unexpected behavior but in data driven models like the ones I
am using maybe we **just need to add some samples in training** which exhibit this
strange behavior and hope that the model is able to recognize it in the test
set when something similar happens? Obviously this should not make the model to
always predict this healing effect also when it is not there. In any case if
there is a clear difference in the shape of the signals in the extracted
feature is easier for the model to learn this distinction.

### `RUL` Healing in lithium-ion batteries

I have already heard about regeneration phenomena in the field of `RUL`
estimation on lithium-ion batteries. In particular this is what I wrote in
`PdM_Survey.md` about a paper that deals with this challenge:

>[!info] [*Physics-informed deep learning with multi-resolution for ensemble
>prediction of lithium-ion battery health
>status*](https://iopscience.iop.org/article/10.1088/1361-6501/ada849/meta) →
>In the field of Lithium-ion batteries this method considers also the
>**Capacity Regeneration Phenomena** (`CRP`), so this is a peculiar case in
>which there is not only degradation but at some point the equipment can start
>to regenerate its life? In order to deal with this challenge they use a
>**physics-informed** `DL` which is used to **detect global and local degradation
>features**. The **global ones are used to predict the `RUL` and to provide
>interpretability** in the results while the **local ones can be useful to detect
>the `CRP` phenomena**. Finally they use an ensamble of both global and local
>features.

#### Physics-informed deep learning with multi-resolution for ensemble prediction of lithium-ion battery health status*

Here I report some notes I get while reading the paper to get some potential ideas on how to solve the `RUL` healing problem.

The simplest approach used in the literature is to smooth the `RUL` signal
(i.e. the battery capacity degradation signal) to remove the oscillations
produced by `CRP`. This may help in creating a signal which is more
monotonically decreasing (which makes the `RUL` estimation easier) but can have
negative effects on the accuracy of this estimate → we can have high deviation
in the prediction which causes a wrong `RUL` estimate.

##### Proposed Method

To solve the problem in the paper they propose a **multi-resolution
decomposition** of the battery capacity degradation data (i.e. the `RUL`
signal) → so, somehow similarly to [[Notes/rul_healing#Idea by
Francesco|Francesco's Idea]], we are **decomposing the `RUL` signal**.

The `RUL` is decomposed into: **trend**, **seasonal** and **residual** components, where:

- **Trend** → overall (**global**) degradation trend
- **Seasonal and Residual** → together they indicate the `CRP` phenomena

Some more technical details:
- Smoothing method → robust locally weigthed regression, where locally weigthed scatterplot smoothing is fitted with a local polynomial regression. This is a method to smooth a scatterplot (i.e. so a set of 2 features?) → looking at the complicated explanation this should be something similar to `loess` regression, so a different polynomial is fitted in different segments of the `x` axis.
- After the smoothing process (which I guess is just a pre processing step) we can proceed with the decomposition of the battery capacity time series which we call $Y_v$ which is decomposed into trend $T_v$, seasonality $S_v$ and residual $R_v$ such that:

$$
    Y_v = T_v + S_v + R_v
$$

>[!info]
> See pag 5 of the paper for all the details on the degradation algorithm.

The result of this process is that starting from a single battery capacity
signal we obtain 3 signals: the trend, seasonal and residual components. At
pag. 6 of the paper there is an example on battery `#5`. For all the batteries
they considered the correlation coefficient between each one of the three
components and the original signal is computed showing how that is always very
close to 1 for the trend component, which is for this reason the most important
variable to predict in order to have a good estimation of the battery capacity.

>[!note]
> So essentially to have a good `RUL` estimate we have to estimate well the degradation
> trend, the `CPR` (which is contained in the seasonal and residual component) is not the
> main effect but it's probably still important to have a precise prediction.

The trend component was estimated with a Gaussian fitting function → here it's
classical statistical learning → we use the data to estimate the value of the
Gaussian's parameters, we plug that into the function and we got our battery
capacity estimate.

Successively they show how fitting the trend with the Gaussian function
produces good results (except for battery `#18` were probably the `CPR`
phenomena is more pronounced) and this function is better than the exponential
and Fourier functions whose results are contained in the Appendix (they are
worse than the results obtained with the Gaussian).

Here we have the important point → with the Gaussian function we have
**physically modelled the battery capacity** → now we can insert this modelling
as a physical constraint in the deep learning model (i.e. `Bi-GRU`) loss
function. This is essentially a form of regularization which should push the
model to produce more feasible solutions. The final loss function is, in fact,
the following:

$$
    L = \text{Adam} \ L_{\text{data}}(y,\hat{y}) + \psi \ L_{\text{physics}}(p,\hat{p})
$$

where:
- $L_{\text{data}}$ is the training loss obtained by the `Bi-GRU` model.
- $L_{\text{physics}}$ is the physical loss (i.e. loss on predicting the trend with the Gaussian function) and $\psi$ is a trade-off parameter to regulate the importance to minimize the physical or empirical loss.

Firstly the model is used to predict the trend, seasonal and residual components, whose predictions are summed together to obtain the final prediction. In any case the complete pipeline of the proposed method is outlined in Fig. 5 of the paper. The pipeline is the following:

1. Extract the battery capacity degradation from the raw data
2. Multi-resolution decomposition → decompose battery capacity into trend, seasonal and residual components
3. We set the hyperparameters of the `Bi-GRU` and we use it to estimate the seasonal and residual components (which are more difficult to estimate using a physical model). For the trend prediction we use the Gaussian function and we train the model using the physics-informed loss function to predict the trend.
4. Finally the three predictions of trend, seasonal and residual are summed together to obtain the final prediction.

>[!important]
> At the end the approach presented in this paper is very similar to the typical approach used in hybrid-based `RUL` estimation. The quantities which are easy to estimate through model-based approaches (i.e. the trend component in this case) is estimated through a physical-model, while the other non-linear effects (i.e. the seasonal and residual components (so the `CPR` in this case)) are estimated through a data-driven method.

>[!question] How to take inspiration from this approach for our `RUL` Healing problem?
> Maybe also in our case we can try to build a sort of physics-based `NN` adding a regularization
> term to the loss function that encapsulated some physical or domain knowledge on the data we have?
> Lucas can give us this background knowledge to help us figure out how to define this $L_{\text{physics}}$ term?
> The issue is that in this case we do not have a physical and natural process that causes the healing effect but it's
> rather an external human intervention (it's unlikely that we will have a seasonal component in the data). How can we deal with that?

### Idea by me: Composition of `FD` + `RUL`

Use a `FD` model that works in parallel to the `RUL` model and the final `RUL`
prediction is the result of a combination (i.e. a sum, a product) of the `RUL`
model and the `FD` model predictions. For example if a certain fault is detected by the
`FD` model we can increase the `RUL` somehow? In this case we have to assume that whenever a
fault is detected someone will go and fix that (and so the `RUL` healing effect is triggered).

### Idea by Francesco

Is it possible to divide the `RUL` by the `RUL` of different components? So the
final `RUL` is a certain combination of the `RUL` of different
components/faults → so for example $RUL_{tot} = RUL_{\text{component_1}} \times
RUL_{\text{component}_2} * \dots * RUL_{component_{n}}$. If we use this idea
then the statistical distribution used to predict the `RUL` (the Weibull
Distribution) should be considered as the combination of the different Weibull
distributions of the different components. At this point however it may become
a bit complex to start working with composition of probability distributions.

The problem with this idea is that we have to find out what are the sensor/s
that are responsible for a certain fault, it may be that multiple sensors are
responsible for a certain fault or a single sensor may be responsible for
multiple faults so determine which ones are connected with which may be a bit
of a mess. If we are able to do this clustering we can then predict the `RUL`
singularly for each one of them and then do this thing of composing the single
components into a single total `RUL`.

### Fault Detection + `RUL` Idea

Can we use Fault Detection to do that? If we use a `XAI` model for Fault
Detection (for example `ACME` that should work also for classification models)
we can see what are the most important features (so sensors) to predict each
different fault. We have to understand weather we have also labels on the
faults but I think that this may be the case because Lucas is using this
approach of doing all toghether: `AD`, Fault Detection and then also `RUL`
estimation. Here hopefully we can achieve a result from `ACME` that identifies
not super overlapped groups of features for each different fault

Possible pipeline:

1. We start from raw data from the sensors
2. Apply some feature engineering transformations to get the health indexes
   (`RMS`, skewness, kurtosis, ...)
3. Do Fault Detection with some `FD` model (i.e. usual `S4` adapted to classification or similar?)
4. Interpret the results with a `XAI` model (e.g. `ACME`) → is it possible to
obtain the most important features for `Fault 1` for `Fault 2` ecc ecc? I don't
know if it is possible to do that on a global level, maybe it's something that
we can do just at the local level.
    - Maybe to find the most important variables for `Fault 1` we can create a
    sort of `AD` dataset where `Fault 1` are the anomalies and all the other
    data are inliers and we can also use `ExIFFI` here to find the most
    important features, then we can do the same with all the other faults → this
    does not make a lot of sense because `EIF` does not need/use labels.
5. The result of Step 4 should be a sort of division of the sensors into
different groups → groups of sensors giving info for the prediction of `Fault
1`, for the prediction of `Fault 2`, ecc ...
6. Train a different `RUL` prediction model for each different group of
sensors?

This idea is better summarized in [[Excalidraw/rul_healing|the `rul_healing` Excalidraw sketch]].



