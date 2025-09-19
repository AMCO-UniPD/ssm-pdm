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

This problem happens usually in real data when we have the RUL or the health
index (here he talked about the RMS so probably he is using some feature
extraction methods to find and Health Index and uses it as the RUL I guess
(something that I never did or considered up to now). The problem is that maybe
during the condition monitoring of the equipment some operators perform
maintenance on other parts of the machine to solve some other minor issues and
this creates a strange "healing" effect on the machine (maybe is something
similar to the Healing effect we have in the batteries datasets for RUL? Maybe
I can look at one of the papers that does that and see how they deal with this
problem), so the RUL momentarily increases and that makes the statistical
methods Lucas usually uses (Weibull Distribution) to "go crazy" and mess up the
whole prediction. 

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
am using maybe we just need to add some samples in training which exhibit this
strange behavior and hope that the model is able to recognize it in the test
set when something similar happens? Obviously this should not make the model to
always predict this healing effect also when it is not there. In any case if
there is a clear difference in the shape of the signals in the extracted
feature is easier for the model to learn this distinction.

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

- Can we use Fault Detection to do that? If we use a `XAI` model for Fault
Detection (for example `ACME` that should work also for classification models)
we can see what are the most important features (so sensors) to predict each
different fault. We have to understand weather we have also labels on the
faults but I think that may be the case because Lucas is using this approach of
doing all toghether: `AD`, Fault Detection and then also `RUL` estimation. Here
hopefully we can achieve a result from `ACME` that identifies not super
overlapped groups of features for each different fault

Possible pipeline:

1. We start from raw data from the sensors
2. Apply some feature engineering transformations to get the health indexes
   (`RMS`, skewness, kurtosis, ...)
3. Do Fault Detection with some `FD` model
4. Interpret the results with a `XAI` model (e.g. `ACME`) → is it possible to
obtain the most important features for `Fault 1` for `Fault 2` ecc ecc? I don't
know if it is possible to do that on a global level, maybe it's something that
we can do just at the local level.
    - Maybe to find the most important variables for `Fault 1` we can create a
    sort of `AD` dataset where `Fault 1` are the anomalies and all the other
    data are inliers and we can also use `ExIFFI` here to find the most
    important features, then we can do the same with all the other faults.
5. The result of Step 4 should be a sort of division of the sensors into
different groups → groups of sensors giving info for the prediction of `Fault
1`, for the prediction of `Fault 2`, ecc ...
6. Train a different `RUL` prediction model for each different group of
sensors?

