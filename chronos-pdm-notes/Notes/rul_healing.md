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

