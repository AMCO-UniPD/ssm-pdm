---
id: pdm_datasets
aliases: []
tags:
  - ssm_pdm
  - chronos-pdm
  - data
---

# New `PdM` Datasets

I found some new `PdM` datasets in the [`IEEE
DataPort`](https://ieee-dataport.org/) platform. It may be interesting to check
them out for a potential `industrial` version of the `sss_pdm` paper. In fact
`CMAPSS` at the end it's a synthetic dataset so proving the efficiency and
precision of the Quantile Regression + `SSM` model on some real-world data may
be a good contribution.

The datasets I found are:

- [Distribution
Transformer](https://ieee-dataport.org/documents/data-driven-predictive-maintenance-distribution-transformers)
→ subscription required
- [Ship Main
Engine](https://ieee-dataport.org/open-access/predictive-maintenance-ships-main-engine-using-ai)
- [Crompton Greeves Three Phase
Pump](https://ieee-dataport.org/documents/crompton-greeves-three-phase-pump-predictive-maintenance-dataset)
→ subscription required
- [SCANIA Component X](https://researchdata.se/en/catalogue/dataset/2024-34)
- [`PHM` Data
Challenge](https://data.phmsociety.org/phm2024-conference-data-challenge/) →
dataset used in the `RUL_CL_KAN` paper.

In the next section I will take some more detailed notes on each one of the
datasets.

---
## Distribution Transformer

The dataset was used in [this
article](https://www.researchgate.net/publication/330491538_Data_Driven_Predictive_Maintenance_of_Distribution_Transformers).
This is a pretty old article (2018) and in fact the methods used to perform
`PdM` are quite archaic → Random Forest. In fact here `PdM` is treated as a
classification task. The proposed framework consists in predicting weather a
certain distribution transformer (i.e. here transformer is not the attention
based model) will fail or not in a certain time horizon. In this dataset in
fact we have the `0` and `1` labels typical of Binary classification datasets.

>[!note] Contribution Idea
> Here we can convert the dataset into a `RUL` dataset.
> If the data are in a time series format we can use the simple Linear Degradation Model and manually add the `RUL` as a new target variable.

Two possible ways to work on this dataset:
- Classification version → keep the dataset as it is and use some more advanced
classification models (i.e. convert `SSM,LSTM,LSTM-CNN`) for classification
- Regression version → As said in the `Contribution Idea` above convert the
dataset to be used for traditional `RUL` estimation and employ the usual
models.

---
## Ship Main Engine

There are not many information about the dataset inside `IEEE DataPort` but
essentially it contains Ship Engine monitoring data, so these should be time
series and converting it into a `RUL` dataset should be possible.

>[!note] Contribution Idea
> As for the Distribution Transformer dataset here I need to carefully check the structure
> of the dataset to see weather it is usable for `RUL` estimation.

---
## Crompton Greeves Three Phase Pump

This is a synthetic dataset actually (generated with `GPT-4o` logic apparently)
but it seems very interesting since there are 26 sensor streams and there is a
detailed description of what each one of them represents. In this case there
are a lot of mechanical terms and they suggest to use extract features and so
on but as usual I think I may start trying to feed everything inside the `SSM`
and make it do all the dirty work 😎.

>[!note] Contribution Idea
> From the description the format of the dataset seems very similar to the `CMAPSS` one,
> so we can directly start with the `RUL` estimation.

---
## `SCANIA` Component X

This dataset is from the `SCANIA` Swedish company and it contains multivariate
time series (exactly what we need for `PdM`) about an anonymous component (for
this reason is called Component X) of a fleet of trucks from `SCANIA`.

>[!note] Contribution Idea
> As for the other datasets we have to convert it into a `RUL` estimation dataset.
> From a quick look at the dataset page there are a lot of different files to download. I have to carefully check them all to understand what are the data we need.

---
# Project first ideas

Let's gather here some first ideas to turn this into a potential thesis/paper
project. It's similar to the `ind_ceruleo` paper (see the `ind_ceruleo`
citation in the `aims5_paper` project). Also in that case they benchmarked some
`DL` based `RUL` approaches on some `PdM` datasets.

- In this case the new thing is the usage of `SSM` models which were not used
there and that are an approach not used a lot in the literature.
- Look better at `CMAPSS` to see the label for the different operating
conditions and exploit that to do the normalization on different subsets using
different means and variances as done in othe papers used in the reviews?
- Maybe we can also add the Quantile Regression thing for uncertainty
estimation?
- Can we compare it to the Bayesian approach used in `uncertainty_bayesian`
paper? Maybe if for both these models we are able to get a confidence interval
we can compare them and see which one is the better one?
- Train `SSM` with a Bayesian method to obtain Bayesian `SSM` and compare it to
Quantile Regression?

---
## Models to use

Some first ideas on the models to use for the benchmark:

- `SSM` based models
    - `S4`
    - `S5`
    - `S4D`
- `LSTM`
- `CNN`
- `LSTM-CNN` → this is the architecture used essentially everywhere in the
literature and I want to try it out → We can implement the simple combination
of the two taking inspiration from some of the multiple papers using it.
- `TCN`

