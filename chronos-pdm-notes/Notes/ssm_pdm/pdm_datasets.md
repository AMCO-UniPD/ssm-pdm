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

- [Distribution Transformer](https://ieee-dataport.org/documents/data-driven-predictive-maintenance-distribution-transformers)
- [Ship Main Engine](https://ieee-dataport.org/open-access/predictive-maintenance-ships-main-engine-using-ai)
- [Crompton Greeves Three Phase Pump](https://ieee-dataport.org/documents/crompton-greeves-three-phase-pump-predictive-maintenance-dataset)
- [SCANIA Component X](https://researchdata.se/en/catalogue/dataset/2024-34)

In the next section I will take some more detailed notes on each one of the datasets.

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
