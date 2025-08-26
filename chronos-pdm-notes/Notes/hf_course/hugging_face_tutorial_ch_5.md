---
id: hugging_face_tutorial_ch_5
aliases: []
tags: []
---

# Hugging Face 🤗 Tutorial Chapter 5

In this note we will take on Chapter 5 of the `HuggingFace` 🤗 `NLP` course, which is about the `Datasets` library. This chapter , I would say, it's very important, in particular for what concerns the `chronos-pdm` project since it let us learn how to work with datasets which are not in the `HuggingFace` dataset Hub. This is exactly the case of the `chronos-pdm` project since (I already checked) the `C-MAPSS` dataset is not present in the Hub and I am pretty sure that any other Predictive Maintenance related dataset is not present here. That's because `PdM` is not very related to `NLP` in general. In any case the chapter should also have a section regarding how to upload a dataset to the Hub, so maybe we can also upload `C-MAPSS` or other datasets we may use to the Hub.

>[!info]
> Reference to the [Chapter 5 on the `HuggingFace` website](https://huggingface.co/learn/nlp-course/chapter5/1)

## What if my dataset isn't on the Hub?

We start immediately with the topic I am most interested in. How to load datasets that rather than being on the Hub are stored locally on out laptop or on a remote server.

### Working with local and remote datasets

Actually at the end there is not a very fancy and complicated way of doing this thing. Essentially we will still use the `load_dataset` method from the 🤗 `datasets` library but we will have to pass some different input arguments to point to the path where the file containing the dataset lives in our file system or the `URL` to the dataset.

Different kind of file types are supported by the `load_dataset` function, as reported in this table:

| **Data format** | **Loading script** | **Example** |
| --- | --- | --- |
| CSV & TSV | `csv` |  `load_dataset("csv",data_files="myfile.csv")` |
| Text files | `text` | `load_dataset("text", data_files='my_file.txt')` |
| JSON & JSON Lines | `json` | `load_dataset("json", data_files='my_file.jsonl')` |
| Pickled DataFrames | `pandas` | `load_dataset("pandas", data_files='my_dataframe.pkl')` |

So essentially we have to pass the file type as the first argument and then the path to the file as the `data_files` argument, which, as we will see in a bit, may also contain the paths to multiple files.

### Loading a local dataset

In the course they consider as an example dataset the [SQuAD-it dataset](https://github.com/crux82/squad-it/) which is hosted in `GitHub` and it's a dataset for Question Answering in Italian.

>[!note]
> Now I will use this dataset in the commands I will write down here but then I will try to load another datast contained in my local filesystem to see if it works. I will use this `SQuAD-it` dataset as the example for the remote dataset loading.

Since the dataset is hosted on `GitHub` we will have to download it in our system first. This can be easily done with the `wget` command:


```zsh
wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
```

This will download two compressed `json.gz` files in our system, we can decompress them with `gzip`:

```zsh
!gzip -dkv SQuAD_it-*.json.gz
```

Now we have the two files `SQuAD-it-train.json` and `SQuAD-it-test.json` in our system. We can load them with the `load_dataset` function.

Since we have a `json` file we have to understand weather that is a regular `json` format (which is a nested dictionary (dictionaries inside a dictionary)) or weather it is in the `json` Lines format (line-separated `json`). As it normally happens in Question Answering dataset this is in the `json` format and all the text is contained in the `data` field. So we will load it in the following way:

```python
from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
```

By default loading local files will create a `DatasetDict` object with a `train` split. In fact if we inspect the variable `squad_it_dataset` we will see:


```python
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
})
```

as usual it shows us te features in the `features` attribute of the `Dataset` object and the number of rows in the `num_rows` attribute. We can access one of the rows to see how a sample is structured. This is done doing `squad_it_dataset["train"][0]`:


```python
{
    "title": "Terremoto del Sichuan del 2008",
    "paragraphs": [
        {
            "context": "Il terremoto del Sichuan del 2008 o il terremoto...",
            "qas": [
                {
                    "answers": [{"answer_start": 29, "text": "2008"}],
                    "id": "56cdca7862d2951400fa6826",
                    "question": "In quale anno si è verificato il terremoto nel Sichuan?",
                },
                ...
            ],
        },
        ...
    ],
}
```

Actually the structure is not straight forward to understand, however in the `README` on the `GitHub` repo everything is well explained.

Ok great we have our dataset, but we would like to have a single `DatasetDict` object containing both the `train` and `test` split, which however are in two different files. No problem! We can pass the `data_files` argument as a dictionary that maps each one of the two files we have to its split name:


```python
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
squad_it_dataset
```

Now we have:


```python
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
    test: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 48
    })
})
```

>[!note]
> The `data_files` argument is very flexible. We can pass a single path, a list of paths, a dictionary as did above. We can also pass glob files matching a specified pattern according to the rules used in the Unix shell. For example we can pass all the `json` files in a directory inside a single split by passing `data_files=*.json`. More details on the 🤗 `Datasets` [documentation](https://huggingface.co/docs/datasets/loading#local-and-remote-files)

Actually, as it happens also with `pd.read_csv`, the `load_dataset` method supports auto decompression of files so we could have passed directly the path to the compressed `json` files without having to decompress them first:


```python
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

This is very handy in case we have very huge dataset files and decompressing them would take a lot of time and space on our local machine.

### Loading my local dataset

Now I will try to load a dataset that I have already locally saved in my system. I will use one of the many datasets I used in the `ExIFFI` paper, which are stored as `csv` files. In particular I will load the `diabetes.csv` file.

So I saved the `diabetes.csv` file in the `data` directory of the `chronos-pdm` project. I will load it in the following way:


```python
from datasets import load_dataset

# Loading a local dataset
datapath=os.path.join(os.getcwd(),'data','diabetes.csv')
dataset = load_dataset('csv', data_files=datapath)
```

Now `dataset` contains:

```python

DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'age', 'bmi', 'HbA1c_level', 'blood_gluc
ose_level', 'Target'],
        num_rows: 100000
    })
})
```

If I inspect a sample doing `dataset["train"][0] I get:

```python

{'Unnamed: 0': 0,
 'age': 80.0,
 'bmi': 25.19,
 'HbA1c_level': 6.6,
 'blood_glucose_level': 140,
 'Target': 0}
```

Considering that Anomaly Detection is mainly on unsupervised task we are happy having just the `train` split. However I am wondering: if instead of having one file for the `train` split and one for the `test` one I have everything together in a single file, is there a function like `train_test_split` in `sklearn` to create the split in the `DatasetDict` object?

>[!success] There is `train_test_split`
> Looking at the 🤗 `Datasets` documentation I found the [`train_test_split` function](https://huggingface.co/docs/datasets/process)

### Loading a remote dataset

Sometimes it may happen that huge datasets are hosted on a remote server and maybe downloading them locally is not convenient because they are too big. In this case we can just pass the `URL` of the files to the `data_files` parameter. For example in our `SQuAD-it` example we can use the `GitHub` `URL`:


```python
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

This will return us the same `DatasetDict` object we have seen before but it saves us from downloading the files locally and de compressing it.

>[!success]
> I was able to successfully load the `SQuAD-it` dataset using this remote dataset strategy. However doing `squad_it_dataset["train"][0]` I get a much longer and much more complex dictionary than the one shown above and reported on the course. Maybe they have truncated some of the output because it is very long and copying everything on the course section would have made it unreadable. Yes, that's probably it, I check the beginning of sample 0 of the `train` split and it starts with `terremoto del Sichuan` as in the example above.

## Time to slice and dice

In this section we will explore all the data wrangling we can do with the 🤗 `Dataset` library since most of the time some cleaning and pre processing operations will be needed before feeding our data to the models.

### Slicing and dicing our data

The 🤗 `Dataset` library provides a lot of functions and ways to slice and dice our data, as other libraries like `pandas` do. We have already seen the `Dataset.map()` function and in this section we will explore other functions to manipulate our data.

As an example dataset we will use the [Drug Review Dataset](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com) hosted on the [UCI Irvine Machine Learning Repository](https://archive.ics.uci.edu/). 

Before loading it we will download and unzip it doing:


```zsh
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
unzip drugsCom_raw.zip
```

As a result we now have two files called: `drugsComTest_raw.tsv` and `drugsComTrain_raw.tsv`. Their format is `tsv`, which is the same thing as `csv` but instead of having comma separated values we have tab separated values. So in order to load it with `load_dataset` we will use the `csv` script and we will pass the `\t` as the delimiter:


```python
from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
```

When we load a new dataset a good thing to do is to look at some samples contained in it to get and idea on how the data is structured and what kind of data are contained in there. We can do that chaining the `Dataset.shuffle()` and `Dataset.select()` methods:


```python
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
drug_sample[:3]
```

>[!note]
> Actually simply looking at the first 3 samples of the dataset without doing this shuffling and selecting will be equally useful, but whatever let's do it this way.

The result is:


```python
{'Unnamed: 0': [87571, 178045, 80482],
 'drugName': ['Naproxen', 'Duloxetine', 'Mobic'],
 'condition': ['Gout, Acute', 'ibromyalgia', 'Inflammatory Conditions'],
 'review': ['"like the previous person mention, I&#039;m a strong believer of aleve, it works faster for my gout than the prescription meds I take. No more going to the doctor for refills.....Aleve works!"',
  '"I have taken Cymbalta for about a year and a half for fibromyalgia pain. It is great\r\nas a pain reducer and an anti-depressant, however, the side effects outweighed \r\nany benefit I got from it. I had trouble with restlessness, being tired constantly,\r\ndizziness, dry mouth, numbness and tingling in my feet, and horrible sweating. I am\r\nbeing weaned off of it now. Went from 60 mg to 30mg and now to 15 mg. I will be\r\noff completely in about a week. The fibro pain is coming back, but I would rather deal with it than the side effects."',
  '"I have been taking Mobic for over a year with no side effects other than an elevated blood pressure.  I had severe knee and ankle pain which completely went away after taking Mobic.  I attempted to stop the medication however pain returned after a few days."'],
 'rating': [9.0, 3.0, 10.0],
 'date': ['September 2, 2015', 'November 7, 2011', 'June 5, 2013'],
 'usefulCount': [36, 13, 128]}
```

Here the `seed` was set on `Dataset.shuffle()` for reproducibility purposes and the `range(1000)` was passed to `Dataset.select()` to select the first 1000 samples of the dataset.

Essentially each sample contains:

- `drugName` → the name of the drug,
- `condition` → the condition it is used for,
- `review` → the review of the patient,
- `rating` → the rating the patient gave to the drug (between 1 and 10),
- `date` → the date of the review
- `usefulCount` → the number of people that found the review useful.

Looking carefully at these 3 samples we have extracted we can already see some strange thing:

- `Unnamed: 0` column → This is the usual annoying column automatically generated when we read `csv` or `tsv` files which is the index of the row, like an anonymized index for each patient.
- `condition` column → This column contains both uppercase and lower case lables.
- In the `review` column there are several special characters like Python line separators like `\r\n` and HTML entities like `&#039;`.

Actually if the `Unnamed: 0` column contains a unique `ID` value for each patient it could be useful. In order to check that we can use the `Dataset.unique()` method which will return a list with all the unique values contained in a column. If the length of the unique values in `Unnamed: 0` is equal to the number of rows in the dataset we can safely assume that it is a unique identifier for each patient. This can be verified with:


```python
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
```

`Unnamed: 0` can be indeed considered as a patient `ID` looking at this result:

```zsh
Split length: 161297
Unique Unnamed: 0 length: 161297
Split length: 53766
Unique Unnamed: 0 length: 53766
```

However the name `Unnamed: 0` is not very intuitive, so we can use the `Dataset.rename_column()` method to rename it to `patient_id`:


```python
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
drug_dataset
```

With the `unique` method we can also check some other interesting properties of the dataset like the number of unique drugs and conditions in the training and test sets.

For what concerns drugs:

```zsh
##################################################
Number of unique drugs in the training set: 3436
##################################################
Number of unique drugs in the test set: 2637
##################################################
```

For what concerns conditions:

```zsh
##################################################
Number of unique conditions in the training set: 885
##################################################
Number of unique conditions in the test set: 709
##################################################
```

Addressing the second concern we had about the data, we want now to normalize the `condition` column by converting all the strings contained in `condition` to lower case. To convert a string to lower cae we simply need to use the `lower()` method. Since we want to do that for all the rows (i.e. samples) of the dataset we can use the `map()` function:


```python
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

drug_dataset.map(lowercase_condition)
```

Unfortunately doing that we will get the following error:


```zsh
AttributeError: 'NoneType' object has no attribute 'lower'
```

This means that there are some `None` conditions on which the `lower` method cannot be used since they are not strings. 

To remove the `NaN` values we can use the `Dataset.filter()` function, which works similarly to `Dataset.map()`. It expects a function that performs the filtering operation on a single example of the dataset and then it will apply that operation to all the rows of the dataset.

We can define the following function and then passing it to `Dataset.filter()`:


```python
def filter_nones(x):
    return x["condition"] is not None

drug_dataset = drug_dataset.filter(filter_nones)
```

This will work without any problem but here they suggest a faster way of doing that which is using Python **`lambda` functions**. These kind of functions come in handy whenever we want to define a simple function that we will use just ones or a few times, as in this case since we will apply this function only now that we have to remove the `None` values. 

The general syntax for `lambda` function is the following:


```python
lambda <arguments>: <expression>
```

where:

- `lambda` → Python keyword to define a `lambda` function,
- `<arguments>` → the arguments the function will take,
- `<expression>` → the expression that will be evaluated and returned by the function.

For example we can define a simple function that does the square of a number as a `lambda` function by doing:


```python
lambda x:x*x
```

If we want to apply it to an input we have to wrap it and the input in parenthesis:


```python
(lambda x:x*x)(2)
```

We will have to repeat `lambda x:x*x` every time we want to execute it on an input because with `lambda` function we are not assining a name to the function. However if we have a function that has to be called multiple times in our code we will use a normal function definition with `def`.

We can also define `lamnbda` functions with multiple input arguments separating them with a comma:


```python
(lambda base, height: 0.5 * base * height)(4, 8)
```

So we can use these `lamda` functions as the arguments to the `Dataset.map()` and `Dataset.filter()` functions. For example we can use a `lambda` function to remove the `None` values from the `condition` column:


```python
drug_dataset=drug_dataset.filter(lambda x:x["condition"] is not None)
```

Now that the `None` are removed with can use `map()` to convert all the strings in the `condition` column to lower case:

```python
drug_dataset=drug_dataset.map(lambda x:{"condition":x["condition"].lower()})
```

### Creating new columns

When dealing with datasets containing customer reviews (like this one) a common practice is to count the number of words per review to get an idea of the review. In fact we may have super simple reviews, just saying `Good` or `Bad` and we may have very long reviews. For the moment we will count the words by using the `split()` function which will split the string contained in the `review` column by the whitespaces. 

We can do that with the `map()` function. So first of all we define a function on a single example:


```python
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
```

Now the `compute_review_length` function is returnig a dictionary whose key is not a column of the dataset. Thus when we pass this function to `Dataset.map()` a new column named `review_length` will be created in the dataset.


```python
drug_dataset=drug_dataset.map(compute_review_length)
```

Now we can check the the new column was added looking at the content of one sample in `drug_dataset`:

```python
{'patient_id': 206461,
 'drugName': 'Valsartan',
 'condition': 'left ventricular dysfunction',
 'review': '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"',
 'rating': 9.0,
 'date': 'May 20, 2012',
 'usefulCount': 27,
 'review_length': 17}
```

If we sort this new `review_length` column we can see what is the length of the shortest and longest review in the dataset:


```python
drug_dataset["train"].sort("review_length")[:3]
```
Let's look at the first 3 samples of the sorted column to see the shortest reviews:

```zsh
{'patient_id': [103488, 23627, 20558],
 'drugName': ['Loestrin 21 1 / 20', 'Chlorzoxazone', 'Nucynta'],
 'condition': ['birth control', 'muscle spasm', 'pain'],
 'review': ['"Excellent."', '"useless"', '"ok"'],
 'rating': [10.0, 1.0, 6.0],
 'date': ['November 4, 2008', 'March 24, 2017', 'August 20, 2016'],
 'usefulCount': [5, 2, 10],
 'review_length': [1, 1, 1]}
```

Indeed we have some reviews with just a single word. These can still be useful for a task like sentiment analysis (if we have a review saying `Good` or `Bad` it is still informative to classifiy a review as positive or negative) but if we want for example to predict the `condition` column starting from the review a single word review will be useless.

>[!note]
> Passing `reverse=True` we can sort the column in descending order over the `review_length` column to see which are the longest reviews in the dataset.

Here are the review lengths of the longest reviews:


```python
 'review_length': [1894, 1162, 1107]}
```

Pretty long reviews!

>[!note]
> Using the `Dataset.map()` function may not be always the best way of adding a column to our dataset. Another way of doing that is using the `Dataset.add_column()` method where we can pass an entire list of `np.array` with all the values for the new column. 

To solve the problem of very short review we can filter out reviews with less than 30 words. This can be done using the `filter` method:


```python
# With the function definition
def filter_short_reviews(example):
    return example["review_length"] >= 30

drug_dataset = drug_dataset.filter(filter_short_reviews)

# With the lambda function
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] >= 30)
```

The last annoying thing we have to deal with are the `HTML` codes contained inside the reviews. We can do that using the `html` python library that let's use escape these special characters and transform them into normal text:


```python
import html

text = "I&#039;m a transformer called BERT"
html.unescape(text)
```

This will return:

```zsh
"I'm a transformer called BERT"
```

We can use the `map()` function to apply this `unescape` transformation to all the reviews in the dataset:


```python
drug_dataset=drug_dataset.map(lambda x:{"review":html.unescape(x["review"])})
```

### The `map()` method's superpowers

We have already seen that the `map()` method is very powerful when we want to apply an operation to all the rows in the dataset. However we didn't even scratched the surface of what it can do. We will go more in details in this section.

#### The `batched` argument

As we have seen when we firslty introduced the `map()` method to tokenize the sample of our dataset in [[hugging_face_tutorial_ch_3|chapter 3]] we saw that we can pass a `batched` argument to it. This is a boolean argument that if set to `True` will execute the function passed to `Dataset.map()` to a batch of samples of the dataset. We can also set the `batch_size` to use, by default it is equal to 1000. Previously the `map()` command used to run the `html.unescape` function on the `review` column took some to run. If we perform it in this way it will be much faster:


```python
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)
```

In fact the differene from `batched=False` (which is the default value) is that now the `map()` method receives a dictionary with all the dataset columns as the keys, but instead of processing a single value at a time, a list of values (i.e. a batch of samples) is procesed in one go.

Here the command above is faster also because we are using a list comprehension instead of `{"review": html.unescape(x["review"])}` which is in general much faster than using a `for` loop.

Using the `batched=True` argument will be very useful in Chapter 6 when we will see how to use and create fast tokenizers that can use the `map` function to tokenize all the samples in the dataset in a very fast way. For example we can use the following function to tokenize the reviews in the dataset:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)
```

In Chapter 3 we saw that a tokenizer can take also a list of strings in input, so we can actually pass multiple strings to tokenize in the `example` argument of `tokenize_function`. This means thatw we can actually use `map()` also without `batched=True` if we pass a list of strings to a tokenizer. But which one is faster? We can try to perform some time experiments.

The time experiments we can perform are the following:

- Time taken with `batched=True`
- Time taken with `batched=False`
- Time taken passing `use_fast=False` when loading the tokenizer

>[!note]
> Here in the course they perform the time experiments on a notebook using the `%time` `%%time` commands to record the time for the execution of a single line or an entire cell, respectively. I will instead use a `python` script and I will use the `time` library to record the time taken for the execution of the `tokenize_function` function. Let's see what results I get.

The results are reported in this table:

| Options | Fast tokenizer | Slow tokenizer |
| --- | --- | --- |
| `batched=True` | 12.74 s | 55.5 s |
| `batched=False` | 110.5 s  |129.17 s  |

So fast tokenizers + `batched=True` is really fast! In general the fast tokenizers are faster than the slow ones because in the background they are executing the tokenization of the strings passed in input using `Rust` which is a language that makes it easy to parallelize code execution. 

In general the fast tokenizer is about 30x time faster than the slow one and 6x time faster execution is obtained if we use `batched=True`. We can't in fact parallelize the tokenization of a single sample but we can do in parallel the tokenization of a batch of samples.

It may happen that the tokenizer we are using does not have the fast `Rust` version implemented. In this case the `Dataset.map()` function still gives the possibility of parallelizing operations using the `num_proc` argument where we have to pass the number of processes to execute in parallel we want to spawn. This parallelization is not as fast as the `Rust` one but can still help. We can achieve it in this way:


```python
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)


def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)


tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)
```

Here they performed some other experiments with `num_proc` and found out that the best value was 8. Here I won't go on and do this test myself (otherwise I will never finish this chapter 😓) so I will just report here the results:

| Options | Fast tokenizer | Slow tokenizer |
| --- | --- | --- |
| `batched=True` | 10.8 s | 4 min 41 s |
| `batched=False` | 59.2 s | 5 min 3 s |
| `batched=True,num_proc=8` | 6.52 s | 41.3 s |
| `batched=False,num_proc=8` | 9.49 s | 45.2 s |

We can how a substantial improvement is obtained in the slow tokenizers and also in the fast ones. However for the fast tokenizers there are some other values of `num_proc` where that did not help at all. In general it is not recommended to use Python `multiprocessing` (this is what is being used in the background when we set the `num_proc` argument) with fast tokenizers with `batched=True`.

>[!note]
> In general it is not recommended to use `num_proc` in a function where there is some parallelization already implemented. In fact too much parallelization can increase the overhead needed to coordinate all the parallel processes which can then increase the execution time.

The `Dataset.map()` method starts to be more and more powerful, but there are still other features we have not explored yet. For example we can use `Dataset.map()` to change the number of elements in our dataset, for example to **create several training features from a single example**. This is something we will need in Chapter 7 when we will see how this is needed as the pre processing step of several `NLP` tasks. 

>[!note]
> In here with `example` we refer to a single sample of the dataset, that is represented by a set of *features* (i.e. a set of columns) that we will feed to the model. However in some contexts, like in question answering, multiple features can be extracted from a single example and belong to a single column.

Let's see how to do this thing. For example here we will tokenize our samples and truncate them to a maximum length of 128 tokens, but we ask to the tokenizer also to return *all* the chuncks of text, not just the first one (so the first 128 tokens). This can be done adding the `return_overflowing_tokens=True` argument to the `tokenizer` function.


```python
def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
```

If we apply this transformation on a single sample:


```python
result = tokenize_and_split(drug_dataset["train"][0])
[len(inp) for inp in result["input_ids"]]
```

We get the following shapes:


```python
[128,49]
```

In this way we have extracted two features. The first one are the first 128 tokens (up to which we truncated) and then we have 49 overflowing tokens that are not discarded.

Now we can do this for all the samples in our dataset:


```python
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
```

Unfortunately this returns thew following error:


```zsh
ArrowInvalid: Column 1 named condition expected length 1463 but got length 1000
```

There is a shape mismatch between column `condition` and the `review` column which is the one we are modifying with the tokenizer. In fact, as we have seen applying this `tokenize_and_split` function to a single sameple, we may produce multiple tokenized reviews than the ones we have in the original dataset. In particular here the old shape is 1000 because we have seen that by default the `batch_size` used by `Dataset.map()` is 1000. Now we have two options to solve this problem:

- We can remove the old columns from the dataset and replace them with the new ones
- Make the old columns the same size as the new ones.

We can perform the first solution passing the `remove_columns` argument to the `map()` function:


```python
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
```

This works without error and we can now check that our `tokenized_dataset` has more columns than the original one:


```python
len(tokenized_dataset["train"]), len(drug_dataset["train"])
```


```python
(206772, 138514)
```

The second solution is the one of making the columns of the original dataset to have the same increased size as the new ones. In order to do this we need to use the field `overflow_to_sample_mapping` that is returned by the `tokenizer` 
when we instantiate it with `return_overflowing_tokens=True`. This field gives us a mapping from a new feature index to the index of the sample it originated from. In this way we can associate to each key in the original dataset with a list of values of the right size by repeating the values of each example as many times as new features are generated. It is easier to understand what it i exactly happening looking at the new version of the `tokenize_and_split` function we have to create to do this:


```python
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result
```

Defining the function in this way the `map()` method can be used without errors and without having to specify the `remove_columns` argument.


```python
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
tokenized_dataset
```

Now the dataset structure looks like this:


```python
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 206772
    })
    test: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 68876
    })
})
```

We have the same increased number of rows (206772) we obtained before but now we have kept also all the field of the original dataset (i.e. `condition,date,rating,...`) which may be useful if we need to apply some post-processing operations on the dataset.

On the other hand with the first approach we saw (i.e. the one using the `remove_columns` argument) we will lose all the old fields of the dataset.

In fact this is the structure of the `DatasetDict` we get if we use the `remove_columns` approach:


```python
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'over
flow_to_sample_mapping'],
        num_rows: 228656
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'over
flow_to_sample_mapping'],
        num_rows: 76239
    })
})
```

As we can see we have only the fields added by the tokenizer: `input_ids,token_type_ids,attention_maks` and `overflow_to_sample_mapping`.

Now we have seen how to perform several operations on dataset which maybe helpful in pre procesing. In general most of the basic pre processing steps can be achieved with 🤗 `Datasets` but in case we need to use more specific stuff such as `groupby` we may need to user `pandas`. In the next section we will see how 🤗 `Datasets` is percetly compatible with libraries such as `pandas,numpy,PyTorch,TensorFlow` and `Jax`.

### From `Dataset` to `DataFrame` and back

In order to work on 🤗 `Datasets` object with third-party libraries we can change the output format of the object with the `set_format` method. The default format is Apache Arrow. If we instead want to pass to a `pd.DataFrame` object we can do:


```python
drug_dataset.set_format("pandas")
```

Now when we access some samples of the `drug_dataset` object we get a `pd.DataFrame` object instead of a dictionary.

To create a `pd.DataFrame` object from the training set we can do: 

```python
train_df = drug_dataset["train"][:]
```

>[!note]
> What happens under the hood when we use `set_format("pandas")` is that the dunder method `__get_item()__` is changed to return in output a `pd.DataFrame` but if we try to access to the type of `drug_dataset["train"]` we will see that it is still a `Dataset` object. So in order to obtain a `pd.DataFrame` object we have to use the `[:]` operator and assign the result to a new variable.

Now `train_df` is a `pd.DataFrame` so we can use all the `pandas` function on it. For example here we can perform this fancy chain of functions to compute the class distribution among the `condition` column:


```python
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "condition": "frequency"})
)
frequencies.head()
```

Now that we have performed this `pandas` analysis we can convert the `frequencies` `pd.DataFrame` object to a `Dataset` object using the `from_pandas` method:


```python
from datasets import Dataset

freq_dataset = Dataset.from_pandas(frequencies)
freq_dataset
```

Now we can visualize it as:


```python
Dataset({
    features: ['condition', 'frequency'],
    num_rows: 819
})
```

If we want to reset the format of our dataset back to the default `arrow` format we can do:


```python
drug_dataset.reset_format()
```

### Creating a validation set

Ok now we want to create a validation set in order to train our model. We already have a test set but it is always a good practive to use a validation set to check the performances during training, visualize weather we are overfitting the data, fine tuniong hyperparameters and so on. The test set should be used just at the end to check the model performances. 

Here we can use the `train_test_split` function available in 🤗 `Datasets`, which is based on the famous one from `sklearn`.

Here we will apply it on the training set with a `train_size=0.8`. This function automatically screated a `train` and `test` split on the resulting dataset, thus we will rename the `test` split to `validation` since we already have the `test` split and we will then add the `test` split to the `DatasetDict` object.


```python
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
drug_dataset_clean
```

Now the `DatasetDict` object looks like this:


```python
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 46108
    })
})
```

Ok now we have a dataset that is ready to be trained on! We will see in Section 5 how to load a dataset to the Hub, but for the moment let's see how to save it in our local machine.

### Saving a dataset

Even though the dataset are always cached and saved in our `~/.cache/huggingface/datasets/` path in some cases (i.e the cache is deleted) it may be useful to save the dataset to the disk. There are different methods to save a dataset to the disk depending on the format in which we want to save it:

| Data format | Function | 
| --- | --- |
| Arrow | `Dataset.save_to_disk()` |
| CSV | `Dataset.to_csv()` |
| JSON | `Dataset.to_json()` |

If we want to save to the disk our `drug_dataset_clean` object in the `arrow` format we can do:


```python
drug_dataset_clean.save_to_disk("drug-reviews")
```

This will create a `drug-reviews` folder in our working directory with the following structure:


```txt
drug-reviews/
├── dataset_dict.json
├── test
│   ├── dataset.arrow
│   ├── dataset_info.json
│   └── state.json
├── train
│   ├── dataset.arrow
│   ├── dataset_info.json
│   ├── indices.arrow
│   └── state.json
└── validation
    ├── dataset.arrow
    ├── dataset_info.json
    ├── indices.arrow
    └── state.json
```

which is a bit complicated. Essewntially we have a sub folder for each split (i.e. `train`,`validation` and `test`) and in each one of them we have some metadata in `dataset_info.json` and `state.json`. In fact the Arrow format is a fancy table with columns and rows which is optimized to build high-performance applications processing and transporting large datasets.

Once the dataset is saved we can load it using `load_from_disk`:


```python
from datasets import load_from_disk

drug_dataset_reloaded = load_from_disk("drug-reviews")
drug_dataset_reloaded
```

For what concerns the `csv` and `json` formats here we have to save a separate file for each split. We can do that iterating over the `items` of the `DatasetDict` object:


```python
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")
```

with the code above we are saving each split as a `jsonl` file, which stands for the `json` line format. Essentially in this format each row of the dataset is stored as a single line of `json`. 

To load these kind of files we can use the `load_dataset` method we have seen [[hugging_face_tutorial_ch_5#loading-a-local-dataset|before]]:

```python
data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)
```

Ok, we have finished this long chapter on data wrangling 💪. 

>[!todo] Things to try out
> Now here are some things to try out to practice with these new things we learned:
> - Train a classifier on the `drug_dataset_clean` dataset that can predict the patient condition based on the drug review.
> - Use the `summarization` pipeline to generate summaries of the reviews.

In the next section we will see how to train models on huge datast without bloowing out our laptop or our remote server. 


