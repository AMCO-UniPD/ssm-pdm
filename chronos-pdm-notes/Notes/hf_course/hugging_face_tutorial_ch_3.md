---
id: hugging_face_tutorial_ch_3
aliases: []
tags: []
---

# Hugging Face 🤗 Tutorial Chapter 3

In this note we will take on Chapter 3 of the `HuggingFace` 🤗 `NLP` tutorial. This chapter talks about **fine-tuning**, a vey important topic that may come very in handy for the `chronos-pdm` project.

>[!info]
> Reference to the [Chapter 3 on the `HuggingFace` website](https://huggingface.co/learn/nlp-course/chapter3/1)

## Processing the data

With the things we have seen up to now we are able to load a model and a tokenizer using a checkpoint from the model hub and we can do inference with it. But what about training the model to improve its performances on a specific task? Here they start writing down a simple extension of the code in order to train our model with a `PyTorch` like training:

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

Obviously this training won't do much because we have just used these batch of two sequences. We need a proper dataset where all the `(input,label)` pairs are stored. 

In this chapter we will use a small dataset for the `SequenceClassification` task which is called `MRPC` (Microsoft Research Paraphrase Corpus). It contains 5801 pairs of sentences with a label saying weather they are paraphrased or not (i.e. if they have the same meaning or not).

>[!note]
> The `MRPC` dataset is part of the `GLUE` benchmark. An accademic benchmark to evaluate `ML` models on 10 text classification taks.

### Loading a dataset from the Hub

In `HuggingFace` 🤗 we do not have only the Model Hub but there is also the [Dataset Hub](https://huggingface.co/datasets) which contains a lot of dataset we can use to train and fine tune our models. 

The datasets can be managed with the `datasets` library. 

>[!warning]
> Make sure to install the `datasets` library with `pip install datasets`.

Once the library is installed we can load the `MRPC` dataset with the following code:

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
```

If we print out the content of `raw_dataset` we can see that it is a `DatasetDict` object with the following structure:


```python
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

We have a field for each split of the dataset (`train`, `validation`, `test`) and each split is an instance of `Dataset` which contains the `features` and `num_rows` fields. So from here we can already see that the `MRPC` dataset has 3668 training examples, 408 validation examples and 1725 test examples. We can see that there are 2 main features (the two sentences) and then for each of them we have the `index` (with which we can access a specific sample in the dataset I guess) and the `label` field which contains the label of the pair of sentences.

>[!note]
> Similarly to what happens when we load models, the first time we load a dataset this will be saved and cached locally at the path `~.cache/huggingface/datasets`. In this way the next time the dataset it's loaded it will be loaded from the cache and not from the web.

We can now use the fields of the dictionary to easily access some samples of the dataset:

```python
raw_train_dataset= raw_datasets["train"]
print(raw_train_dataset[0])
```

Let's see what a sample looks like:


```python
{'idx': 0,
 'label': 1,
 'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
 'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'}
```

The labels are already stored as integers, so we do not have to do any pre processing. But what class is 1 referring to? We can check by accessing the `features` field of the dataset:

```python
raw_train_dataset.features
```

```python
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}
```

The labels are instances of class `ClassLabel` and we can see that class 0 is `not_equivalent` and class 1 is `equivalent` (in fact reading at the two sentences from index 0 they seemed pretty similar)

### Preprocessing a dataset

In order to feed the dataset to the model we need to preprocess it. How can we do it? With the `tokenizer` of course. Since it can take multiple sentences in input we can pass it the entire `sentence_1` and `sentence_2` features to it:

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
```

However this is not what we want because we need to process the sentences as a pair, so they have to be passed together to the model. In this way we are passing two separate and independent sentences. Fortunately we can pass two sentences to the tokenizer and it will treat them as a pair:


```python
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```

The results is:


```python
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

We can see that we have a `102` token in the middle that represents the end of the first sentence and the beginning of the second one. 

We have already discussed about the `attention_mask` and the padding but here we can now also understand the role of the `token_type_ids` field that is used to distinguish the two sentences. In fact all the tokens of the first sentence are paired with a 0 in `token_type_ids` while we have a 1 for the tokens of the second sentence.

If we convert the `ID`s to tokens we can see the format in which the model expects the inputs to be:


```python
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
```

So the format is: `[CLS] sentence_1 [SEP] sentence_2 [SEP]`. We can also align it with the `token_type_ids` to see how the two sentences are separated.


```python
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
```

>[!note]
> The `token_type_ids` field is not present in all the tokenizer, it depends on the specific checkpoint. For example in the `DistilBERT` checkpoint we do not have this field because it was not used in the model pre training. Here with `bert-base-cased` it has already seen `token_type_ids` during pre training.

The `bert-base-cased` checkpoint we are using was trained on two pre training objectives:

- Masked Language Modelling → Predict the masked tokens
- Next Sentence Prediction → Predict if the second sentence follows the first one. Here we need a pair of sentence and that's why we have the `token_type_ids` field.

>[!note]
> In any case in practice we do not really care about the field present inside the tokenizer, we just need to load it with the same checkpoint of the model and we will be fine becasue the tokenizer will be perfectly aligned with the model.

Ok now how to tokenize the entire dataset? We know that `tokenize` works also with lists or batches of sentences so we just need to pass the two lists of sentences to it as a pair. Here we will also need to add the `paddding` and `truncation` options to the tokenizer:


```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

The problem with this approach, as I already experienced myself in the code, is that the data are stored as a list containing all values of the kind we have seen in the previous example when we tokenized a single pair. This is not very easy to handle. Moreover this method will work **only if we have enough `RAM` to store the data**. On the other hand 🤗 `datasets` are stored as `Apache Arrow` files stored in the disk and we keep only the samples we asked to load.

The approach we will use to tokenize our dataset is to us ethe `Dataset.map()` function which gives more flexibility to us if we want to do some additional preprocessing steps other than just the tokenization. In order to use the `Dataset.map()` function we need to write a function that applies all our preprocesing steps to an `example` from our dataset:


```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```

In fact here the `map` function it's like the standard python `map` function which applies a specific function to a set of inputs.

The `tokenize_function` does the same thing we have done previously to tokenize a single pair of sentences, we just added the `truncation` argument to manage long sentences. In any case here `example` can also be a list of sentences. In this way we can process an entire mini batch of sentences and passing the argument `batched=True` to the `map` function we can process the entire dataset in batches.

>[!note]
> The `tokenizer` is backed by a tokenizer written in `Rust`, which is very fast but only if we give a lot of inputs at once.

>[!question] What about padding?
> Why didn't we insert also the padding argument? That's because if we apply padding to the entire dataset than all sequences will be padded to have the same length of the longest sequence in the datasets and this can create very long sequences if there is an unbalance in the sequence lengths. It is more convenient to apply padding only to the mini batches, so all sequences in the same batch will have the same length.

So now we can apply the `map` function. We pass `batched=True` so that multiple pairs of sequences are processed at once forming a mini batch:


```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

The way the `daytasets` 🤗 library applies this transfromation is by adding the fields of the `tokenizer` inside the `features` field of the `DatasetDict` object so that we have the same object but with more information. 


```python
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})
```

We can see that in `features` we now also have `attention_mask,input_ids,token_type_ids` which are the fields added by the tokenizer.

>[!note]
> Passing the argument `num_proc` to `map` we can also parallelize the tokenization process using multiprocessing. In this case it was not needed because out `tokenizer` is already doing multi-threading, but if we have a slower tokenizer not backed by Rust that may be useful.

### Dynamic Padding

The last thing we need to do is to pad all the sequences inside the different batches, this technique is called **dynamic padding**.

In order to put samples together inside a batch we need to use a so called *collate function*, which is used also in the creation of `torch` `Dataloaders` and it normally converts the list of samples into `torch.tensor` and concatenates them. However this is not possible for us since we have elements with different lenghts and so we have to use padding in order to use mini batches.

We want to pad dynamically depending on the maximum sequence length on each batch to speed up the traning. 

>[!note]
> Dynamic padding does not work if we are training our model on a `TPU` since this hardware prefers to have fixed size tensors, even if that means padding all the sequences to the same length.

Fortunately there is already a collate function in 🤗, called `DataCollatorWithPadding`. We have to instantiate it passing a `tokenizer` (which is needed to the collator to know which is the padding token), then we have to say weather the padding should be performed on the left or on the right:


```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

Now let's try to create a batch. We will take some samples from our dataset and we will pass them to the `data_collator`. We will remove the `idx`, `sentence1,sentence2` fields since they are not necessary anymore (we have infact the `input_ids` now) and moreover they contain strings which cannot be used to create a tensor:


```python
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
```
Here if we look at the length of the sentence pairs in our batch we can see that, without surprise, they are all different:

```python
[50, 59, 47, 67, 59, 50, 62, 32]
```

We can see that the longest sequence is 67 tokens long, so after padding with `data_collator` we should have all the sequences of the same length of 67. Let's see:


```python
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```

```python
{'attention_mask': torch.Size([8, 67]),
 'input_ids': torch.Size([8, 67]),
 'token_type_ids': torch.Size([8, 67]),
 'labels': torch.Size([8])}
```

Yep, all 67 💪

Ok now we have organized the dataset in batches and we are ready to use it to train the model.

## Fine-tuning a model with the `Trainer` API

Now we are ready to fine-tune our model and in 🤗 there is a specific `API` called `Trainer` to do that.

>[!note]
> Here obviously we have to remember that performing a fine tuning on a `CPU` will take a lot of time, so we need to have a `GPU` at our disposal to do that. In my case I have `acquario3` so no problems.

In the code chunck below we have a review of all the steps we did up to now to remember the starting point:


```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Training

The first step towards training the mode is to define an object of the `TrainingArguments` class which is used to store all the hyperparameters of the training and a directory where to save the model and all its checkpoints. In our simple example we will use the default values for all the hyperparameters:so we just need to specify the directory where to save the model:

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```

>[!info]
> It is also possible to automatically save the model checkpoints to the 🤗 hub passing the argument `push_to_hub=True` to the `TrainingArguments` constructor.

Let's now create the model we will train, we will use the usual `AutoModelForSequenceClassification` class and we will pass the `num_labels` argument to it to specify the number of classes we have in our dataset, which is 2 in our case:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

>[!note]
> Now when we instantiate the model we get a warning. In fact `bert-base-uncased` was not pretrained for a sequence classification task with 2 labels (it was trained on Masked Language Modelling where for each mask the labels were the entire vocabulary). So now the `AutoModelForSequenceClassification` will automatically discard the model head from pre training and add a new randomly initialized head for the sequence classification task. So in the warning it suggests us to train the model because its head is randomly initialized and won't produce good performances.


Now that we have all the ingredients we can create an instance of the `Trainer` class passing the training arguments, the model, the train and validation datasets, the tokenizer and the data collator:


```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

Actually in this case the `data_collator=data_collator` line is not necessary because out tokenizer will automatically use a `DataCollatorWithPadding` object if we pass it to the `Trainer`.

Now to train the model we simply have to call the `train` method of the `Trainer` object:

```python
trainer.train()
```

This starts the fine-tuning process that with this dataset should take a couple of minutes on a `GPU`. It will print out the training loss every 500 steps by default but that is not very informative of how the training is going. 

In fact here we called the `train` method with all the default arguments. There are a lot of things we can do to have live feedback on the training process:

- We can set the `evaluation_strategy` parameter to something like `steps` (to evaluate every `eval_steps`) or to `epochs` to evaluate at the end of each epoch.
- Moreover it may be more useful to print out a metric rather than a loss value which is not very intuitive. We can do that with a `compute_metrics()` function.

### Evaluation

In this section the idea is to understand how to build a function we will call `compute_metrics` that will be passed inside the `Trainer` object to make the model compute some metrics on the validation set while training so that we can see how the model is performing. We will compute these metrics at every epochs, to set this up we will add the `evaluation_strategy="epochs"` argument to the `TrainingArguments` object.

This function should take an object of type `EvalPrediction` which is a named tuple with the fields `predictions` and `label_ids`. The function will return a dictionary with strings (with the metric names) as keys and floats as values (the value of the metrics).

To get predictions from the model we use the `predict` method of a `Trainer` object:


```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```
The `predictions` object is organized in the following way:

- `predictions` is a tensor of shape `(num_samples,num_labels)` containing the predicted logits for each class.
- `label_ids` is a tensor of shape `(num_samples,)` containing the true labels of the samples.
- `metrics` is a dictionary containing the metrics computed by the model.

In particular the `metrics` dictionary looks like this:


```python
{'test_loss': 0.7428162693977356,
 'test_runtime': 10.7068,
 'test_samples_per_second': 38.106,
 'test_steps_per_second': 1.588}
```

These however are not very intuitive metrics and thus we will now create the `compute_metrics` function which will add some fields to the `metrics` dictionary with the metrics we want.

In order to transform the logits contained in the `predictions` field we have to transform them into labels. This can be done taking the `argmax` (i.e. index of the maximum value) along the second dimension:


```python
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)
```

Now that we have the predicted labels we can compare them to the target ones (i.e. the ones in the field `label_ids`). We will use the `evaluate` 🤗 library which let us load the metrics for the `MRPC` dataset and with the `compute` method we can get the metrics values:

```python
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

The results is:

```python
{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}
```

Here the `evaluate` class loaded the `accuracy` and `f1` score which are the metrics used for the `GLUE` benchmark.

Wrapping everything up we can now define the `compute_metrics` function:


```python
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

Finally our new training loop will look like this:


```python
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

The model needs to be re initialized otherwise we will go on training a model that we have already trained for some time.

Ok now we have all the basic ingredient to train a model with the `Trainer` API. There are obviously a lot of other options offered by the `Trainer` class that I can check for myself I guess. Now in the next section the course will explain how to achieve the same result but writing the entire training loop in `pytorch`.


## A full training loop in `PyTorch`

Now we will so how we can achieve the same training loop but instead of using the `Trainer` we will do everything in `pytorch`. This means that we will have to write some more additional code to handle some technical things that the `Trainer` does for us. 

>[!important] 
> This thing of writing the training loop in `PyTorch` seems useless at a first sight since we have everything already simple and done by the `Trainer` `API` however writing the loop in `PyTorch` we have more control on every step and we can personalize it as we want. I think that it may also be useful because for the specific case of the `chronos-pdm` project if we want to fine tune `CHRONOS` on `RUL` data (e.g. `C-MAPSS`) we won't probably have that `RUL` data contained in the 🤗 `Datasets` library and thus we may have to deal with the data as raw `PyTorch` tensors and write the training loop by ourselves. But maybe is there a way to load our dataset insid the Dataset Hub so that it is saved as a 🤗 `Dataset` object and so then we can use all the `API` of 🤗 to train the model. I will have to check this out.

>[!success] Obviously there is a solution in 🤗
> In [Chapter 5 section 2](https://huggingface.co/learn/nlp-course/chapter5/2) there is properly the solution for us if the dataset we want to use it's not in the Hub. Here they will explain how to load datasets from local machine or from a remote server.

Let's recall the starting point for creating our full training loop in `PyTorch`. We have loaded our dataset and tokenizer and have used the latter to tokenize the dataset also dealing with padding and truncation:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Prepare for training

Before writing down the `torch` training loop we have to define some objects. Obviously we have to define the `Dataloader`s we will iterate over in the loop. However before doing that we have to handle some pre processing steps that are normally managed by the `Trainer` `API` for ourselves:

- Remove useless features from the dataset → `sentence1,sentence2` are not needed anymore since we have the `input_ids` field and moreover they are strings so can't be used to create `torch` tensors.
- Rename the column `label` to `labels`. In fact the model expects the argument to be called `labels`.
- Set the format of the dataset to return `torch` tensors and not lists.

Fortunately the `tokenized_dataset` object we have has a method to deal with each one of these things.


```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
```

With the `column_names` parameter in the last line we can check that we have all the columns we need:


```python
["attention_mask", "input_ids", "labels", "token_type_ids"]
```

Now we can define the `Dataloader`s:


```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

>[!note]
> Note that in creating the `DataLoader`s we are also passing the `data_collator` object to the `collate_fn` argument. I have never used this argument before in `DataLoader` but since we have previously created it we can use it here so that it will handle all dynamic the padding and truncation for us.

We can check the shapes of the components of each batch by doing:


```python
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
```

Obtaining:


```python
{'attention_mask': torch.Size([8, 65]),
 'input_ids': torch.Size([8, 65]),
 'labels': torch.Size([8]),
 'token_type_ids': torch.Size([8, 65])}
```

Here the results may change because of the `shuffle=True` argument passed in the `train_dataloader` that will created different batches and so there will be a different maximum length to use for the dynamic padding.

Now it's time to defin the model, that is the usual one and we pass the batch to it to see the output:


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
```

The output is:


```python
tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])
```

In all 🤗 `transformers` models when we pass the labels in input to the model (in this case inside the `batch` we passed there is the `labels` key) the model will automatically return the `loss` as an attribute of the `outputs` tensor together with the logits that in this case are of shape `(8,2)` because we have 8 samples in the mini batch and 2 classes.

Now we are missing just two ingredients to get to our training loop: the optimizer and the learning rate scheduler. Since we wanto to exactly reproduce the training loop previously done with the `Trainer` we will use its default optimizer which is `AdamW`.

>[!note]
> The `AdamW` optimizer is a variant of `Adam` which sgligthly modifies the weight decay regularization. Details [on the paper that introduced it](https://arxiv.org/abs/1711.05101)


```python
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

The learning rate scheduler used by default by `Trainer` is just a linear decay scheduler where the learning rate decays from `5e-5` to 0. In order to define it we need to now the number of training steps which is the number of epochs multiplied by the length of our `dataloader`. By default the `Trainer` performs 3 epochs, so we have:


```python
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
```

### The training loop

One last thing, we need to use `GPU` to make the training time reasonable so we have to define a `device` to which we will send the model.


```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
```

We can now finally define the training lop. We also add a progress bar with `tqdm`.


```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

>[!note]
> Interestingly here instead of just doing `for epoch in progress_bar` they are doing `for epoch in range(num_epochs)` and then the progress bar is updated at every iteration by the command `progress_bar.update(1)`. I would have used the first solution but I guess that in this way it is way the code is more readable.

This is a very basic training loop that does not give us any information on how the model is progressing in its performances as we go on with the epochs. In order to do that it's better if we create also an evaluation loop.

### The evaluation loop

Here we can still use the 🤗 `evaluate` library that let's us load the metrics for the specific dataset we are using and with `metrics.compute()` computes them. However here we can also accumulate the metrics over the batches using the `add_batch()` method (this method essentially does the same thing that I am doing in the training and evaluation loops I have in the `ad_mg` project when I do `train_loss+=loss.item()` and then I do the mean. Here we do `add_batch` inside the loop and once we are out of it we do `metrics.compute()` to get the final metrics.


```python
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

and we obtain something in the form of:


```python
{'accuracy': 0.8431372549019608, 'f1': 0.8907849829351535}
```

### Supercharge your training loop with 🤗 `Accelerate`

The code we just wrote works fine on a single `TPU` or `GPU` but if we have multiple `GPU`s or `TPU`s available we can speed up our training using the `Accelerate` library by 🤗. This library handles all the boring technicalities of distributed training for us and will distribute the computations among all the pieces of hardware we have but just adding small changes to the code.

The main changes to the code are the following:


```python
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

1. Import the `Accelerator` library and instantiate and `Acceleratore` object:


```python
from accelerate import Accelerator
accelerator = Accelerator()
```

>[!note]
> `Accelerator` also handles all the `device` stuff so we can also remove the lines where we define `device` and send the model to it. In any case if we want to maintain those lines we can do that, we just have to substitute `device` with `accelerator.device` in the code.

2. Define the `dataloader`s, the model, the scheduler and the optimizer in one go with `accelerator.prepare()`:


```python
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)
```

This is the step where the main work by `Accelerate` happens. It puts the `dataloaders`, model, scheduler and optimizer to the samew container in order to distribute the computations among the different pieces of hardware we have.

3. Remove the  first line inside the training loop where we send the `batch` to the `device`, or we can keep it substituting `device` with `accelerator.device`:

4. Finally we have to replace the `loss.backward()` with `accelerator.backward(loss)` which is the method that will handle the backpropagation for us.

Now we can put all the code together inside a `train.py` script. In order to try out this distributed training we have to launch the following command in the terminal:


```zsh
accelerate config
```

I think that this command works also if we do not have the `py` script ready, it is something we can do just after having installed the `accelerate` package. It will ask use some questions in order to configure the `Accelerate` library. Finally to launch our `Accelerate` training script we have to do:


```zsh
accelerate launch train.py
```

It is also possible to use `Accelerate` on a notebook (for example we can use it in Google Colab so that we can use the Google `GPU`s or `TPU`s) by plugging all the coe inside a function (we can call it `training_function()`) and then run a cell with the following code:


```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```

>[!note]
> There are other examples inside the 🤗 [`Accelerate` repository](https://github.com/huggingface/accelerate/tree/main/examples)

>[!warning]
> I realized that if we use the `Trainer` `API` for fine tuning the models that is using `Accelerate` in the background by default. This is good since our training will be really fast, however working on a shared remote server like `acquario3` running processes on all the `GPU`s while there are other users running their own processes is not very safe since we can slow down the server for everyone or making the processes of other users crash. There should be a way to limit the number of `GPU`s used by `Accelerate` but at this point it is probaly easier to use a standard `torch` training with a single device as we have seen in the previous sections.

And this ends the Chapter 3 💪
