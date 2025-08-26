---
id: hugging_face_tutorial_ch_1_2
aliases: []
tags: []
---

# Hugging Face 🤗 Tutorial Chapter 1 and 2

I am looking at the `NLP` Course of Hugging Face and I am trying out the basic stuff about the `transformers` library. 

>[!note]
> See all the examples in the `transformer_test.py` python script inside the `experiments` folder.

Here I report the notes regarding one part of Chapter 1 (the `pipeline` function) which is mainly theoretical and introduces the `Transformer` model and the different kind of architecture we can do with it (i.e. Encoder Only, Decoder Only, Encoder-Decoder).

>[!info]
> For full details on Chapter 1 [see it directly on the `HuggingFace` website](https://huggingface.co/learn/nlp-course/chapter1/1)

## `pipeline` function

The most basic thing to do in `transformers` is to use the models available in the Model Hub and do inference with them throught the `pipeline` function which is a wrapper doing all the necessary pre processing and post processing steps in the background.

To use the `pipeline` function we have to pass in input a task, a model to execute that task. For example to perform `sentiment-analysis`we can do:


```python
sa = pipeline("sentiment-analysis",device=device)
sentiment_analysis=sa(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

>[!note]
> Even if we do not specifically pass a `model` parameter it will use a default model for the task we are performing. However if we do that in the output it throws a warning saying that using the default model is not recommended.

>[!important]
> Here I also passed the `device` argument passing one betweee `cuda:0,cuda:1,cuda:2` (at least in `acquario3` since we have 3 `GPU`). If wee do not pass the `device` it will use the `CPU` by default, which may take some more time.

>[!note]
> The first time we call the `pipeline` function it will download the model from the Model Hub and cache it in the `~/.cache/huggingface/hub/` directory. In this way the next time we call the function it will not download the model again and we can do inference faster.


### `text-generation` Task

The `text-generation` taks is the one we are using every day with `ChatGPT` or `DeepSeek` `UI` interfaces. I wanted to test out some of the cool and powerful models but I had some problems. 

#### `DeepSeek-V3` Model 🐳

The first model I found in the Model Hub filtering for the `text-generation` task was `DeepSeek-V3` and I used this code to generate some text:


```python
deepseek = pipeline(model="deepseek-ai/DeepSeek-V3",device=device)
text_gen=deepseek("In this course, we will teach you how to")

print("#"*50)
print("Text Generation Results:")
print(text_gen[0]["generated_text"])
print("#"*50)
```

However after running the code I got an error related to the `fp8` quantization which is not available as a quantization method in the `transformers` library. Reading the details on the Model Card it is written that this model is not available in `transformers` and in fact they explain other ways to use the model.

#### `Llama-3.2-1B` Model 🦙

Another model I tried is the `Llama-3.2-1B` which is supposed to be open-source. However after running the code below I got an error message saying that the model is not available if I do not have the license, which is not free 😠.

```python
model_id = "meta-llama/Llama-3.2-1B"

llama = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

text_gen=llama("In this course, we will teach you how to")

print("#"*50)
print("Text Generation Results:")
print(text_gen[0]["generated_text"])
print("#"*50)
```

At this point since all the cool models I try give me some strange errors I will us the default model to generate text 😠. which is `gpt2`. 

In any case we can control the `max_length` of the generated text and the number of sentences to generate with `num_return_sequences`. For example to generate two sentences of 15 words we can do:

```python
generator = pipeline("text-generation",
                     device=device)
text_gen=generator("In this course, we will teach you how to",
                   num_return_sequences=2,
                   max_length=15)

print("#"*50)
print("Text Generation Results:")
print(text_gen[0]["generated_text"])
print(text_gen[1]["generated_text"])
print("#"*50)
```

## Behind the `pipeline` function

Now we are in Chapter 2 of the `NLP` course where we go deeper inside the `transformers` library. In particular we will start by seeing what happens in the background of the `pipeline` function. These are all the necessary pre processing and post processing steps normally needed when we work with a `transformer` model.

>[!info]
> In case something is missing go check that out [directly on the `HuggingFace` website](https://huggingface.co/learn/nlp-course/chapter2/1)

### Tokenizer

The first step is to make the input text understandable by the model and this is done throught a **tokenization** step. In this step the text is transformed into a sequence of tokens that are part of a pre built vocabulary of a fixed size. Each token is associated with a unique integer index. These are the numbers that are passed in input to the model. 

More in details the steps we are doing are the following:

- Split the input words into subwords, symbols called **tokens**
- Map each token to an integer
- Adding additional inputs that may be useful for the model

#### `AutoTokenizer` class

In the `pipeline` function we are doing inference with a model that is already pre trained and saved on the Model Hub. So we have to use the same tokenizer that was used to train the model. These are stored in the `AutoTokenizer` class. This class works similarly to `pipeline`, we still have to pass a string containing the model id we are interested in. For example let's consider the default model for the `sentiment-analysis` task, its tokenizer can be obtained with the following code:


```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

Now that we have the tokenizer we can pass input text to it and we will obtain the tokens `ID`s that we can pass in input to the model without worrying about the `DL` backend framework that is used (i.e. `PyTorch` or `TensorFlow`). However we need to make sure that we are passing these tokens in form of tensors. With the `return_tensors` argument we can specify what kind of tensors we want, for example we can pass `pt` to get `PyTorch` tensors. By default this method will return a list of lists.


```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

The results are:


```bash
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

We can notice some things:

- Both `input_ids` start with token 101 and end with token 102. These are probably special tokens used to mark the beginning and the end of the input text.
- 0 padding was used in the second sentence in order to make its `input_ids` and `attention_mask` list match the size of the first sentence, which is longer. 
- The attention mask signals the tokens that have to be considered inside the Attention layer. The ones marked with 0 will not be considered in the second sentence because they correspond to the padding tokens.

#### `AutoModel` class

Now that we have the tokens we need the model in order to feed it with them. This is provided by the `AutoModel` class that works very similarly to `AutoTokenizer`, still with the model id in a string and with the `from_pretrained` method. For example to get the model for the `sentiment-analysis` task we can do:

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

>[!warning]
> This is not the final model that perform the `sentiment-analysis` task but just the `transformer` part so its output will be the last hidden state. In order to get the final output we have to add an *head* to the model, the head changes depending on the specific task we want to perform.

The hidden state returned by this model has the following shape:

- **Batch size** → the number of sequences passed in input (e.g. 2 in our example)
- **Sequence length** → the number of tokens in the longest sequence passed in input (e.g. 16 in our example)
- **Hidden size** → the size of the hidden states of the model (e.g. 768 for `distilbert-base-uncased`)

The hidden size is the high dimensional thing. In fact after the `input_ids` are passed in input to the transformer there is the embedding layer that associated an high dimensional embedding representation to each token. This is a fixed learned mapping which can be learned from scratch or ca n be initialized with pre trained embeddings. Then there is also the addition of the positional encoding values that are needed to keep track of the position of a token in the sequence.

In order to get the shape of the hidden state we can do:

```python
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

and we get:


```bash
torch.Size([2, 16, 768])
```

##### `AutoModelForSequenceClassification` class

We can use the class `AutoModelForSequenceClassification` to load the complete model comprising also the *head* to perform the taks of `sentiment-analysis`. There are several of these classes inside `transformers` depending on the task we want to solve.


```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```

Now the model outputs have shape `torch.Size([2, 2])`. In fact we have 2 sequences and two ouputs are produced for each sequence. In fact this is a Binary Classification task and thus the model produces the logits for the two classes.

The logits are the following:


```bash
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```

They do not give us useful information on what is the classification of the model for the input sentences, we have to process them throught the `softmax` function to get the probabilities of the two classes. 


```python
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

Now the results make more sense:


```python
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```

The first sequence is classified as `positive` with a probability of `0.9598` and the second sequence is classified as `negative` with a probability of `0.9995`.

The get the label id we can do:

```python
model.config.id2label
```

obtaining:


```bash
Tokenization Results:
Input ids:
First sentence: tensor([  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
         2607,  2026,  2878,  2166,  1012,   102])
Second sentence: tensor([ 101, 1045, 5223, 2023, 2061, 2172,  999,  102,    0,    0,    0,    0,
           0,    0,    0,    0])
Attention mask:
First sentence: tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
Second sentence: tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
##################################################
##################################################
Model Output:
Logits:
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>)
Probabilities:
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward0>)
Predicted Labels:
['POSITIVE', 'NEGATIVE']
##################################################
```

## Models

With the `AutoModel` class we can load a pre trained model contained in the Model Hub. Passing a correct model id string this class will automatically build the model with the correct architecture. However if we know the specific architecture we want to use (without focusing on a specific task) we can also create a model with the architecture using the specific class for that architecture. For example we can create a `Bert` model as: 

```python
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
```

The `config` is essentially the same thing as the `ModelConfig` class that I am using in the `AD_MG` project, it let's use define all the hyperparameters of the model. Printing out its values we can see that it contains the ususal hyperparameters like `hidden_size`,`num_hidden_layers`,...


```python
BertConfig {
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.45.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```

Obviously here we have passed all the default configurations but we can pass some key value arguments to set the value we want for the hyperparameters.

To create a `BertModel` with a specific configuration we will simply do:

```python
model=BertModel(config)
```

### `from_pretrained` method

Obviously if we create an instanc of `BertModel` like this we have a model with randomly initialized weights and we should train it in order to make it produce meaningful results. Obviously the pre training task is not something that you do with some simple `GPU`s so we will resort to using the model checkpoints contained in the Model Hub. We can load the weights of the pre trained model using the `from_pretrained` method:


```python
model = BertModel.from_pretrained("bert-base-uncased")
```

This is exactly the same thing we previously did with the `AutoModel` class. In fact from now on **we will always use the `AutoModel` class** which is checkpoint-agnostic. In fact it works also with other `checkpoint` (i.e. model is strings) like `distilbert-base-uncased` or `roberta-base`.

Obviosuly using `from_pretrained` we cannot set the `config` we want but we have to use the ones used by the authors of the pre trained model. Now we can use this pre trained model to do inference or to fine tune it on our specific task. Starting from a pre trained mdoel even with a small dataset and with a non enormous numbe of epochs we can get good results. As we have already seen the first time we run `from_pretrained` the model is cached in our system in `~/.cache/huggingface/transfomers/` so that the next time we call it the model does not have to be downloaded again.

### Saving models 

Saving a model is very easy, we have to use the `save_pretrained` method passing in input the path where we want the model to be saved.

```python
model.save_pretrained("path/to/save")
```

This produces two files inside the path we have specified:

- `config.json` → contains the configuration of the model
- `pytorch_model.bin` → contains the weights of the model. This is like the `state_dict` of a `torch` model that we save with `torch.save`.

Essentialy with `config.json` we construct the model archtitecture and with `pytorch_model.bin` we insert the weights inside it.

### Loading models

To load a model we can use the `from_pretrained` method passing in input the path where the model is saved in our machine instead of passing the model id of the model checkpoint in the Model Hub as we did previously.

```python
model = BertModel.from_pretrained("path/to/save")
```

### Using a model for inference

We have actually already seen how to use a model for inference.

- We start from a set of input sentences
- These sentences have to pass throught the tokenizer in order to be converted into sequences of input `ID`s that are understandable by the model. We will see more in details how this process is carried out in the next section.
- Once we have the input `ID`s before passing them to the model we have to make sure they are `tensors`, since this is the only data type supported by `transformers` models. 

In order to tranform some data into `tensor` they have to be in a rectangular shape, so a collection of elements all of the same shape. By default the tokenizer will return the input `ID`s as a list of lists, as follows:


```python
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
```

Since these are already in rectangular shape we can simply convert them into a `torch tensor` doing:

```python
input=torch.tensor(encoded_sequences)
```

Then to do inference with the model is super simple, it's like doing the `forward` step on a `torch` model:

```python
outputs = model(input)
```

## Tokenizers

Now we will take a deep look at **tokenizers** which constitute a very important step in the usage of `transformers` models. These algorithms are used to convert the input text into a sequences of input `ID`s that are understandable by the model.

There are different approaches to tokenize a text. They have their pros and conds and in particular we want to obtain a representation that is not too big and that is also meaningful. 

### Word-based tokenizers

The simplest thing we can do that convert a text into a series of tokens is to divide it into words, so we `split` at every whitespace in the prompt. This can be easily done in `python` using the `split()` method: 


```python
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
```

This produces the following output:

```python
['Jim', 'Henson', 'was', 'a', 'puppeteer']
```

The next step is to use a vocabulary, which will be the collection of all the unique tokens across our input sequences, to assign an unique input `ID` to each token. 

There are several drawbacks in using a word-based tokenizer. First of all the number of possible word s is very huge (i.e. 500,000 in English). Moreover similar words (i.e. `dog`,`dogs`,`run`,`running`) will have different `ID`s and thus will be considered as two completely different things by the model

Moreover we have to consider that we also need to include some special tokens inside our vocabulary. One of these is the *unknown* token, usually represented by `[UNK]` that is used to represent words that are not present in the vocabulary.

>[!note]
> In fact we may not be able to maintain a 500,000 tokens dictionary (as it would be if we use the entire set of English words)

Clearly we would like to have the least number of words to be mapped to the `[UNK]` token.

### Character-based tokenizers

Another approach is to look at characters as tokens. 

In this way we have a much smaller vocabulary (since there are twenty-ish letters in the English alphabet) and we can also include special characters and punctuation as tokens.

A drawback is obviously the fact that a character alone is not that meaningful, at least in English or European languages. In other languages like Chinese or Japanese this approach may be more meaningful. Having characters full or meaning or not in any case with a character-based tokenizer we have a huge number of tokens per sentence, at least 10 times more. This is a huge problem for `Transformer` based models which have a quadratic complexity in the number of tokens in the input sequence inside the Attention layer. That's the main reason why character-level tokenizers are not used in practice.

### Subword tokenizers

As usual at this point we have to use an hybrid approach, **subword tokenizers**. In these tokenizers frequently used words should not be splitted into multiple tokens while complex words can be divided into multiple meaningful tokens. 

>[!example]
> The word `annoyingly` can be divided into two tokens: `annoying` and `ly` 

There are other methods like:

- Byte-level `BPE` (Byte Pair Encoding) used in `GPT2`
- `WordPiece` used in `BERT`
- `SentencePiece` or `Unigram` used in multilingual models

>[!important]
> Some very interesting things about tokenizers (and also the explanation of Byte Pair Encoding) can be found in the `Week 1.md` note inside `transformers_tutorial/LLM Course` folder. These are the notes from the `LLM` course on Coursera.

### Loading and saving tokenizers

Loading and saving tokenizers is similar to models. Also in this case we will use the `from_pretrained` method to load a tokenizer from the Model Hub or from a path in our machine. 

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
```

It's preferable to sue the `AutoTokenizer` class which is checkpoint-agnostic. 

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

The `save_pretrained` method is used to save a tokenizer in a specific path in our machine. 

```python
tokenizer.save_pretrained("path/to/save")
```

When we save a tokenizer we will save the algorithm used to tokenize the input (which is similar to the model architecture) and the vocabulary (which is similar to the model weights).

Once we have a tokenizer we can feed it with a sentence and it will spit out a sequence with the input `ID`s as well with other information:


```python
tokenizer("Using a Transformer network is simple")
```

The output is:


```python
{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Let's now focus on how the input `ID`s are generated by the tokenizer.

### Tokenizer Encoding

The first step in a tokenizer is the **encoding step**. This is composed of two steps:

- **Tokenization** → the input text is divided into tokens
- The tokens are converted into `ID`s

The conversion into tokens may change from model to model and also the vocabulary (which is crucial for the secon stepm to produce the input `ID`s) may change. For this reason when we load a tokenizer we have to pass the model checkpoint in order for the tokenizer to load the correct algorithm and vocabulary for the specific model we are working with.

#### Tokenization

The tokenization step can be achieved using the `tokenize` method of the tokenizer. This method will return a list of tokens that are part of the vocabulary of the tokenizer. 

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

which will output something like:

```python
['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
```

we can see that the `transformer` word is divided into two tokens `transform` and `##er`, while the other tokens are actuall all full words. These are frequent words so it makes sense that they are kept alone.

#### From tokens to input `ID`s

The conversion to input `ID`s is managed by the `convert_tokens_to_ids()` method:


```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

which will output something like:

```python
[7993, 170, 11303, 1200, 2443, 1110, 3014]
```

This outputs can then be packed into a `tensor` and passed in input to the model.

>[!note]
> Interestingly here we do not have the 101 and 102 token at the beginning and at the end of the sequence. Maybe they are added automatically by some other functions.

### Decoding

The **decoding step** is used to pass from a sequence of input `ID`s to a textual output. This is used to understand the output of the model and provide it into a digestible format for humans. In fact this method does not only produced the text divided into tokens (i.e. the output of the `tokenize` method) but it also groups togethert tokens that are part of the same word to produce a text like the input one we started from. The method that does that is `decode`:

```python
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
```

## Handling multiple sequences

Up to now we have worked with inputs that were single sequences, but what about when we have multiple sequences, or sequences of different lengths? We have to imagine that in pre training and fine tuning a model we essentially have to repeat the processes we have seen up to now for a huge number of sequences which most likely have different lengths.

As a matter of facts `transformers` models expects multiples sequences in input by default, thay are actually not built to accept single inputs since there are very rare cases (like the small little introductory example we have done in this tutorial) in which these models take just a single output.

In fact let's say that we have loaded a tokenizer and a model and with the tokenizer we have converted an input prompt into a sequence of input `ID`s. If we now convert this sequence into a `torch.tensor` and fed it in input to the model we will get an error:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)
# This line will fail.
model(input_ids)
```

```bash
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
```

In fact now we have tried to do by hand what the `tokenizer` object does automatically but we have missed one step. After having generated the list of input `ID`s an additional dimension is added to it in case we have a single input, so it becomes a mini-batch of size 1.

In fact the code:


```python
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
```

returns:

```bash
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102]])
```

The `[[` tells us that an additional dimension was added. Here the shape of this tensor should be `torch.Size([1, 16])` where `1` is the batch size and `16` is the length of the longest sequence in the batch.

>[!note]
> Differently from the output we obtained by hand we also have the 101 and 102 initial and final tokens. Even though it is not mentioned in the course yet I think that this is another step automatically managed by the `tokenizer` object.

>[!info]
> The process of sensing multiple sequences in input to a model is called *batching*

In case we have a single sequence we can perform batching in two ways:

- Adding an additional dimension doing: `torch.tensor([ids])
- Doubling the input `ID`s sequence: `torch.tensor([ids,ids])`

#### Sequences of different lengths

In general it is very common to have to deal with multiple sequences of different lengths. This is a problem because if we put together our sequences in a list of lists they won't have the rectangular shape needed to be convertd into tensors if they have different lengths.

For example this batch of sequences:


```python
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
```

Will throw an error if we try to convert it into a tensor. 

The solution is to use padding. We will substitue all the missing indexes in the shorter sequence/s with a `padding_id` value (i.e. normally 0) until it matches the length of the longest sequence in the batch.

So in that simple example, padding will result in:


```python
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```

Each tokenizer has its `padding_id` stored in `tokenizer.pad_token_id`. Let's use it to pad our sequence and send it to the model for inference:


```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
```

The output is:


```python
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
tensor([[ 1.5694, -1.3895],
        [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)
```

It is a bit strange because in the last `tensor` we should have the second raw of logits equal to the one of the previous outout `tensor` (since they are both representing the logits of the second sequence) but this is not the case. This happened because the padded token (i.e. 100 in this case) was interpreted as an input token by the Attention layer and not just as a placeholder. 

#### Attention Mask

To avoid the issue described above we need to tell to the Attention layers to ignore the padding tokens. This is done through the **Attention Mask** we have already seen in the outputs of the `tokenizer` object. 

This is a sequence with the same length of all the others containing `1` in correspondance of the tokens that have to be considered inside the Attention layer and `0` in correspondance of the padding tokens or tokens we do not want to attend.

Let's create and attention mask to ignore the padding tokens:


```python
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

The output is:

```python
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
```

Now the two logits associated to the second sequence correspond to the ones we obtained in the previous output.

>[!note]
> I tried to create a list of lists with the two sentences:
> - "I've been waiting for a HuggingFace course my whole life."
> - "I hate this so much!"
> and I passed it directly to the tokenizer `tokenizers(prompt,return_tensors="pt")` (without doing the intermediate steps by hand) and it gave me an error saying to pass `padding=True` and `truncation=True`. So it means that this padding thing is not done automatically. So now we understand the meaning of the `padding` argument we saw in the previous chapter when we used the `tokenizer` object.

>[!warning] Exercise results
> In the course it asks to do an exercise where we have to do the same thing as in the example (where adding padding and changing the attention mask is necessary to obtain correct results) but there is one thing that it is not telling. In fact I correctly batched together the `input_ids`, obtained using `tokenizer.tokenize` + `tokenizer.convert_tokens_to_ids`, applying padding to the second shorter one. 
> However feeding these `batched_ids` to the model does not produce the same outputs as the ones obtained using the automatic `tokenizer(prompts,padding=True,truncation=True,return_tensors="pt")` method, and that's because we are missing the 101 and 102 initial and final tokens.

>[!success]
> I discovered, thanks to Copilot, that these 101 and 102 special tokens can be accessed doing `tokenizer.cls_token_id` and `tokenizer.sep_token_id`.

>[!warning]
> This passage of adding `tokenizer.cls_token_id` and `tokenizer.sep_token_id` to the beginning and the end of the sequence has to be performed **before applying the padding**. In fact the token 102 (i.e. `tokenizer.sep_token_id`) has to preeced the padding tokens.

### Longer sequences

In `Transformer` models we have also to consider that we cannot use any sequence length we want. In fact there is a limit to the maximum sequence length these models can process. This is also known as the **context window** in `LLM` terminology.

Here we can discover the meaning of the `truncation` argument contained inside the `tokenizer` `forward` method. In fact the simplest approach to deal with sequences exceeding the context window is to truncate them at the maximum lenght and maybe pass the rest of the sequence in a second batch.

## Putting it all together

In the last chapter we have explored at what happens inside the `tokenizer` `API` and try to do everything by hand. It took some time but actually in real cases we won't need to go through all those intermediate steps in fact instances of the class `AutoTokenizer` can do everything in automatic for us and will prepare the inputs to the model in the correct way.

This `tokenizer` object is very powerful, in fact it can:

- Tokenize a single sequence:

```python
sequence="I've been waiting for a HuggingFace course my whole life."
tokenizer(sequence)
```

- Tokenize multiple sequences:

```python
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
]
tokenizer(sequences)
```

- It can pad sequences according to different objectives:

```python
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

- It can truncate sequences:


```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

- It can also handle the conversion to specific type of tensors:


```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

Here it finally talks about the `[CLS]` (i.e. 101) and `[SEP]` (i.e. 102) special tokens. These special tokens have to be added in the right places and this also depends on the specific tokenizer. Some of them may have different names for these special tokens and may also use only one at the beginning or one at the end. In any case the `AutoTokenizer` class will always take care of that depending on the specifics of the tokenizer we are using.

### Tokenizer to produce model outputs

Note that we did not finish yet in listing all the functionalities of  `AutoTokenizer` `API`. This class in fact is used also to produce the predictions of the model into human understandable text in tasks where the model has to predict text (i.e. question answering, summarization, text generation,...). In the example seen during this chapter we focus on the `sentiment-analysis` task we so we did not see this part of the `API`.

In the `text-generation` task for example the final steps of the pipeline are:

- The model produces logits for each word in the vocabulary
- These are turned into probabilities throught the `softmax` function. Now we have a probability distribution over the next word to be generated. In fact in `text-generation` at each step the model has to predicty the next word based on the context (i.e. all the words seen up to now).
- At this point a token is selected based on the probablity values. We have seen in the `LLM Course` that there are different strategy for this selection (i.e. selecting the token with the highest probability may not always be the best choice) and we obtain its input `ID`.
- At this point with the tokenizer we have to convert this input `ID` (plus all the ones of the preceeding words) into their textual representation.

