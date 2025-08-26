---
id: hugging_face_tutorial_ch_4
aliases: []
tags: []
---

# Hugging Face 🤗 Tutorial Chapter 4

In this note we will take on Chapter 4 of the `HuggingFace` 🤗 course on `NLP`. This chapter is more focused on how to navigate and use the `HuggingFace 🤗` Hub, which is the main website. In particular here we will focus on models and tokenizers. So we will see how to load, use and save pre trained models. In Chapter 5 instead we will focus on datasets.

>[!info]
> Reference to the [Chapter 4 on the `HuggingFace` website](https://huggingface.co/learn/nlp-course/chapter4/1)

## The `HuggingFace` Hub

In this very short introductory section they explain all the functionalities of the `HuggingFace` Hub. Here over 10,000 models are contained and freely available to the community. Each model comes as a sort of `github` repository (and there is also probably a real `github` repo over than the page on `HuggingFace` Hub). 

There are not only models from the 🤗`transformers` library but also `Flair` and `AllenNLP` models for `NLP`, `Asteroid` and `pyannote` for speech and `timm` for vision. 

Another very interesting thing is that whenever a model is shared it automatically deploys a hosted Inference API that let's everyone test the model out for free.

Then they suggest to watch [this video](https://www.youtube.com/watch?v=XvSGPZFEjDY) to get familiar on how to navigate the Model Hub.

## Using pretrained models

With the Model Hub it's very easy to access to model for any task and the model can be used in just a few lines of code.

Let's consider and example. We want a French model to perform Mask Filling, so the task where we want to predic the masked tokens in a sentence. In the Model Hub we can find the `camembert-base` checkpoint. As we have seen in the previous chapters we just need the checkpoint to instantiate the model and use it for example with the `pipeline` function:


```python
from transformers import pipeline

camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
```

The results are the following. In this case it returns the 5 most probable mask filling tokens sampled from the distribution outputted by the model head (obtained after having applied the `softmax` function to the logits):


```python
[
  {'sequence': 'Le camembert est délicieux :)', 'score': 0.49091005325317383, 'token': 7200, 'token_str': 'délicieux'}, 
  {'sequence': 'Le camembert est excellent :)', 'score': 0.1055697426199913, 'token': 2183, 'token_str': 'excellent'}, 
  {'sequence': 'Le camembert est succulent :)', 'score': 0.03453313186764717, 'token': 26202, 'token_str': 'succulent'}, 
  {'sequence': 'Le camembert est meilleur :)', 'score': 0.0330314114689827, 'token': 528, 'token_str': 'meilleur'}, 
  {'sequence': 'Le camembert est parfait :)', 'score': 0.03007650189101696, 'token': 1654, 'token_str': 'parfait'}
]
```

Here in particular each output is a dictionary with the following fields:
- `sequence` → The input prompt sequence with the predicted token instead of the `<mask>` token. 
- `score` → The `softmax` score of the predicted token
- `token` → The vocabulary `ID` of the predicted token
- `token_str` → The string version of the predicted token

Here we have to be careful when selecting the model. In fact here we used the `fill-mask` token and the results are fine and that's because the `camembert-base` model was trained on the `fill-mask` task and thus its head is the head of a model that has to fill the mask. However if we use the checkpoint of a model trained on `text-classification` (such as `bert-base-uncased`) that the performances will be much worse. In fact, as we have seen in [[hugging_face_tutorial_ch_3|chapter 3]], when we load a pre trained model on a task it was not trained on its head will be substituted with the appropriate head for the selected task (i.e the model head of `bert-base-uncased` (which is the model head for a binary classification task) will be substituted with the head of a mask filling model (which will have an outut node per each token in the vocabulary)), which will however have randomly initialized weights and thus will not perform very well.

So we have to look at `Task` tags in the model card when we search for a model and want to test it through the `pipeline` function.

Let's try to do an experiment and see what happens if I try to perform a `text-classification` task with `camembert-base` as the checkpoint.

We can also instatiante a model directly using its specific model architecture. In this case we have also to load the tokenizer since we will need it to pre process the input data before feeding the model with them.


```python
from transformers import CamembertTokenizer, CamembertForMaskedLM

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
```

However it is recommended to use the `Auto*` model and tokenizer classes which are task specific and architecture agnostic.


```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```

>[!important]
> When we use a model from the Model Hub it is always recommended to read carefully the model card to understand how it was trained, its limits and biases, the dataset on which it was trained on. These are all important information we should now also to understand what to expect from the model. These information may help in the choice of the data on which to fine tune the model on.

## Sharing pre trained models 

In this section we will explore how to share the weights of a model on the Model Hub and how to manage the repository associated with it in order to make it easy for the other users to use the model.

There are three ways to create a new model repository:

- Using the `push_to_hub` API
- Using the `huggingface_hub` Python package
- Using the web interface

Once the repository is create we can upload files to it using `git` or `git-lfs` as in a normal `git` repo.

Let's explore these methods more in details.

### Using the `push_to_hub` API

This is considered the easiest way to upload files to the Hub. 

First thing to do is to generate some authentication tokens so that the `huggingface_hub` `API` knows who we are and what namespaces we can write to. In order to do that we have to be inside an environment where `transformers` is installed (for me right not this is the `hf` `conda` 🐍 environment) and use this command:


```zsh
huggingface-cli login
```

Now we should be prompted for the username and password and then our tokens will be saved to our cache and we can now start to create files and repositories.

Actually it seems that I have to generate the token first and then when I do `huggingface-cli login` I have to insert the token. So it's like in `github` that they have passed from the password to the token authentication.

I will keep here the `test` Access token I just generated:

```txt
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
```

After we inser the token on the `cli` command that is saved in `~/.cache/huggingface/token`

>[!note]
> Here `token` is simply a text file that contains the token string as it is written above.

As we have already seen in [[hugging_face_tutorial_ch_3#fine-tuning-a-model-with-the-trainer-api|chapter 3]] we can push a model to the hub through the `Trainer` API simply passing the `push_to_hub=True` argument to the `TrainingArguments` class. There are however some other arguments we can pass:


```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
)
```

Here we have also passed:

- `bert-finetuned-mrpc` → This is actually the name of the folder where the model is saved at certain moments during training depending on the `save_strategy` parameter. In case we pass `push_to_hub=True` this will be the name of the repository that will be created on the Hub with all the files that are normally generated when we save a model. In case `push_to_hub=False` than all those files will be saved in a folder on our local machine. In case we do not want to save the model with the `bert-finetunes-mrpc` name we can specificy a specific name for the Hub repository passing it through the parameter `hub_model_id="a_different_name"`. Moreover we can also pass something like `hub_model_id="my_organization/my_repo-name"` if we are part of an organization.
- `save_strategy` → The frequency with which we save a model, here with `epoch` we save the model at the end of each epoch.

Once training is finished we need also to add the `trainer.push_to_hub()` command to complete the push operation, which will upload the last version of the model (i.e. the one obtained at the last epoch). Moreover it will als automatically generate a default model card that will simply contain a list of all the hyperparameters that were used during training  and some metric results.

We can also use the `push_to_hub()` method on any model, tokenizer and configuration object we have loaded from the Hub in order to save them in our own repository after we have performed changes on it. This is very cool 💪. Let's see an example.

Let's say that we want to do some fine tuning or modifications on a model from the Hub, like the `camembert-base` model we used in the previous example (this is exactly what I will do with `chronos` for the `chronos-pdm` project) 


```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

After having done all the training, modifications, fine tuning or whatever we can call `push_to_hub()` to the model and tokenizer objects. For example we can do:


```python
model.push_to_hub("dummy_model")
```

This will create a new repository on the Hub with the name `dummy_model` and will upload the model weights, the configuration file and the model card. The same can be done for the tokenizer object. If I do it with the tokenizer passing the same name that also the tokenizer file will be added to repo to have a complete set of files:

```python
tokenizer.push_to_hub("dummy_model")
```

If we are part of an orgnization we can pass its name to the `organization` parameter:


```python
model.push_to_hub("dummy_model", organization="my_organization")
```

We can also specificy a particular authentication token passing the `use_auth_token` parameter:


```python
tokenizer.push_to_hub("dummy-model", organization="huggingface", use_auth_token="<TOKEN>")
```

>[!question] Why passing the auth token?
> Passing the authentication tokens seems useful. However when I created the auth token previously it asked me to specify which kind of permission I wanted to use (write/read access to my repositories, to public ones ecc ecc). So this is useful because I may have differ auth tokens with different permissions and here I can specify the one I want to use.

>[!note]
> The great thing about the `push_to_hub` method is that it takes care of creating the repository, pushing files to it and even it creates a starting point for the model card and no additional manual handling is needed. This is not possible with the other two methods which require some manual handling.

A very powerful thing is that we can use the `push_to_hub()` method also on a `Trainer` object. This will upload to an existing (or a new) repository the model, its configuration, the tokenizer and also created a draft for the model card. This is a lot of information inserted at once! 💪

>[!important]
> There are several options and input arguments possible for the `push_to_hub` method, check them out in the [documentation](https://huggingface.co/docs/transformers/model_sharing).

### Using the `huggingface_hub` Python package

This python package is what there is behind the `push_to_hub` method. It offers an API to manage and intereact with the repositories we have in the Hub. Also in this case it is necessary to perform the login operation before using it, in the same way we did in the previous section for `push_to_hub`.

There are several methods offered by the `huggingace_hub` library:


```python
from huggingface_hub import (
    # User management
    login,
    logout,
    whoami,

    # Repository creation and management
    create_repo,
    delete_repo,
    update_repo_visibility,

    # And some methods to retrieve/change information about the content
    list_models,
    list_datasets,
    list_metrics,
    list_repo_files,
    upload_file,
    delete_file,
)
```

It also offers the very powerful `Repository` class that can be used to manage a local repository. We will explore it in the next sections when we'll see how to manage file inside a repository after having created it.

In order to create a repository in the Hub we can use the `create_repo` method:


```python
from huggingface_hub import create_repo

create_repo("dummy-model")
```

Also in this case we can pass the `organization` argument if we want to initialize the repo inside an organization.


```python
from huggingface_hub import create_repo

create_repo("dummy-model", organization="huggingface")
```

Other useful arguments we can pass are the following:

- `private` → If we want to create a private repository
- `token` → If we want to override the token stored in the cache by a given token
- `repo_type` → This specifies the repository type. Other than `models` we can in fact also create `datasets` or `space` repositories.

Now that we have seen how to create a repository let's see how to add files to it.

>[!info]
> I will skip to insert here the notes on how to create a new repository throught the web interface. It is very similar to the `UI` of `github`.

### Uploading the model files

The system under the file managment inside a 🤗 repository is based on `git` for regular files and on `git-lfs` (Git Large File Storage) for large files.

Let's now explore three different ways of uploading files to the hub: through `huggingface_hub` or throught `git` commands.

#### The `upload_file` approach

This approach does not require to have a `git` or `git-lfs` installation on our system because performs the upload as an `HTTP POST` request. The limitation is that it works with files with a size up to 5 GB, so if we want to upload a file that is heavier than 5 GB we have to resort to the other two methods. In any case the `API` is the following:


```python
from huggingface_hub import upload_file

upload_file(
    "<path_to_file>/config.json",
    path_in_repo="config.json",
    repo_id="<namespace>/dummy-model",
)
```

Other useful arguments are:

- `token` → To override the token stored in the cache
- `repo_type` → We can pass `dataset` or `space` if we are not uploading a model file

#### The `Repository` class

This class manages a local repository in a `git`-like manner, so it requires us to have a `git` and `git-lfs` installation on our machine. It is very useful because it manages in the background all the possible errors that `git` may throw at us when dealing with large files or other things that we need to have when working with these models that are normally not possible to do in `git`.

The repository we have created with the commands seen in the previous section is hosted on the Hub, so it is a remote repository like the ones we may have on our `github` account. So first of all we can clone it in a folder locally on our machine:


```python
from huggingface_hub import Repository

repo = Repository("<path_to_dummy_folder>", clone_from="<namespace>/dummy-model")
```

This is essentially equivalent to the `git clone <repository_url>` command we use in `git`.

This will create the folder `<path_to_dummy_folder>` in our working directory and will clone the repository in it. Initially this folder will only contain the `.gitattributes` file that is the only file created when we instantiate the repo with the `create_repo` command.

From this point onwards we can leverage all the traditional `git` comands:


```python
repo.git_pull()
repo.git_add()
repo.git_commit()
repo.git_push()
repo.git_tag()
```

>[!info]
> To check all the other methods available in `Repository`, check the [documentation](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub#advanced-programmatic-repository-management)

Let's consider the case in which we have a model and a tokenizer we wish to save to the Hub. We have create the repo and cloned it locally. The first thing to do will be to check if there is something to pull from the remote repository:


```python
repo.git_pull()
```

We can now save the model and tokenizer locally in the repo root folder using the `save_pretrained` method:


```python
model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
```

Then we can do the usual 3 step process of `git` to push a file to the remote repository: `add`, `commit` and `push`:


```python
repo.git_add()
repo.git_commit("Add model and tokenizer files")
repo.git_push()
```

### The `git`-based approach

This approach uses directly `git` and `git-lfs` and its the very barebone approach without all the abstractions provided by the other two methods. There are a few caveats with this method we will consider a more complex use case to see how to manage them.

First of all we need to have `git` and `git-lfs` installed. `git` is usually installed by default on most systems, while `git-lfs` may not be. We can simply install it through:


```zsh
git lfs install
```

We can clone the remote repository locally by doing:


```zsh
git clone https://huggingface.co/<namespace>/<your-model-id>
```

>[!note]
> Note that here the format of the `url` is different from the one we usually have with `git` repositories. However the structure is more or less the same:
> - We have `https://huggingface.co` as the base url instead of `https://github.com`
> - Then we have the `namespace` of the user or organization that owns the repository (this will be `FrizzoDavide` in my case) 
> - Then we have the `model-id` that is the name of the repository we have created, this is equal to `github`

So if I create a repository with the name `dummy` the command will be something like:


```zsh
git clone https://huggingface.co/FrizzoDavide/dummy
```

Now we have the `dummy` folder in our machine and we can `cd` into it and do whatever we want. The content of the folder depends on the approach we used to initialize the repository:

- If we used `create_repo` we should have only the `.gitattributes` file
- If we created the repo throught the web interface we will also have the `README.md` file, together with `.gitattributes`

Now adding regular size file (i.e. a few MBs) works exactly the same as in `git` (I wonder weather `lazygit` will work here), but if we want to push large files we have to use `git-lfs` commands.

For example let's say that we have loaded a model and a tokenizer from the hub and we have worked with them a little bit:


```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Do whatever with the model, train it, fine-tune it...

model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
```

Since we have saved the model and tokenizer with `save_pretrained` we will see some more files doing `ls`:


```zsh
config.json  pytorch_model.bin  README.md  sentencepiece.bpe.model  special_tokens_map.json tokenizer_config.json  tokenizer.json
```

Looking at the sizes of the files using `ls -lh` we can see that `pytorch_model.bin` is the largest one and has a size of about 400 MB. This is a large file for `git` and so it will be tracked by `git lfs`.

>[!note]
> The `.gitattributes` files is built to automatically track files with certain extensions (e.g. `.bin`,`.h5`) with `git-lfs` without the need for us to perform this setup by hand.

Now we can go on as in any `git` repo adding the files to the staging area:


```python
git add .
```

Now if we look at the repository status doing `git status` we have:


```zsh
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
  modified:   .gitattributes
	new file:   config.json
	new file:   pytorch_model.bin
	new file:   sentencepiece.bpe.model
	new file:   special_tokens_map.json
	new file:   tokenizer.json
	new file:   tokenizer_config.json
```

We can see that the `.bin` file containing the model weights is not tracked by `git`. In fact, since it is a large file, it is trakced by `git-lfs` and thus to see it in the staging area we have to do `git lfs status`:


```zsh

On branch main
Objects to be pushed to origin/main:


Objects to be committed:

	config.json (Git: bc20ff2)
	pytorch_model.bin (LFS: 35686c2)
	sentencepiece.bpe.model (LFS: 988bc5a)
	special_tokens_map.json (Git: cb23931)
	tokenizer.json (Git: 851ff3e)
	tokenizer_config.json (Git: f0f7783)

Objects not staged for commit:

```

Now we have all the staged files in the list and inside the round brackets we can see that all of them are handled by `Git` while `pytorch_model.bin` is handles b `LFS`. 

Now we can commit the changes:


```zsh
git commit -m "First model version"
```

and we have:


```zsh
[main b08aab1] First model version
 7 files changed, 29027 insertions(+)
  6 files changed, 36 insertions(+)
 create mode 100644 config.json
 create mode 100644 pytorch_model.bin
 create mode 100644 sentencepiece.bpe.model
 create mode 100644 special_tokens_map.json
 create mode 100644 tokenizer.json
 create mode 100644 tokenizer_config.json
```

Finally we can push with `git push`. This command may take some time depending on out internet connection and how big the files we are pushing are.

Now loking at the model repository from the Hub we can see all the added files and the `UI`, like in `github`, let use browse the commits and the diffs in the files.

>[!success] `lazygit` works here too 🎉
> I created a new repository on the Hub and cloned it locally. Now I am running a simple `torch` training script to fine tune the `bert-base-uncased` model on the `sst2` task and then I will push the created files to the Hub. If I go inside the `bert_sst2` folder in `acquario3` (that is where I have my repo) and I run `lazygit` it seems to work.
> Yes, it works! Now that the experiment has finished I was able to see the new created files as files to add and commit throught `lazygit`. Then I did everything through the `git` cli because I wanted to see if the steps coincided with the ones I have written here. And they did! 🎉

## Building the Model Card

The model card is as important as the model and the tokenizer inside a 🤗 repository. It is essentiallty the same thing as the `README.md` file in a `github` repository. 

Giving instructions to the other people that may acess our model repository on how the model was trained, how it performed, how the tokenzier works and how to access and use it in python is very important. It makes it easier for the other people to use our model and to understand what we have done. 

In fact differently from the traditional `github` repositories here in general there are not the `py` script file where someone can try to understand what we have done reading the code. There are in fact only the files that are created when the model and tokenizers are saved.

For example I am watching the [repo of `DeepSeek r1` in the hub](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main) and looking at the files there are a bunch of `.safetensors` files.

>[!note]
> I think that the `.safetensors` file are equivalent to the `.bin` files that were cited in the previous sections. In fact when I tried out the repo I had no `.bin` file and the file that was handled by `LFS` for me was the `.safetensors` file. These files should contain all the model weights and that's the reason why they are so big. The ones on the `DeepSeek r1` repo are all around 4.3 GB.

Actually here in this repo there is also a `py` file called `configuration_deepseek.py` which however seems pretty complex so it's clearly better to have a well documented Model Card.

Actually the concept of model card was introduced in a [paper by Google](https://arxiv.org/abs/1810.03993) and most of the indications contained in this sections are taken from that paper.

Normally we start with a brief high-level description of what the model is used for and then it is divided in the following sections:

- Model description
- Intended uses & limitations
- How to use
- Limitations and bias
- Training data
- Training procedure
- Evaluation results

Now the chapter becomes a bit discoursive and says pretty obvious stuff, I don't want to rewrite everything in here, so [look here](https://huggingface.co/learn/nlp-course/chapter4/4) for more details.

### Model card metadata

Once we enter inside the page of a model we can see all the tags on the top that specify the task, language, dataset, license, paper of the model. In order to add them we have to specify the model metadata at the beginning of the model card Markdown file, in the header/frontmatter specifically. For example if we enter in the `README.md` file inside the repo of the `camembert-base` model card we can see:


```python
---
language: fr
license: mit
datasets:
- oscar
---
```

which says that the model is trained on a french dataswet, the license is `mit` and the dataset used is `oscar`.


