# toxicity_detector_app
Web application that can identify toxicity of the comments / set of phrases written in English.

## Deployment of application

This requires Docker to be installed on your machine. If you don't have it, please follow instructions on https://docs.docker.com/install/

1. Clone master branch of this repository and open its directory.
2. Run command `docker build -f Dockerfile -t NAME_OF_IMAGE . `
3. Run command `docker run -p 1080:1080 NAME_OF_IMAGE` or `docker run -p -d 1080:1080 NAME_OF_IMAGE` if you want to run Docker container in detached mode.

When container is running you can make requests to it through `curl` in bash or `requests` library in Python.

Example of `curl` request:
```
curl -X POST -H "Content-Type: application/json" \
> -d '{"data":[{"id":1, "text": "I was told that you can make a great burgers. Is it true?"}, {"id":2, "text": "I was told that you are a piece of shit stuffed into a rotten burger. Is it true?"}]}' \
> http://0.0.0.0:1080/toxicity_clf
```

Output:
```
{"1":"{'toxic': 0.0012, 'obscene': 0.0002, 'threat': 1e-04, 'insult': 0.0007, 'identity_hate': 0.0}","2":"{'toxic': 0.924, 'obscene': 0.8531, 'threat': 0.0008, 'insult': 0.6875, 'identity_hate': 0.0024}"}
```

Example of request through Python `requests` library:

Open root directory of the cloned repository and run `python rest_api/api_test.py` (this may require installation of libraries imported in file api_test.py)

***Please notice, that examples in `api_test.py` where created for testing the model and do not reflect my opinion regarding any identities mentioned in the example.***

Output:
```
 {
 "1": "{'toxic': 0.003, 'obscene': 0.0008, 'threat': 0.0005, 'insult': 0.0015, 'identity_hate': 1e-04}",
 "2": "{'toxic': 0.9964, 'obscene': 0.7519, 'threat': 0.0017, 'insult': 0.9904, 'identity_hate': 0.0057}",
 "3": "{'toxic': 0.9194, 'obscene': 0.7017, 'threat': 0.0005, 'insult': 0.5592, 'identity_hate': 0.0039}",
 "4": "{'toxic': 0.0158, 'obscene': 0.0033, 'threat': 0.0004, 'insult': 0.0107, 'identity_hate': 0.0002}",
 "5": "{'toxic': 0.6821, 'obscene': 0.1681, 'threat': 0.0004, 'insult': 0.2278, 'identity_hate': 0.004}",
 "6": "{'toxic': 0.9651, 'obscene': 0.8883, 'threat': 0.0008, 'insult': 0.7631, 'identity_hate': 0.0018}",
 "7": "{'toxic': 0.9311, 'obscene': 0.0145, 'threat': 0.8862, 'insult': 0.2523, 'identity_hate': 0.0625}",
 "8": "{'toxic': 0.7402, 'obscene': 0.0035, 'threat': 0.0015, 'insult': 0.212, 'identity_hate': 0.5906}",
 "9": "{'toxic': 0.9993, 'obscene': 0.0333, 'threat': 0.0063, 'insult': 0.9981, 'identity_hate': 0.8824}",
 "10": "{'toxic': 0.0008, 'obscene': 0.0002, 'threat': 1e-04, 'insult': 0.0004, 'identity_hate': 0.0}",
 "11": "{'toxic': 0.617, 'obscene': 0.0037, 'threat': 0.0508, 'insult': 0.2092, 'identity_hate': 0.0048}"
}
```

## Training the model

### Dataset

Initially the model was trained on dataset from Toxic Comment Classification Challenge hosted by Kaggle (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#description)

Jupyter Notebook `exploratory_notebooks/jigsaw_eda.ipynb' contains exploratory data analysis of the dataset, which added few approaches to preprocessing steps.

After initial model was made it gave poor predictions for comments that contain threats. One of the reason was very few cases of threat class in original dataset. Thus it was decided to use bigger set from Jigsaw Unintended Bias in Toxicity Classification challenge (https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification).
The bigger dataset helped to improve subjective estimation of test phrases from file api_test.py.

### Dataset preprocessing

Before giving the dataset to the algorithm it needs to be prepared. This can be done by loading Jigsaw dataset into folder `raw_data` and the running `data_preparer.py`. Some steps in this preparation file are Jigsaw 2 specific, thus in case of using it for any other dataset the code should be properly adjusted.

During the preparation process the dataset is split for train, validation and test sets. Preprocessing pipeline and tokenizer are fitted on train set and then used to transform validation and test set. Thus the process imitates real-world situation when we don't see new data until we asked to make some predictions on it.

Overall preprocessing can be split into four phases:

1. First step is to preprocess the text in each sample. It is done using sklearn Pipeline where each step is a custom transformer that changes the dataset according to its instructions and then pass it to the next step. As the result Pipeline can be fit on a training data, pickled and then used for transformation of a new data.

2. Keras Tokenizer receives text corpus and take specified number of words into "vocabulary" based on how often they appear. Then each word in a text corpus is presented as an index that this word has in the vocabulary. This tokenization helps to preserve the order of words in text corpus, which is what we need for applying RNN or CNN. Thus the second phase is to train Tokenizer on whole corpus of train set. 

3. For hierarchical attention data should be transformed into 3D vector before it is fed to neural network. This vector must have following dimensions:
    1. Samples
    2. Equal number of sentences per each sample
    3. Equal number of words per each sentence (each word is presented by its index in vocabulary)

This transformation is done by first splitting each sample on sentences with the help of NLTK sentence tokenizer. Then each sentence is split into tokens (words and punctuation). 

### Model training

Model can be trained by running `python model_trainer.py` command. 

The process of training transforms the dataset into 4D vector by replacing each token with 300 dimensional representation for pre-trained GloVe.6B.300d based on Wikipedia articles.

Network goes into each sample and look through sentences in it as "subsamples". Each "subsample" has specific number of words presented by 300d vector. Bidirectional GRU goes through each sentence and creates hidden state for each word-token. Using Self-attention neural network gets weights for each hidden state and calculates context vector by multiplying hidden states by their weights. Then context vector is summed to get a representation of a sentence.

Then the same is done but on a sentence level to get vector representation of whole sample. Thus the name "hierarchical" attention. 

The whole scheme is presented below, except the softmax as the last step. As it is a multiclass problem we use 5 units and sigmoid activation for last dense layer of the network.

![Alt text](./ha.png?raw=true "Hierarchical attention")

Source: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

### Visualization of training process

`model_trainer.py` creates file with visualization of model train vs validation accuracy and loss (`val_acc_loss_charts\val_loss_baseline.png`).

### Model test

Testing of the model can be done with help of command `python model_test.py` which takes test set for the check. The metric applied is ROC AUC averaged for each class.
But the most interesting testing can be done by live feeding model with different comments. 

## Dataset and thus trained model peculiarity

Here is the citing of explanations to the dataset on Kaggle: 
```
"Each comment in Train has a toxicity label (target), and models should predict the target toxicity for the Test data. 
This attribute (and all others) are fractional values which represent the fraction of human raters who believed the 
attribute applied to the given comment. 
For evaluation, test set examples with target >= 0.5 will be considered to be in the positive class (toxic)."
```

Thus trained model learned some subjectivity of raters. We may hope that by averaging subjective assessments of many people we can get an objective point of view. But can opinion of many be considered as one that is free from any bias? Well this is not something I'm not going to discuss here :)
