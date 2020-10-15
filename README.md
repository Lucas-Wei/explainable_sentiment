# Explainable Sentiment
### RoBERTa based support word extractor 

Explainable Sentiment extracts support word from text according to sentiment label.

<img src="https://github.com/Lucas-Wei/explainable_sentiment/tree/master/material/explainable_sentiment.gif">

## Installation
Git clone and  (python>=3.6)
```bash
$ git clone https://github.com/Lucas-Wei/explainable_sentiment.git
$ cd explainable_sentiment
$ pip install -r ./requirements.txt
```

## Setup
Download the config files for the pre-trained RoBERTa model
```bash
$ wget https://explainable-sentiment.s3.amazonaws.com/config/roberta-base.zip
```
Unzip it into config fold
```bash
$ unzip roberta-base.zip -d ./config
```

## Docker

Build docker image.
If you have a CUDA-compatible NVIDIA graphics card,
```bash
$ docker build -t explainable_sentiment:cpu .
```

## How To Use
Run the application on your local machine:
```bash
$ python ./src/app.py
```
Or run the application in Docker container:
```bash
$ docker container run --rm -p 8501:8501 explainable_sentiment:cpu
```
