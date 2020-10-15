# Explainable Sentiment
### RoBERTa based support word extractor 

Explainable Sentiment extracts support word from text according to sentiment label.

<img src="https://github.com/Lucas-Wei/explainable_sentiment/blob/master/material/explainable_sentiment.gif">

## Installation
Git clone and install dependencies (python>=3.6)
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

## How To Use
### Local
```bash
$ python ./src/app.py
```
### Docker
1. Build docker image. (If you have a CUDA-compatible NVIDIA graphics card, use the Dockerfile in the fold "Dockerfiles")
```bash
$ docker build -t explainable_sentiment:cpu .
```
2. Run application in Docker container.
```bash
$ docker container run --rm -p 8501:8501 explainable_sentiment:cpu
```
