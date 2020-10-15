<h1 align="center">
  <br>
  Explainable Sentiment
  <br>
</h1>

Explainable Sentiment extracts support word from text according to sentiment label.

## Modes
Explainable sentiment has 2 modes: input single text or upload a .csv file

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
Download the fine-tuned trained model
```bash
$ wget https://explainable-sentiment.s3.amazonaws.com/config/RoBERTa_best.pth
$ mv RoBERTa_best.pth ./config/roberta-pths
```

## How To Use
### Local
```bash
$ streamlit run ./src/app.py
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
