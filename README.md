# Explainable Sentiment
### RoBERTa based support word extractor 

Explainable Sentiment extracts support word from text according to sentiment label.

## Installation
Git clone (python>=3.6)
```bash
$ git clone https://github.com/Lucas-Wei/explainable_sentiment.git
$ cd explainable_sentiment
$ pip install -r ./requirements.txt
```

Build docker image.
If you have a CUDA-compatible NVIDIA graphics card,
```bash
$ docker build -t explainable_sentiment .
```

```bash
$ docker container run --rm -p 8501:8501 explainable_sentiment
```