FROM python:3.6

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

EXPOSE 8501

COPY requirements.txt /tmp

COPY ./src/app.py /tmp/src
COPY ./src/inference.py /tmp/src
COPY ./src/dataset.py /tmp/src
COPY ./src/models.py /tmp/src
COPY ./config /tmp/config
COPY ./test /tmp/test
WORKDIR /tmp

RUN pip install --upgrade pip
RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
CMD ['streamlit', 'run', 'app.py']