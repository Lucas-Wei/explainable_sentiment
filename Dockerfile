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

COPY ./src /tmp/src
COPY ./config /tmp/config
COPY ./test /tmp/test
WORKDIR /tmp/src

RUN pip install -r ../requirements.txt
CMD ["streamlit", "run", "app.py"]