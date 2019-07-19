FROM python:3.6-slim

RUN apt-get update && \
    apt-get -y install libpq-dev python3-dev libgomp1 procps

COPY ./requirements.txt /toxicity/requirements.txt
WORKDIR /toxicity/

RUN pip install -r requirements.txt

COPY . /toxicity

ENV PYTHONPATH=${PYTHONPATH}:/toxicity/

RUN python3 -m nltk.downloader punkt averaged_perceptron_tagger wordnet
EXPOSE 1020
ENTRYPOINT ["bash", "./rest_api/boot.sh"]
