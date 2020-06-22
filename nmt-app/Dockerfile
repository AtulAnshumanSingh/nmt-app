FROM python:3.6
WORKDIR /app
COPY requirements.txt /app

COPY nmt.py /app

COPY vocab.py /app

COPY utils.py /app

COPY VOCAB_FILE /app

COPY es_en /app

COPY train.es /app

COPY train.en /app

RUN mkdir -p /app/templates/

COPY templates/home.html /app/templates/

RUN pip install -r ./requirements.txt

COPY app.py /app

CMD ["python", "app.py"]~
