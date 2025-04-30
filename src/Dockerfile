FROM python:3.10

WORKDIR /opt/ml_recommender

COPY requirements.txt requirements.txt

RUN  pip install --upgrade pip \
     && pip install -r requirements.txt  --no-cache-dir

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8035

ENTRYPOINT ["bash", "entrypoint.sh"]
