
FROM python:3.9

LABEL maintainer "Ivanpuzako@gmail.com"

RUN pip install mlflow && \
    pip install awscli --upgrade --user && \
    pip install boto3 numpy pandas scikit-learn

ENV PORT 5000

COPY mlflow-gridsearch.py /
COPY data.csv /
CMD python /mlflow-gridsearch.py