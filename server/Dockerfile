
FROM python:3.9

LABEL maintainer "Ivanpuzako@gmail.com"


RUN pip install mlflow && \
    pip install awscli --upgrade --user && \
    pip install boto3

ENV PORT 5000

COPY files/run.sh /
ENTRYPOINT [ "./run.sh" ]