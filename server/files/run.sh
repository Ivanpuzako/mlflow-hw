#!/bin/sh

set -e

if [ -z "$FILE_DIR" ]; then
  echo >&2 "FILE_DIR must be set"
  exit 1
fi

if [ -z "$AWS_BUCKET" ]; then
  echo >&2 "AWS_BUCKET must be set"
  exit 1
fi

mkdir -p "$FILE_DIR"
mlflow server --host 0.0.0.0 --default-artifact-root s3://${AWS_BUCKET}/artifacts --backend-store-uri /home/logging/