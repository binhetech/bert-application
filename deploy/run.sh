docker run -t --rm -p 8501:8501 \
    -v $(pwd)/model/:/models/model \
    -e MODEL_NAME=model \
    tensorflow/serving &