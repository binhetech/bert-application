docker run -t --rm -p 8501:8501 \
  -v /home/hebin/python_work/bert-application/fine_tuned/garbledSents-cased/saved_model:/models/garbledSents \
  -e MODEL_NAME=garbledSents \
  tensorflow/serving &
