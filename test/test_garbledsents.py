import sys

sys.path.append("../")
import pandas as pd
from tqdm import tqdm
from sequence_classifier import BertSequenceClassifier
from sklearn.metrics import classification_report


def test_predict(cls):
    text = "Take one of my friends as an example, he works very hard to get a high mark in school to please his parents. " \
           "But, his parents stopped to reward him with pocket money, he lost his interest in studying."
    out = cls.predict(text)
    print("labels={}".format(out))


def test_set(cls, fileIn="/home/hebin/python_work/bert-application/glue/garbledSents/test.tsv"):
    data = pd.read_csv(fileIn, sep="\t")[["sample_sent", "label"]]
    y_pred = []
    print("data shape={}".format(data.shape))
    for i in tqdm(range(data.shape[0])):
        out = cls.predict_sample(data.iloc[i]["sample_sent"])
        y_pred.append(out)
    rp = classification_report(data["label"], y_pred)
    print("report:\n{}".format(rp))


if __name__ == "__main__":
    cls = BertSequenceClassifier("garbledsents")

    test_predict(cls)

    test_set(cls)
