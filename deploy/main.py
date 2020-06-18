import sys

sys.path.append("../")

import json
import requests
import spacy
import tokenization
from run_classifier_imbalance import GarbledSentsProcessor

SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner"])


class BertSequenceClassifier(object):

    def __init__(self,
                 task_name="garbledsents",
                 vocab_file="../models/cased_L-12_H-768_A-12/vocab.txt",
                 do_lower_case=False,
                 max_seq_length=128,
                 url='http://localhost:8501/v1/models/garbledSents:predict'):
        processors = {
            "garbledsents": GarbledSentsProcessor
        }

        task_name = task_name.lower()

        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        self.processor = processors[task_name]()
        # 获取标签列表
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self._url = url

    def predict(self, text):
        sentences = [i.text for i in SPACY_NLP(text).sents]
        features = self.processor.extract_features(sentences, self.max_seq_length, self.tokenizer, class_weight=None)
        headers = {"content-type": "application/json"}
        data = json.dumps({"signature_name": "serving_default", "instances": features})
        postrslt = requests.post(self._url, data=data, headers=headers)
        result = json.loads(postrslt.text)
        result = [i["predictions"] for i in result["predictions"]]
        return result


if __name__ == "__main__":
    cls = BertSequenceClassifier()
    text = "Take one of my friends as an example, he works very hard to get a high mark in school to please his parents. " \
           "But, his parents stopped to reward him with pocket money, he lost his interest in studying."
    out = cls.predict(text)
    print("labels={}".format(out))
