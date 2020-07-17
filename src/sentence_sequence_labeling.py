import tensorflow as tf
from transformers import BertModel, TFBertModel
import csv
import os
import tokenization
import torch

from transformers import pipelines
from transformers import BertForSequenceClassification, BertTokenizer
from bilstm_crf import BiLSTM_CRF

# model = BertForSequenceClassification.from_pretrained('bert-base-cased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

input_ids = tf.concat(
    [tokenizer.encode(tokens, max_length=12, pad_to_max_length=True) for
     tokens in ["ok, thank you", "please do it"]], axis=0)

print(input_ids)

flags = tf.flags

FLAGS = flags.FLAGS


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: list of string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) list of string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar='"'):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class DiscProcessor(DataProcessor):
    """Processor for the Discourse Identifier data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, FLAGS.train_file)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, FLAGS.eval_file)), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, FLAGS.predict_file)), "test")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(13)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        enum = 0
        text_as, labels = [], []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if i <= 3:
                print("i={}, line={}".format(i, line))
            if line.strip() == "":
                # new example
                guid = "%s-%s" % (set_type, enum)
                examples.append(InputExample(guid=guid, text_a=text_as, text_b=None, label=labels))
                enum += 1
                text_as, labels = [], []
            else:
                if set_type == "test":
                    text_a = tokenization.convert_to_unicode(line[0])
                    label = "0"
                else:
                    text_a = tokenization.convert_to_unicode(line[0])
                    label = tokenization.convert_to_unicode(line[1])
                text_as.append(text_a)
                labels.append(label)
        return examples

def model_fn_builder():

    return


def main():
    # processor = DiscProcessor()

    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
        0)  # Batch size 1
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, labels=labels)
    loss, logits = outputs[:2]
    print(loss)
    print(logits)


if __name__ == "__main__":
    main()
