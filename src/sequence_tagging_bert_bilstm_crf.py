# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("../")

import os
import collections
import csv
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
import tf_metrics
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.contrib import *
from tensorflow.contrib.distribute import AllReduceCrossDeviceOps
from tensorflow.python import keras
from tensorflow.contrib.layers.python.layers import initializers

import torch

from bilstm_crf import BiLSTM_CRF

from transformers import TFBertModel
from transformers import BertTokenizer

model = TFBertModel.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

L = keras.layers

import custom_optimization
import modeling
import optimization
import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

# 是否进行字符串转小写操作
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool("do_early_stopping", True, "Whether to do early stopping")

flags.DEFINE_integer(
    "max_seq_length", 64,
    "The maximum total input sequence length")

flags.DEFINE_integer(
    "max_token_length", 128,
    "The maximum total input token length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_string("train_file", "train.tsv", "train file name in data_dir.")
flags.DEFINE_string("eval_file", "dev.tsv", "evaluate file name in data_dir.")
flags.DEFINE_string("predict_file", "test.tsv", "test file name in data_dir.")

flags.DEFINE_string("classifier_mode", "binary", "binary or multi-class for classifier.")

# 初始学习率 for Adam 优化器
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

# 总训练轮数
flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

# early stopping parameters
flags.DEFINE_integer("min_steps", 1000, "The minimum steps for early stopping")
flags.DEFINE_integer("max_steps_without_increase", 5000,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("max_steps_without_decrease", 5000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("class_weight", None, "Whether to use class weight(balanced or None).")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "num_gpu_cores", 2,
    "OTotal number of GPU cores to use.")
flags.DEFINE_bool("use_fp16", False, "Whether to use fp16.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, texts, labels=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          texts: list of string. The untokenized text of the sequence.
          labels: (Optional) list of string. The labels of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.texts = texts
        self.labels = labels


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


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
                if len(lines) > 1000:
                    break
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
        tag2id = {'B_C0': 0, 'B_C1': 1, 'B_I0': 2, 'B_M0': 3, 'B_M1': 4, 'B_S0': 5, 'B_S1': 6, 'B_S2': 7, 'B_T0': 8,
                  'B_T1': 9,
                  'E_C0': 10, 'E_C1': 11, 'E_I0': 12, 'E_M0': 13, 'E_M1': 14, 'E_S0': 15, 'E_S1': 16, 'E_S2': 17,
                  'E_T0': 18,
                  'E_T1': 19, 'I_C0': 20, 'I_I0': 21, 'I_M0': 22, 'I_S0': 23, 'I_S1': 24, 'I_S2': 25, 'I_T0': 26,
                  'I_T1': 27,
                  # 'O': 28,
                  'S_C0': 29, 'S_C1': 30, 'S_C2': 31, 'S_I0': 32, 'S_M0': 33, 'S_M1': 34, 'S_M2': 35,
                  'S_S0': 36,
                  'S_S1': 37, 'S_S2': 38, 'S_T0': 39, 'S_T1': 40}
        # O(other) 标签放在索引0位置上
        label_list = ['O'] + [str(i) for i in tag2id.keys()]
        print("{} label_list={}".format(len(label_list), label_list))
        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        enum = 0
        texts, labels = [], []
        for (i, line) in enumerate(lines):
            if i <= 3:
                print("i={}, line={}".format(i, line))
            if len(examples) > 10:
                print("i={}, examples[-1].texts={}".format(i, examples[-1].texts))
                break
            if len(line) < 2:
                # new example
                guid = "%s-%s" % (set_type, enum)
                examples.append(InputExample(guid=guid, texts=texts, labels=labels))
                enum += 1
                texts, labels = [], []
            else:
                if set_type == "test":
                    text = tokenization.convert_to_unicode(line[0])
                    label = "0"
                else:
                    text = tokenization.convert_to_unicode(line[0])
                    label = tokenization.convert_to_unicode(line[1])
                texts.append(text)
                labels.append(label)
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, max_token_length, tokenizer):
    """
    Converts a single `InputExample` into a single `InputFeatures`.

    Args:
        ex_index: int, example index
        example: SequenceExample
        label_list: list
        max_seq_length: int, 最大序列长度，即句子个数， 默认64
        max_token_length: int, 最大句子中单词token长度，一般为128
        tokenizer: 分词器

    """

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[[tokenizer.pad_token_id] * max_token_length] * max_seq_length,
            label_ids=[0] * max_seq_length,
            is_real_example=False,
        )

    # 构建标签文本-索引id
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    # 打印label_map
    if ex_index == 0:
        tf.logging.info("{} label_map={}".format(len(label_map), label_map))

    input_ids = tf.constant([tokenizer.encode(tokens, max_length=max_token_length, pad_to_max_length=True) for
                             tokens in example.texts[:max_seq_length]])

    if input_ids.shape[0] < max_seq_length:
        padding_input_ids = tf.constant(tokenizer.pad_token_id,
                                        shape=[max_seq_length - input_ids.shape[0], max_token_length])
        input_ids = tf.concat([input_ids, padding_input_ids], axis=0)

    label_ids = [label_map[i] for i in example.labels]
    if len(label_ids) > max_seq_length:
        label_ids = label_ids[: max_seq_length]
    else:
        label_ids += [0] * (max_seq_length - len(label_ids))

    assert input_ids.shape[0] == max_seq_length
    assert input_ids.shape[1] == max_token_length
    assert len(label_ids) == max_seq_length

    if ex_index < 3:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: {}".format(example.guid))
        tf.logging.info("{} input_ids: {}".format(input_ids.shape, input_ids))
        tf.logging.info("{} labels_ids: {}".format(len(label_ids), label_ids))

    # 转换成InputFeatures对象
    feature = InputFeatures(
        input_ids=input_ids,
        label_ids=label_ids,
        is_real_example=True,
    )
    return feature


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, max_token_length, tokenizer,
                                            output_file):
    """
    Convert a set of `InputExample`s to a TFRecord file.
    1. Feature -> Features（字典） -> Example
    2. Feature -> Features（字典）                    -> context        +
                                                                        } -> SequenceExample
       Feature -> FeatureList -> FeatureLists (dict) -> feature_lists  +

    Args:
        examples: list, 样本s
        label_list: list of label, 全部标签类列表
        max_seq_length: 最大序列长度
        tokenizer: tokenizer
        output_file: string, TF records输出文件

    """

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list, max_seq_length, max_token_length, tokenizer)

        # Context features for the entire sequence
        context = tf.train.Features(
            feature={
                "label_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feature.label_ids)),
                "is_real_example": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[feature.is_real_example]))})

        # Feature lists for the sequential features of the example
        with tf.Session():
            # print("context={}".format(context))
            feat_dict = {"input_ids": tf.train.FeatureList(
                feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=i)) for i in feature.input_ids.eval()])}
            # print("feat_dict={}".format(feat_dict))
        feature_lists = tf.train.FeatureLists(feature_list=feat_dict)

        tf_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, num_labels):
    """
    Creates an `input_fn` closure to be passed to TPUEstimator.
    基于文件的数据输入函数.

    """

    # 上下文特征列匹配
    context_features = {
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    # 输入序列特征列匹配
    sequence_features = {
        "input_ids": tf.FixedLenSequenceFeature([FLAGS.max_token_length], tf.int64),
    }

    def _decode_record(record, context_features, sequence_features):
        """
        Decodes a record to a TensorFlow example.
        解析tf.record文件数据
        """
        # 解析单个样本
        context_output, feature_list_output = tf.parse_single_sequence_example(record, context_features,
                                                                               sequence_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        example = {}
        # 将样本中int64数据类型转换为int32
        # print("context_output: ")
        for name in list(context_output.keys()):
            t = context_output[name]
            # print("key={}, value={}".format(name, t))

            # if t.dtype == tf.int64:
            #     t = tf.to_int32(t)
            example[name] = t

        # graph_tensor = tf.Graph().as_default()
        # init = tf.initialize_all_variables()
        # with tf.Session() as sess:
        #     # sess.run(init)

        print("\n\nfeature_list_output: ")
        for name in list(feature_list_output.keys()):
            t = feature_list_output[name]
            print("key={}, value={}".format(name, t))
            # input_ids
            # t = torch.tensor(sess.run(t).eval(), dtype=torch.long)
            example[name] = t
            print("name={}, t={}".format(name, t.shape))

        # print("example={}".format(example))
        return example

    def input_fn(params):
        """
        The actual input function.
        生成batch样本.

        """
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        dataset = tf.data.TFRecordDataset(input_file)
        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=100)

        # 应用批处理batch size
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, context_features, sequence_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

        # iterator = dataset.make_one_shot_iterator()
        # d = iterator.get_next()
        # print("dataset={}".format(type(dataset)))
        # print("iterator={}".format(type(iterator)))
        # print("d={}".format(type(d)))
        return dataset

    return input_fn


def serving_input_receiver_fn():
    """
    Serving input_fn that builds features from placeholders.

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    input_ids = tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_ids')
    label_ids = tf.placeholder(dtype=tf.int64, shape=[None, None], name='label_ids')
    is_real_example = tf.placeholder(dtype=tf.int64, shape=[None, None], name='is_real_example')

    features = {'input_ids': input_ids,
                "label_ids": label_ids,
                "is_real_example": is_real_example,
                }
    receiver_tensors = {'input_ids': input_ids,
                        "label_ids": label_ids,
                        "is_real_example": is_real_example,
                        }
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 label_ids, num_labels, use_one_hot_embeddings, label_weights,
                 cell="lstm",
                 num_layers=1,
                 dropout_rate=0.1,
                 ):
    """
    Create a model architecture.

    Args:
        bert_config:
        is_training:
        input_ids:
        input_mask:
        segment_ids:
        label_ids: [batch_size=32, ]
        num_labels: int, 类别个数
        use_one_hot_embeddings:
        label_weights: [batch_size=32, num_labels=2]

    """
    # output_layer: [batch_size=32, hidden_size=768]
    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    # with tf.Session():
    #     print(
    #         "{} input_ids={}".format(input_ids.shape, tf.reshape(input_ids, [-1, FLAGS.max_token_length]).eval().shape))

    # with tf.Session() as sess:
    #     input_ids = torch.tensor(sess.run(input_ids).eval(), dtype=torch.long)
    #     sess.close()

    input_ids = tf.reshape(input_ids, [-1, FLAGS.max_token_length])
    _, embedding = model(input_ids)

    print("input_ids={}".format(input_ids.shape))
    # hidden_size: 768

    print("{} embedding".format(embedding.shape))

    hidden_size = embedding.shape[-1].value

    max_seq_length = embedding.shape[1].value

    # 算序列真实长度
    used = tf.sign(tf.abs(input_ids))
    true_lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的真实序列长度

    print("{} true_lengths".format(true_lengths))
    # 添加Bi-LSTM+CRF Layer
    blstm_crf = BiLSTM_CRF(embedding_inputs=embedding, hidden_units_num=hidden_size, cell_type=cell,
                           num_layers=num_layers,
                           dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                           sequence_length=max_seq_length, tag_indices=label_ids, sequence_lengths=true_lengths,
                           is_training=is_training)

    (loss, per_example_loss, logits, probabilities) = blstm_crf.add_blstm_crf_layer(crf_only=True)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, fp16=FLAGS.use_fp16):
    """
    Returns `model_fn` closure for TPUEstimator.

    Args:
        num_labels: int, 类别个数

    """

    def model_fn(features, labels, mode, params, config=None):  # pylint: disable=unused-argument
        """
        The `model_fn` for TPUEstimator.
        模型有训练，验证和测试三种阶段，而且对于不同模式，对数据有不同的处理方式。例如在训练阶段，我们需要将数据喂给模型，
        模型基于输入数据给出预测值，然后我们在通过预测值和真实值计算出loss，最后用loss更新网络参数，
        而在评估阶段，我们则不需要反向传播更新网络参数，换句话说，mdoel_fn需要对三种模式设置三套代码。

        Args:
            features: dict of Tensor, This is batch_features from input_fn,`Tensor` or dict of `Tensor` (depends on data passed to `fit`
            labels: This is batch_labels from input_fn. features, labels是从输入函数input_fn中返回的特征和标签batch
            mode: An instance of tf.estimator.ModeKeys
            params: Additional configuration for hyper-parameters. 是一个字典，它可以传入许多参数用来构建网络或者定义训练方式等


        Return:
            tf.estimator.EstimatorSpec

        """

        print("features={}".format(features))
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        label_ids = features["label_ids"]
        if "is_real_example" in features:
            # 类型强制转换为tf.float32
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            # 创建一个将所有元素都设置为1的张量Tensor.
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        # 根据mode判断是否为训练模式
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 基于特征数据创建模型，并计算loss等
        print("create_model:\ninput_ids={}".format(input_ids.shape))
        print("label_ids={}".format(label_ids.shape))
        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, None, None, label_ids,
            num_labels, use_one_hot_embeddings, None)

        print("total_loss={}".format(total_loss))
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        # 训练模式
        if mode == tf.estimator.ModeKeys.TRAIN:
            if FLAGS.num_gpu_cores > 1:
                train_op = custom_optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, fp16=fp16)

                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold=scaffold_fn)
            else:
                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
        # 评估模式
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                # add more metrics
                pr, pr_op = tf.metrics.precision(labels=label_ids, predictions=predictions, weights=is_real_example)
                re, re_op = tf.metrics.recall(labels=label_ids, predictions=predictions, weights=is_real_example)
                # if FLAGS.classifier_mode == "multi-class":
                #     # multi-class
                #     # pr, pr_op = tf_metrics.precision(label_ids, predictions, num_labels, average="macro")
                #     # re, re_op = tf_metrics.recall(label_ids, predictions, num_labels, average="macro")
                #     f1 = tf_metrics.f1(label_ids, predictions, num_labels, average="macro")
                # else:
                #     # binary classifier
                #     f1 = tf.contrib.metrics.f1_score(label_ids, predictions)
                #     # f1, f1_op = (2 * pr * re) / (pr + re)  # f1-score for binary classification
                # 返回结果：dict: {key: value(tuple: (metric_tensor, update_op)) }
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    "eval_precision": (pr, pr_op),
                    "eval_recall": (re, re_op),
                    # "eval_f1": f1,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits, is_real_example])
            if FLAGS.num_gpu_cores > 1:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=metric_fn(per_example_loss, label_ids, logits, is_real_example),
                    scaffold=scaffold_fn,
                )
            else:
                # eval on single-gpu only
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            # tf.estimator.ModeKeys.PREDICT 预测模式
            # 基于logits计算最大的概率所在索引的label
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            if FLAGS.num_gpu_cores > 1:
                # 多GPUs
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={"probabilities": probabilities, "predictions": predictions})
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={"probabilities": probabilities, "predictions": predictions},
                    scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def print_cls_report(fileTest=os.path.join(FLAGS.data_dir, FLAGS.train_file),
                     fileResult=os.path.join(FLAGS.output_dir, "test_results.tsv"),
                     sep="\t",
                     header="label"):
    try:
        y_true = pd.read_csv(fileTest, sep=sep)[header]
        y_pred = [float(i.strip()) for i in open(fileResult, "r").readlines()]
        rp = classification_report(y_true, y_pred, digits=4)
        print("report:\n{}".format(rp))
    except Exception as e:
        print("Error: {}".format(repr(e)))
    return


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "disc": DiscProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # local_device_protos = device_lib.list_local_devices()
    # FLAGS.num_gpu_cores = min(sum([1 for d in local_device_protos if d.device_type == 'GPU']), FLAGS.num_gpu_cores)
    tf.logging.info(f"Info: {FLAGS.num_gpu_cores} GPUs found")

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    # 获取标签列表
    label_list = processor.get_labels()
    num_labels = len(label_list)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=num_labels,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    if FLAGS.use_tpu and FLAGS.tpu_name:
        tf.logging.info(f"***** {FLAGS.tpu_name} TPU Running *****")
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)

    elif FLAGS.num_gpu_cores > 1:
        # multi-gpus 训练
        tf.logging.info(f"***** {FLAGS.num_gpu_cores} GPUs Running *****")
        # 1.先定义分布式训练的镜像策略：MirroredStrategy
        dist_strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=FLAGS.num_gpu_cores,  # 使用gpu的个数
            cross_device_ops=AllReduceCrossDeviceOps('nccl', num_packs=FLAGS.num_gpu_cores),  # 各设备之间的数据操作方式
            # cross_device_ops=AllReduceCrossDeviceOps('hierarchical_copy'),
        )
        # 2.设置会话session配置
        session_config = tf.ConfigProto(
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))

        # 3.设置运行配置run_config
        run_config = tf.contrib.tpu.RunConfig(
            train_distribute=dist_strategy,
            eval_distribute=dist_strategy,
            model_dir=FLAGS.output_dir,
            session_config=session_config,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=15)
        # 4.构造CPU、GPU评估器对象
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": FLAGS.train_batch_size})
    else:
        # 单GPU或者CPU训练
        tf.logging.info("***** Single GPU/CPU Running *****")
        tpu_cluster_resolver = None
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train and FLAGS.do_eval:
        '''
        pipeline: train examples -> (write tf_record) -> (read tf_record) -> train 
        '''
        # ------------- 1.Train ------------- #
        # 1.1 将文本预处理，并存储为tf.record格式文件
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length,
                                                FLAGS.max_token_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        # 1.2 基于tf.record格式文件进行迭代batch处理样本
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True, num_labels=len(label_list))
        # ------------- 2.Eval ------------- #
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, FLAGS.max_token_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder, num_labels=len(label_list))

        if FLAGS.do_early_stopping:
            # 1.3 early stopping
            early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
                estimator=estimator,
                metric_name='eval_loss',
                max_steps_without_decrease=FLAGS.max_steps_without_decrease,
                eval_dir=None,
                min_steps=FLAGS.min_steps,
                run_every_secs=None,
                run_every_steps=FLAGS.save_checkpoints_steps)
            train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
                                                hooks=[early_stopping_hook])
            eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps, throttle_secs=0)

            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
            tf.logging.info("***** Training & Evaluating completed*****")
        else:
            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
            tf.logging.info("***** Training completed*****")

        # export SavedModel format for TF serving
        export_dir_base = os.path.join(FLAGS.output_dir, 'saved_model')
        estimator.export_saved_model(export_dir_base, serving_input_receiver_fn)
        tf.logging.info("***** SavedModel export completed*****")

    elif FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, FLAGS.max_token_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder, num_labels=len(label_list))

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        tf.logging.info("***** Evaluation completed*****")

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, FLAGS.max_token_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder, num_labels=len(label_list))

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                # 根据输出的索引id找到对应的label
                y_pred = label_list[prediction["predictions"]]
                if i <= 3:
                    print(f"probabilities={probabilities}")
                    print("y_pred={}".format(y_pred))
                if i >= num_actual_predict_examples:
                    break
                # output_line = "\t".join(
                #     [str(class_probability) for class_probability in probabilities] + [str(y_pred)]) + "\n"
                writer.write(str(y_pred) + "\n")
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples
        tf.logging.info("***** Prediction completed*****")
        # 打印输出分类任务的性能指标
        print_cls_report(fileTest=os.path.join(FLAGS.data_dir, FLAGS.predict_file),
                         fileResult=os.path.join(FLAGS.output_dir, "test_results.tsv"),
                         sep="\t",
                         header="label")


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
