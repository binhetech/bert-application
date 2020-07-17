import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow.contrib import rnn


class BiLSTM_CRF(object):
    """
    Bi-LSTM+CRF model class.
    """

    def __init__(self, embedding_inputs, hidden_units_num, cell_type, num_layers, dropout_rate,
                 initializers, num_labels, sequence_length, tag_indices, sequence_lengths, is_training):
        """
        Bi-LSTM+CRF网络初始化方法.

        Args:
            embedding_inputs: embedding input, [batch_size, max_sequence_length, embedding_dim]
            hidden_units_num: LSTM的隐含单元个数
            cell_type: RNN类型（LSTM OR GRU will be add in feature）
            num_layers: RNN的层数
            dropout_rate: dropout rate
            initializers: variable init class
            num_labels: 标签数量
            sequence_length: int, 序列最大长度
            tag_indices: 真实标签id, [batch_size, max_seq_len]
            sequence_lengths: [batch_size] 每个batch下序列的真实长度
            is_training: 是否是训练过程

        Return:
            None

        """
        self.hidden_units_num = hidden_units_num
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedding_inputs = embedding_inputs
        self.initializers = initializers
        self.sequence_length = sequence_length
        self.num_labels = num_labels
        self.tag_indices = tag_indices
        self.sequence_lengths = sequence_lengths
        self.embedding_dims = embedding_inputs.shape[-1].value
        self.is_training = is_training

    def add_blstm_crf_layer(self, crf_only):
        """
        bi-lstm-crf网络.

        Return:

        """
        if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            self.embedding_inputs = tf.nn.dropout(self.embedding_inputs, self.dropout_rate)

        if crf_only:
            # 只有CRF Layer
            logits = self.project_crf_layer(self.embedding_inputs)
        else:
            # bi-lstm
            lstm_output = self.blstm_layer(self.embedding_inputs)
            # project
            logits = self.project_bilstm_layer(lstm_output)
        # crf
        loss, per_example_loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        probabilities, _ = crf.crf_decode(potentials=logits, transition_params=trans,
                                          sequence_length=self.sequence_lengths)
        # pred_ids
        return (loss, per_example_loss, logits, probabilities)

    def _witch_cell(self):
        """
        添加RNN单元.

        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.hidden_units_num)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_units_num)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN.

        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.dropout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return cell_fw, cell_bw

    def blstm_layer(self, embedding_chars):
        """

        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            # 双向RNN
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                                        sequence_length=self.sequence_length,
                                                                        dtype=tf.float32)
            # 拼接LSTM输出
            outputs = tf.concat([output_fw, output_bw], axis=-1)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits。

        Args:
            lstm_outputs: [batch_size, num_steps, emb_size]

        Return: [batch_size, num_steps, num_tags]

        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_units_num * 2, self.hidden_units_num],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_units_num], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_units_num * 2])
                hidden = tf.nn.xw_plus_b(output, W, b)

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_units_num, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.sequence_length, self.num_labels])

    def project_crf_layer(self, embedding_inputs):
        """
        hidden layer between input layer and logits.

        Args:
            embedding_inputs: [batch_size, num_steps, emb_size]

        Return:
            [batch_size, num_steps, num_tags]

        """
        with tf.variable_scope("project"):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(embedding_inputs,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.sequence_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss.

        Args:
            logits: [batch_size, max_seq_len, num_tags]

        Return:
            scalar loss

        """
        with tf.variable_scope("crf_loss"):
            transition_params = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            if self.tag_indices is None:
                return None, transition_params
            else:
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.tag_indices,
                    transition_params=transition_params,
                    sequence_lengths=self.sequence_lengths)
                # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs * label_weights, axis=-1)
                per_example_loss = -log_likelihood
                loss = tf.reduce_mean(-log_likelihood)
                return loss, per_example_loss, transition_params
