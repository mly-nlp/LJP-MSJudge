import tensorflow as tf
from tf_utils.ops import gelu, get_shape_list
from tf_utils.bert_modeling import attention_layer, create_attention_mask_from_input_mask, transformer_model, \
    get_assignment_map_from_checkpoint
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleLSTMCell

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
Epsilon = 1e-5


class Model(object):
    """
    Ai Dialogue model by wangtianyi
    """

    def __init__(self, config):
        self.config = config

        with tf.name_scope("placeholder"):
            # batch_size, max_sentence_num, max_sequence_length
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name="input_x")
            self.input_sentences_lens = tf.placeholder(tf.int32, [None, None],
                                                       name="input_sentences_lens")  # batch_size, max_sentence_num
            self.input_sample_lens = tf.placeholder(tf.int32, [None], name="input_sample_lens")  # batch_size

            self.input_role = tf.placeholder(tf.int32, [None, None], name="input_role")  # batch_size, max_sentence_num
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.training = tf.placeholder(tf.bool, name="training")

            # ground truth
            self.input_fact = tf.placeholder(tf.float32, [None, self.config.fact_num], name="input_fact")

            # batch, max_claim_num
            self.input_claim_y = tf.placeholder(tf.int32, [None, None], name="input_claim_y")

            # batch_size, max_sentence_num
            self.input_sample_mask = tf.sequence_mask(self.input_sample_lens, name="input_sample_mask")

            # batch_size, max_sentence_num, max_sequence_length
            self.input_sentences_mask = tf.sequence_mask(self.input_sentences_lens, name="input_sentences_mask")

            batch_size, max_sentence_num, max_sequence_length = get_shape_list(self.input_x)

        # Embedding layer
        with tf.name_scope("embedding"):
            with tf.device("/cpu:0"):
                self.word_table = tf.Variable(self.config.pre_trained_word_embeddings, trainable=False,
                                              dtype=tf.float32, name='word_table')

                self.role_table = tf.Variable(
                    tf.truncated_normal([self.config.role_num + 1, self.config.role_edim], stddev=self.config.init_std),
                    trainable=True, dtype=tf.float32, name='role_table'
                )

                self.fact_table = tf.Variable(
                    tf.truncated_normal([self.config.fact_num, self.config.fact_edim], stddev=self.config.init_std),
                    trainable=True, dtype=tf.float32, name='fact_table'
                )

        with tf.variable_scope("dialogue_representation"):
            sample_embedding = tf.nn.embedding_lookup(self.word_table, self.input_x, name='sample_embedding')
            role_embedding = tf.nn.embedding_lookup(self.role_table, self.input_role, name='role_embedding')

            with tf.variable_scope("utterance_rnn"):
                tiled_role_embedding = tf.multiply(
                    tf.ones([batch_size, max_sentence_num, max_sequence_length, self.config.role_edim],
                            dtype=tf.float32),
                    tf.expand_dims(role_embedding, axis=2)  # batch_size, max_sentence_num,
                )
                sample_embedding = tf.concat([sample_embedding, tiled_role_embedding], axis=-1)

                sample_embedding = tf.reshape(sample_embedding,
                                              [-1, max_sequence_length, sample_embedding.get_shape()[-1].value])

                mask = tf.sequence_mask(tf.reshape(self.input_sentences_lens, [-1]), maxlen=max_sequence_length)
                mask = tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)
                sample_embedding = tf.multiply(sample_embedding, mask)
                cell_fw = MultiRNNCell(
                    [CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
                     range(self.config.rnn_layer_num)]
                )
                cell_bw = MultiRNNCell(
                    [CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
                     range(self.config.rnn_layer_num)]
                )

                (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw, cell_bw=cell_bw, inputs=sample_embedding,
                    dtype=tf.float32, sequence_length=tf.reshape(self.input_sentences_lens, [-1])
                )
                utterance_memory_embeddings = tf.concat([output_fw, output_bw], axis=2)

                # RNN attention
                utterance_memory_embeddings = tf.multiply(utterance_memory_embeddings, mask)
                utterance_memory_embeddings = tf.nn.dropout(utterance_memory_embeddings,
                                                            keep_prob=self.dropout_keep_prob,
                                                            name="utterance_memory_embeddings")
                sample_text_final_state, _ = self.attention_mechanism(utterance_memory_embeddings,
                                                                      tf.squeeze(mask, axis=-1))

                # batch, max_sentence_num, 2 * h_size
                sample_text_final_state = tf.reshape(sample_text_final_state, [batch_size, max_sentence_num,
                                                                               2 * self.config.rnn_hidden_size])

                final_states = tf.concat([role_embedding, sample_text_final_state], axis=2)

            with tf.variable_scope("dialogue_rnn"):
                mask = tf.cast(tf.expand_dims(self.input_sample_mask, axis=-1), dtype=tf.float32)
                final_states = tf.multiply(mask, final_states)

                cell_fw = MultiRNNCell(
                    [CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
                     range(self.config.rnn_layer_num)]
                )
                cell_bw = MultiRNNCell(
                    [CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
                     range(self.config.rnn_layer_num)]
                )

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw, cell_bw=cell_bw, inputs=final_states,
                    dtype=tf.float32, sequence_length=self.input_sample_lens
                )
                outputs = tf.concat(outputs, axis=2)

                sample_hidden_states = tf.multiply(outputs, mask)

                self.dialogue_representation = tf.nn.dropout(sample_hidden_states, keep_prob=self.dropout_keep_prob)

        with tf.variable_scope("fact_prediction"):
            fact_att_out, _ = self.attention_mechanism(self.dialogue_representation,
                                                       tf.to_float(self.input_sample_mask))
            fact_att_fc = tf.layers.dense(fact_att_out, 512, activation=tf.nn.relu)
            fact_att_fc = tf.nn.dropout(fact_att_fc, self.dropout_keep_prob)
            fact_logits = tf.layers.dense(fact_att_fc, self.config.fact_num, name="fact_logits")
            self.fact_predict_prob = tf.nn.sigmoid(fact_logits, name="fact_predict_prob")

            # batch, fact_num
            self.fact_predict = tf.round(self.fact_predict_prob, name="fact_predict")

            # fact_loss
            self.fact_loss = tf.losses.sigmoid_cross_entropy(logits=fact_logits, multi_class_labels=self.input_fact)

        with tf.variable_scope("fact_representation"):

            fact_idx = tf.zeros((batch_size, 1), dtype=tf.int32) + tf.expand_dims(
                tf.range(0, self.config.fact_num, dtype=tf.int32), axis=0)  # [batch, fact_num]

            # batch, fact_num, h_size
            fact_embedding = tf.nn.embedding_lookup(self.fact_table, ids=fact_idx)

            # do mask training过程用的ground_truth，inference过程用的预测的事实
            fact_mask = tf.cond(self.training, lambda: self.input_fact, lambda: self.fact_predict)
            fact_embedding = tf.to_float(tf.expand_dims(fact_mask, axis=-1)) * fact_embedding  # batch, fact_num, h_size

            fact_embedding = tf.layers.dense(fact_embedding, 2 * self.config.rnn_hidden_size, activation=tf.nn.relu)
            self.fact_embedding = tf.nn.dropout(fact_embedding, self.dropout_keep_prob)

            # also can add rnn to self.fact_embedding
            self.fact_representation = self.fact_embedding  # batch, fact_num, 2 * rnn_h_size

        # now we get self.dialogue_representation and self.fact_representation
        with tf.name_scope("claim_placeholder"):
            # batch_size, max_sentence_num, max_sequence_length
            self.input_claim_x = tf.placeholder(tf.int32, [None, None, None], name="input_claim_x")
            self.input_claim_sentences_lens = tf.placeholder(tf.int32, [None, None],
                                                             name="input_claim_sentences_lens")  # batch_size, max_sentence_num
            self.input_claim_lens = tf.placeholder(tf.int32, [None], name="input_claim_lens")  # batch_size

            # batch_size, max_sentence_num
            self.claim_sample_mask = tf.sequence_mask(self.input_claim_lens, name="claim_sample_mask", dtype=tf.float32)

            # batch_size, max_sentence_num, max_sequence_length
            self.claim_sentences_mask = tf.sequence_mask(self.input_claim_sentences_lens, name="claim_sentences_mask")

            _, max_claim_num, max_claim_length = get_shape_list(self.input_claim_x)

        with tf.variable_scope("claim_representation"):
            claim_embedding = tf.nn.embedding_lookup(self.word_table, self.input_claim_x, name='claim_embedding')

            with tf.variable_scope("claim_rnn"):
                claim_embedding = tf.reshape(claim_embedding,
                                             [-1, max_claim_length, claim_embedding.get_shape()[-1].value])

                mask = tf.sequence_mask(tf.reshape(self.input_claim_sentences_lens, [-1]), maxlen=max_claim_length)
                mask = tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)
                claim_embedding = tf.multiply(claim_embedding, mask)

                cell_fw = MultiRNNCell(
                    [CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
                     range(self.config.rnn_layer_num)]
                )
                cell_bw = MultiRNNCell(
                    [CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
                     range(self.config.rnn_layer_num)]
                )

                (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw, cell_bw=cell_bw, inputs=claim_embedding,
                    dtype=tf.float32, sequence_length=tf.reshape(self.input_claim_sentences_lens, [-1])
                )
                claim_memory_embeddings = tf.concat([output_fw, output_bw], axis=2)

                # RNN attention
                claim_memory_embeddings = tf.multiply(claim_memory_embeddings, mask)
                claim_memory_embeddings = tf.nn.dropout(claim_memory_embeddings, keep_prob=self.dropout_keep_prob,
                                                        name="claim_memory_embeddings")
                claim_final_state, _ = self.attention_mechanism(claim_memory_embeddings, tf.squeeze(mask, axis=-1))

                # batch, max_claim_num, 2 * h_size
                claim_final_state = tf.reshape(claim_final_state,
                                               [batch_size, max_claim_num, 2 * self.config.rnn_hidden_size])
                claim_final_state = claim_final_state * tf.expand_dims(self.claim_sample_mask, axis=-1)

        with tf.variable_scope("memory_network"):
            for i in range(self.config.hops):
                with tf.variable_scope("hop") as scope:
                    if i > 0:
                        # 保持每一层都复用变量
                        tf.get_variable_scope().reuse_variables()

                    # memory from dialogue
                    cd_attention_mask = create_attention_mask_from_input_mask(claim_final_state,
                                                                              tf.to_int32(self.input_sample_mask))

                    # batch, max_claim_num, 2 * h_size
                    attention_out_between_dia_claim = attention_layer(
                        from_tensor=claim_final_state,
                        to_tensor=self.dialogue_representation,
                        attention_mask=cd_attention_mask,
                        num_attention_heads=1,
                        size_per_head=2 * self.config.rnn_hidden_size,
                        from_seq_length=self.input_claim_lens
                    )

                    # cf_attention_mask = create_attention_mask_from_input_mask(claim_final_state,
                    #                                                           tf.to_int32(self.fact_predict))
                    # 这里注释掉诉讼请求和事实之间掉mask，是因为有些情况下如果预测不含任何事实，那么attention加权将为0，会引发NaN问题

                    # batch, max_claim_num, 2 * h_size
                    attention_out_between_fact_claim = attention_layer(
                        from_tensor=claim_final_state,
                        to_tensor=self.fact_representation,
                        num_attention_heads=1,
                        size_per_head=2 * self.config.rnn_hidden_size,
                        from_seq_length=self.input_claim_lens
                    )

                    # Gate / Interaction
                    W1 = tf.layers.dense(claim_final_state, 1, use_bias=False)
                    W2 = tf.layers.dense(attention_out_between_dia_claim, 1, use_bias=False)
                    W3 = tf.layers.dense(attention_out_between_fact_claim, 1, use_bias=False)
                    gate = tf.nn.sigmoid(W1 + W2 + W3)
                    update_vec = gate * attention_out_between_dia_claim + (1 - gate) * attention_out_between_fact_claim

                    claim_final_state = tf.add(
                        tf.layers.dense(update_vec, 2 * self.config.rnn_hidden_size, activation=tf.nn.relu),
                        tf.layers.dense(claim_final_state, 2 * self.config.rnn_hidden_size, activation=tf.nn.relu)
                    )

                    # batch, max_claim_num, 2 * h_size
                    claim_final_state = tf.nn.dropout(claim_final_state, self.dropout_keep_prob)

                    # interaction between claims, use RNN or self attention
                    attention_out_across_claims = attention_layer(
                        from_tensor=claim_final_state,
                        to_tensor=claim_final_state,
                        num_attention_heads=1,
                        size_per_head=2 * self.config.rnn_hidden_size,
                        from_seq_length=self.input_claim_lens
                    )
                    claim_final_state += tf.nn.dropout(attention_out_across_claims,
                                                       self.dropout_keep_prob)  # 残差网络
                    claim_final_state = claim_final_state * tf.expand_dims(self.claim_sample_mask, axis=-1)
            self.claim_final_state = claim_final_state

        with tf.variable_scope("Decoder"):
            self.claim_logits = tf.layers.dense(self.claim_final_state, 3, name="claim_logits")
            self.claim_predict_prob = tf.nn.softmax(self.claim_logits, name="laim_predict_prob")
            self.claim_predict = tf.argmax(self.claim_predict_prob, axis=2, name="claim_predict")

            self.claim_loss = seq2seq.sequence_loss(logits=self.claim_logits, targets=self.input_claim_y,
                                                    weights=self.claim_sample_mask)

            self.loss = self.claim_loss + self.fact_loss

    @staticmethod
    def attention_mechanism(inputs, x_mask=None):
        """
        Attention mechanism layer.

        :param inputs: outputs of RNN/Bi-RNN layer (not final state)
        :param x_mask:
        :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
        """
        # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        _, sequence_length, hidden_size = get_shape_list(inputs)

        v = tf.layers.dense(
            inputs, hidden_size,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.tanh,
            use_bias=True
        )
        att_score = tf.layers.dense(
            v, 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            use_bias=False
        )  # batch_size, sequence_length, 1

        att_score = tf.squeeze(att_score, axis=-1) * x_mask + VERY_NEGATIVE_NUMBER * (
                1 - x_mask)  # [batch_size, sentence_length
        att_score = tf.expand_dims(tf.nn.softmax(att_score), axis=-1)  # [batch_size, sentence_length, 1]
        att_pool_vec = tf.matmul(tf.transpose(att_score, [0, 2, 1]), inputs)  # [batch_size,  h]
        att_pool_vec = tf.squeeze(att_pool_vec, axis=1)

        return att_pool_vec, att_score

