import datetime
import os
import sys
import json
import tensorflow as tf


sys.path.append("..")
import pickle as pkl
from pprint import pprint
import logging
from collections import deque

import numpy as np

from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
# from utils.data_helper import Vocabulary, WordTable
from models.model import Model
from tf_utils.optimization import create_optimizer
import jieba

from keras.preprocessing.text import Tokenizer
import joblib
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from collections import Counter
from sklearn.metrics import classification_report

# Parameters
# ==================================================
# Data loading params
tf.flags.DEFINE_string("all_data_file","./data/judgment_prediction_dataset.json",
                       "Data source for training.")
tf.flags.DEFINE_string("word_embedding_file", "./word_models/embedding_matrix.npy",
                       "Pre train embedding file.")
tf.flags.DEFINE_string("tokenizer_model_file", "./word_models/tokenizer.pickle",
                       "tokenizer model file.")


# Embedding params
tf.flags.DEFINE_integer("edim", 300, "Dimensionality of word embedding (default: 300)")
tf.flags.DEFINE_integer("embedding_dense_size", 300, "Dimensionality of word embedding dense layer (default: 300)")
tf.flags.DEFINE_boolean("use_role_embedding", True, "Use role embedding or not  (default:True)")
tf.flags.DEFINE_integer("role_num", 4, "How many roles  (default: 3)")
tf.flags.DEFINE_integer("role_edim", 300, "Dimensionality of role embedding  (default: 100)")
tf.flags.DEFINE_integer("fact_edim", 300, "Dimensionality of fact embedding  (default: 100)")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 140, "Max sentence sequence length (default: 140)")
tf.flags.DEFINE_integer("fact_num", 10, "Number of classes (default: 10)")
# tf.flags.DEFINE_string("facts","是否存在还款行为,是否约定利率,是否约定借款期限,是否约定保证期间,是否保证人不承担担保责任,是否保证人担保,是否约定违约条款,是否约定还款期限,是否超过诉讼时效,是否借款成立","facts name")
tf.flags.DEFINE_string("facts","Agreed Loan Period,Couple Debt,Limitation of Action,Liquidated Damages,Repayment Behavior,Interest Dispute,Term of Guarantee,Term of Repayment,Guarantee Liability,Loan Established","facts name")
# memmory network used
tf.flags.DEFINE_integer("hops", 3, "Memory network hops")

# transformer used
tf.flags.DEFINE_integer("heads", 4, "multi-head attention (default: 4)")

# rnn used
tf.flags.DEFINE_integer("rnn_hidden_size", 128, "rnn hidden size (default: 300)")
tf.flags.DEFINE_integer("rnn_layer_num", 1, "rnn layer num (default: 1)")

tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")



# Training parameters

tf.flags.DEFINE_integer("max_decoder_steps", 100, "max_decoder_steps")
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 50)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "initial learning rate (default: 1e-3)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("warm_up_steps_percent", 0.05, "Warm up steps percent (default: 5%)")

# tf.flags.DEFINE_string("checkpoint_path", "", "Checkpoint file path without extension, as list in file 'checkpoints'")
# tf.flags.DEFINE_string("pre_train_lm_checkpoint_path", "lm_runs_v2/2019-03-08_16-01-48/checkpoints/model-186000",
#                        "Checkpoint file path from pre trained language model.'")

tf.flags.DEFINE_string("cuda_device", "0", "GPU used")

FLAGS = tf.flags.FLAGS
FLAGS.facts = list(map(str, FLAGS.facts.split(",")))

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device


def train(data):
    train_data_set = data["train_data_set"]
    train_hand_out_set = data["train_hand_out_set"]
    valid_data_set = data["valid_data_set"]
    test_data_set = data["test_data_set"]

    train_num_samples = len(train_data_set)
    batch_num = (train_num_samples * FLAGS.num_epochs) // FLAGS.batch_size + 1


    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True

    # session_conf = tf.ConfigProto()
    # session_conf.gpu_options.allow_growth = True
    with tf.Session(graph=tf.Graph(), config=session_conf) as session:
        model = Model(FLAGS)

        # Define Training procedure
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step,
        #                                            FLAGS.decay_rate, staircase=True)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # train_op = optimizer.minimize(model.loss, global_step=global_step)

        # bert loss
        train_op, learning_rate, global_step = create_optimizer(
            model.loss, FLAGS.learning_rate, num_train_steps=batch_num,
            num_warmup_steps=int(batch_num * FLAGS.warm_up_steps_percent), use_tpu=False
        )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch normalization
        # Output directory for models
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "trail_gen_runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=os.path.join(checkpoint_dir, "log.txt"),
                            filemode='w+')
        logging.info(FLAGS)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        session.run(tf.global_variables_initializer())
        logging.info("no self attention")
        def get_result(target, preds, mode):
            macro_f1 = f1_score(target, preds, average="macro")
            macro_precision = precision_score(target, preds, average="macro")
            macro_recall = recall_score(target, preds, average="macro")

            micro_f1 = f1_score(target, preds, average="micro")
            print(mode + ",val," + str("micro_f1 %.4f," % micro_f1) + str("macro_f1 %.4f," % macro_f1)  + str(
                "-macro_precision %.4f," % macro_precision) + str("-macro_recall %.4f" % macro_recall))
            logging.info(mode + "val," + str("micro_f1 %.4f," % micro_f1) + str("macro_f1 %.4f," % macro_f1)  + str(
                "-macro_precision %.4f," % macro_precision) + str("-macro_recall %.4f" % macro_recall))

            from collections import Counter
            sorted_target = sorted(Counter(target).items())
            sorted_preds = sorted(Counter(preds).items())

            logging.info("ground: (0, {:d}), (1, {:d}), (2, {:d}) ".format(sorted_target[0][1], sorted_target[1][1],
                                                                           sorted_target[2][1]))
            logging.info("pred  : (0, {:d}), (1, {:d}), (2, {:d}) ".format(sorted_preds[0][1], sorted_preds[1][1],
                                                                           sorted_preds[2][1]))
            print("ground:", sorted_target)
            print("pred  :", sorted_preds)

            target_names = ['驳回诉请', '部分支持', "支持诉请"]
            if mode == "test":
                print(classification_report(target, preds, target_names=target_names, digits=4))
            logging.info(classification_report(target, preds, target_names=target_names, digits=4))
            return micro_f1+macro_f1

        def get_fact_result(target, preds):
            macro_f1 = f1_score(target, preds, average="macro")
            macro_precision = precision_score(target, preds, average="macro")
            macro_recall = recall_score(target, preds, average="macro")

            micro_f1 = f1_score(target, preds, average="micro")
            print("val," + str("micro_f1 %.4f," % micro_f1) + str("macro_f1 %.4f," % macro_f1)  + str(
                "-macro_precision %.4f," % macro_precision) + str("-macro_recall %.4f" % macro_recall))
            logging.info("val," + str("micro_f1 %.4f," % micro_f1) + str("macro_f1 %.4f," % macro_f1)  + str(
                "-macro_precision %.4f," % macro_precision) + str("-macro_recall %.4f" % macro_recall))
        def _do_train_step(input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch,
                           input_claims_batch, input_claims_num_batch, input_claims_len_batch, output_fact_labels_batch,
                           output_claims_labels_batch):
            """
            A single training step
            """
            feed_dict = {model.input_x: input_x_batch,
                         model.input_role: input_role_batch,
                         model.input_sample_lens: input_sample_lens_batch,
                         model.input_sentences_lens: input_sentences_lens_batch,
                         model.input_claim_x: input_claims_batch,
                         model.input_claim_sentences_lens: input_claims_len_batch,
                         model.input_claim_lens: input_claims_num_batch,
                         model.input_fact: output_fact_labels_batch,
                         model.input_claim_y: output_claims_labels_batch,
                         model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                         model.training: True
                         }
            fetches = [update_ops, train_op, learning_rate, global_step, model.loss, model.claim_loss, model.fact_loss, model.claim_predict, model.fact_predict]
            _, _, lr, step, loss, claim_loss, fact_loss, claim_predict, fact_predict = session.run(fetches=fetches, feed_dict=feed_dict)
            logging.info("loss {:g}, claim_loss {:g}, fact_loss {:g}".format(loss, claim_loss, fact_loss))

        def _do_valid_step(data_set, mode):
            """
            Evaluates model on a valid set
            """
            num_samples = len(data_set)
            div = num_samples % FLAGS.batch_size
            batch_num = num_samples // FLAGS.batch_size + 1 if div != 0 else num_samples // FLAGS.batch_size

            tf_data_set = tf.data.Dataset.from_generator(
                lambda: data_set,
                (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
            ).padded_batch(FLAGS.batch_size,
                           padded_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([]),
                                          tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([]),
                                          tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])),
                           padding_values=(0, 0, 0, 0, 0, 0, 0, 0, 0))  # pad index 0
            valid_iterator = tf_data_set.make_one_shot_iterator()
            valid_one_batch = valid_iterator.get_next()

            losses = []
            claim_losses = []
            fact_losses = []
            claim_predicts = []
            fact_predicts = []
            ground_claim_labels = []
            ground_fact_labels = []
            ground_length = []
            for _ in range(batch_num):
                input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch, \
                input_claims_batch, input_claims_num_batch, input_claims_len_batch, output_fact_labels_batch, \
                output_claims_labels_batch = session.run(valid_one_batch)

                # decoder_input_x_batch = tf.keras.preprocessing.sequence.pad_sequences(
                #     decoder_input_x_batch, maxlen=FLAGS.max_decoder_steps, padding="post", truncating="post", value=0
                # )
                # decoder_output_x_batch = tf.keras.preprocessing.sequence.pad_sequences(
                #     decoder_output_x_batch, maxlen=FLAGS.max_decoder_steps, padding="post", truncating="post", value=0
                # )

                feed_dict = {model.input_x: input_x_batch,
                             model.input_role: input_role_batch,
                             model.input_sample_lens: input_sample_lens_batch,
                             model.input_sentences_lens: input_sentences_lens_batch,
                             model.input_claim_x: input_claims_batch,
                             model.input_claim_sentences_lens: input_claims_len_batch,
                             model.input_claim_lens: input_claims_num_batch,
                             model.input_fact: output_fact_labels_batch,
                             model.input_claim_y: output_claims_labels_batch,
                             model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                             model.training: False
                             }
                fetches = [model.loss, model.claim_loss, model.fact_loss, model.claim_predict, model.fact_predict]
                loss, claim_loss, fact_loss, claim_predict, fact_predict = session.run(fetches=fetches,
                                                                                       feed_dict=feed_dict)
                losses.append(loss)
                claim_losses.append(claim_loss)
                fact_losses.append(fact_loss)
                claim_predicts.extend(claim_predict.tolist())
                fact_predicts.extend(fact_predict.tolist())
                ground_length.extend(input_claims_num_batch.tolist())
                ground_claim_labels.extend(output_claims_labels_batch.tolist())
                ground_fact_labels.extend(output_fact_labels_batch.tolist())
            # print("ground_claim_labels:",ground_claim_labels)
            # ground_length =sum(ground_length,[])
            new_ground_claim_labels = []
            new_claim_predicts = []
            for i in range(len(ground_length)):
                new_ground_claim_labels.append(ground_claim_labels[i][:ground_length[i]])
                new_claim_predicts.append(claim_predicts[i][:ground_length[i]])
            new_ground_claim_labels = sum(new_ground_claim_labels, [])
            new_claim_predicts = sum(new_claim_predicts, [])

            score = get_result(new_ground_claim_labels, new_claim_predicts, mode)

            # print fact result
            if mode == "test":
                fact_predicts = [[int(y) for y in x] for x in fact_predicts]
                ground_fact_predicts = []
                for i in range(len(data_set)):
                    ground_fact_predicts.append(data_set[i][7])
                for i in range(len(FLAGS.facts)):
                    ground = []
                    preds = []
                    for j in range(len(data_set)):
                        ground.append(ground_fact_predicts[j][i])
                        preds.append(fact_predicts[j][i])
                    logging.info(FLAGS.facts[i])
                    print(FLAGS.facts[i])
                    get_fact_result(ground, preds)
                ground_all = sum(ground_fact_predicts, [])
                fact_predicts_all = sum(fact_predicts, [])
                logging.info("total:")
                print("total:")
                get_fact_result(ground_all, fact_predicts_all)
            return score
        tf_train_data_set = tf.data.Dataset.from_generator(
            lambda: train_data_set,
            (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
        ).shuffle(len(train_data_set)).repeat(FLAGS.num_epochs).padded_batch(
            FLAGS.batch_size,
            padded_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([]),
                           tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([]),
                           tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])),
            padding_values=(0, 0, 0, 0, 0, 0, 0, 0, 0))  # pad index 0
        train_iterator = tf_train_data_set.make_one_shot_iterator()
        train_one_batch = train_iterator.get_next()

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=FLAGS.num_checkpoints)
        losses = deque([])
        losses_steps = deque([])
        # Generate batches
        logging.info("\nTraining:")
        print("\nTraining:")
        score = 0
        batch_to_print=0
        for epoch in range(FLAGS.num_epochs):
            for batch_id in tqdm(range(batch_num//FLAGS.num_epochs)):
                input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch, \
                input_claims_batch, input_claims_num_batch, input_claims_len_batch, output_fact_labels_batch, \
                output_claims_labels_batch = session.run(
                    train_one_batch)
                try:
                    _do_train_step(input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch,
                                   input_claims_batch, input_claims_num_batch, input_claims_len_batch,
                                   output_fact_labels_batch,
                                   output_claims_labels_batch)
                except Exception as e:
                    print(e)

                current_step = tf.train.global_step(session, global_step)
                current_lr = session.run(learning_rate)
                logging.info("epoch_no %d, batch_no %d, global_step %d, learning_rate %.5f." % (epoch, batch_id+1, current_step, current_lr))
                batch_to_print+=1
                if batch_to_print % 1000 == 0:
                    _do_valid_step(valid_data_set, "batch")
            logging.info("\nEvaluation:")
            print("\nEvaluation:")
            new_score = _do_valid_step(valid_data_set, "val")
            if new_score > score:
                print(str(new_score)+">"+str(score)+",saved")
                path = saver.save(session, checkpoint_prefix)
                logging.info("Saved model checkpoint to {}\n".format(path))
                score = new_score
            else:
                print(str(new_score)+"<"+str(score)+",not save model")

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(trainable_vars)
        saver.restore(session, checkpoint_prefix)
        logging.info("\ntest:")
        print("\ntest:")
        _do_valid_step(test_data_set, "test")
def load_data():
    with open(FLAGS.all_data_file, 'r', encoding="utf-8") as load_f:
        data = json.load(load_f)
    jsonContent = data
    print(len(jsonContent))

    #         vocab = Vocabulary()
    #         vocab.load(FLAGS.vocab_model_file, keep_words=FLAGS.vocab_size)

    import random

    def _do_vectorize(jsonContent):
        keys = list(jsonContent.keys())
        tokenizer = Tokenizer()
        tokenizer = joblib.load("./word_models/tokenizer.pickle")

        sentence_nums = []
        dialogue_word_ids = []
        sentences_lens = []
        roles = []
        claims = []
        claims_num = []
        claims_len = []
        for key in keys:
            dialogue = jsonContent[key]["court_debate"].split("\n")
            dialogue = tokenizer.texts_to_sequences(dialogue)
            # if len(dialogue) > FLAGS.max_sequence_num:
            #     dialogue = dialogue[:FLAGS.max_sequence_num]
            sentence_nums.append(len(dialogue))
            sentences_lens.append([len(x) for x in dialogue])
            roles_text = jsonContent[key]["roles"]
            # if len(roles_text) > FLAGS.max_sequence_num:
            #     roles_text = roles_text[:FLAGS.max_sequence_num]
            roles_num = []
            for role in roles_text:
                if role == "审":
                    roles_num.append(1)
                elif role == "原":
                    roles_num.append(2)
                elif role == "被":
                    roles_num.append(3)
                elif role == "其他":
                    roles_num.append(4)
                else:
                    print("error")
            roles.append(roles_num)

            dialogue = tf.keras.preprocessing.sequence.pad_sequences(
                dialogue, maxlen=FLAGS.max_sequence_length, padding="post", truncating="post", value=0
            )
            claims_cut = jsonContent[key]["claims"]
            # claims_cut = []
            # for claim in claims_split:
            #     claim_cut = jieba.cut(claim)
            #     claims_cut.append(" ".join(claim_cut))
            claims_sequences = tokenizer.texts_to_sequences(claims_cut)
            claims_len.append([len(x) for x in claims_sequences])
            claims_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                claims_sequences, maxlen=FLAGS.max_sequence_length, padding="post", truncating="post", value=0
            )
            claims.append(claims_sequences)
            claims_num.append(len(claims_cut))
            dialogue_word_ids.append(dialogue)
        print(len(dialogue_word_ids))
        return dialogue_word_ids, roles, sentence_nums, sentences_lens, claims, claims_num, claims_len

    def _do_label_vectorize(jsonContent):
        keys = list(jsonContent.keys())

        fact_labels = []
        claims_labels = []

        # facts = ["是否存在还款行为", "是否约定利率", "是否约定借款期限", '是否约定保证期间', '是否保证人不承担担保责任', '是否保证人担保', '是否约定违约条款', '是否约定还款期限',
        #          '是否超过诉讼时效', '是否借款成立']

        for key in keys:
            fact_label_sample = jsonContent[key]["fact_labels"]
            fact_label_decode = []
            claim_label_decode = []
            for fact in FLAGS.facts:
                label = 0
                if fact_label_sample[fact] != 0:
                    label = 1
                fact_label_decode.append(label)
            for label in jsonContent[key]["claims_labels"]:
                if label == "Reject":
                    claim_label_decode.append(0)
                elif label == "Partially Support":
                    claim_label_decode.append(1)
                elif label == "Support":
                    claim_label_decode.append(2)
                else:
                    print(label)
                    print("claim label error")

            fact_labels.append(fact_label_decode)
            claims_labels.append(claim_label_decode)
        return fact_labels, claims_labels

    # encode x
    dialogue_word_ids, roles, sentence_nums, sentences_lens, claims, claims_num, claims_len = _do_vectorize(jsonContent)

    fact_labels, claims_labels = _do_label_vectorize(jsonContent)

    data = list(zip(dialogue_word_ids, roles, sentence_nums, sentences_lens, claims, claims_num, claims_len,
                    fact_labels, claims_labels))

    return data


def main(argv=None):
    print("\nParameters:")

    pprint(FLAGS.flag_values_dict())

    print("\nLoading data...")
    all_data = load_data()
    embedding_matrix = np.load("./word_models/embedding_matrix.npy")

    print(embedding_matrix.shape)
    FLAGS.pre_trained_word_embeddings = embedding_matrix
    # FLAGS.label_names = label_names
    train_data, other_data = train_test_split(all_data, test_size=0.2, random_state=2019)
    _, train_hand_out = train_test_split(train_data, test_size=0.1, random_state=2019)
    test_data, valid_data = train_test_split(other_data, test_size=0.5, random_state=2019)

    print("train_data %d, valid_data %d, test_data %d." % (
        len(train_data), len(valid_data), len(test_data)))

    data_dict = {
        "train_data_set": train_data,
        "test_data_set": test_data,
        "train_hand_out_set":train_hand_out,
        "valid_data_set": valid_data
    }
    print("\nSampling data...")
    pass

    print("\nTraining...")
    train(data_dict)


if __name__ == '__main__':
    tf.app.run()
