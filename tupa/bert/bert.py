import tensorflow as tf
import numpy as np

import collections
import csv
import os
from . import modeling
from . import optimization
from . import tokenization

from tupa.classifiers.nn.sub_model import SubModel

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "")
flags.DEFINE_string("output_file", None, "")
flags.DEFINE_string("layers", "-1,-2,-3,-4", "")
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


class Bert():
    '''
    Implementation of BERT.
    '''

    def __init__(self, labels):
        print('You are in BERT implementation class.')
        print(labels)
        tf.logging.set_verbosity(tf.logging.INFO)
        layer_indexes = [int(x) for x in FLAGS.layers.split(",")]
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    def read_passage(self, passage):
        """
        Transfer TUPA / UCCA formatted files to two lists (words and tags).
        :param passage: UCCA formatted passage.
        :return: two lists.
        """
        pass

    def train(self, passage=None, dev=None, test=None):
        pass
