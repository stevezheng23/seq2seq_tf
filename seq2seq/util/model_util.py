import collections

import numpy as np
import tensorflow as tf

from model.seq2seq import *
from model.seq2seq_attention import *
from util.default_util import *
from util.data_util import *
from util.seq2seq_util import *

__all__ = ["TrainModel", "EvalModel", "InferModel", "EncodeModel",
           "create_train_model", "create_eval_model", "create_infer_model", "create_encode_model",
           "get_model_creator", "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel",
    ("graph", "model", "data_pipeline", "src_embedding", "trg_embedding"))):
    pass

class EvalModel(collections.namedtuple("EvalModel",
    ("graph", "model", "data_pipeline", "src_embedding", "trg_embedding"))):
    pass

class InferModel(collections.namedtuple("InferModel",
    ("graph", "model", "data_pipeline", "src_input", "trg_input", "src_embedding", "trg_embedding"))):
    pass

class EncodeModel(collections.namedtuple("EncodeModel",
    ("graph", "model", "data_pipeline", "src_input", "src_embedding"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (_, _, src_embedding, trg_embedding, src_vocab_size, trg_vocab_size,
            src_vocab_index, trg_vocab_index, _, trg_vocab_inverted_index) = prepare_seq2seq_data(logger,
            hyperparams.data_src_train_file, hyperparams.data_trg_train_file,
            hyperparams.data_src_vocab_file, hyperparams.data_trg_vocab_file,
            hyperparams.data_src_embedding_file, hyperparams.data_trg_embedding_file,
            hyperparams.data_src_full_embedding_file, hyperparams.data_trg_full_embedding_file,
            hyperparams.data_src_vocab_size, hyperparams.data_trg_vocab_size,
            hyperparams.model_encoder_embed_dim, hyperparams.model_decoder_embed_dim,
            hyperparams.data_unk, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.data_share_vocab, hyperparams.model_pretrained_embedding)
    
        logger.log_print("# create train data pipeline")
        data_pipeline = create_seq2seq_pipeline(hyperparams.data_src_train_file, hyperparams.data_trg_train_file,
            src_vocab_index, trg_vocab_index, hyperparams.data_src_max_length, hyperparams.data_trg_max_length,
            hyperparams.data_src_reverse, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.train_batch_size, hyperparams.train_random_seed, hyperparams.train_enable_shuffle)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size,
            src_vocab_index=src_vocab_index, trg_vocab_index=trg_vocab_index,
            trg_vocab_inverted_index=trg_vocab_inverted_index,
            mode="train", scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline,
            src_embedding=src_embedding, trg_embedding=trg_embedding)

def create_eval_model(logger,
                      hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare evaluation data")
        (_, _, src_embedding, trg_embedding, src_vocab_size, trg_vocab_size,
            src_vocab_index, trg_vocab_index, _, trg_vocab_inverted_index) = prepare_seq2seq_data(logger,
            hyperparams.data_src_eval_file, hyperparams.data_trg_eval_file,
            hyperparams.data_src_vocab_file, hyperparams.data_trg_vocab_file,
            hyperparams.data_src_embedding_file, hyperparams.data_trg_embedding_file,
            hyperparams.data_src_full_embedding_file, hyperparams.data_trg_full_embedding_file,
            hyperparams.data_src_vocab_size, hyperparams.data_trg_vocab_size,
            hyperparams.model_encoder_embed_dim, hyperparams.model_decoder_embed_dim,
            hyperparams.data_unk, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.data_share_vocab, hyperparams.model_pretrained_embedding)
        
        logger.log_print("# create evaluation data pipeline")
        data_pipeline = create_seq2seq_pipeline(hyperparams.data_src_eval_file, hyperparams.data_trg_eval_file,
            src_vocab_index, trg_vocab_index, hyperparams.data_src_max_length, hyperparams.data_trg_max_length,
            hyperparams.data_src_reverse, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.train_eval_batch_size, hyperparams.train_random_seed, False)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size,
            src_vocab_index=src_vocab_index, trg_vocab_index=trg_vocab_index,
            trg_vocab_inverted_index=trg_vocab_inverted_index,
            mode="eval", scope=hyperparams.model_scope)
        
        return EvalModel(graph=graph, model=model, data_pipeline=data_pipeline,
            src_embedding=src_embedding, trg_embedding=trg_embedding)

def create_infer_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare inference data")
        (src_input, trg_input, src_embedding, trg_embedding, src_vocab_size, trg_vocab_size,
            src_vocab_index, trg_vocab_index, _, trg_vocab_inverted_index) = prepare_seq2seq_data(logger,
            hyperparams.data_src_eval_file, hyperparams.data_trg_eval_file,
            hyperparams.data_src_vocab_file, hyperparams.data_trg_vocab_file,
            hyperparams.data_src_embedding_file, hyperparams.data_trg_embedding_file,
            hyperparams.data_src_full_embedding_file, hyperparams.data_trg_full_embedding_file,
            hyperparams.data_src_vocab_size, hyperparams.data_trg_vocab_size,
            hyperparams.model_encoder_embed_dim, hyperparams.model_decoder_embed_dim,
            hyperparams.data_unk, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.data_share_vocab, hyperparams.model_pretrained_embedding)
        
        logger.log_print("# create inference data pipeline")
        data_pipeline = create_seq2seq_infer_pipeline(src_vocab_index, hyperparams.data_src_max_length,
            hyperparams.data_src_reverse, hyperparams.data_pad)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size,
            src_vocab_index=src_vocab_index, trg_vocab_index=trg_vocab_index,
            trg_vocab_inverted_index=trg_vocab_inverted_index,
            mode="infer", scope=hyperparams.model_scope)
        
        return InferModel(graph=graph, model=model,
            data_pipeline=data_pipeline, src_input=src_input, trg_input=trg_input,
            src_embedding=src_embedding, trg_embedding=trg_embedding)

def create_encode_model(logger,
                        hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare encoding data")
        (src_input, src_embedding, src_vocab_size,
            src_vocab_index, _) = prepare_data(logger,
            hyperparams.data_src_eval_file, hyperparams.data_src_vocab_file,
            hyperparams.data_src_embedding_file, hyperparams.data_src_full_embedding_file,
            hyperparams.data_src_vocab_size, hyperparams.model_encoder_embed_dim,
            hyperparams.data_unk, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.model_pretrained_embedding)
        
        logger.log_print("# create encoding data pipeline")
        data_pipeline = create_seq2seq_infer_pipeline(src_vocab_index, hyperparams.data_src_max_length,
            hyperparams.data_src_reverse, hyperparams.data_pad)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            src_vocab_size=src_vocab_size, trg_vocab_size=None,
            src_vocab_index=src_vocab_index, trg_vocab_index=None,
            trg_vocab_inverted_index=None, mode="encode", scope=hyperparams.model_scope)
        
        return EncodeModel(graph=graph, model=model, data_pipeline=data_pipeline,
            src_input=src_input, src_embedding=src_embedding)

def get_model_creator(model_type):
    if model_type == "vanilla":
        model_creator = Seq2Seq
    elif model_type == "attention":
        model_creator = Seq2SeqAttention
    else:
        raise ValueError("can not create model with unsupported model type {0}".format(model_type))
    
    return model_creator

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model):
    with model.graph.as_default():
        model.model.restore(sess)
