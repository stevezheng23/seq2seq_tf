import collections
import math

import numpy as np
import tensorflow as tf

from model.seq2seq import *
from model.seq2seq_attention import *
from util.data_util import *
from util.model_util import *
from util.eval_util import *
from util.eval_logger import *

__all__ = ["TrainModel", "EvalModel", "InferModel", "create_train_model", "create_eval_model", "create_infer_model",
           "get_model_creator", "train_intrinsic_eval", "train_extrinsic_eval", "train_decode_eval", "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel",
    ("graph", "model", "data_pipeline", "src_embedding", "trg_embedding"))):
    pass

class EvalModel(collections.namedtuple("EvalModel",
    ("graph", "model", "data_pipeline", "src_embedding", "trg_embedding"))):
    pass

class InferModel(collections.namedtuple("InferModel",
    ("graph", "model", "data_pipeline", "src_input", "trg_input", "src_embedding", "trg_embedding"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (_, _, src_embedding, trg_embedding, src_vocab_size, trg_vocab_size,
         src_vocab_index, trg_vocab_index, trg_vocab_inverted_index) = prepare_data(logger,
            hyperparams.data_src_train_file, hyperparams.data_trg_train_file,
            hyperparams.data_src_vocab_file, hyperparams.data_trg_vocab_file,
            hyperparams.data_src_embedding_file, hyperparams.data_trg_embedding_file,
            hyperparams.data_src_full_embedding_file, hyperparams.data_trg_full_embedding_file,
            hyperparams.data_src_vocab_size, hyperparams.data_trg_vocab_size,
            hyperparams.model_encoder_embed_dim, hyperparams.model_decoder_embed_dim,
            hyperparams.data_unk, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.data_share_vocab, hyperparams.model_pretrained_embedding)
    
        logger.log_print("# create train data pipeline")
        data_pipeline = create_data_pipeline(hyperparams.data_src_train_file, hyperparams.data_trg_train_file,
            src_vocab_index, trg_vocab_index, hyperparams.data_src_max_length, hyperparams.data_trg_max_length,
            hyperparams.data_src_reverse, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.train_batch_size, hyperparams.train_random_seed)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger, hyperparams, data_pipeline, src_vocab_size, trg_vocab_size,
            src_vocab_index, trg_vocab_index, trg_vocab_inverted_index,
            "train", hyperparams.model_pretrained_embedding)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline,
            src_embedding=src_embedding, trg_embedding=trg_embedding)

def create_eval_model(logger,
                      hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare evaluation data")
        (_, _, src_embedding, trg_embedding, src_vocab_size, trg_vocab_size,
         src_vocab_index, trg_vocab_index, trg_vocab_inverted_index) = prepare_data(logger,
            hyperparams.data_src_eval_file, hyperparams.data_trg_eval_file,
            hyperparams.data_src_vocab_file, hyperparams.data_trg_vocab_file,
            hyperparams.data_src_embedding_file, hyperparams.data_trg_embedding_file,
            hyperparams.data_src_full_embedding_file, hyperparams.data_trg_full_embedding_file,
            hyperparams.data_src_vocab_size, hyperparams.data_trg_vocab_size,
            hyperparams.model_encoder_embed_dim, hyperparams.model_decoder_embed_dim,
            hyperparams.data_unk, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.data_share_vocab, hyperparams.model_pretrained_embedding)
        
        logger.log_print("# create evaluation data pipeline")
        data_pipeline = create_data_pipeline(hyperparams.data_src_eval_file, hyperparams.data_trg_eval_file,
            src_vocab_index, trg_vocab_index, hyperparams.data_src_max_length, hyperparams.data_trg_max_length,
            hyperparams.data_src_reverse, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.train_eval_batch_size, hyperparams.train_random_seed)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger, hyperparams, data_pipeline, src_vocab_size, trg_vocab_size,
            src_vocab_index, trg_vocab_index, trg_vocab_inverted_index,
            "eval", hyperparams.model_pretrained_embedding)
        
        return EvalModel(graph=graph, model=model, data_pipeline=data_pipeline,
            src_embedding=src_embedding, trg_embedding=trg_embedding)

def create_infer_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare inference data")
        (src_input, trg_input, src_embedding, trg_embedding, src_vocab_size, trg_vocab_size,
         src_vocab_index, trg_vocab_index, trg_vocab_inverted_index) = prepare_data(logger,
            hyperparams.data_src_eval_file, hyperparams.data_trg_eval_file,
            hyperparams.data_src_vocab_file, hyperparams.data_trg_vocab_file,
            hyperparams.data_src_embedding_file, hyperparams.data_trg_embedding_file,
            hyperparams.data_src_full_embedding_file, hyperparams.data_trg_full_embedding_file,
            hyperparams.data_src_vocab_size, hyperparams.data_trg_vocab_size,
            hyperparams.model_encoder_embed_dim, hyperparams.model_decoder_embed_dim,
            hyperparams.data_unk, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.data_share_vocab, hyperparams.model_pretrained_embedding)
        
        logger.log_print("# create inference data pipeline")
        data_pipeline = create_infer_pipeline(src_vocab_index,
            hyperparams.data_src_max_length, hyperparams.data_src_reverse, hyperparams.data_pad)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger, hyperparams, data_pipeline, src_vocab_size, trg_vocab_size,
            src_vocab_index, trg_vocab_index, trg_vocab_inverted_index,
            "infer", hyperparams.model_pretrained_embedding)
        
        return InferModel(graph=graph, model=model,
            data_pipeline=data_pipeline, src_input=src_input, trg_input=trg_input,
            src_embedding=src_embedding, trg_embedding=trg_embedding)

def get_model_creator(model_type):
    if model_type == "vanilla":
        model_creator = Seq2Seq
    elif model_type == "attention":
        model_creator = Seq2SeqAttention
    else:
        raise ValueError("can not create model with unsupported model type {0}".format(hyperparams.model_type))
    
    return model_creator

def train_intrinsic_eval(logger,
                         sess,
                         model,
                         src_embedding,
                         trg_embedding):
    load_model(sess, model)
    sess.run(model.data_pipeline.initializer)
    
    loss = 0.0
    word_count = 0
    sample_size = 0
    while True:
        try:
            eval_result = model.model.evaluate(sess,
                src_embedding, trg_embedding)
            loss += eval_result.loss * eval_result.batch_size
            word_count += eval_result.word_count
            sample_size += eval_result.batch_size
        except  tf.errors.OutOfRangeError:
            break
    
    perplexity = math.exp(loss/word_count)
    intrinsic_eval_result = IntrinsicEvalLog(metric="perplexity",
        score=perplexity, sample_size=sample_size)
    logger.update_intrinsic_eval(intrinsic_eval_result)
    logger.check_intrinsic_eval()

def train_extrinsic_eval(logger,
                         sess,
                         model,
                         src_input,
                         trg_input,
                         src_embedding,
                         trg_embedding,
                         batch_size,
                         metric,
                         eval_id):
    load_model(sess, model)
    sess.run(model.data_pipeline.initializer,
        feed_dict={model.model.src_inputs_placeholder: src_input,
            model.model.batch_size_placeholder: batch_size})
    
    predict = []
    while True:
        try:
            decode_result = model.model.decode(sess,
                src_embedding, trg_embedding)
            predict.extend(decode_result.sample_sentence)
        except  tf.errors.OutOfRangeError:
            break
    
    sample_size = len(predict)
    score = evaluate(predict, trg_input, metric)
    extrinsic_eval_result = ExtrinsicEvalLog(metric=metric,
        score=score, sample_output=predict, sample_size=sample_size)
    logger.update_extrinsic_eval(extrinsic_eval_result)
    logger.check_extrinsic_eval()
    logger.check_extrinsic_eval_detail(eval_id)

def train_decode_eval(logger,
                      sess,
                      model,
                      src_input,
                      trg_input,
                      src_embedding,
                      trg_embedding,
                      sample_size,
                      random_seed):
    np.random.seed(random_seed)
    sample_ids = np.random.randint(0, len(src_input)-1, size=sample_size)
    src_sample_inputs = [src_input[sample_id] for sample_id in sample_ids]
    trg_sample_inputs = [trg_input[sample_id] for sample_id in sample_ids]
    
    load_model(sess, model)
    sess.run(model.data_pipeline.initializer,
        feed_dict={model.model.src_inputs_placeholder: src_sample_inputs,
            model.model.batch_size_placeholder: sample_size})
    
    decode_result = model.model.decode(sess,
        src_embedding, trg_embedding)
    decode_eval_result = DecodeEvalLog(sample_input=src_sample_inputs,
        sample_output=decode_result.sample_sentence, sample_reference=trg_sample_inputs)
    logger.update_decode_eval(decode_eval_result)
    logger.check_decode_eval()

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model):
    with model.graph.as_default():
        model.model.restore(sess)
