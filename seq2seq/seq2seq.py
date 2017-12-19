import argparse
import collections
import math
import time

import numpy as np
import tensorflow as tf

from util.debug_logger import *
from util.train_logger import *
from util.eval_logger import *
from util.default_util import *
from util.param_util import *
from util.data_util import *
from util.model_util import *
from util.eval_util import *
from model.seq2seq import *

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
            hyperparams.data_src_vocab_size, hyperparams.data_trg_vocab_size,
            hyperparams.model_encoder_embed_dim, hyperparams.model_decoder_embed_dim,
            hyperparams.data_unk, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.data_share_vocab, hyperparams.model_pretrained_embedding)
    
        logger.log_print("# create train data pipeline")
        data_pipeline = create_data_pipeline(hyperparams.data_src_train_file, hyperparams.data_trg_train_file,
            src_vocab_index, trg_vocab_index, hyperparams.data_src_max_length, hyperparams.data_trg_max_length,
            hyperparams.data_src_reverse, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.train_batch_size, hyperparams.train_random_seed)
        
        if hyperparams.model_type == "vanilla":
            model = Seq2Seq(logger, hyperparams, data_pipeline, src_vocab_size, trg_vocab_size,
                src_vocab_index, trg_vocab_index, trg_vocab_inverted_index,
                "train", hyperparams.model_pretrained_embedding)
        else:
            raise ValueError("can not create model with unsupported model type {0}".format(hyperparams.model_type))
        
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
            hyperparams.data_src_vocab_size, hyperparams.data_trg_vocab_size,
            hyperparams.model_encoder_embed_dim, hyperparams.model_decoder_embed_dim,
            hyperparams.data_unk, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.data_share_vocab, hyperparams.model_pretrained_embedding)
        
        logger.log_print("# create evaluation data pipeline")
        data_pipeline = create_data_pipeline(hyperparams.data_src_eval_file, hyperparams.data_trg_eval_file,
            src_vocab_index, trg_vocab_index, hyperparams.data_src_max_length, hyperparams.data_trg_max_length,
            hyperparams.data_src_reverse, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.train_eval_batch_size, hyperparams.train_random_seed)
        
        if hyperparams.model_type == "vanilla":
            model = Seq2Seq(logger, hyperparams, data_pipeline, src_vocab_size, trg_vocab_size,
                src_vocab_index, trg_vocab_index, trg_vocab_inverted_index,
                "eval", hyperparams.model_pretrained_embedding)
        else:
            raise ValueError("can not create model with unsupported model type {0}".format(hyperparams.model_type))
        
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
            hyperparams.data_src_vocab_size, hyperparams.data_trg_vocab_size,
            hyperparams.model_encoder_embed_dim, hyperparams.model_decoder_embed_dim,
            hyperparams.data_unk, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.data_share_vocab, hyperparams.model_pretrained_embedding)
        
        logger.log_print("# create inference data pipeline")
        data_pipeline = create_infer_pipeline(src_vocab_index,
            hyperparams.data_src_max_length, hyperparams.data_src_reverse, hyperparams.data_pad)
        
        if hyperparams.model_type == "vanilla":
            model = Seq2Seq(logger, hyperparams, data_pipeline, src_vocab_size, trg_vocab_size,
                src_vocab_index, trg_vocab_index, trg_vocab_inverted_index,
                "infer", hyperparams.model_pretrained_embedding)
        else:
            raise ValueError("can not create model with unsupported model type {0}".format(hyperparams.model_type))
        
        return InferModel(graph=graph, model=model,
            data_pipeline=data_pipeline, src_input=src_input, trg_input=trg_input,
            src_embedding=src_embedding, trg_embedding=trg_embedding)

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model):
    with model.graph.as_default():
        model.model.restore(sess)

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
                         metric):
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
    score = evaluate(predict, trg_input, metric)
    sample_size = len(predict)
    extrinsic_eval_result = ExtrinsicEvalLog(metric=metric,
        score=score, sample_size=sample_size)
    logger.update_extrinsic_eval(extrinsic_eval_result)
    logger.check_extrinsic_eval()

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

def train(logger,
          hyperparams):
    logger.log_print("##### create train model #####")
    train_model = create_train_model(logger, hyperparams)
    logger.log_print("##### create eval model #####")
    eval_model = create_eval_model(logger, hyperparams)
    logger.log_print("##### create infer model #####")
    infer_model = create_infer_model(logger, hyperparams)
    
    train_sess = tf.Session(graph=train_model.graph)
    eval_sess = tf.Session(graph=eval_model.graph)
    infer_sess = tf.Session(graph=infer_model.graph)
    
    train_src_embedding = convert_embedding(train_model.src_embedding)
    train_trg_embedding = convert_embedding(train_model.trg_embedding)
    eval_src_embedding = convert_embedding(eval_model.src_embedding)
    eval_trg_embedding = convert_embedding(eval_model.trg_embedding)
    infer_src_input = infer_model.src_input
    infer_trg_input = infer_model.trg_input
    infer_src_embedding = convert_embedding(infer_model.src_embedding)
    infer_trg_embedding = convert_embedding(infer_model.trg_embedding)
    
    logger.log_print("##### start model training #####")
    summary_writer = tf.summary.FileWriter(
        hyperparams.train_summary_output_dir, train_model.graph)
    
    init_model(train_sess, train_model)
    init_model(eval_sess, eval_model)
    init_model(infer_sess, infer_model)
    
    global_step = 0
    train_model.model.save(train_sess, global_step)
    train_logger = TrainLogger(hyperparams.data_log_output_dir)
    eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    for epoch in range(hyperparams.train_num_epoch):
        train_sess.run(train_model.data_pipeline.initializer)
        step_in_epoch = 0
        while True:
            try:
                start_time = time.time()
                train_result = train_model.model.train(train_sess,
                    train_src_embedding, train_trg_embedding)
                end_time = time.time()

                global_step = train_result.global_step
                step_in_epoch += 1
                train_logger.update(train_result, epoch, step_in_epoch, end_time-start_time)

                if step_in_epoch % hyperparams.train_step_per_stat == 0:
                    train_logger.check()
                if step_in_epoch % hyperparams.train_step_per_ckpt == 0:
                    train_model.model.save(train_sess, global_step)
                if step_in_epoch % hyperparams.train_step_per_eval == 0:
                    train_intrinsic_eval(eval_logger, eval_sess, eval_model,
                        eval_src_embedding, eval_trg_embedding)
            except tf.errors.OutOfRangeError:
                train_logger.check()
                train_model.model.save(train_sess, global_step)
                train_intrinsic_eval(eval_logger, eval_sess, eval_model,
                    eval_src_embedding, eval_trg_embedding)
                train_extrinsic_eval(eval_logger, infer_sess,
                    infer_model, infer_src_input, infer_trg_input,
                    infer_src_embedding, infer_trg_embedding,
                    hyperparams.train_eval_batch_size, hyperparams.train_eval_metric)
                train_decode_eval(eval_logger, infer_sess,
                    infer_model, infer_src_input, infer_trg_input,
                    infer_src_embedding, infer_trg_embedding,
                    hyperparams.train_decode_sample_size, global_step)
                break

    summary_writer.close()
    logger.log_print("##### finish model training #####")

def main(args):
    hyperparams = load_hyperparams(args.config)
    logger = DebugLogger(hyperparams.data_log_output_dir)
    
    tf_version = check_tensorflow_version()
    logger.log_print("# tensorflow verison is {0}".format(tf_version))
    
    if (args.mode == 'train'):
        train(logger, hyperparams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
