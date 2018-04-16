import argparse
import os.path
import time

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.param_util import *
from util.model_util import *
from util.train_logger import *
from util.eval_util import *
from util.eval_logger import *
from util.debug_logger import *
from util.summary_writer import *
from util.result_writer import *

def add_arguments(parser):
    parser.add_argument("--mode", help="mode to run", required=True)
    parser.add_argument("--config", help="path to json config", required=True)

def intrinsic_eval(logger,
                   summary_writer,
                   sess,
                   model,
                   src_embedding,
                   trg_embedding,
                   global_step):
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
    
    metric = "perplexity"
    perplexity = safe_exp(loss/word_count)
    summary_writer.add_value_summary(metric, perplexity, global_step)
    intrinsic_eval_result = IntrinsicEvalLog(metric=metric,
        score=perplexity, sample_size=sample_size)
    logger.update_intrinsic_eval(intrinsic_eval_result)
    logger.check_intrinsic_eval()

def extrinsic_eval(logger,
                   summary_writer,
                   sess,
                   model,
                   src_input,
                   trg_input,
                   src_embedding,
                   trg_embedding,
                   batch_size,
                   metric,
                   global_step):
    load_model(sess, model)
    sess.run(model.data_pipeline.initializer,
        feed_dict={model.model.src_inputs_placeholder: src_input,
            model.model.batch_size_placeholder: batch_size})
    
    predict = []
    while True:
        try:
            infer_result = model.model.infer(sess,
                src_embedding, trg_embedding)
            predict.extend(infer_result.sample_sentence)
        except  tf.errors.OutOfRangeError:
            break
    
    sample_size = len(predict)
    score = evaluate_from_data(predict, trg_input, metric)
    summary_writer.add_value_summary(metric, score, global_step)
    extrinsic_eval_result = ExtrinsicEvalLog(metric=metric,
        score=score, sample_output=predict, sample_size=sample_size)
    logger.update_extrinsic_eval(extrinsic_eval_result)
    logger.check_extrinsic_eval()
    logger.check_extrinsic_eval_detail(global_step)

def decode_eval(logger,
                summary_writer,
                sess,
                model,
                src_input,
                trg_input,
                src_embedding,
                trg_embedding,
                sample_size,
                random_seed,
                global_step):
    np.random.seed(random_seed)
    sample_ids = np.random.randint(0, len(src_input)-1, size=sample_size)
    src_sample_inputs = [src_input[sample_id] for sample_id in sample_ids]
    trg_sample_inputs = [trg_input[sample_id] for sample_id in sample_ids]
    
    load_model(sess, model)
    sess.run(model.data_pipeline.initializer,
        feed_dict={model.model.src_inputs_placeholder: src_sample_inputs,
            model.model.batch_size_placeholder: sample_size})
    
    infer_result = model.model.infer(sess,
        src_embedding, trg_embedding)
    if infer_result.summary is not None:
        summary_writer.add_summary(infer_result.summary, global_step)
    decode_eval_result = DecodeEvalLog(sample_input=src_sample_inputs,
        sample_output=infer_result.sample_sentence, sample_reference=trg_sample_inputs)
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
    
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
    infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)
    
    logger.log_print("##### start training #####")
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    train_summary_writer = SummaryWriter(train_model.graph, os.path.join(summary_output_dir, "train"))
    eval_summary_writer = SummaryWriter(eval_model.graph, os.path.join(summary_output_dir, "eval"))
    infer_summary_writer = SummaryWriter(infer_model.graph, os.path.join(summary_output_dir, "infer"))
    
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
                    train_model.src_embedding, train_model.trg_embedding)
                end_time = time.time()
                
                global_step = train_result.global_step
                step_in_epoch += 1
                train_logger.update(train_result, epoch, step_in_epoch, end_time-start_time)
                
                if step_in_epoch % hyperparams.train_step_per_stat == 0:
                    train_logger.check()
                    train_summary_writer.add_summary(train_result.summary, global_step)
                if step_in_epoch % hyperparams.train_step_per_ckpt == 0:
                    train_model.model.save(train_sess, global_step)
                if step_in_epoch % hyperparams.train_step_per_eval == 0:
                    intrinsic_eval(eval_logger, eval_summary_writer, eval_sess, eval_model,
                        eval_model.src_embedding, eval_model.trg_embedding, global_step)
            except tf.errors.OutOfRangeError:
                train_logger.check()
                train_model.model.save(train_sess, global_step)
                intrinsic_eval(eval_logger, eval_summary_writer, eval_sess, eval_model,
                    eval_model.src_embedding, eval_model.trg_embedding, global_step)
                extrinsic_eval(eval_logger, infer_summary_writer, infer_sess,
                    infer_model, infer_model.src_input, infer_model.trg_input,
                    infer_model.src_embedding, infer_model.trg_embedding,
                    hyperparams.train_eval_batch_size,
                    hyperparams.train_eval_metric, global_step)
                decode_eval(eval_logger, infer_summary_writer, infer_sess,
                    infer_model, infer_model.src_input, infer_model.trg_input,
                    infer_model.src_embedding, infer_model.trg_embedding,
                    hyperparams.train_decode_sample_size,
                    hyperparams.train_random_seed + global_step, global_step)
                break

    train_summary_writer.close_writer()
    eval_summary_writer.close_writer()
    infer_summary_writer.close_writer()
    logger.log_print("##### finish training #####")

def evaluate(logger,
             hyperparams):
    logger.log_print("##### create eval model #####")
    eval_model = create_eval_model(logger, hyperparams)
    logger.log_print("##### create infer model #####")
    infer_model = create_infer_model(logger, hyperparams)
    
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
    infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)
    
    logger.log_print("##### start evaluation #####")
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    eval_summary_writer = SummaryWriter(eval_model.graph, os.path.join(summary_output_dir, "eval"))
    infer_summary_writer = SummaryWriter(infer_model.graph, os.path.join(summary_output_dir, "infer"))
    
    init_model(eval_sess, eval_model)
    init_model(infer_sess, infer_model)
    
    global_step = 0
    eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    intrinsic_eval(eval_logger, eval_summary_writer, eval_sess, eval_model,
        eval_model.src_embedding, eval_model.trg_embedding, global_step)
    extrinsic_eval(eval_logger, infer_summary_writer, infer_sess,
        infer_model, infer_model.src_input, infer_model.trg_input,
        infer_model.src_embedding, infer_model.trg_embedding,
        hyperparams.train_eval_batch_size,
        hyperparams.train_eval_metric, global_step)
    decode_eval(eval_logger, infer_summary_writer, infer_sess,
        infer_model, infer_model.src_input, infer_model.trg_input,
        infer_model.src_embedding, infer_model.trg_embedding,
        hyperparams.train_decode_sample_size, hyperparams.train_random_seed, global_step)
    
    eval_summary_writer.close_writer()
    infer_summary_writer.close_writer()
    logger.log_print("##### finish evaluation #####")

def encode(logger,
           hyperparams):
    logger.log_print("##### create encode model #####")
    encode_model = create_encode_model(logger, hyperparams)
    
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    encode_sess = tf.Session(config=config_proto, graph=encode_model.graph)
    
    logger.log_print("##### start encoding #####")
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    encode_summary_writer = SummaryWriter(encode_model.graph, os.path.join(summary_output_dir, "encode"))
    result_writer = ResultWriter(hyperparams.data_result_output_dir)
    
    init_model(encode_sess, encode_model)
    load_model(encode_sess, encode_model)
    encode_sess.run(encode_model.data_pipeline.initializer,
        feed_dict={encode_model.model.src_inputs_placeholder: encode_model.src_input,
            encode_model.model.batch_size_placeholder: hyperparams.train_eval_batch_size})
    
    encoding = []
    while True:
        try:
            encode_result = encode_model.model.encode(encode_sess, encode_model.src_embedding)
            batch_size = encode_result.encoder_output_length.shape[0]
            batch_encoding = [(encode_result.encoder_output_length[i].tolist(),
                encode_result.encoder_outputs[i,:encode_result.encoder_output_length[i],:].tolist(),
                encode_result.encoder_embedding[i,:encode_result.encoder_output_length[i],:].tolist())
                for i in range(batch_size)]
            encoding.extend(batch_encoding)
        except  tf.errors.OutOfRangeError:
            break
    
    encoding_size = len(encoding)
    encoding_sample = encode_model.src_input
    encoding_length = [encoding[i][0] for i in range(encoding_size)]
    
    encoding_vector = None
    if hyperparams.model_encoder_encoding == "context":
        encoding_vector = [encoding[i][1] for i in range(encoding_size)]
    elif hyperparams.model_encoder_encoding == "embedding":
        encoding_vector = [encoding[i][2] for i in range(encoding_size)]
    
    encoding = [{ "sample": encoding_sample[i], "max_length": encoding_length[i], 
        "encoding_type": hyperparams.model_encoder_encoding,
        "encoding_vector": encoding_vector[i] if encoding_vector != None else None }
        for i in range(encoding_size)]
    result_writer.write_result(encoding, "encode", 0)
    
    encode_summary_writer.close_writer()
    logger.log_print("##### finish encoding #####")

def main(args):
    hyperparams = load_hyperparams(args.config)
    logger = DebugLogger(hyperparams.data_log_output_dir)
    
    tf_version = check_tensorflow_version()
    logger.log_print("# tensorflow verison is {0}".format(tf_version))
    
    if (args.mode == 'train'):
        train(logger, hyperparams)
    elif (args.mode == 'eval'):
        evaluate(logger, hyperparams)
    elif (args.mode == 'encode'):
        encode(logger, hyperparams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
