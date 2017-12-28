import argparse
import time

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.param_util import *
from util.train_util import *
from util.train_logger import *
from util.eval_logger import *
from util.debug_logger import *

def add_arguments(parser):
    parser.add_argument("--mode", help="mode to run", required=True)
    parser.add_argument("--config", help="path to json config", required=True)

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
    
    logger.log_print("##### start model training #####")
    train_summary_writer = tf.summary.FileWriter(
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
                    train_intrinsic_eval(eval_logger, eval_sess, eval_model,
                        eval_model.src_embedding, eval_model.trg_embedding)
            except tf.errors.OutOfRangeError:
                train_logger.check()
                train_model.model.save(train_sess, global_step)
                train_intrinsic_eval(eval_logger, eval_sess, eval_model,
                    eval_model.src_embedding, eval_model.trg_embedding)
                train_extrinsic_eval(eval_logger, infer_sess,
                    infer_model, infer_model.src_input, infer_model.trg_input,
                    infer_model.src_embedding, infer_model.trg_embedding,
                    hyperparams.train_eval_batch_size,
                    hyperparams.train_eval_metric, global_step)
                train_decode_eval(eval_logger, infer_sess,
                    infer_model, infer_model.src_input, infer_model.trg_input,
                    infer_model.src_embedding, infer_model.trg_embedding,
                    hyperparams.train_decode_sample_size, global_step)
                break

    train_summary_writer.close()
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
