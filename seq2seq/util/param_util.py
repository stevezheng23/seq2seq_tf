import argparse
import codecs
import json

import numpy as np
import tensorflow as tf

__all__ = ["add_arguments", "create_default_hyperparams", "load_hyperparams"]

def add_arguments(parser):
    """add arguments to parser"""
    parser.add_argument("--mode", help="mode to run", required=True)
    parser.add_argument("--config", help="path to json config", required=True)

def create_default_hyperparams():
    """create default hyperparameters"""
    hyperparams = tf.contrib.training.HParams(
        data_src_train_file="",
        data_trg_train_file="",
        data_src_eval_file="",
        data_trg_eval_file="",
        data_src_vocab_file="",
        data_trg_vocab_file="",
        data_src_embedding_file="",
        data_trg_embedding_file="",
        data_src_full_embedding_file="",
        data_trg_full_embedding_file="",
        data_src_vocab_size=30000,
        data_trg_vocab_size=30000,
        data_src_max_length=50,
        data_trg_max_length=50,
        data_src_reverse=False,
        data_share_vocab=False,
        data_sos="<s>",
        data_eos="</s>",
        data_pad="<pad>",
        data_unk="<unk>",
        data_log_output_dir="",
        train_random_seed=0,
        train_batch_size=64,
        train_eval_batch_size=1024,
        train_eval_metric="bleu",
        train_decode_sample_size=1,
        train_num_epoch=20,
        train_ckpt_output_dir="",
        train_summary_output_dir="",
        train_step_per_stat=1000,
        train_step_per_ckpt=100000,
        train_step_per_eval=100000,
        train_clip_norm=5.0,
        train_optimizer_type="adam",
        train_optimizer_learning_rate=0.001,
        train_optimizer_decay_mode="exponential_decay",
        train_optimizer_decay_rate=0.95,
        train_optimizer_decay_step=10000,
        train_optimizer_decay_start_step=100000,
        train_optimizer_momentum_beta=0.9,
        train_optimizer_rmsprop_beta=0.999,
        train_optimizer_rmsprop_epsilon=1e-8,
        train_optimizer_adadelta_rho=0.95,
        train_optimizer_adadelta_epsilon=1e-8,
        train_optimizer_adagrad_init_accumulator=0.1,
        train_optimizer_adam_beta_1=0.9,
        train_optimizer_adam_beta_2=0.999,
        train_optimizer_adam_epsilon=1e-08,
        model_type="vanilla",
        model_pretrained_embedding=False,
        model_encoder_type="bi",
        model_encoder_embed_dim=300,
        model_encoder_num_layer=1,
        model_encoder_unit_dim=500,
        model_encoder_unit_type="lstm",
        model_encoder_hidden_activation="tanh",
        model_encoder_residual_connect=False,
        model_encoder_forget_bias=1.0,
        model_encoder_dropout=0.1,
        model_decoder_type="uni",
        model_decoder_embed_dim=300,
        model_decoder_num_layer=2,
        model_decoder_unit_dim=500,
        model_decoder_unit_type="lstm",
        model_decoder_hidden_activation="tanh",
        model_decoder_projection_activation=None,
        model_decoder_residual_connect=False,
        model_decoder_forget_bias=1.0,
        model_decoder_dropout=0.1,
        model_decoder_decoding="greedy",
        model_decoder_max_len_factor=2.0,
        model_decoder_len_penalty_factor=0.0,
        model_decoder_beam_size=5
    )
    return hyperparams

def load_hyperparams(config_file):
    """load hyperparameters from config file"""
    if tf.gfile.Exists(config_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(config_file, "rb")) as file:
            hyperparams = create_default_hyperparams()
            hyperparams_dict = json.load(file)
            hyperparams.set_from_map(hyperparams_dict)
            
            return hyperparams
    else:
        raise FileNotFoundError("config file not found")
