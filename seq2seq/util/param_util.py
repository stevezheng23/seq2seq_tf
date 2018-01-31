import argparse
import codecs
import json
import math
import os.path

import numpy as np
import tensorflow as tf

__all__ = ["create_default_hyperparams", "load_hyperparams",
           "generate_search_lookup", "search_hyperparams", "create_hyperparams_file"]

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
        data_result_output_dir="",
        train_random_seed=0,
        train_batch_size=128,
        train_eval_batch_size=1024,
        train_eval_metric="bleu",
        train_decode_sample_size=3,
        train_num_epoch=20,
        train_ckpt_output_dir="",
        train_summary_output_dir="",
        train_step_per_stat=10,
        train_step_per_ckpt=100,
        train_step_per_eval=100,
        train_clip_norm=5.0,
        train_optimizer_type="adam",
        train_optimizer_learning_rate=0.001,
        train_optimizer_decay_mode="exponential_decay",
        train_optimizer_decay_rate=0.95,
        train_optimizer_decay_step=1000,
        train_optimizer_decay_start_step=10000,
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
        model_scope="seq2seq",
        model_pretrained_embedding=False,
        model_encoder_type="bi",
        model_encoder_embed_dim=300,
        model_encoder_num_layer=1,
        model_encoder_unit_dim=512,
        model_encoder_unit_type="lstm",
        model_encoder_hidden_activation="tanh",
        model_encoder_residual_connect=False,
        model_encoder_forget_bias=1.0,
        model_encoder_dropout=0.1,
        model_decoder_type="uni",
        model_decoder_embed_dim=300,
        model_decoder_num_layer=2,
        model_decoder_unit_dim=512,
        model_decoder_unit_type="lstm",
        model_decoder_hidden_activation="tanh",
        model_decoder_projection_activation="",
        model_decoder_residual_connect=False,
        model_decoder_forget_bias=1.0,
        model_decoder_dropout=0.1,
        model_decoder_attention_type="bahdanau",
        model_decoder_attention_dim=512,
        model_decoder_decoding="greedy",
        model_decoder_max_len_factor=2.0,
        model_decoder_len_penalty_factor=0.0,
        model_decoder_beam_size=5,
        device_num_gpus=1,
        device_default_gpu_id=0,
        device_log_device_placement=False,
        device_allow_soft_placement=False,
        device_allow_growth=False,
        device_per_process_gpu_memory_fraction=0.95
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

def generate_search_lookup(search, search_lookup=None):
    search_lookup = search_lookup if search_lookup else {}
    search_type = search["stype"]
    data_type = search["dtype"]
    
    if search_type == "uniform":
        range_start = search["range"][0]
        range_end = search["range"][1]
        if data_type == "int":
            search_sample = np.random.randint(range_start, range_end)
        elif data_type == "float":
            search_sample = (range_end - range_start) * np.random.random_sample() + range_start
        else:
            raise ValueError("unsupported data type {0}".format(data_type))
    elif search_type == "log":
        range_start = math.log(search["range"][0], 10)
        range_end = math.log(search["range"][1], 10)
        if data_type == "float":
            search_sample = math.pow(10, (range_end - range_start) * np.random.random_sample() + range_start)
        else:
            raise ValueError("unsupported data type {0}".format(data_type))
    elif search_type == "discrete":
        search_set = search["set"]
        search_sample = np.random.choice(search_set)
    elif search_type == "lookup":
        search_key = search["key"]
        if search_key in search_lookup:
            search_sample = search_lookup[search_key]
        else:
            raise ValueError("search key {0} doesn't exist in look-up table".format(search_key))
    else:
        raise ValueError("unsupported search type {0}".format(search_type))
    
    data_scale = search["scale"] if "scale" in search else 1.0
    data_shift = search["shift"] if "shift" in search else 0.0
    
    if data_type == "int":
        search_sample = int(data_scale * search_sample + data_shift)
    elif data_type == "float":
        search_sample = float(data_scale * search_sample + data_shift)
    elif data_type == "string":
        search_sample = str(search_sample)
    elif data_type == "boolean":
        search_sample = bool(search_sample) 
    else:
        raise ValueError("unsupported data type {0}".format(data_type))
    
    return search_sample

def search_hyperparams(hyperparams, config_file, num_group):
    """search hyperparameters based on search config"""
    if tf.gfile.Exists(config_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(config_file, "rb")) as file:
            hyperparams_group = []
            np.random.seed(hyperparams.train_random_seed)
            search_setting = json.load(file)
            hyperparams_search_setting = search_setting["hyperparams"]
            variables_search_setting = search_setting["variables"]
            for i in range(num_group):
                variables_search_lookup = {}
                for key in variables_search_setting.keys():
                    variables_search = variables_search_setting[key]
                    variables_search_lookup[key] = generate_search_lookup(variables_search)
                hyperparams_search_lookup = {}
                for key in hyperparams_search_setting.keys():
                    hyperparams_search = hyperparams_search_setting[key]
                    hyperparams_search_lookup[key] = generate_search_lookup(hyperparams_search, variables_search_lookup)
                
                hyperparams_sample = tf.contrib.training.HParams(hyperparams.to_proto())
                hyperparams_sample.set_from_map(hyperparams_search_lookup)
                hyperparams_group.append(hyperparams_sample)
            
            return hyperparams_group
    else:
        raise FileNotFoundError("config file not found")

def create_hyperparams_file(hyperparams_group, config_dir):
    """create config files from groups of hyperparameters"""
    if not tf.gfile.Exists(config_dir):
        tf.gfile.MakeDirs(config_dir)
    
    for i in range(len(hyperparams_group)):
        config_file = os.path.join(config_dir, "config_hyperparams_{0}.json".format(i))
        with codecs.getwriter("utf-8")(tf.gfile.GFile(config_file, "w")) as file:
            hyperparams_json = json.dumps(hyperparams_group[i].values(), indent=4)
            file.write(hyperparams_json)
