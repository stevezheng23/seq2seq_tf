import codecs
import collections
import os.path

import numpy as np
import tensorflow as tf

__all__ = ["DataPipeline", "create_data_pipeline", "create_infer_pipeline",
           "load_pretrained_embedding", "create_vocab_table", "create_vocab_file", "load_input", "prepare_data"]

class DataPipeline(collections.namedtuple("DataPipeline",
    ("initializer", "source_input", "target_input", "target_output",
     "source_input_length", "target_input_length", "target_output_length",
     "source_input_placeholder", "target_input_placeholder",
     "target_output_placeholder", "batch_size_placeholder"))):
    pass

def create_data_pipeline(src_file,
                         trg_file,
                         src_vocab_index,
                         trg_vocab_index,
                         src_max_length,
                         trg_max_length,
                         src_reverse,
                         sos,
                         eos,
                         pad,
                         batch_size,
                         random_seed):
    """create data pipeline based on config"""
    src_pad_id = tf.cast(src_vocab_index.lookup(tf.constant(pad)), tf.int32)
    trg_pad_id = tf.cast(trg_vocab_index.lookup(tf.constant(pad)), tf.int32)
    trg_sos_id = tf.cast(trg_vocab_index.lookup(tf.constant(sos)), tf.int32)
    trg_eos_id = tf.cast(trg_vocab_index.lookup(tf.constant(eos)), tf.int32)
    
    src_dataset = tf.data.TextLineDataset([src_file])
    trg_dataset = tf.data.TextLineDataset([trg_file])
    dataset = tf.data.Dataset.zip((src_dataset, trg_dataset))
    
    buffer_size = batch_size * 1000
    dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.map(lambda src, trg:
        (tf.string_split([src], delimiter=' ').values, tf.string_split([trg], delimiter=' ').values))
    dataset = dataset.filter(lambda src, trg:
        tf.logical_and(tf.size(src) > 0, tf.size(trg) > 0))
    dataset = dataset.map(lambda src, trg:
        (src[:src_max_length], trg[:trg_max_length]))
    
    if src_reverse == True:
        dataset = dataset.map(lambda src, trg:
            (tf.reverse(src, axis=[0]), trg))
    
    dataset = dataset.map(lambda src, trg:
        (tf.cast(src_vocab_index.lookup(src), tf.int32),
         tf.cast(trg_vocab_index.lookup(trg), tf.int32)))
    dataset = dataset.map(lambda src, trg: (src,
        tf.concat(([trg_sos_id], trg), 0),
        tf.concat((trg, [trg_eos_id]), 0)))
    dataset = dataset.map(lambda src, trg_input, trg_output:
        (src, trg_input, trg_output, tf.size(src),
         tf.size(trg_input), tf.size(trg_output)))
    
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([])),
        padding_values=(
            src_pad_id,
            trg_pad_id,
            trg_pad_id,
            0,
            0,
            0))
    
    iterator = dataset.make_initializable_iterator()
    (src_input_ids, trg_input_ids, trg_output_ids,
         src_input_len, trg_input_len, trg_output_len) = iterator.get_next()
    
    return DataPipeline(initializer=iterator.initializer,
        source_input=src_input_ids, target_input=trg_input_ids, target_output=trg_output_ids,
        source_input_length=src_input_len, target_input_length=trg_input_len, target_output_length=trg_output_len,
        source_input_placeholder=None, target_input_placeholder=None,
        target_output_placeholder=None, batch_size_placeholder=None)

def create_infer_pipeline(src_vocab_index,
                          src_max_length,
                          src_reverse,
                          pad):
    """create inference pipeline based on config"""
    src_pad_id = tf.cast(src_vocab_index.lookup(tf.constant(pad)), tf.int32)
    
    src_data_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

    src_dataset = tf.data.Dataset.from_tensor_slices(src_data_placeholder)
    src_dataset = src_dataset.map(lambda src: tf.string_split([src], delimiter=' ').values)
    src_dataset = src_dataset.map(lambda src: src[:src_max_length])
    
    if src_reverse == True:
        src_dataset = src_dataset.map(lambda src: tf.reverse(src, axis=[0]))
    
    src_dataset = src_dataset.map(lambda src: tf.cast(src_vocab_index.lookup(src), tf.int32))
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))
    
    src_dataset = src_dataset.padded_batch(
        batch_size=batch_size_placeholder,
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([])),
        padding_values=(
            src_pad_id,
            0))
    
    iterator = src_dataset.make_initializable_iterator()
    src_input_ids, src_input_len = iterator.get_next()
    
    return DataPipeline(initializer=iterator.initializer,
        source_input=src_input_ids, target_input=None, target_output=None,
        source_input_length=src_input_len, target_input_length=None, target_output_length=None,
        source_input_placeholder=src_data_placeholder, target_input_placeholder=None,
        target_output_placeholder=None, batch_size_placeholder=batch_size_placeholder)

def load_pretrained_embedding(embedding_file,
                              embedding_size,
                              unk,
                              sos,
                              eos,
                              pad):
    """load pre-trained embeddings from embedding file"""
    if tf.gfile.Exists(embedding_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(embedding_file, "rb")) as file:
            embedding = {}
            for line in file:
                items = line.strip().split(' ')
                if len(items) != embedding_size + 1:
                    continue
                word = items[0]
                vector = [float(x) for x in items[1:]]
                if word not in embedding:
                    embedding[word] = vector
            
            if unk not in embedding:
                embedding[unk] = np.random.rand(embedding_size)
            if sos not in embedding:
                embedding[sos] = np.random.rand(embedding_size)
            if eos not in embedding:
                embedding[eos] = np.random.rand(embedding_size)
            if pad not in embedding:
                embedding[pad] = np.random.rand(embedding_size)
            
            return embedding
    else:
        raise FileNotFoundError("embedding file not found")

def create_embedding_file(embedding_file,
                          embedding_table):
    """create embedding file based on embedding table"""
    embedding_dir = os.path.dirname(embedding_file)
    if not tf.gfile.Exists(embedding_dir):
        tf.gfile.MakeDirs(embedding_dir)
    
    if not tf.gfile.Exists(embedding_file):
        with codecs.getwriter("utf-8")(tf.gfile.GFile(embedding_file, "w")) as file:
            for vocab in embedding_table.keys():
                embed = embedding_table[vocab]
                embed_str = " ".join(map(str, embed))
                file.write("{0} {1}\n".format(vocab, embed_str))

def load_vocab_table(vocab_file,
                     vocab_size,
                     vocab_lookup,
                     unk,
                     sos,
                     eos,
                     pad):
    """load vocab table from vocab file"""
    if tf.gfile.Exists(vocab_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as file:
            vocab = {}
            for line in file:
                items = line.strip().split('\t')
                
                item_size = len(items)
                if item_size > 1:
                    vocab[items[0]] = int(items[1])
                elif item_size > 0:
                    vocab[items[0]] = 1
            
            if unk in vocab:
                del vocab[unk]
            if sos in vocab:
                del vocab[sos]
            if eos in vocab:
                del vocab[eos]
            if pad in vocab:
                del vocab[pad]
            
            if vocab_lookup is not None:
                vocab = { k: vocab[k] for k in vocab.keys() if k in vocab_lookup }
            
            sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
            sorted_vocab = [unk, sos, eos, pad] + sorted_vocab
            
            vocab_table = sorted_vocab[:vocab_size]
            vocab_size = len(vocab_table)
            
            vocab_index = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(vocab_table), default_value=0)
            vocab_inverted_index = tf.contrib.lookup.index_to_string_table_from_tensor(
                mapping=tf.constant(vocab_table), default_value=unk)
            
            return vocab_table, vocab_size, vocab_index, vocab_inverted_index
    else:
        raise FileNotFoundError("vocab file not found")

def create_vocab_table(text_file,
                       vocab_size,
                       vocab_lookup,
                       unk,
                       sos,
                       eos,
                       pad):
    """create vocab table from text file"""
    if tf.gfile.Exists(text_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(text_file, "rb")) as file:
            vocab = {}
            for line in file:
                words = line.strip().split(' ')
                for word in words:
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        vocab[word] += 1
            
            if unk in vocab:
                del vocab[unk]
            if sos in vocab:
                del vocab[sos]
            if eos in vocab:
                del vocab[eos]
            if pad in vocab:
                del vocab[pad]
            
            if vocab_lookup is not None:
                vocab = { k: vocab[k] for k in vocab.keys() if k in vocab_lookup }
            
            sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
            sorted_vocab = [unk, sos, eos, pad] + sorted_vocab
            
            vocab_table = sorted_vocab[:vocab_size]
            vocab_size = len(vocab_table)
            
            vocab_index = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(vocab_table), default_value=0)
            vocab_inverted_index = tf.contrib.lookup.index_to_string_table_from_tensor(
                mapping=tf.constant(vocab_table), default_value=unk)
            
            return vocab_table, vocab_size, vocab_index, vocab_inverted_index
    else:
        raise FileNotFoundError("text file not found")

def create_vocab_file(vocab_file,
                      vocab_table):
    """create vocab file based on vocab table"""
    vocab_dir = os.path.dirname(vocab_file)
    if not tf.gfile.Exists(vocab_dir):
        tf.gfile.MakeDirs(vocab_dir)
    
    if not tf.gfile.Exists(vocab_file):
        with codecs.getwriter("utf-8")(tf.gfile.GFile(vocab_file, "w")) as file:
            for vocab in vocab_table:
                file.write("{0}\n".format(vocab))

def load_input(text_file):
    """load data from text file"""
    input_table = []
    if tf.gfile.Exists(text_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(text_file, "rb")) as file:
            for line in file:
                input_table.append(line.strip())
            input_size = len(input_table)
            
            return input_table, input_size
    else:
        raise FileNotFoundError("text file not found")

def prepare_data(logger,
                 src_file,
                 trg_file,
                 src_vocab_file,
                 trg_vocab_file,
                 src_embedding_file,
                 trg_embedding_file,
                 src_full_embedding_file,
                 trg_full_embedding_file,
                 src_vocab_size,
                 trg_vocab_size,
                 src_embed_dim,
                 trg_embed_dim,
                 unk,
                 sos,
                 eos,
                 pad,
                 share_vocab,
                 pretrained_embedding):
    src_input = None
    trg_input = None
    if tf.gfile.Exists(src_file):
        logger.log_print("# loading source input from {0}".format(src_file))
        src_input, src_input_size = load_input(src_file)
        logger.log_print("# source input has {0} lines".format(src_input_size))
    if tf.gfile.Exists(src_file):
        logger.log_print("# loading target input from {0}".format(trg_file))
        trg_input, trg_input_size = load_input(trg_file)
        logger.log_print("# target input has {0} lines".format(trg_input_size))
    
    src_embedding = None
    if pretrained_embedding == True:
        if tf.gfile.Exists(src_embedding_file):
            logger.log_print("# loading source embeddings from {0}".format(src_embedding_file))
            src_embedding = load_pretrained_embedding(src_embedding_file,
                src_embed_dim, unk, sos, eos, pad)
        elif tf.gfile.Exists(src_full_embedding_file):
            logger.log_print("# loading source embeddings from {0}".format(src_full_embedding_file))
            src_embedding = load_pretrained_embedding(src_full_embedding_file,
                src_embed_dim, unk, sos, eos, pad)
        logger.log_print("# source embeddings has {0} words".format(len(src_embedding)))
    
    if tf.gfile.Exists(src_vocab_file):
        logger.log_print("# loading source vocab from {0}".format(src_vocab_file))
        (src_vocab_table, src_vocab_size, src_vocab_index,
            src_vocab_inverted_index) = load_vocab_table(src_vocab_file,
            src_vocab_size, src_embedding, unk, sos, eos, pad)
    elif tf.gfile.Exists(src_file):
        logger.log_print("# creating source vocab from {0}".format(src_file))
        (src_vocab_table, src_vocab_size, src_vocab_index,
            src_vocab_inverted_index) = create_vocab_table(src_file,
            src_vocab_size, src_embedding, unk, sos, eos, pad)
        logger.log_print("# creating source vocab file {0}".format(src_vocab_file))
        create_vocab_file(src_vocab_file, src_vocab_table)
    else:
        raise ValueError("{0} or {1} must be provided".format(src_vocab_file, src_file))
    logger.log_print("# source vocab table has {0} words".format(src_vocab_size))
    
    if src_embedding is not None:
        src_embedding = { k: src_embedding[k] for k in src_vocab_table if k in src_embedding }
        logger.log_print("# source embeddings has {0} words after filtering".format(len(src_embedding)))
        if not tf.gfile.Exists(src_embedding_file):
            logger.log_print("# creating source embeddings file {0}".format(src_embedding_file))
            create_embedding_file(src_embedding_file, src_embedding)
    
    if share_vocab == True:
        logger.log_print("# sharing vocab between source and target data")
        trg_embedding = src_embedding
        trg_vocab_size = src_vocab_size
        trg_vocab_index = src_vocab_index
        trg_vocab_inverted_index = src_vocab_inverted_index
        
        return (src_input, trg_input, src_embedding, trg_embedding, src_vocab_size, trg_vocab_size,
            src_vocab_index, trg_vocab_index, trg_vocab_inverted_index)
    
    trg_embedding = None
    if pretrained_embedding == True:
        if tf.gfile.Exists(trg_embedding_file):
            logger.log_print("# loading target embeddings from {0}".format(trg_embedding_file))
            trg_embedding = load_pretrained_embedding(trg_embedding_file,
                trg_embed_dim, unk, sos, eos, pad)
        elif tf.gfile.Exists(trg_full_embedding_file):
            logger.log_print("# loading target embeddings from {0}".format(trg_full_embedding_file))
            trg_embedding = load_pretrained_embedding(trg_full_embedding_file,
                trg_embed_dim, unk, sos, eos, pad)
        logger.log_print("# target embeddings has {0} words".format(len(trg_embedding)))
    
    if tf.gfile.Exists(trg_vocab_file):
        logger.log_print("# loading target vocab from {0}".format(trg_vocab_file))
        (trg_vocab_table, trg_vocab_size, trg_vocab_index,
            trg_vocab_inverted_index) = load_vocab_table(trg_vocab_file,
            trg_vocab_size, trg_embedding, unk, sos, eos, pad)
    elif tf.gfile.Exists(trg_file):
        logger.log_print("# creating target vocab from {0}".format(trg_file))
        (trg_vocab_table, trg_vocab_size, trg_vocab_index,
            trg_vocab_inverted_index) = create_vocab_table(trg_file,
            trg_vocab_size, trg_embedding, unk, sos, eos, pad)
        logger.log_print("# creating target vocab file {0}".format(trg_vocab_file))
        create_vocab_file(trg_vocab_file, trg_vocab_table)
    else:
        raise ValueError("{0} or {1} must be provided".format(trg_vocab_file, trg_file))
    logger.log_print("# target vocab table has {0} words".format(trg_vocab_size))
    
    if trg_embedding is not None:
        trg_embedding = { k: trg_embedding[k] for k in trg_vocab_table if k in trg_embedding }
        logger.log_print("# target embeddings has {0} words after filtering".format(len(trg_embedding)))
        if not tf.gfile.Exists(trg_embedding_file):
            logger.log_print("# creating target embeddings file {0}".format(trg_embedding_file))
            create_embedding_file(trg_embedding_file, trg_embedding)
    
    return (src_input, trg_input, src_embedding, trg_embedding, src_vocab_size, trg_vocab_size,
        src_vocab_index, trg_vocab_index, trg_vocab_inverted_index)
