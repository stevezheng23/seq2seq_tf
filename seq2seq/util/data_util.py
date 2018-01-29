import codecs
import collections
import os.path

import numpy as np
import tensorflow as tf

__all__ = ["DataPipeline", "create_src_pipeline", "create_seq2seq_pipeline",
           "load_pretrained_embedding", "create_embedding_file", "convert_embedding",
           "load_vocab_table", "create_vocab_table", "create_vocab_file",
           "load_input", "prepare_data", "prepare_seq2seq_data"]

class DataPipeline(collections.namedtuple("DataPipeline",
    ("initializer", "source_input", "target_input", "target_output",
     "source_input_length", "target_input_length", "target_output_length",
     "source_input_placeholder", "target_input_placeholder",
     "target_output_placeholder", "batch_size_placeholder"))):
    pass

def create_src_pipeline(src_vocab_index,
                        src_max_length,
                        src_reverse,
                        pad):
    """create source data pipeline based on config"""
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

def create_seq2seq_pipeline(src_file,
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
    """create seq2seq data pipeline based on config"""
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

def convert_embedding(embedding_lookup):
    if embedding_lookup is not None:
        embedding = [v for k,v in embedding_lookup.items()]
    else:
        embedding = None
    
    return embedding

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
                 input_file,
                 vocab_file,
                 embedding_file,
                 full_embedding_file,
                 vocab_size,
                 embed_dim,
                 unk,
                 sos,
                 eos,
                 pad,
                 pretrained_embedding):
    """prepare input data from input files"""
    input_data = None
    if tf.gfile.Exists(input_file):
        logger.log_print("# loading input data from {0}".format(input_file))
        input_data, input_size = load_input(input_file)
        logger.log_print("# input data has {0} lines".format(input_size))
    
    embedding_data = None
    if pretrained_embedding == True:
        if tf.gfile.Exists(embedding_file):
            logger.log_print("# loading embeddings from {0}".format(embedding_file))
            embedding_data = load_pretrained_embedding(embedding_file, embed_dim, unk, sos, eos, pad)
        elif tf.gfile.Exists(full_embedding_file):
            logger.log_print("# loading embeddings from {0}".format(full_embedding_file))
            embedding_data = load_pretrained_embedding(full_embedding_file, embed_dim, unk, sos, eos, pad)
        logger.log_print("# embeddings has {0} words".format(len(embedding_data)))
    
    if tf.gfile.Exists(vocab_file):
        logger.log_print("# loading vocabs from {0}".format(vocab_file))
        (vocab_table, vocab_size, vocab_index,
            vocab_inverted_index) = load_vocab_table(vocab_file,
            vocab_size, embedding_data, unk, sos, eos, pad)
    elif tf.gfile.Exists(input_file):
        logger.log_print("# creating vocabs from {0}".format(input_file))
        (vocab_table, vocab_size, vocab_index,
            vocab_inverted_index) = create_vocab_table(input_file,
            vocab_size, embedding_data, unk, sos, eos, pad)
        logger.log_print("# creating vocab file {0}".format(vocab_file))
        create_vocab_file(vocab_file, vocab_table)
    else:
        raise ValueError("{0} or {1} must be provided".format(vocab_file, input_file))
    logger.log_print("# vocab table has {0} words".format(vocab_size))
    
    if embedding_data is not None:
        embedding_data = { k: embedding_data[k] for k in vocab_table if k in embedding_data }
        logger.log_print("# embeddings has {0} words after filtering".format(len(embedding_data)))
        if not tf.gfile.Exists(embedding_file):
            logger.log_print("# creating embedding file {0}".format(embedding_file))
            create_embedding_file(embedding_file, embedding_data)
        embedding_data = convert_embedding(embedding_data)
    
    return input_data, embedding_data, vocab_size, vocab_index, vocab_inverted_index

def prepare_seq2seq_data(logger,
                         src_input_file,
                         trg_input_file,
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
    """prepare seq2seq data from source & target input files"""
    logger.log_print("# prepare source data")
    (src_input, src_embedding, src_vocab_size,
        src_vocab_index, src_vocab_inverted_index) = prepare_data(logger,
        src_input_file, src_vocab_file, src_embedding_file, src_full_embedding_file,
        src_vocab_size, src_embed_dim, unk, sos, eos, pad, pretrained_embedding)
    
    logger.log_print("# prepare target data")
    if share_vocab == True:
        trg_input = None
        if tf.gfile.Exists(trg_input_file):
            logger.log_print("# loading input data from {0}".format(trg_input_file))
            trg_input, trg_input_size = load_input(input_file)
            logger.log_print("# input data has {0} lines".format(input_size))
        
        logger.log_print("# sharing vocab between source and target data")
        trg_embedding = src_embedding
        trg_vocab_size = src_vocab_size
        trg_vocab_index = src_vocab_index
        trg_vocab_inverted_index = src_vocab_inverted_index
    else:
        (trg_input, trg_embedding, trg_vocab_size,
            trg_vocab_index, trg_vocab_inverted_index) = prepare_data(logger,
            trg_input_file, trg_vocab_file, trg_embedding_file, trg_full_embedding_file,
            trg_vocab_size, trg_embed_dim, unk, sos, eos, pad, pretrained_embedding)
    
    return (src_input, trg_input, src_embedding, trg_embedding, src_vocab_size, trg_vocab_size,
        src_vocab_index, trg_vocab_index, src_vocab_inverted_index, trg_vocab_inverted_index)
