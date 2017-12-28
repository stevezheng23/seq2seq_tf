import numpy as np
import tensorflow as tf

from model.seq2seq import *

__all__ = ["Seq2SeqAttention"]

class Seq2SeqAttention(Seq2Seq):
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 src_vocab_size,
                 trg_vocab_size,
                 src_vocab_index,
                 trg_vocab_index,
                 trg_vocab_inverted_index=None,
                 mode="train",
                 pretrained_embedding=False):
        """sequence-to-sequence model with attention"""
        super(Seq2SeqAttention, self).__init__(logger=logger,
            hyperparams=hyperparams, data_pipeline=data_pipeline,
            src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size,
            src_vocab_index=src_vocab_index, trg_vocab_index=trg_vocab_index,
            trg_vocab_inverted_index=trg_vocab_inverted_index,
            mode=mode, pretrained_embedding=pretrained_embedding)
        
        self.hyperparams = hyperparams
        self.mode = mode
        
        if self.mode == "infer":
            self.infer_summary = self._get_infer_summary()
    
    def _convert_decoder_cell(self,
                              cell,
                              unit_dim,
                              encoder_outputs,
                              encoder_output_length):
        """convert decoder cell"""
        decoding = self.hyperparams.model_decoder_decoding
        alignment_history = (self.mode == "infer" and decoding == "greedy")
        
        attention_mechanism = self._create_attention_mechanism(unit_dim,
            encoder_outputs, encoder_output_length,
            self.hyperparams.model_decoder_attention_type)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell=cell,
            attention_mechanism=attention_mechanism, alignment_history=alignment_history,
            attention_layer_size=self.hyperparams.model_decoder_attention_dim)
        
        return cell
    
    def _convert_decoder_state(self,
                               state,
                               cell):
        """convert decoder state"""
        batch_size = self.batch_size
        decoding = self.hyperparams.model_decoder_decoding
        beam_size = self.hyperparams.model_decoder_beam_size
        if self.mode == "infer":
            if decoding == "beam_search" and beam_size > 0:
                batch_size = self.batch_size * beam_size
        
        state = cell.zero_state(batch_size=batch_size,
            dtype=tf.float32).clone(cell_state=state)
        
        return state
    
    def _get_infer_summary(self):
        """get infer summary"""
        decoding = self.hyperparams.model_decoder_decoding
        summary = tf.no_op()
        if self.mode == "infer":
            if decoding == "greedy":
                summary = self._create_attention_summary(self.decoder_final_state)
        
        return summary
    
    def _create_attention_summary(self,
                                  state):
        """create attention summary"""
        attention_mapping = state.alignment_history.stack()
        attention_mapping = tf.expand_dims(tf.transpose(attention_mapping, [1, 2, 0]), -1)
        attention_mapping *= 255
        summary = tf.summary.image("attention_mapping", attention_mapping)
        
        return summary
    
    def _create_attention_mechanism(self,
                                    unit_dim,
                                    attention_memory,
                                    attention_memory_length,
                                    attention_type):
        """create attention mechanism based on attention type"""
        if attention_type == "bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=unit_dim,
                memory=attention_memory, memory_sequence_length=attention_memory_length)
        elif attention_type == "bahdanau_normed":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=unit_dim,
                memory=attention_memory, memory_sequence_length=attention_memory_length, normalize=True)
        elif attention_type == "luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=unit_dim,
                memory=attention_memory, memory_sequence_length=attention_memory_length)
        elif attention_type == "luong_scaled":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=unit_dim,
                memory=attention_memory, memory_sequence_length=attention_memory_length, scale=True)
        else:
            raise ValueError("unsupported attention type {0}".format(attention_type))

        return attention_mechanism
