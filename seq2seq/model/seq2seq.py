import collections
import os.path

import numpy as np
import tensorflow as tf

from util.seq2seq_util import *

__all__ = ["TrainResult", "EvaluateResult", "InferResult", "DecodeResult", "EncodeResult", "Seq2Seq"]

class TrainResult(collections.namedtuple("TrainResult",
    ("loss", "learning_rate", "global_step", "batch_size", "summary"))):
    pass

class EvaluateResult(collections.namedtuple("EvaluateResult", ("loss", "batch_size", "word_count"))):
    pass

class InferResult(collections.namedtuple("InferResult",
    ("logits", "sample_id", "sample_word", "batch_size", "summary"))):
    pass

class DecodeResult(collections.namedtuple("DecodeResult",
    ("logits", "sample_id", "sample_word", "sample_sentence", "batch_size", "summary"))):
    pass

class EncodeResult(collections.namedtuple("EncodeResult",
    ("encoder_outputs", "encoder_output_length", "batch_size"))):
    pass

class Seq2Seq(object):
    """sequence-to-sequence vanilla model"""
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
                 pretrained_embedding=False,
                 scope="seq2seq"):
        """initialize seq2seq model"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.logger = logger
            self.hyperparams = hyperparams
            
            self.data_pipeline = data_pipeline
            self.src_vocab_size = src_vocab_size
            self.trg_vocab_size = trg_vocab_size
            self.src_vocab_index = src_vocab_index
            self.trg_vocab_index = trg_vocab_index
            self.trg_vocab_inverted_index = trg_vocab_inverted_index
            self.mode = mode
            self.pretrained_embedding = pretrained_embedding
            self.scope = scope
            
            self.num_gpus = self.hyperparams.device_num_gpus
            self.default_gpu_id = self.hyperparams.device_default_gpu_id
            self.logger.log_print("# {0} gpus are used with default gpu id set as {1}"
                .format(self.num_gpus, self.default_gpu_id))
            
            """get batch inputs from data pipeline"""
            src_inputs = self.data_pipeline.source_input
            trg_inputs = self.data_pipeline.target_input
            src_input_length = self.data_pipeline.source_input_length
            trg_input_length = self.data_pipeline.target_input_length
            self.batch_size = tf.size(src_input_length)
            
            if self.mode == "encode" or self.mode == "infer":
                self.src_inputs_placeholder = self.data_pipeline.source_input_placeholder
                self.batch_size_placeholder = self.data_pipeline.batch_size_placeholder
            
            """build graph for seq2seq model"""
            self.logger.log_print("# build graph for seq2seq model")
            if self.mode == "encode":               
                self.logger.log_print("# build encoder for seq2seq model")
                (encoder_outputs, _, encoder_output_length,
                    self.encoder_embedding_placeholder) = self._build_encoder(src_inputs, src_input_length)
                self.encoder_outputs = encoder_outputs
                self.encoder_output_length = encoder_output_length
            else:
                (logits, sample_id, _, decoder_final_state, self.encoder_embedding_placeholder,
                    self.decoder_embedding_placeholder) = self._build_graph(src_inputs,
                    trg_inputs, src_input_length, trg_input_length)
                self.decoder_final_state = decoder_final_state
            
            if self.mode == "infer":
                self.infer_logits = logits
                self.infer_sample_id = sample_id
                self.infer_sample_word = self.trg_vocab_inverted_index.lookup(
                    tf.cast(self.infer_sample_id, tf.int64))
                
                self.infer_summary = self._get_infer_summary()
            
            if self.mode == "train" or self.mode == "eval":
                logit_length = self.data_pipeline.target_output_length
                self.word_count = tf.reduce_sum(logit_length)
                
                """compute optimization loss"""
                self.logger.log_print("# setup loss computation mechanism")
                labels = self.data_pipeline.target_output
                loss = self._compute_loss(logits, labels, logit_length)
                self.train_loss = loss
                self.eval_loss = loss
                
                """apply learning rate decay"""
                self.logger.log_print("# setup learning rate decay mechanism")
                self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                    initializer=tf.zeros_initializer, trainable=False)
                self.learning_rate = tf.get_variable("learning_rate", dtype=tf.float32,
                    initializer=tf.constant(self.hyperparams.train_optimizer_learning_rate), trainable=False)
                decayed_learning_rate = self._apply_learning_rate_decay(self.learning_rate)
                
                """initialize optimizer"""
                self.logger.log_print("# initialize optimizer")
                self.optimizer = self._initialize_optimizer(decayed_learning_rate)
                
                """minimize optimization loss"""
                self.logger.log_print("# setup loss minimization mechanism")
                self.update_model, self.clipped_gradients, self.gradient_norm = self._minimize_loss(self.train_loss)
                
                """create summary"""
                self.train_summary = self._get_train_summary()
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            self.ckpt_dir = self.hyperparams.train_ckpt_output_dir
            self.ckpt_name = os.path.join(self.ckpt_dir, "model_ckpt")
            self.ckpt_saver = tf.train.Saver()
    
    def _create_encoder_cell(self,
                             num_layer,
                             unit_dim,
                             unit_type,
                             activation,
                             forget_bias,
                             residual_connect,
                             drop_out):
        """create encoder cell"""
        cell = create_rnn_cell(num_layer, unit_dim, unit_type, activation,
            forget_bias, residual_connect, drop_out, self.num_gpus, self.default_gpu_id)
        
        return cell
    
    def _convert_encoder_state(self,
                               state):
        """convert encoder state"""
        encoder_type = self.hyperparams.model_encoder_type
        num_layer = self.hyperparams.model_encoder_num_layer
        if encoder_type == "bi":
            if num_layer > 1:
                state_list = []
                for i in range(num_layer):
                    state_list.append(state[0][i])
                    state_list.append(state[1][i])
                state = tuple(state_list)
        
        decoding = self.hyperparams.model_decoder_decoding
        beam_size = self.hyperparams.model_decoder_beam_size
        if self.mode == "infer":
            if decoding == "beam_search" and beam_size > 0:
                state = tf.contrib.seq2seq.tile_batch(state, multiplier=beam_size)
        
        return state
    
    def _convert_encoder_outputs(self,
                                 outputs,
                                 output_length):
        """convert encoder outputs"""
        encoder_type = self.hyperparams.model_encoder_type
        if encoder_type == "bi":
            outputs = tf.concat(outputs, -1)
        
        decoding = self.hyperparams.model_decoder_decoding
        beam_size = self.hyperparams.model_decoder_beam_size
        if self.mode == "infer":
            if decoding == "beam_search" and beam_size > 0:
                outputs = tf.contrib.seq2seq.tile_batch(outputs, multiplier=beam_size)
                output_length = tf.contrib.seq2seq.tile_batch(output_length, multiplier=beam_size)
        
        return outputs, output_length
    
    def _build_encoder(self,
                       inputs,
                       input_length):
        """build encoder for seq2seq model"""
        embed_dim = self.hyperparams.model_encoder_embed_dim
        encoder_type = self.hyperparams.model_encoder_type
        num_layer = self.hyperparams.model_encoder_num_layer
        unit_dim = self.hyperparams.model_encoder_unit_dim
        unit_type = self.hyperparams.model_encoder_unit_type
        hidden_activation = self.hyperparams.model_encoder_hidden_activation
        residual_connect = self.hyperparams.model_encoder_residual_connect
        forget_bias = self.hyperparams.model_encoder_forget_bias
        drop_out = self.hyperparams.model_encoder_dropout
        
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            """create embedding for encoder"""
            self.logger.log_print("# create embedding for encoder")
            embedding, embedding_placeholder = create_embedding(self.src_vocab_size,
                embed_dim, self.pretrained_embedding)
            embedding_lookup = tf.nn.embedding_lookup(embedding, inputs)
            
            """create hidden layer for encoder"""
            self.logger.log_print("# create hidden layer for encoder")
            if encoder_type == "uni":
                cell = self._create_encoder_cell(num_layer, unit_dim, unit_type, hidden_activation,
                    forget_bias, residual_connect, drop_out)
                outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=embedding_lookup,
                    sequence_length=input_length, dtype=tf.float32)
            elif encoder_type == "bi":
                fwd_cell = self._create_encoder_cell(num_layer, unit_dim, unit_type, hidden_activation,
                    forget_bias, residual_connect, drop_out)
                bwd_cell = self._create_encoder_cell(num_layer, unit_dim, unit_type, hidden_activation,
                    forget_bias, residual_connect, drop_out)
                outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell, cell_bw=bwd_cell,
                    inputs=embedding_lookup, sequence_length=input_length, dtype=tf.float32)
            else:
                raise ValueError("unsupported encoder type {0}".format(encoder_type))
            
            final_state = self._convert_encoder_state(final_state)
            outputs, output_length = self._convert_encoder_outputs(outputs, input_length)
            
            return outputs, final_state, output_length, embedding_placeholder
        
    def _create_decoder_cell(self,
                             num_layer,
                             unit_dim,
                             unit_type,
                             activation,
                             forget_bias,
                             residual_connect,
                             drop_out):
        """create decoder cell"""
        cell = create_rnn_cell(num_layer, unit_dim, unit_type, activation, 
            forget_bias, residual_connect, drop_out, self.num_gpus, self.default_gpu_id)
        
        return cell
    
    def _convert_decoder_cell(self,
                              cell,
                              unit_dim,
                              encoder_outputs,
                              encoder_output_length):
        """convert decoder cell"""
        return cell
    
    def _convert_decoder_state(self,
                               state,
                               cell):
        """convert decoder state"""
        return state
    
    def _build_decoder(self,
                       inputs,
                       init_state,
                       input_length,
                       encoder_outputs,
                       encoder_output_length):
        """build decoder for seq2seq model"""
        embed_dim = self.hyperparams.model_decoder_embed_dim
        encoder_type = self.hyperparams.model_decoder_type
        num_layer = self.hyperparams.model_decoder_num_layer
        unit_dim = self.hyperparams.model_decoder_unit_dim
        unit_type = self.hyperparams.model_decoder_unit_type
        hidden_activation = self.hyperparams.model_decoder_hidden_activation
        projection_activation = create_activation_function(self.hyperparams.model_decoder_projection_activation)
        residual_connect = self.hyperparams.model_decoder_residual_connect
        forget_bias = self.hyperparams.model_decoder_forget_bias
        drop_out = self.hyperparams.model_decoder_dropout
        decoding = self.hyperparams.model_decoder_decoding
        len_penalty_factor = self.hyperparams.model_decoder_len_penalty_factor
        beam_size = self.hyperparams.model_decoder_beam_size
        trg_sos_id = tf.cast(self.trg_vocab_index.lookup(tf.constant(self.hyperparams.data_sos)), tf.int32)
        trg_eos_id = tf.cast(self.trg_vocab_index.lookup(tf.constant(self.hyperparams.data_eos)), tf.int32)
        
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            """create embedding for decoder"""
            self.logger.log_print("# create embedding for decoder")
            embedding, embedding_placeholder = create_embedding(self.trg_vocab_size,
                embed_dim, self.pretrained_embedding)
            
            """create hidden layer for decoder"""
            self.logger.log_print("# create hidden layer for decoder")
            cell = self._create_decoder_cell(num_layer, unit_dim, unit_type,
                hidden_activation, forget_bias, residual_connect, drop_out)
            cell = self._convert_decoder_cell(cell, unit_dim, encoder_outputs, encoder_output_length)
            init_state = self._convert_decoder_state(init_state, cell)
            
            """create projection layer for decoder"""
            self.logger.log_print("# create projection layer for decoder")
            projector = tf.layers.Dense(units=self.trg_vocab_size, activation=projection_activation)
            
            if self.mode == "infer":
                max_len = tf.cast(tf.round(tf.cast(tf.reduce_max(encoder_output_length), tf.float32) *
                    self.hyperparams.model_decoder_max_len_factor), tf.int32)
                start_tokens = tf.fill([self.batch_size], trg_sos_id)
                end_token = trg_eos_id
                if decoding == "beam_search" and beam_size > 0:
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell, embedding=embedding,
                        start_tokens=start_tokens, end_token=end_token, initial_state=init_state,
                        beam_width=beam_size, output_layer=projector, length_penalty_weight=len_penalty_factor)
                    outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_len)
                    sample_id = outputs.predicted_ids
                    projections = tf.no_op()
                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                        start_tokens=start_tokens, end_token=end_token)
                    decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell, helper=helper,
                        initial_state=init_state, output_layer=projector)
                    outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_len)
                    sample_id = outputs.sample_id
                    projections = outputs.rnn_output
            else:
                embedding_lookup = tf.nn.embedding_lookup(embedding, inputs)
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=embedding_lookup, sequence_length=input_length)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell, helper=helper,
                    initial_state=init_state, output_layer=projector)
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                sample_id = outputs.sample_id
                projections = outputs.rnn_output
            
            return projections, sample_id, final_state, embedding_placeholder
    
    def _build_graph(self,
                     src_inputs,
                     trg_inputs,
                     src_input_length,
                     trg_input_length):
        """build graph for seq2seq model"""       
        """encoder: encode source inputs to get encoder outputs"""
        self.logger.log_print("# build encoder for seq2seq model")
        (encoder_outputs, encoder_final_state, encoder_output_length,
            encoder_embedding_placeholder) = self._build_encoder(src_inputs, src_input_length)
        
        """decoder: decode target outputs based target inputs and encoder outputs"""
        self.logger.log_print("# build decoder for seq2seq model")
        (logits, sample_id, decoder_final_state,
            decoder_embedding_placeholder) = self._build_decoder(trg_inputs,
            encoder_final_state, trg_input_length, encoder_outputs, encoder_output_length)
        
        return logits, sample_id, encoder_outputs, decoder_final_state, encoder_embedding_placeholder, decoder_embedding_placeholder
    
    def _compute_loss(self,
                      logits,
                      labels,
                      logit_length):
        """compute optimization loss"""
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        mask = tf.sequence_mask(logit_length, maxlen=tf.shape(logits)[1], dtype=logits.dtype)
        loss = tf.reduce_sum(cross_entropy * mask) / tf.to_float(self.batch_size)
        
        return loss
    
    def _apply_learning_rate_decay(self,
                                   learning_rate):
        """apply learning rate decay"""
        decay_mode = self.hyperparams.train_optimizer_decay_mode
        decay_rate = self.hyperparams.train_optimizer_decay_rate
        decay_step = self.hyperparams.train_optimizer_decay_step
        decay_start_step = self.hyperparams.train_optimizer_decay_start_step
        
        if decay_mode == "exponential_decay":
            decayed_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                global_step=(self.global_step - decay_start_step), decay_steps=decay_step, decay_rate=decay_rate)
        elif decay_mode == "inverse_time_decay":
            decayed_learning_rate = tf.train.inverse_time_decay(learning_rate=learning_rate,
                global_step=(self.global_step - decay_start_step), decay_steps=decay_step, decay_rate=decay_rate)
        else:
            raise ValueError("unsupported decay mode {0}".format(decay_mode))
        
        decayed_learning_rate = tf.cond(tf.less(self.global_step, decay_start_step),
            lambda: learning_rate, lambda: decayed_learning_rate)
        
        return decayed_learning_rate
    
    def _initialize_optimizer(self,
                              learning_rate):
        """initialize optimizer"""
        optimizer_type = self.hyperparams.train_optimizer_type
        if optimizer_type == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                momentum=self.hyperparams.train_optimizer_momentum_beta)
        elif optimizer_type == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                decay=self.hyperparams.train_optimizer_rmsprop_beta,
                epsilon=self.hyperparams.train_optimizer_rmsprop_epsilon)
        elif optimizer_type == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                rho=self.hyperparams.train_optimizer_adadelta_rho,
                epsilon=self.hyperparams.train_optimizer_adadelta_epsilon)
        elif optimizer_type == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                initial_accumulator_value=self.hyperparams.train_optimizer_adagrad_init_accumulator)
        elif optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                beta1=self.hyperparams.train_optimizer_adam_beta_1, beta2=self.hyperparams.train_optimizer_adam_beta_2,
                epsilon=self.hyperparams.train_optimizer_adam_epsilon)
        else:
            raise ValueError("unsupported optimizer type {0}".format(optimizer_type))
        
        return optimizer
    
    def _minimize_loss(self,
                       loss):
        """minimize optimization loss"""
        """compute gradients"""
        grads_and_vars = self.optimizer.compute_gradients(loss)
        
        """clip gradients"""
        gradients = [x[0] for x in grads_and_vars]
        variables = [x[1] for x in grads_and_vars]
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.hyperparams.train_clip_norm)
        grads_and_vars = zip(clipped_gradients, variables)
        
        """update model based on gradients"""
        update_model = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        
        return update_model, clipped_gradients, gradient_norm
    
    def _get_train_summary(self):
        """get train summary"""
        return tf.summary.merge([tf.summary.scalar("learning_rate", self.learning_rate),
            tf.summary.scalar("train_loss", self.train_loss), tf.summary.scalar("gradient_norm", self.gradient_norm)])
    
    def train(self,
              sess,
              src_embedding,
              trg_embedding):
        """train seq2seq model"""
        if self.pretrained_embedding == True:
            _, loss, learning_rate, global_step, batch_size, summary = sess.run([self.update_model,
                self.train_loss, self.learning_rate, self.global_step, self.batch_size, self.train_summary],
                feed_dict={self.encoder_embedding_placeholder: src_embedding,
                    self.decoder_embedding_placeholder: trg_embedding})
        else:
            _, loss, learning_rate, global_step, batch_size, summary = sess.run([self.update_model,
                self.train_loss, self.learning_rate, self.global_step, self.batch_size, self.train_summary])
        
        return TrainResult(loss=loss, learning_rate=learning_rate,
            global_step=global_step, batch_size=batch_size, summary=summary)
    
    def evaluate(self,
                 sess,
                 src_embedding,
                 trg_embedding):
        """evaluate seq2seq model"""
        if self.pretrained_embedding == True:
            loss, batch_size, word_count = sess.run([self.eval_loss, self.batch_size, self.word_count],
                feed_dict={self.encoder_embedding_placeholder: src_embedding,
                    self.decoder_embedding_placeholder: trg_embedding})
        else:
            loss, batch_size, word_count = sess.run([self.eval_loss, self.batch_size, self.word_count])
        
        return EvaluateResult(loss=loss, batch_size=batch_size, word_count=word_count)
    
    def _get_infer_summary(self):
        """get infer summary"""
        return tf.no_op()
    
    def infer(self,
              sess,
              src_embedding,
              trg_embedding):
        """infer seq2seq model"""
        if self.pretrained_embedding == True:
            logits, sample_id, sample_word, batch_size, summary = sess.run([self.infer_logits,
                self.infer_sample_id, self.infer_sample_word, self.batch_size, self.infer_summary],
                feed_dict={self.encoder_embedding_placeholder: src_embedding,
                    self.decoder_embedding_placeholder: trg_embedding})
        else:
            logits, sample_id, sample_word, batch_size, summary = sess.run([self.infer_logits,
                self.infer_sample_id, self.infer_sample_word, self.batch_size, self.infer_summary])
        
        return InferResult(logits=logits, sample_id=sample_id,
            sample_word=sample_word, batch_size=batch_size, summary=summary)
    
    def decode(self,
               sess,
               src_embedding,
               trg_embedding):
        """decode seq2seq model"""
        infer_result = self.infer(sess, src_embedding, trg_embedding)
        decoding = self.hyperparams.model_decoder_decoding
        beam_size = self.hyperparams.model_decoder_beam_size
        
        if decoding == "beam_search" and beam_size > 0:
            logits, _, _, batch_size, summary = infer_result
            sample_id = infer_result.sample_id[:,:,0]
            sample_word = infer_result.sample_word[:,:,0]
        else:
            logits, sample_id, sample_word, batch_size, summary = infer_result
        
        trg_eos = self.hyperparams.data_eos
        sample_sentence = [convert_decoding(sample.tolist(), trg_eos) for sample in sample_word]
        
        return DecodeResult(logits=logits, sample_id=sample_id, sample_word=sample_word,
            sample_sentence=sample_sentence, batch_size=batch_size, summary=summary)
    
    def encode(self,
               sess,
               src_embedding):
        """encode seq2seq model"""
        if self.pretrained_embedding == True:
            (encoder_outputs, encoder_output_length,
                batch_size) = sess.run([self.encoder_outputs, self.encoder_output_length, self.batch_size],
                feed_dict={self.encoder_embedding_placeholder: src_embedding})
        else:
            (encoder_outputs, encoder_output_length,
                batch_size) = sess.run([self.encoder_outputs, self.encoder_output_length, self.batch_size])
        
        return EncodeResult(encoder_outputs=encoder_outputs,
            encoder_output_length=encoder_output_length, batch_size=batch_size)
    
    def save(self,
             sess,
             global_step):
        """save checkpoint for seq2seq model"""
        self.ckpt_saver.save(sess, self.ckpt_name, global_step=global_step)
    
    def restore(self,
                sess):
        """restore seq2seq model from checkpoint"""
        ckpt_file = tf.train.latest_checkpoint(self.ckpt_dir)
        if ckpt_file is not None:
            self.ckpt_saver.restore(sess, ckpt_file)
        else:
            raise FileNotFoundError("latest checkpoint file doesn't exist")
