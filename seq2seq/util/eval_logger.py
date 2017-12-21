import codecs
import collections
import os.path
import time

import numpy as np
import tensorflow as tf

__all__ = ["IntrinsicEvalLog", "ExtrinsicEvalLog", "DecodeEvalLog", "EvalLogger"]

class IntrinsicEvalLog(collections.namedtuple("IntrinsicEvalLog", ("metric", "score", "sample_size"))):
    pass

class ExtrinsicEvalLog(collections.namedtuple("ExtrinsicEvalLog", ("metric", "score", "sample_size"))):
    pass

class DecodeEvalLog(collections.namedtuple("DecodeEvalLog", ("sample_input", "sample_output", "sample_reference"))):
    pass

class EvalLogger(object):
    """evaluation logger"""    
    def __init__(self,
                 output_dir):
        """initialize evaluation logger"""
        """intrinsic evaluation result"""
        self.intrinsic_metric = 0.0
        self.intrinsic_score = 0.0
        self.intrinsic_sample_size = 0
        
        """extrinsic evaluation result"""
        self.extrinsic_metric = 0.0
        self.extrinsic_score = 0.0
        self.extrinsic_sample_size = 0
        
        """decoding evaluation result"""
        
        if not tf.gfile.Exists(output_dir):
            tf.gfile.MakeDirs(output_dir)
        self.log_file = os.path.join(output_dir, "eval_{0}.log".format(time.time()))
        self.log_writer = codecs.getwriter("utf-8")(tf.gfile.GFile(self.log_file, mode="a"))
    
    def update_intrinsic_eval(self,
                              eval_result):
        """update evaluation logger based on intrinsic evaluation result"""
        self.intrinsic_metric = eval_result.metric
        self.intrinsic_score = eval_result.score
        self.intrinsic_sample_size = eval_result.sample_size
    
    def update_extrinsic_eval(self,
                              eval_result):
        """update evaluation logger based on extrinsic evaluation result"""
        self.extrinsic_metric = eval_result.metric
        self.extrinsic_score = eval_result.score
        self.extrinsic_sample_size = eval_result.sample_size
    
    def update_decode_eval(self,
                           eval_result):
        """update evaluation logger based on decoding evaluation result"""
        self.sample_decode_input = eval_result.sample_input
        self.sample_decode_output = eval_result.sample_output
        self.sample_decode_reference = eval_result.sample_reference
    
    def check_intrinsic_eval(self):
        """check intrinsic evaluation result"""       
        log_line = "{0}={1}, sample size={2}".format(self.intrinsic_metric,
            self.intrinsic_score, self.intrinsic_sample_size).encode('utf-8')
        self.log_writer.write("{0}\r\n".format(log_line))
        print(log_line)
        
    def check_extrinsic_eval(self):
        """check extrinsic evaluation result"""
        log_line = "{0}={1}, sample size={2}".format(self.extrinsic_metric,
            self.extrinsic_score, self.extrinsic_sample_size).encode('utf-8')
        self.log_writer.write("{0}\r\n".format(log_line))
        print(log_line)
        
    def check_decode_eval(self):
        """check decoding evaluation result"""
        input_size = len(self.sample_decode_input)
        output_size = len(self.sample_decode_output)
        reference_size = len(self.sample_decode_reference)
        
        if input_size != output_size or input_size != reference_size:
            raise ValueError("size of decoding input, output and reference don't match")
        
        for i in range(input_size):
            decode_input = self.sample_decode_input[i]
            decode_output = self.sample_decode_output[i]
            decode_reference = self.sample_decode_reference[i]
            log_line = "sample {0} - input: {1}".format(i+1, decode_input).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            log_line = "sample {0} - output: {1}".format(i+1, decode_output).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            log_line = "sample {0} - reference: {1}".format(i+1, decode_reference).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
