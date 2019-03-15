import codecs
import collections
import os.path
import time

import numpy as np
import tensorflow as tf

__all__ = ["IntrinsicEvalLog", "ExtrinsicEvalLog", "DecodeEvalLog", "EvalLogger"]

class IntrinsicEvalLog(collections.namedtuple("IntrinsicEvalLog", ("metric", "score", "sample_size"))):
    pass

class ExtrinsicEvalLog(collections.namedtuple("ExtrinsicEvalLog", ("metric", "score", "sample_output", "sample_size"))):
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
        self.extrinsic_sample_output = None
        self.extrinsic_sample_size = 0
        
        """decoding evaluation result"""
        self.decode_sample_input = None
        self.decode_sample_output = None
        self.decode_sample_reference = None
        
        self.output_dir = output_dir
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MakeDirs(self.output_dir)
        self.log_file = os.path.join(self.output_dir, "eval_{0}.log".format(time.time()))
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
        self.extrinsic_sample_output = eval_result.sample_output
        self.extrinsic_sample_size = eval_result.sample_size
    
    def update_decode_eval(self,
                           eval_result):
        """update evaluation logger based on decoding evaluation result"""
        self.decode_sample_input = eval_result.sample_input
        self.decode_sample_output = eval_result.sample_output
        self.decode_sample_reference = eval_result.sample_reference
    
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
    
    def check_extrinsic_eval_detail(self,
                                    eval_id):
        """check extrinsic evaluation detail result"""
        eval_detail_file = os.path.join(self.output_dir, "eval_{0}_{1}.detail".format(eval_id, time.time()))
        with codecs.getwriter("utf-8")(tf.gfile.GFile(eval_detail_file, mode="w")) as eval_detail_writer:
            if self.extrinsic_sample_output is None:
                return
            for sample_output in self.extrinsic_sample_output:
                eval_detail_writer.write("{0}\r\n".format(sample_output))
    
    def check_decode_eval(self):
        """check decoding evaluation result"""
        input_size = len(self.decode_sample_input)
        output_size = len(self.decode_sample_output)
        reference_size = len(self.decode_sample_reference)
        
        if input_size != output_size or input_size != reference_size:
            raise ValueError("size of decoding input, output and reference don't match")
        
        for i in range(input_size):
            decode_input = self.decode_sample_input[i]
            log_line = "sample {0} - input: {1}".format(i+1, decode_input).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            decode_output = self.decode_sample_output[i]
            log_line = "sample {0} - output: {1}".format(i+1, decode_output).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            decode_reference = self.decode_sample_reference[i]
            log_line = "sample {0} - reference: {1}".format(i+1, decode_reference).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
