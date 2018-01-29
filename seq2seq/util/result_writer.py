import codecs
import os.path

import numpy as np
import tensorflow as tf

__all__ = ["ResultWriter"]

class ResultWriter(object):
    """result writer"""    
    def __init__(self,
                 output_dir):
        """initialize result writer"""       
        self.output_dir = output_dir
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MakeDirs(self.output_dir)
    
    def write_result(self,
                     results,
                     result_tag,
                     result_id=0):
        """write result to file"""
        result_file = os.path.join(self.output_dir, "{0}_{1}.result".format(result_tag, result_id))
        with codecs.getwriter("utf-8")(tf.gfile.GFile(result_file, mode="w")) as result_writer:
            for result in results:
                result_writer.write("{0}\r\n".format(result))
