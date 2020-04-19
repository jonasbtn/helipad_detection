import time
import random
from pathlib import Path
from SMWinservice import SMWinservice

import tensorflow as tf

class TestTensorflow(SMWinservice):
    _svc_name_ = "PythonCornerExample"
    _svc_display_name_ = "Python Corner's Winservice Example"
    _svc_description_ = "That's a great winservice! :)"

    def start(self):
        self.isrunning = True

    def stop(self):
        self.isrunning = False

    def main(self):
        hello = tf.constant("hello TensorFlow!")

        sess = tf.Session()

        print(sess.run(hello))

if __name__ == '__main__':
    TestTensorflow.parse_command_line()