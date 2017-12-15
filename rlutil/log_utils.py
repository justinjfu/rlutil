import os
import random
from sklearn.externals import joblib
import json
import contextlib

import rllab.misc.logger as rllablogger
import tensorflow as tf
import numpy as np

from rlutil.hyperparametrized import extract_hyperparams

THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
RLUTIL_DIR = os.path.dirname(THIS_FILE_DIR)
DATA_DIR = os.path.join(RLUTIL_DIR, 'data')

@contextlib.contextmanager
def rllab_logdir(algo=None, dirname=None):
    if dirname:
        rllablogger.set_snapshot_dir(dirname)
    dirname = rllablogger.get_snapshot_dir()
    rllablogger.add_tabular_output(os.path.join(dirname, 'progress.csv'))
    if algo:
        with open(os.path.join(dirname, 'params.json'), 'w') as f:
            params = extract_hyperparams(algo)
            json.dump(params, f)
    yield dirname
    rllablogger.remove_tabular_output(os.path.join(dirname, 'progress.csv'))

def load_policy(fname):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        snapshot_dict = joblib.load(fname)
    pol_params = snapshot_dict['policy_params']
    tf.reset_default_graph()
    return pol_params
