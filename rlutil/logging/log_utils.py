import os
import random
import json
import contextlib

import rlutil.logging.logger as rllablogger
from rlutil.logging.hyperparametrized import extract_hyperparams

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

