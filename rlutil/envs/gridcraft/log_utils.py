import os
import errno
import contextlib
import matplotlib.image
import json
import pickle
import logging
import random
import joblib

import rllab.misc.logger as rllablogger
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


IMG_DIR = 'img'
VIDEO_DIR = 'video'
RLLAB_DIR = 'rllab_log'
_snapshot_dir = None


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def set_snapshot_dir(dirname):
    global _snapshot_dir
    _snapshot_dir = dirname
    mkdir_p(_snapshot_dir)


def get_snapshot_dir():
    return _snapshot_dir or None

def rllab_snapshot_dir():
    return rllablogger.get_snapshot_dir()


def logger_active():
    return get_snapshot_dir() is not None


@contextlib.contextmanager
def rllab_logdir(run_name='run', algo=None, flatten=False):
    if flatten:
        rllab_dir = get_snapshot_dir()
    else:
        rllab_dir = get_log_subdir(os.path.join(RLLAB_DIR, run_name))
    rllablogger.set_snapshot_dir(rllab_dir)
    rllablogger.add_tabular_output(os.path.join(rllab_dir, 'progress.csv'))
    if algo:
        with open(os.path.join(rllab_dir, 'params.json'), 'w') as f:
            params = extract_hyperparams(algo)
            json.dump(params, f)
    yield rllab_dir
    rllablogger.remove_tabular_output(os.path.join(rllab_dir, 'progress.csv'))



def get_log_subdir(subdir, rllabdir=False):
    if not logger_active():
        raise NotImplementedError()
    if rllabdir:
        dirname = os.path.join(rllab_snapshot_dir(), subdir)
    else:
        dirname = os.path.join(get_snapshot_dir(), subdir)
    mkdir_p(dirname)
    return dirname



def record_image(name, img, itr=None, cmap='afmhot', subdir=IMG_DIR):
    if not logger_active():
        return
    if itr is not None:
        name += '_itr%d' % itr
    filename = os.path.join(get_log_subdir(subdir), name)
    matplotlib.image.imsave(filename+'.png', img, cmap=cmap)


def record_fig(name, itr=None, subdir=IMG_DIR, rllabdir=False, close=True, big_figures=True):
    if not logger_active():
        return
    if itr is not None:
        name += '_itr%d' % itr
    filename = os.path.join(get_log_subdir(subdir, rllabdir=rllabdir), name)
    if big_figures:
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(20, 20)
    plt.savefig(filename, dpi=100)
    if close:
        plt.close()

def record_paths(paths, name, task_id=None, itr=None, subdir='path_data'):
    """
    Dumps a list of paths
    :param paths: A list of paths (which are dictionaries)
    :param name:
    :param task_id:
    :param itr:
    :param subdir:
    :return:
    """
    if task_id is not None:
        name += '_subtask%s' % task_id
    if itr is not None:
        name += '_itr%d' % itr
    name += '.pkl'
    filename = os.path.join(get_log_subdir(subdir), name)
    LOGGER.debug('Writing path to %s', filename)

    # filter?
    paths = [path for path in paths if np.sum(path['rewards'])>60]

    with open(filename, 'wb') as f:
        pickle.dump(paths, f)


def append_progress(data_line, name='progress.csv', itr=None, append=True):
    if not logger_active():
        return
    filename = os.path.join(get_snapshot_dir(), name)
    mode = 'w'
    if append:
        mode = 'a'
    with open(filename, mode=mode) as f:
        f.write(','.join(data_line)+'\n')


__KEYS = ['Iteration']
__VALUES = [0]

def record_tabular(key, value):
    __KEYS.append(key)
    __VALUES.append(value)

def advance_iteration(itr=0):
    if itr==0:
        append_progress(__KEYS, itr=itr, append=False)
    append_progress([str(val) for val in __VALUES],itr=itr)

    __KEYS.clear()
    __VALUES.clear()
    __KEYS.append('Iteration')
    __VALUES.append(itr+1)


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

