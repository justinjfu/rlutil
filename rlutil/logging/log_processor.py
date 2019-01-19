"""
A collection of utility functions for reading and sorting log files
"""
import os
from os import path
import collections
import json
import csv
import numpy as np
import pandas


def _find_logs_recursive(root_dir):
    """ Iterate log directories recursively. """
    progress_file = path.join(root_dir, 'progress.csv')
    params_file = path.join(root_dir, 'params.json')

    if path.isfile(params_file) and path.isfile(progress_file):
        yield root_dir
    else:
        for dirname in os.listdir(root_dir):
            dirname = path.join(root_dir, dirname)
            if path.isdir(dirname):
                for log_dir in _find_logs_recursive(dirname):
                    yield log_dir


ExperimentLog = collections.namedtuple('ExperimentLog', ['params', 'progress', 'log_dir'])


def iterate_experiments(root_dir, filter_fn=None):
    """
    Returns an iterator through ExperimentLog objects.

    filter_fn is a function which takes in a params dictionary as an argument and returns
    False if the experiment should be skipped. Using this filter argument will be faster than
    python's filter on this iterator because progress file loading will be skipped.

    Args:
        root_dir: String name of root directory to walk through
        filter_fn: (dict) -> bool filter function

    Returns:
        An iterator through ExperimentLog tuples. Contains two fields
            params: A dictionary of experiment parameters
            progress: A dictionary from keys to numpy arrays of logged values
    """
    for log_dir in _find_logs_recursive(root_dir):
        progress_file = path.join(log_dir, 'progress.csv')
        params_file = path.join(log_dir, 'params.json')

        with open(params_file, 'r') as f:
            params = json.load(f)

        if not filter_fn(params):
            continue

        progress_dict = collections.defaultdict(list)
        with open(progress_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in row:
                    progress_dict[key].append(float(row[key]))
        progress = {key:np.array(progress_dict[key]) for key in progress_dict}
        
        if len(progress) == 0:
            print('WARN: empty log file in %s' % log_dir)
            continue


        yield ExperimentLog(params, progress, log_dir)


def partition_params(all_experiments, split_key):
    """
    Partitions parameters into groups according to split_key

    Returns:
        A dictionary from values of split_key to lists of ExperimentLog objects
    """
    partition_params = collections.defaultdict(list)
    for exp in all_experiments:
        partition_params[exp.params[split_key]].append(exp)
    return partition_params


def reduce_last(l):
    return l[-1]

def reduce_first(l):
    return l[0]

def reduce_mean(l):
    return np.mean(l)


def aggregate_partitions(partitions, reduce_fn=reduce_last, aggregate_fn=reduce_mean):
    """
    Aggregate partitions into a pandas data frame.
    The column keys will be the log values, and the row keys (stored in the 'split_key' column) will be the split key values.

    reduce_fn reduces along the progress file. The default is reduce_last (use the last value logged)
    aggregate_fn reduces along experiments. The default is reduce_mean (average the value of reduce_fn across experiments)
    """
    col_keys = list(partitions.keys()) 
    rows = list(partitions[col_keys[0]][0].progress.keys())

    row_key_to_val = collections.defaultdict(list)
    for col_key in col_keys:
        exps = partitions[col_key]
        # aggregate exps
        aggs = {row_key: aggregate_fn([reduce_fn(exp.progress[row_key]) for exp in exps]) for row_key in rows}
        
        for row_key in aggs:
            row_key_to_val[row_key].append(aggs[row_key])
    
    row_key_to_val['split_key'] = col_keys
    frame = pandas.DataFrame(data=row_key_to_val)
    return frame


if __name__ == "__main__":
    import sys
    dirname = sys.argv[1]
    for log in iterate_experiments(dirname):
        print(log)
        break

