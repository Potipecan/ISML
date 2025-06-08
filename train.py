from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import csv
import gzip
import argparse
from pathlib import Path
import json
import numpy as np
from pandas import DataFrame

passthrough = 'passthrough'

class DataFrameEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return f'{type(obj).__name__}()'
    
def load_data(data_file):
    p = Path(data_file)
    if p.suffix == '.gz':
        open_f = gzip.open
    elif p.suffix == '.csv':
        open_f = open
    else:
        raise ValueError(f"Invalid data file suffix: {p.suffix}")

    with open_f(data_file, 'rt') as df:
        dim = tuple(map(int, next(df).split(';')))  # get width and height from first line
        reader = csv.reader(df, delimiter=';')
        y = []
        x = []

        for row in reader:
            y.append(row[0])
            x.append(list(map(float, row[1:])))

        return dim, x, y


def get_model_pipeline(config):
    pipe_config = config['pipeline']
    
    pipe = []
    grid = {}
    for k, v in pipe_config.items():
        if 'vals' in v:
            cls = list(map(lambda x: eval(str(x)), v['vals']))
            grid[k] = cls
            pipe.append((k, "passthrough"))
            
        if 'classname' in v:
            cls = eval(v["classname"])
            params = v.get("params", {})
            pipe.append((k, cls(**params)))
        
        if 'grid' in v:
            for p, vals in v['grid'].items():
                e_vals = list(map(lambda x: eval(str(x)), vals))
                grid[f"{k}__{p}"] = e_vals
            

    pipe = Pipeline(pipe)
    return pipe, grid


def train(args_results, conf, data, validate_conf):
    config = json.load(open(conf, 'r'))
    try:
        pipe, grid = get_model_pipeline(config)
    except Exception as ex:
        print(ex)
        return 1

        
    # data loading and preprocessing
    dim, x, y = load_data(data)
        
    # training
    gs = GridSearchCV(estimator=pipe, param_grid=grid, verbose=1, n_jobs=-1)
    gs.fit(x, y)
    
    
    # results
    results = {
        "config": config,
        "results": gs.cv_results_
    }
    try:
        with open(args_results, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, cls=DataFrameEncoder)
    except ValueError:
        print(results)
        return 1    
    
    return 0


def main():
    parser = argparse.ArgumentParser("train")
    parser.add_argument('--validate_conf', action='store_true')
    parser.add_argument('-o', '--results', default="train_results.json")
    parser.add_argument("data", type=str)
    parser.add_argument("conf", type=str)
    args = parser.parse_args()

    conf = args.conf
    validate_conf = args.validate_conf
    data = args.data
    args_results = args.results

    exit(train(args_results, conf, data, validate_conf))


if __name__ == "__main__":
    main()
