from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import csv
import gzip
import argparse
from pathlib import Path
import json


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


def get_model(config):
    model = eval(config['model'])

    if not 'hyper_grid' in config:
        return model(**config.get("hyper", {})), False
    else:
        return RandomizedSearchCV(model(**config.get("hyper", {})), config['hyper_grid'], n_iter=10, cv=3,
                                  n_jobs=-1), True


def train(args_results, conf, data, validate_conf):
    config = json.load(open(conf, 'r'))
    try:
        model, hyper_search = get_model(config)
    except Exception as ex:
        print(ex)
        exit(1)
    if validate_conf:
        exit(0)
    dim, x, y = load_data(data)
    split_params = {
        "test_size": 0.2,
        "train_size": 0.8
    }
    split_params.update(config.get("split_params", {}))

    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        **split_params)
    model.fit(x_train, y_train)
    if hyper_search:
        clf = model.best_estimator_
    else:
        clf = model
    # results
    y_pred = clf.predict(x_test)
    results = {
        "hyperparams": clf.get_params(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    try:
        with open(args_results, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
    except ValueError:
        print(results)


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

    train(args_results, conf, data, validate_conf)


if __name__ == "__main__":
    main()
