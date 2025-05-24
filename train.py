from sklearn import neural_network, model_selection, neighbors, naive_bayes, tree, linear_model, ensemble, metrics, svm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import csv
import gzip
import os
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

    with open_f(data_file, 'r') as df:
        dim = tuple(map(int, next(df).split(b';')))  # get width and height from first line
        reader = csv.reader(df, delimiter=';')
        y = []
        x = []

        for row in reader:
            y.append(int(row[1]))
            x.append(list(map(float, row[2:])))

        return dim, x, y


def get_model(config):
    model = eval(config['model'])
    
    has_hyper = 'hyper' in config
    has_grid = 'hyper_grid' in config
    
    if has_hyper == has_grid:
        raise ValueError("Config error: 'hyper' and 'hyper_grid' cannot be defined in the same config")
    
    if has_hyper:
        return model(**config['hyper']), False
    else:
        return RandomizedSearchCV(model(), config['hyper_grid'], n_iter=10, cv=3, n_jobs=-1), True


def main():
    parser = argparse.ArgumentParser("train")
    parser.add_argument('--validate_conf', action='store_true')
    parser.add_argument('-o', '--results', default="train_results.json")
    parser.add_argument("data", type=str)
    parser.add_argument("conf", type=str)
    args = parser.parse_args()

    config = json.load(open(args.conf, 'r'))
    try:
        model, hyper_search = get_model(config)
    except Exception as ex:
        print(ex)
        exit(1)

    if args.validate_conf:
        exit(0)

    dim, x, y = load_data(args.data)

    metaparams = {
        "test_size": 0.2
    }

    metaparams.update(config.get("metaparams", {}))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=metaparams['test_size'])
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
        with open(args.results, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
    except ValueError:
        print(results)


if __name__ == "__main__":
    main()
