import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import csv
from zipfile import ZipFile
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
        reader = csv.reader(df)
        dim = tuple(map(int, next(reader))) # get width and height from first line
        y = []
        x = []
        
        for row in reader:
            y.append(int(row[0]))
            x.append(list(map(float, row[2:])))
    
        return dim, x, y        
        
def get_model(m_str: str, m_params: dict | None):
    if m_params is None:
        m_params = {}

    # noinspection PyUnreachableCode
    match m_str:
        case 'svm':
            return sklearn.svm.SVC(**m_params)
        case 'sgd':
            return sklearn.linear_model.SGDClassifier(**m_params)
        case 'decision_tree':
            return sklearn.tree.DecisionTreeClassifier(**m_params)
        case 'knn':
            return sklearn.neighbors.KNeighborsClassifier(**m_params)
        case 'ridge':
            return sklearn.linear_model.RidgeClassifier(**m_params)
        case 'gnb':
            return sklearn.naive_bayes.GaussianNB(**m_params)
        case 'mlp':
            return sklearn.neural_network.MLPClassifier(**m_params)
        case 'ada_boost':
            return sklearn.ensemble.AdaBoostClassifier(**m_params)
        case 'grad_boost':
            return sklearn.ensemble.GradientBoostingClassifier(**m_params)
        case 'random_forest':
            return sklearn.ensemble.RandomForestClassifier(**m_params)
        #     
        case _:
            raise ValueError(f"Unknown model string: {m_str}")
        
def main():
    parser = argparse.ArgumentParser("train")
    parser.add_argument('--validate_conf', action='store_true')
    parser.add_argument('-o', '--results', default="train_results.json")
    parser.add_argument("data", type=str)
    parser.add_argument("conf", type=str)
    args = parser.parse_args()

    config = json.load(args.conf)
    try:
        model = get_model(config['model'], config.get('model_params'))
    except Exception as ex:
        print(ex)
        exit(1)

    if args.validate_conf:
        exit(0)

    dim, x, y = load_data(args.data)
    
    metaparams = {
        "test_size": 0.4
    }
    
    metaparams.update(config.get("metaparams", {}))
    
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=metaparams['test_size'])
    model.fit(x_train, y_train)
    
    # results
    y_pred = model.predict(y_test)
    
    results = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'accuracy': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    try:
        json.dump(results, args.results, ensure_ascii=False)
    except ValueError:
        print(results)
    
    
if __name__ == "__main__":
    main()