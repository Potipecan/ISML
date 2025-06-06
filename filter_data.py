from train import load_data
from typing import Literal
from scipy.stats import zscore
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.utils import shuffle
from csv import reader, writer
import argparse
from pathlib import Path

default_thresh = {
    'z': 3.0,
    'iqr': 1.5,
    'iso': 0.01
}

def sort_data(X, Y):
    keys = set(Y)
    sorted_data = {k: [] for k in list(keys)}
    
    for x, y in zip(X, Y):
        sorted_data[y].append(x)
    
    return sorted_data

def filter_z_score(x: np.ndarray, thresh):
    scores = zscore(x, axis=0)
    
    mask = np.max(np.abs(scores), axis=1) < thresh
    return x[mask]


def filter_iqr(x: np.ndarray, thresh):

    Q1 = np.percentile(x, 25, axis=0)
    Q3 = np.percentile(x, 75, axis=0)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - thresh * IQR
    upper_bound = Q3 + thresh * IQR
    
    # Flag as True if within bounds
    in_bounds = (x >= lower_bound) & (x <= upper_bound)
    
    # Keep samples where all features are in bounds
    sample_mask = in_bounds.all(axis=1)
    
    return x[sample_mask]


def filter_iso(x: np.ndarray, thresh):
    iso = IsolationForest(contamination=thresh)
    preds = iso.fit_predict(x)  # -1 = outlier, 1 = inlier
    mask = preds == 1
    
    return x[mask]
    
filter_dict = {
    'z': filter_z_score,
    'iqr': filter_iqr,
    'iso': filter_iso
}

def filter_outliers(x, y, mode: Literal['z', 'iqr', 'iso'], threshold = None):
    if threshold is None:
        threshold = default_thresh[mode]

    sorted_data = sort_data(x, y)
    ff = filter_dict[mode]
    fx = []
    fy = []
    for k, d in sorted_data.items():
        filtered_data = ff(np.array(d), threshold)
        fx.extend(filtered_data)
        fy.extend([k] * len(filtered_data))
    
    fx, fy = shuffle(fx, fy)
    return fx, fy


def export(file, x, y, dims):
    with open(file, 'w', encoding='utf-8') as f:
        wr = writer(f, delimiter=';')
        wr.writerow(dims)
        wr.writerows(map(lambda t: [t[1]] + t[0].tolist(), zip(x, y)))
   
   
def main():
    parser = argparse.ArgumentParser("filter_data")
    parser.add_argument('-m', dest='mode', choices=['z', 'iqr', 'iso'], required=True)
    parser.add_argument('-t', dest='thresh', required=False, type=float)
    parser.add_argument('-o', dest='output', required=False)
    parser.add_argument("file")
    
    args = parser.parse_args()
    if args.output is None:
        p = Path(args.file)
        args.output = f'{p.parent}/{p.stem}-filtered_{args.mode}{p.suffix}' 

    dim, x, y = load_data(args.file)
    
    fx, fy = filter_outliers(x, y, args.mode, args.thresh)
    export(args.output, fx, fy, dim)
    

if __name__ == '__main__':
    main()
    