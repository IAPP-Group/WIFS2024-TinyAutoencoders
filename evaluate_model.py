import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC


ENCODERS = ['adm', 'biggan', 'glide', 'midjourney', 'sdv4', 'sdv5', 'vqdm', 'wukong']


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, nargs='+')
    parser.add_argument('-n', '--n-imgs', type=int, required=True)
    parser.add_argument('base_path', type=Path)
    
    return parser


def get_rec_errors(parent_path):
    results = defaultdict(dict)
    for encoder in ENCODERS:
        for dataset in ENCODERS:
            results[encoder][dataset] = np.load(f'{parent_path}/rec_errors_encoder_{encoder}_dataset_{dataset}.npy')
    return dict(results)


def get_xs_ys(rec_errors):
    xs = []
    ys = []
    
    for i, dataset in enumerate(ENCODERS):
        features = []
        for encoder in ENCODERS:
            features.append(np.mean(rec_errors[encoder][dataset], axis=1).reshape([-1, 1]))
        features = np.hstack(features)
        xs.append(features)
        ys.extend([i] * features.shape[0])
    
    xs = np.vstack(xs)
    ys = np.asarray(ys)

    return xs, ys


def run_on_paths(train_path, test_path):
    scores_train = get_rec_errors(train_path)
    scores_test = get_rec_errors(test_path)

    xs_train, ys_train = get_xs_ys(scores_train)
    xs_test, ys_test = get_xs_ys(scores_test)

    clf = LinearSVC(max_iter=1000000, dual=False)
    clf = GridSearchCV(clf, param_grid={ 'C': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100] }, cv=5, refit=True)
    clf.fit(xs_train, ys_train)
    
    ys_pred = clf.predict(xs_test)

    return ys_test, ys_pred

    
def main(args):
    accuracies = []
    f1s = []
    for seed in args.seed:
        train_path = args.base_path / f'Train/{seed}/{args.n_imgs}'
        test_path = args.base_path / f'Test/{seed}/{args.n_imgs}'

        ys_true, ys_pred = run_on_paths(train_path, test_path)
        accuracies.append(balanced_accuracy_score(ys_true, ys_pred))
        f1s.append(f1_score(ys_true, ys_pred, average='weighted'))

    print(f"N={args.n_imgs} - Accuracy={np.mean(accuracies):.2f} +/- {np.std(accuracies):.2f} - F1={np.mean(f1s):.2f} +/ {np.std(f1s):.2f}")


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())