import pandas as pd
import sys
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from datetime import datetime
from sklearn.metrics import accuracy_score
import numpy as np



def main():
    start = datetime.now()
    print start

    sub_train = pd.read_csv('data/sub_train.csv')
    validate = pd.read_csv('data/validate.csv')

    # target is 'booking_bool'
    # train features
    X_train = sub_train.iloc[:,:50].fillna(-100)
    X_train.drop(['position', 'date_time'], axis=1, inplace=True)
    X_train = X_train.values
    # train labels
    y_train = sub_train[['booking_bool']]
    y_train = np.ravel(y_train.values)

    # test features
    X_test = validate.iloc[:,:50].fillna(-100)
    X_test.drop(['position', 'date_time'], axis=1, inplace=True)
    X_test = X_test.values
    # test labels
    y_test = validate[['booking_bool']]
    y_test = np.ravel(y_test.values)

    print 'Learning Stage:'
    print 'Decision Tree'
    param_grid = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 3, 4, 5], 'random_state': [4321]}

    # clf0 = GridSearchCV(DecisionTreeClassifier(), param_grid)
    clf0 = DecisionTreeClassifier(min_samples_split=5, max_depth=None, random_state=4321)
    clf0.fit(X_train, y_train)
    prediction0_on_train = clf0.predict(X_train)
    prediction0_on_test = clf0.predict(X_test)

    print 'Logistic Regression'
    clf1 = LogisticRegression(solver='liblinear', n_jobs=-1)
    clf1.fit(X_train, y_train)
    prediction1_on_train = clf1.predict(X_train)
    prediction1_on_test = clf1.predict(X_test)

    # print 'feature importances:'
    # print 'decision tree: ', clf0.feature_importances_

    print 'Quality Stage:'


    # simple benchmark
    w0 = np.ones(len(y_train))
    for idx, i in enumerate(np.bincount(y_train)):
        if idx == 0:
            w0[y_train == idx] = 1.0 / 0.972260230118
        else:
            w0[y_train == idx] = 1.0 / (1 - 0.972260230118)

    w1 = np.ones(len(y_test))
    for idx, i in enumerate(np.bincount(y_test)):
        if idx == 0:
            w1[y_test == idx] = 1.0 / 0.972053497245
        else:
            w1[y_test == idx] = 1.0 / (1 - 0.972053497245)

    train_bench = np.zeros((len(y_train),), dtype=np.int)
    test_bench = np.zeros((len(y_test),), dtype=np.int)
    print 'benchmark (all zeros) score on train: ', accuracy_score(y_train, train_bench, sample_weight=w0)
    print 'benchmark (all zeros) score on test: ', accuracy_score(y_test, test_bench, sample_weight=w1)


    print 'Decision Tree'
    accuracy0_on_sub_train = accuracy_score(y_train, prediction0_on_train, sample_weight=w0)
    print 'subtrain accuracy score: ', accuracy0_on_sub_train

    accuracy0_on_sub_test = accuracy_score(y_test, prediction0_on_test, sample_weight=w1)
    print 'subtest accuracy score: ', accuracy0_on_sub_test


    print 'Logistic Regression'
    accuracy1_on_sub_train = accuracy_score(y_train, prediction1_on_train, sample_weight=w0)
    print 'subtrain accuracy score: ', accuracy1_on_sub_train

    accuracy1_on_sub_test = accuracy_score(y_test, prediction1_on_test, sample_weight=w1)
    print 'subtest accuracy score: ', accuracy1_on_sub_test


    time = datetime.now() - start
    print 'total processing time: %d' % int(time.seconds)


if __name__ == '__main__':
    main()