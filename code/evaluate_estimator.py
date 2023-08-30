#!/usr/bin/env python3
import numpy as np
import sklearn.datasets

def make_quadratic():
    X = np.random.normal(size=(1000,2))
    y = (X**2).sum(axis=1)
    return X,y

#def jackknife_evaluator(model, data_generator = sklearn.datasets.make_friedman1):
#def jackknife_evaluator(model, data_generator = sklearn.datasets.make_regression):
def jackknife_evaluator(model, data_generator = make_quadratic):
    X,y = data_generator()
    signed_errors = []
    for i in range(len(y)):
        X_train = np.delete(X,i,axis=0)
        y_train = np.delete(y,i,axis=0)
        X_test  = X[i]
        y_test  = y[i]
        signed_errors.append(model.fit(X_train,y_train).predict(X_test.reshape(1,-1)) - y_test)
    signed_errors = np.array(signed_errors)
    return signed_errors.mean(), signed_errors.var()
