#!/usr/bin/env python3
import numpy as np
import pandas as pd
from evaluate_estimator import jackknife_evaluator
import plotly.subplots
import plotly.express as px
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.neighbors import RadiusNeighborsRegressor as RNR
from tqdm import tqdm

def evaluate_models(models):
    biases = []
    variances = []
    for model in tqdm(models):
        bias, variance = jackknife_evaluator(model)
        biases.append(bias); variances.append(variance)
    return np.array(biases), np.array(variances)

if __name__=='__main__':
    fig = plotly.subplots.make_subplots(rows=1, cols=2)

    k_vals = np.linspace(1,60,50).astype(int)
    models = [KNR(n_neighbors=k) for k in k_vals]
    biases, variances = evaluate_models(models)
    knr_results = pd.DataFrame({'model':'knr','parameter':k_vals,'bias squared':biases**2,'variance':variances,'mse':biases**2+variances})

    r_vals = np.linspace(1,10,50)
    models = [RNR(radius=r) for r in r_vals]
    biases, variances = evaluate_models(models)
    rnr_results = pd.DataFrame({'model':'rnr','parameter':r_vals,'bias squared':biases**2,'variance':variances,'mse':biases**2+variances})

    results = pd.concat([knr_results,rnr_results])

    fig = px.line(results, x='parameter', y=['bias squared', 'variance', 'mse'], facet_col='model')
    fig.update_xaxes(matches=None)
    fig.show()
