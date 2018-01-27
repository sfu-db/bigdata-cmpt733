from pyspark.ml import Estimator
from pyspark.ml.tuning import ParamGridBuilder

import numpy as np
import matplotlib; matplotlib.use('Agg') # don't fail when on headless server
import matplotlib.pyplot as plt
import re

def get_classname(instance):
    """Extract class name of type name string"""
    typestr = str(type(instance))
    try:
        return re.match(".*\\.(\\S+)'", typestr).group(1) # match between . and '
    except:
        return re.match(".*'(\\S+)'.*", typestr).group(1) # match between two '

def get_estimator_name(pl):
    """Concatenate names of estimators in the pipeline to create a name"""
    try:
        return ":".join(get_classname(est) for est in pl.getStages() if isinstance(est,Estimator))
    except:
        return get_classname(pl)

def estimator_gridbuilder(estimator, paramnames_values):
    """Help to abbreviate ParamGridBuilder construction from dict"""
    pgb = ParamGridBuilder()
    for pn, vals in paramnames_values.items():
        assert hasattr(vals, '__iter__'), "List of values required for each parameter name"
        pgb.addGrid(estimator.getParam(pn), vals)
    return estimator, pgb

def get_best_tvs_model_params(model):
    """Returns (metricname, score, parameters) tuple
    for `model` estimated via TrainValidationSplit."""
    score_params = zip(model.validationMetrics,
                       model.getEstimatorParamMaps())
    ev = model.getEvaluator()
    scorescale = 1 if ev.isLargerBetter() else -1
    return ((ev.getMetricName(),) +
            max(score_params,
                key=lambda met_parm: met_parm[0] * scorescale))

plt.rcParams["axes.axisbelow"] = False # so grid shows on top of figure

def hist2d(df, col1, col2, bins=(100,100), cmap='gist_heat_r', fraction=1.):
    """Show bi-variate binned histogram (works like a scatterplot for many points).
    This displays the empirical density function of columns named `col1` and `col2` of 
    dataframe `df` sampled w/o replacement by `fraction` passing `bins` argument
    to numpy.histogram2d and `cmap` to matplotlib.pyplot.pcolormesh.
    Save the figure using plt.savefig(outputfilename)
    """
    if fraction != 1.:
        dfc = df.sample(True, fraction).collect()
    else:
        dfc = df.collect()
    c1v = [r[col1] for r in dfc]
    c2v = [r[col2] for r in dfc]
    H, xedges, yedges = np.histogram2d(c1v, c2v, bins=bins)
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H, cmap=plt.get_cmap(cmap))
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.axis('tight')
    plt.grid('on')

