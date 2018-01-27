# e.g. run via: spark-submit weather_hint.py /courses/732/tmax-2
# to produce a png that shows temperature measurement density for a lat long grid

# also usable via pyspark with PYSPARK_DRIVER_PYTHON=ipython
# %pylab
# from weather_hint import *
# get_data("/courses/732/tmax-2")
# ...

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row, functions, Column
from pyspark.sql.types import *

from pyspark.ml import Pipeline, Estimator
from pyspark.ml.feature import SQLTransformer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.regression import (LinearRegression,
                                   GBTRegressor,
                                   RandomForestRegressor,
                                   DecisionTreeRegressor)
import sys
from weather_tools import *

input = None
output = None
try:
    input = sys.argv[1]
    output = sys.argv[2]
except:
    pass

spark = SparkSession.builder.appName('weather related prediction').getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

schema = StructType([
    StructField('station', StringType(), False),
    StructField('date', DateType(), False),
    # StructField('dayofyear', IntegerType(), False),
    StructField('latitude', FloatType(), False),
    StructField('longitude', FloatType(), False),
    StructField('elevation', FloatType(), False),
    StructField('tmax', FloatType(), False),
])

def get_data(inputloc, tablename='data'):
    data = spark.read.csv(inputloc, schema=schema)
    data.createOrReplaceTempView(tablename)
    return data

def make_weather_trainers(trainRatio,
                          estimator_gridbuilders,
                          metricName=None):
    """Construct a list of TrainValidationSplit estimators for weather data
       where `estimator_gridbuilders` is a list of (Estimator, ParamGridBuilder) tuples
       and 0 < `trainRatio` <= 1 determines the fraction of rows used for training.
       The RegressionEvaluator will use a non-default `metricName`, if specified.
    """
    feature_cols = ['latitude', 'longitude', 'elevation', 'doy']
    column_names = dict(featuresCol="features",
                        labelCol="tmax",
                        predictionCol="tmax_pred")

    # getDOY = # TODO: engineer a day of year feature 'doy' from schema
    getDOY = SQLTransformer(
        statement="SELECT *, dayofyear(date) AS doy FROM __THIS__")

    feature_assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=column_names["featuresCol"])
    ev = (RegressionEvaluator()
          .setLabelCol(column_names["labelCol"])
          .setPredictionCol(column_names["predictionCol"])
    )
    if metricName:
        ev = ev.setMetricName(metricName)
    tvs_list = []
    for est, pgb in estimator_gridbuilders:
        est = est.setParams(**column_names)

        # pl = # TODO: Construct a pipeline with estimator est
        pl = Pipeline(stages=[getDOY, feature_assembler, est])

        paramGrid = pgb.build()
        tvs_list.append(TrainValidationSplit(estimator=pl,
                                             estimatorParamMaps=paramGrid,
                                             evaluator=ev,
                                             trainRatio=trainRatio))
    return tvs_list

def get_best_weather_model(data):
    train, test = data.randomSplit([0.75, 0.25])
    train = train.cache()
    test = test.cache()

    # e.g., use print(LinearRegression().explainParams()) to see what can be tuned
    estimator_gridbuilders = [
        estimator_gridbuilder(
            LinearRegression(),
            dict(regParam=[.3],         # [0.1, 0.01]
                 elasticNetParam=[.8],  # 0-L2, 1-L1
                 maxIter=[10]
            )),

        # TODO: find better estimators

    ]
    metricName = 'r2'
    tvs_list = make_weather_trainers(.2, # fraction of data for training
                                     estimator_gridbuilders,
                                     metricName)
    ev = tvs_list[0].getEvaluator()
    scorescale = 1 if ev.isLargerBetter() else -1
    model_name_scores = []
    for tvs in tvs_list:
        model = tvs.fit(train)
        test_pred = model.transform(test)
        score = ev.evaluate(test_pred) * scorescale
        model_name_scores.append((model, get_estimator_name(tvs.getEstimator()), score))
    best_model, best_name, best_score = max(model_name_scores, key=lambda triplet: triplet[2])
    print("Best model is %s with validation data %s score %f" % (best_name, ev.getMetricName(), best_score*scorescale))
    return best_model

def main(inputloc, outputloc):
    data = get_data(inputloc)
    model = get_best_weather_model(data)
    print("Best parameters on test data:\n", get_best_tvs_model_params(model))
    data_pred = model.transform(data).drop("features")
    # ATTN: large file output for debugging only
    #data_pred.coalesce(1).write.csv(outputloc, sep=',', mode='overwrite')

    #hist2d(data_pred,'tmax','tmax_pred', max(fraction=5.e5 / data_pred.count(),1))
    #figurename = 'pred_vs_label.png'
    hist2d(data_pred,'longitude','latitude', fraction=max(5.e5 / data_pred.count(),1))
    figurename = 'lat_lng.png'
    
    plt.savefig(figurename)
    print(figurename + ' saved to local directory')

if __name__=='__main__':
    # Note: in current version output is only used for debugging
    main(input, output)
