from pyspark import SparkConf, SparkContext
from pyspark import SparkContext,SQLContext
from pyspark.mllib.common import  _py2java, _java2py
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from numpy import array
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.feature import StandardScalerModel

inputs = "/users/bshadgar/Desktop/out"  # Should be some file on your system
myModelPath = "/users/bshadgar/Desktop/model"
params = "/users/bshadgar/Desktop/parameters"

conf = SparkConf().setAppName('climate regression')
sc = SparkContext(conf=conf)

# Load and parse the data
# each row of data is considered as one sample so that the first element is label and the second is a vector of features
# line format: (station, yy, latitude, longitude,  avg_prcp)
sqlContext = SQLContext(sc)
datadf = sqlContext.read.parquet(inputs)
data = datadf.rdd.map(lambda w: (float(w.avg_prcp), int(w.yy), float(w.latitude), float(w.longitude))).cache()
# Do Normalization and formating in order to fit in LinearRegression library

def parsePoint(line, maxim, minim):
    prcp = (line[0] - minim[0])/(maxim[0]- minim[0])
    year = (line[1] - minim[1])/(maxim[1]- minim[1])
    lati = (line[2] - minim[2])/(maxim[2]- minim[2])
    longt = (line[3] - minim[3])/(maxim[3]- minim[3])
    return LabeledPoint(prcp, [year, lati, longt])

max_prcp = data.max()
min_prcp = data.min()

lat = data.map(lambda x: (x[2])).cache()
min_lat = lat.min()
max_lat = lat.max()

longt =  data.map(lambda x: (x[3])).cache()
min_long = longt.min()
max_long = longt.max()

max_ = [max_prcp[0], float(2050), max_lat, max_long]
min_ = [min_prcp[0], float(1950), min_lat, min_long]

parsedData = data.map(lambda x: parsePoint(x, max_, min_)).cache()


# Split data aproximately into training (60%) and test (40%)
trainData, testData = parsedData.randomSplit([0.6, 0.4], seed = 0)
trainData.cache()
testData.catch()

# Build the model using Trial and error to find out the Parameters.
model = LinearRegressionWithSGD.train(trainData, iterations =5000, step = 0.1 )

# Evaluate the model on test data
valuesAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))

# Save and load model
def save(self, sc, path):
        java_model = sc._jvm.org.apache.spark.mllib.regression.LinearRegressionModel(_py2java(sc, self._coeff), self.intercept)
        java_model.save(sc._jsc.sc(), path)

@classmethod
def load(cls, sc, path):
        java_model = sc._jvm.org.apache.spark.mllib.regression.LinearRegressionModel.load(sc._jsc.sc(), path)
        weights = _java2py(sc, java_model.weights())
        intercept = java_model.intercept()
        model = LinearRegressionModel(weights, intercept)
        return model

# Save parameters i.e. min_ and max_ on disk
f = open(params, "w")
f.write(str(str(min_) + ',' + str(max_)) )
f.close()



model.save(sc, myModelPath)
sameModel = LinearRegressionModel.load(sc, myModelPath)
sample = testData.map(lambda p: p.features)
predictValues = sameModel.predict(sample)




