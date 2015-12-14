from pyspark import SparkConf, SparkContext
from pyspark import SparkContext,SQLContext
from pyspark.mllib.common import callMLlibFunc, _py2java, _java2py, inherit_doc
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from numpy import array
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.feature import StandardScalerModel

# input includes all geographical information about all station, And prediction of model for the given year are stored in output
# input = sys.argv[1]
# myModelPath = sys.argv[2]
# year = sys.argv[3]
# output = sys.argv[4]

input = "/users/bshadgar/Desktop/textformat"
myModelPath = "/users/bshadgar/Desktop/model"
year = '2016'
output = "/users/bshadgar/Desktop/predictions/preds"

conf = SparkConf().setAppName('prediction')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
# Load and parse the data
# line format: (station, latitude, longitude,)


def parsePoint(line):
    return LabeledPoint(line[0], line[1:])

# read data from station file
def  getdata(line):
        line = line.split('  ')
        values = [x.strip() for x in line]
        return values
stations = sc.textFile(input)
stations = stations.map(getdata)
stations = stations.map(lambda (a,b,c): (float(hash(a)), int(year), float(b), float(c))).cache()
stationsDF = sqlContext.createDataFrame(stations)

# create dataset to fit into model
parseData = stations.map(parsePoint)

# load the model
sameModel = LinearRegressionModel.load(sc, myModelPath)

# run the model
stationidAndPreds = parseData.map(lambda p : (p.label,  float(sameModel.predict(p.features))))
stationidAndPredsDF = sqlContext.createDataFrame(stationidAndPreds)

# the result returns a predicted value for each station (stationId) in the given year
# joining the stations rdd with stationidAndPreds to find the latitude and longitude of each station
result = stationsDF.join(stationidAndPredsDF).where(stationidAndPredsDF[0]==stationsDF[0]).select(stationidAndPredsDF[1], stationsDF[2], stationsDF[3])

resultRdd = result.rdd.map(lambda (pred, lat, long): (str(pred) + ',' + str(lat) + ',' + str(long)))
# Save data into text file
resultRdd.coalesce(1).saveAsTextFile(output + year)





