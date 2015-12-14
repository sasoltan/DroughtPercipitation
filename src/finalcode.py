from pyspark import SparkConf, SparkContext,SQLContext,Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import sys, operator
from numpy import array
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.feature import StandardScalerModel
from pyspark.mllib.common import callMLlibFunc, _py2java, _java2py, inherit_doc
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

#Input and Output directory
inputs = sys.argv[1]
yearprediction = sys.argv[2]
output = sys.argv[3]
#Default value,will be changed when reading from file.
maxVal=8000;
#Setting up the spark framework
conf = SparkConf().setAppName('DroughtPrediction')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


#Function that returns labeled data, LabeledPoint is from mllib.regression library
def parsePoint(line):
    return LabeledPoint(line[0], line[1:])


def parsePointPrediction(line, maxim, minim):
    prcp = (line[0] - minim[0])/(maxim[0]- minim[0])*3
    year = (line[1] - minim[1])/(maxim[1]- minim[1])*3
    lati = (line[2] - minim[2])/(maxim[2]- minim[2])*3
    longt = (line[3] - minim[3])/(maxim[3]- minim[3])*3
    return LabeledPoint(prcp, [year, lati, longt])
    
    
#Function that converts string to float.
def strtofloat(x):
    if not(x):
        x=0
        x=float(x)
    else:
        x = float(x)
    return x
    
    
#Reads the stations from GHCN-Stations.txt and writes the results to output
def getStations():
    text = sc.textFile(inputs+"/station/stations")
    words = text.map(lambda l: l.split())
    stationDF = words.map(lambda l: Row(station=(l[0]), latitude=(l[1]), longitude=(l[2])))
    stationschema = sqlContext.inferSchema(stationDF)
    stationschema.registerTempTable("stations")
    outdata = stationschema.repartition(40).rdd.map(lambda w: w.station + "   "+w.latitude+"   "+w.longitude+" ").coalesce(1)
    #stationschema.write.format('parquet').save(output+"/stationparquet")
    outdata.saveAsTextFile(output+"/stationtextformat")
    
    
#Get the countries for final visualization
def getCountries():
    text = sc.textFile(inputs+"/station/countries")
    words = text.map(lambda l: l.split())
    stationDF = words.map(lambda l: Row(station=(l[0]), country=(l[1])))
    stationschema = sqlContext.inferSchema(stationDF)
    stationschema.registerTempTable("stations")
    outdata = stationschema.repartition(40).rdd.map(lambda w: w.station + "   "+w.country).coalesce(1)
    #stationschema.write.format('parquet').save(output+"/stationparquet")
    outdata.saveAsTextFile(output+"/countries")
        
#split each line and returns the values
def  getdata(line):
     line = line.split('  ')
     values = [x.strip() for x in line]
     return values
     
     
# Save and load regression model
#This function is not enabled in spark python yet
#Therefore, we use JAVA Virtual Machine for reading and saving the regression model
#Saving and reading saves substantial amount of time and memory
def save(self, sc, path):
        java_model = sc._jvm.org.apache.spark.mllib.regression.LinearRegressionModel(_py2java(sc, self._coeff), self.intercept)
        java_model.save(sc._jsc.sc(), path)


#Loads the regression method into the program
@classmethod
def load(cls, sc, path):
        java_model = sc._jvm.org.apache.spark.mllib.regression.LinearRegressionModel.load(sc._jsc.sc(), path)
        weights = _java2py(sc, java_model.weights())
        intercept = java_model.intercept()
        model = LinearRegressionModel(weights, intercept)
        return model
        
#Regression model, Linear Regression with Stochastic gradient descent is implemented in this project
# Mllib library is used for regression
#LinearRegressionWithSGD provides the required regression type


def regression():
    #Regression Point
    #Reads the data from the joinedResults directory as a parquet file
    datadf = sqlContext.read.parquet(output+"/joinedResults")
    datadf.show()
    data = datadf.rdd.map(lambda w: (float(w.avg_prcp), int(w.yy), float(w.latitude), float(w.longitude)))
    max_prcp = data.max()
    min_prcp = data.min()
    lat = data.map(lambda x: (x[2])).cache()
    min_lat = lat.min()
    max_lat = lat.max()

    longt =  data.map(lambda x: (x[3])).cache()
    min_long = longt.min()
    max_long = longt.max()
    
    max_ = [max_prcp[0], float(2050), max_lat, max_long]
    min_ = [min_prcp[0], float(1990), min_lat, min_long]
    # change the format to fit in LinearRegression library
    parsedData = data.map(lambda x: parsePointPrediction(x, max_, min_)).cache()
    # Split data aproximately into training (80%) and test (20%)
    trainData, testData = parsedData.randomSplit([0.8, 0.2], seed = 0)
    trainData.cache()
    testData.cache()
    # Build the model using Try and error to find out the Parameters.
    model = LinearRegressionWithSGD.train(trainData, iterations =500, regType="l2", regParam=10, intercept="true"  )
    # Evaluate the model on test data
    valuesAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
    MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    print("Mean Squared Error = " + str(MSE))
    maxVal=max_prcp[0]

    model.save(sc, output+"/modelpath")
    return
#scale the predicted values to 0 and 1 for regression
def scalePoint(line, maxim, minim):
    yy = (line[1] - minim[0])/(maxim[0]- minim[0])*3
    lat = (line[2] - minim[1])/(maxim[1]- minim[1])*3
    log = (line[3] - minim[2])/(maxim[2]- minim[2])*3
    return ([line[0], yy, lat, log])
    
#Scale back the results for output.
def rescale(line):
	a = line[0]
	b = line[1] / 3 * maxVal
	return ([a,b])
#Prediction function
def prediction():
    year=yearprediction
    stations = sc.textFile(output+"/stationtextformat")
    stations = stations.map(getdata).map(lambda x: (x[0], int(year), float(x[1]), float(x[2])))
    lat = stations.map(lambda x: (x[2])).cache()
    min_lat = lat.min()
    max_lat = lat.max()

    longtitude =  stations.map(lambda x: (x[3])).cache()
    min_long = longtitude.min()
    max_long = longtitude.max()

    max_ = [float('2050'), max_lat, max_long]
    min_ = [float('1990'), min_lat, min_long]

    stations = stations.map(lambda x: scalePoint(x, max_, min_)).cache()
    stationsDF = sqlContext.createDataFrame(stations)
    # load the model
    sameModel = LinearRegressionModel.load(sc, output+"/modelpath")
    # run the model
    stationidAndPreds = stations.map(lambda p : (p[0],  float(sameModel.predict(p[1:]))))
    # the result returns a predicted value for each station (stationId) in the given year
    resultRdd = stationidAndPreds.map(rescale)
    rddschema = resultRdd.map(lambda (a,b): Row(station= a, avg_prcp=b)).cache()
    stationidAndPredsDF = sqlContext.createDataFrame(rddschema)
    stationidAndPredsDF.registerTempTable("stationPrediction")
    getCountries()
    countires = sc.textFile(output+"/countries")
    countriesRdd = countires.map(getdata)
    countries = countriesRdd.map(lambda (a,b): Row(station= a, country=b)).cache()
    countriesDF = sqlContext.createDataFrame(countries)
    countriesDF.registerTempTable("StationTable")
    countriesDF.cache()
    shortenstations = sqlContext.sql("SELECT SUBSTR(station, 1, 2) As station,avg_prcp FROM stationPrediction")
    shortenstations.show()
    joinedresult = countriesDF.join(shortenstations).where(countriesDF.station == shortenstations.station).select(shortenstations.avg_prcp, countriesDF.country)
    joinedresult.registerTempTable("joinedresult")
    results = sqlContext.sql("SELECT country, Avg(avg_prcp) as avg_prcp FROM joinedresult GROUP BY country")
    results.registerTempTable("results")
    outrdd=results.repartition(40).rdd.map(lambda l: str(l.country)+","+str(l.avg_prcp)).coalesce(1)
    path = yearprediction
    outrdd.saveAsTextFile(output+'/prediction/'+path)
    
#main function
def main():
    #Constructing the schema for the dataset
    climateSchema = StructType([
          StructField('station', StringType(), False),
          StructField('date', IntegerType(), False),
          StructField('element', StringType(), False),
          StructField('value', IntegerType(), True),
          StructField('mflag', StringType(), True),
          StructField('qflag', StringType(), True),
          StructField('sflag', StringType(), True),
          StructField('obstime', StringType(), True),
          ])
    #Read the dataset and extract the required information
    info = sqlContext.read.format('com.databricks.spark.csv').options(header='false').schema(climateSchema).load(inputs+"/data")
    info.registerTempTable("info")
    #Date is divided to 10000 in order to extract only the year from the date variable
    stationinfo = sqlContext.sql("SELECT station, date, element, value AS prcp, FLOOR(date/10000) as yy FROM info WHERE element='PRCP'")
    stationinfo.registerTempTable("stationinfo")
    stationinfo.cache()
 
    #  create  tables that holds the average of prcp of each station in each year
    yearlyprcp = sqlContext.sql("SELECT station, yy, Avg(prcp) as avg_prcp FROM stationinfo GROUP BY station, yy ")
    yearlyprcp.registerTempTable("prcpMean")
    # Join to station file to add latitude and longitude which is used for feature selection in regression
    getStations()
    stations = sc.textFile(output+"/stationtextformat")
    stations = stations.map(getdata)
    stations = stations.map(lambda (a,b,c): Row(station= a, latitude=float(b), longitude=(c))).cache()
    stationDF = sqlContext.createDataFrame(stations)
    stationDF.registerTempTable("StationTable")
    stationDF.cache()
    yearlyprcp.show()
    result = stationDF.join(yearlyprcp).where(stationDF.station == yearlyprcp.station).select(yearlyprcp.avg_prcp, yearlyprcp.station, yearlyprcp.yy, stationDF.latitude, stationDF.longitude )
    result.show()
    # save into parquet file
    result.write.format('parquet').save(output+"/joinedResults")
    regression()
    prediction()

if __name__ == "__main__":
    main()

# setenv SPARK_HOME /Volumes/projects/big-data/spark-1.5.1-bin-hadoop2.6
# ${SPARK_HOME}/bin/spark-submit --master local --packages com.databricks:spark-csv_2.11:1.2.0
#${SPARK_HOME}/bin/pyspark --master local
##spark-submit --master=yarn-cluster --executor-memory=6g --num-executors=6 --executor-cores=4 --packages com.databricks:spark-csv_2.11:1.2.0 CMPT732/finalcode.py climatedataset/ 2020 /user/saeeds/climateoutput
#${SPARK_HOME}/bin/spark-submit --master "local[*]" --packages com.databricks:spark-csv_2.11:1.2.0 Desktop/outputcsv.py Documents/bigdata 2012 Documents/results