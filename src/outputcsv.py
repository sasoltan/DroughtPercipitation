#This is the code for viweing past data( average Percipitation in recent years)
# This program is not part of the Main Program
# Implemented Only for visualization


from pyspark import SparkConf, SparkContext,SQLContext,Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import sys, operator
import re, string

inputs = sys.argv[1]
output = sys.argv[2]
#Get the station Names form GHCD-stations
def getStations():
    text = sc.textFile(inputs+"/station")
    words = text.map(lambda l: l.split())
    stationDF = words.map(lambda l: Row(station=(l[0]), country=(l[1])))
    stationschema = sqlContext.inferSchema(stationDF)
    stationschema.registerTempTable("stations")
    outdata = stationschema.repartition(40).rdd.map(lambda w: w.station + "   "+w.country).coalesce(1)
    #stationschema.write.format('parquet').save(output+"/stationparquet")
    outdata.saveAsTextFile(output+"/stationtextformat")
#ETL Processing to get the required data
def  getdata(line):
        line = line.split('  ')
        values = [x.strip() for x in line]
        return values

conf = SparkConf().setAppName('climate')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
#Structureed schema for the received database
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

info = sqlContext.read.format('com.databricks.spark.csv').options(header='false').schema(climateSchema).load(inputs+"/data")
info.registerTempTable("info")
stationinfo = sqlContext.sql("SELECT station, date, element, value as prcp, FLOOR(date/10000) as yy FROM info WHERE element='PRCP' ")
stationinfo.registerTempTable("stationinfo")
stationinfo.cache()
#Sort the data by station and year to get the avergae percipatition for each year
yearlyprcp = sqlContext.sql("SELECT station, yy, Avg(prcp) as avg_prcp FROM stationinfo GROUP BY station, yy ")
yearlyprcp.registerTempTable("yearlyprcp")

getStations()
# Get the station information such as latitude and longitude
stations = sc.textFile(output+"/stationtextformat")
stations = stations.map(getdata)
stations = stations.map(lambda (a,b): Row(station= a, country=b)).cache()
stationDF = sqlContext.createDataFrame(stations)
stationDF.registerTempTable("StationTable")
stationDF.cache()
shortenstations = sqlContext.sql("SELECT SUBSTR(station, 1, 2) As station,avg_prcp,yy FROM yearlyprcp")
shortenstations.show()
joinedresult = stationDF.join(shortenstations).where(stationDF.station == shortenstations.station).select(shortenstations.avg_prcp, stationDF.country, shortenstations.yy)
joinedresult.registerTempTable("joinedresult")
results = sqlContext.sql("SELECT country, yy, Avg(avg_prcp) as avg_prcp FROM joinedresult GROUP BY country, yy ")
results.registerTempTable("results")
years = sqlContext.sql("SELECT COUNT(distinct(yy)) FROM results ")
years.show()
yearscount= years.rdd.collect()
r=str(yearscount[0])
r= r.split('=',2)
r = r[1].split(')')
count = int(r[0])
listyear=[2004,2003,2002,2001,2000]
#For each year save all the average percipitation
for x in xrange(count):
    yearspec = results.filter(results.yy==listyear[x])
    outdata = yearspec.select(yearspec.country, yearspec.avg_prcp)
    outrdd=outdata.repartition(40).rdd.map(lambda l: str(l.country)+","+str(l.avg_prcp)).coalesce(1)
    path = str(listyear[x])+".csv"
    outrdd.saveAsTextFile(output+'/prevYears/'+path)

