from pyspark import SparkConf, SparkContext,SQLContext,Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType




# inputs = sys.argv[1]
# inputs2 = sys.argv[2]
# output = sys.argv[3]
inputs = "/users/saeeds/Desktop/a4-weather-1"
output = "/users/saeeds/Desktop/out1"
input2 = "/users/saeeds/Desktop/textformat"

def main():
    conf = SparkConf().setAppName('climate')
    sc = SparkContext(conf=conf)

    sqlContext = SQLContext(sc)
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
    info = sqlContext.read.format('com.databricks.spark.csv').options(header='false').schema(climateSchema).load(inputs)
    info.registerTempTable("info")
    stationinfo = sqlContext.sql("SELECT station, date, element, value, FLOOR(date/10000) as yy FROM info ")
    stationinfo.registerTempTable("stationinfo")
    stationinfo.cache()


    prcpTable = sqlContext.sql("SELECT station, date, value as prcp, yy FROM stationinfo WHERE element='PRCP' ")
    prcpTable.registerTempTable("prcpTable")
    prcpTable.cache()
    # prcpTable.show()

    # create 3 tables that hold the monthly average of min, max temperature and prcp
    yearlyprcp = sqlContext.sql("SELECT station, yy, ROUND(Avg(prcp),0) as avg_prcp FROM prcpTable GROUP BY station, yy ")
    yearlyprcp.registerTempTable("prcpMean")
    # yearlyprcp.show()

    # get information about stations from stations.txt


    def  getdata(line):
        line = line.split('  ')
        values = [x.strip() for x in line]
        return values

    stations = sc.textFile(input2)
    stations = stations.map(getdata)
    stations = stations.map(lambda (a,b,c): Row(station= a, latitude=float(b), longitude=float(c))).cache()
    stationDF = sqlContext.createDataFrame(stations)
    stationDF.registerTempTable("StationTable")
    stationDF.cache()

    # param = sqlContext.sql("SELECT MAX(latitude) as max_lat, Min(latitude) as min_lat, MAX(longitude) as max_long, Min(longitude) as min_long FROM StationTable")
    # param.show()

    # Join to station file to add latitude and longitude and stationID
    result = stationDF.join(yearlyprcp).where(stationDF.station == yearlyprcp.station).select(yearlyprcp.avg_prcp, yearlyprcp.station, yearlyprcp.yy, stationDF.latitude, stationDF.longitude )

    # save into parquet file
    result.write.format('parquet').save(output)
#   result.saveAsTextFile(output)


if __name__ == "__main__":
    main()

# setenv SPARK_HOME /Volumes/projects/big-data/spark-1.5.1-bin-hadoop2.6
# ${SPARK_HOME}/bin/spark-submit --master local --packages com.databricks:spark-csv_2.11:1.2.0

#${SPARK_HOME}/bin/pyspark --master local



#spark-submit --master "local[*]" --packages com.databricks:spark-csv_2.11:1.2.0 CMPT732/ClimateWeather.py projectTest/ user/saeeds/outputclimate
#${SPARK_HOME}/bin/spark-submit --master "local[*]" --packages com.databricks:spark-csv_2.11:1.2.0 Desktop/climateweather.py Documents/bigdata Documents/dataset1   #${SPARK_HOME}/bin/spark-submit --master "local[*]" --packages com.databricks:spark-csv_2.11:1.2.0 Desktop/climateweather.py Documents/bigdata Documents/dataset1