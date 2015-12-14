from pyspark import SparkConf, SparkContext,SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import sys, operator
import json
import sys, operator
import re, string
inputs = sys.argv[1]
output = sys.argv[2]

def strtofloat(x):
    if not(x):
        x=0
        x=float(x)
    else:
        x = float(x)
    return x

def main():
    conf = SparkConf().setAppName('climate')
    sc = SparkContext(conf=conf)
    text = sc.textFile(inputs)
    linere=re.compile("[\w']+")
    words = text.map(lambda l: l.split())
    stationDF = words.map(lambda l: Row(station=(l[0]), latitude=(l[1]), longitude=(l[2])))
    stationschema = sqlContext.inferSchema(stationDF)
    stationschema.registerTempTable("stations")
    outdata = stationschema.repartition(40).rdd.map(lambda w: w.station + "   "+w.latitude+"   "+w.longitude+" ").coalesce(1)
    stationschema.write.format('parquet').save(output+"/parquet")
    outdata.saveAsTextFile(output+"/textformat")
if __name__ == "__main__":
    main()

#${SPARK_HOME}/bin/spark-submit --master "local[*]" desktop/stationNames.py Documents/ghcnd-stations.txt Documents/dataset1