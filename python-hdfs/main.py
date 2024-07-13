import fsspec
import IPython

from pyspark import SparkConf
from pyspark.sql import SparkSession

from pyiceberg.catalog import load_catalog

catalog = load_catalog("iceberg", uri="thrift://hive-metastore:9083")
fs = fsspec.filesystem(protocol="hdfs")

with SparkSession.builder.config(
    conf=SparkConf()
    .setAppName("iceberg")
    .setMaster("spark://spark-master-svc:7077")
    .setAll(
        {
            "spark.jars.packages": "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2",
            "spark.jars": "/opt/hadoop/lib/iceberg-spark-runtime-3.5_2.12-1.5.2.jar",
            "spark.driver.extraClassPath": "/opt/hadoop/lib/*.jar",
            "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            "spark.sql.catalog.iceberg": "org.apache.iceberg.spark.SparkCatalog",
            "spark.sql.catalog.iceberg.uri": "thrift://hive-metastore:9083",
            "spark.driver.port": "10000",
            "spark.driver.host": "pyspark",
            "spark.driver.bindAddress": "0.0.0.0",
        }.items()
    )
).getOrCreate() as spark:
    IPython.embed()
