export CLASSPATH=`hdfs classpath --glob`

HADOOP_USER_NAME=hadoop hdfs dfs -chmod -R 777 /
