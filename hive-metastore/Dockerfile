FROM docker.io/maven:3 AS maven

COPY pom.xml .

RUN mvn dependency:copy-dependencies -DoutputDirectory=/build/lib

FROM docker.io/apache/hive:4.0.0

COPY --from=maven /build/lib/*.jar /opt/hive/lib
COPY --chmod=644 hdfs/hive-site.xml /opt/hive/conf/
COPY --chmod=644 hdfs/core-site.xml /opt/hive/conf/
COPY --chmod=644 hdfs/core-site.xml /opt/hadoop/etc/hadoop/
