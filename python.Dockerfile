FROM docker.io/apache/hadoop:3 AS hadoop

FROM docker.io/python:3

WORKDIR /app

ARG TRINO_VERSION=452

RUN curl -Lo /usr/bin/trino https://repo1.maven.org/maven2/io/trino/trino-cli/${TRINO_VERSION}/trino-cli-${TRINO_VERSION}-executable.jar \
    && chmod 755 /usr/bin/trino

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -yq openjdk-17-jre-headless

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV HADOOP_HOME=/opt/hadoop

RUN pip install fsspec[hdfs] ipython

COPY --from=hadoop /opt/hadoop/lib/native/libhdfs.so $HADOOP_HOME/lib/native/libhdfs.so
