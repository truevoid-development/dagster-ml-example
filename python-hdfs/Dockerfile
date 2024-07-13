FROM docker.io/maven:3 AS maven

COPY pom.xml .

RUN mvn dependency:copy-dependencies -DoutputDirectory=/build/lib

FROM docker.io/curlimages/curl AS curl

ARG TRINO_VERSION=452

WORKDIR /app
RUN curl -Lo trino https://repo1.maven.org/maven2/io/trino/trino-cli/${TRINO_VERSION}/trino-cli-${TRINO_VERSION}-executable.jar

FROM docker.io/python:3.11

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -yq openjdk-17-jre-headless

RUN pip install poetry

WORKDIR /app
COPY pyproject.toml .
COPY poetry.lock .
RUn poetry install --no-root

COPY --from=docker.io/apache/hadoop:3 /opt/hadoop /opt/hadoop
COPY --chmod=644 core-site.xml /opt/hadoop/etc/hadoop/
COPY --from=curl --chmod=755 /app/trino /usr/bin/trino

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin

COPY .bashrc /root/

COPY --from=maven /build/lib/*.jar /opt/hadoop/lib
