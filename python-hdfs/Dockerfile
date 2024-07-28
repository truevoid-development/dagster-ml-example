FROM python:3

ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

RUN --mount=type=cache,target=/root/.cache \
    pip install poetry

WORKDIR /app
COPY pyproject.toml .
COPY poetry.lock .
RUN --mount=type=cache,target=/root/.cache \
    poetry install --no-root

RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

COPY dagster.yaml /tmp/dagster/dagster.yaml
