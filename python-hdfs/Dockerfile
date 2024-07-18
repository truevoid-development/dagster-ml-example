FROM quay.io/jupyter/scipy-notebook:latest

ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

RUN --mount=type=cache,target=/home/jovyan/.cache,uid=1000,gid=100 \
    pip install poetry

COPY --chown=jovyan:users pyproject.toml .
COPY --chown=jovyan:users poetry.lock .
RUN poetry install
