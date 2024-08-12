from typing import Any, Iterator, List

import daft
import dagster
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy
import optuna
import pandas
import polars
import pyarrow
import seaborn
import sklearn
import sklearn.decomposition
import sklearn.metrics
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing

from .resources import IcebergResource, OptunaResource
from .utils import persistent_run_id


class TrainConfig(dagster.Config):
    """Model training configuration."""


@daft.udf(return_dtype=daft.DataType.int64())
class ClassifierModel:
    """Classifier of MNIST images."""

    model: sklearn.base.ClassifierMixin

    def __init__(self, model: sklearn.pipeline.Pipeline) -> None:
        self.model = model

    def predict(self, images: daft.Series):
        """Infer with the classifier."""

        images_arr = images.to_arrow().to_numpy()

        classifications = self.model.predict(images_arr)
        return classifications


def get_or_create_experiment(experiment_name: str) -> str:
    """Retrieve the ID of an existing experiment or create a new one."""

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id

    return mlflow.create_experiment(experiment_name)


@dagster.op(out=dagster.DynamicOut(str))
def parallel_worker(
    context: dagster.OpExecutionContext, optuna_resource: OptunaResource
) -> Iterator[dagster.DynamicOutput]:
    """Return an output for parallelization of model training."""

    experiment = mlflow.set_experiment(
        experiment_id=get_or_create_experiment(context.asset_key.to_user_string())
    )

    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=persistent_run_id(context)
    ) as run:
        yield from (
            dagster.DynamicOutput(run.info.run_id, mapping_key=f"0x{iworker:03x}")
            for iworker in range(optuna_resource.n_workers)
        )


@dagster.op()
def take_one(items: List[Any]) -> Any:
    """Return the first element in the sequence."""

    return items[0]


@dagster.op(
    ins={
        "table": dagster.In(polars.DataFrame),
        "mlflow_run_id": dagster.In(str),
    }
)
def train_model(  # noqa
    context: dagster.OpExecutionContext,
    config: TrainConfig,
    iceberg: IcebergResource,
    optuna_resource: OptunaResource,
    table: polars.DataFrame | pyarrow.Table,
    mlflow_run_id: str,
) -> sklearn.pipeline.Pipeline:
    """Train a model to classify the dataset."""

    train_table = table.filter(polars.col("is_train"))
    train_table = train_table.to_arrow()
    validation_table = table.filter(polars.col("is_validation"))
    validation_table = validation_table.to_arrow()

    train_pixels_arr = numpy.vstack(train_table["pixels"].to_numpy())
    train_classes_arr = train_table["class"].to_numpy()

    validation_pixels_arr = numpy.vstack(validation_table["pixels"].to_numpy())
    validation_classes_arr = validation_table["class"].to_numpy()

    experiment = mlflow.set_experiment(
        experiment_id=get_or_create_experiment(context.asset_key.to_user_string())
    )

    mlflow.autolog()

    def loss_fn(trial: optuna.trial.Trial) -> float:
        """Evaluate a trial model training."""

        trial_name = f"trial-{trial.number:03d}"

        context.log.debug(f"Starting `{trial_name}`.")

        trace_params = {
            "trial": trial.number,
            "run_id": persistent_run_id(context),
        }

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=trial_name,
            nested=True,
        ) as run:
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "avg_precision",
                "n_pca_components": trial.suggest_int("n_pca_components", 10, 30),
                "n_neighbors": trial.suggest_int("n_neighbors", 2, 7),
            }

            mlflow.log_params(params)

            model_pca = sklearn.decomposition.PCA(
                copy=False, n_components=params["n_pca_components"]
            )
            model_knc = sklearn.neighbors.KNeighborsClassifier(n_neighbors=params["n_neighbors"])
            model = sklearn.pipeline.make_pipeline(model_pca, model_knc)
            mlflow.trace(model.fit, attributes=trace_params)(train_pixels_arr, train_classes_arr)

            classes_pred = mlflow.trace(model.predict, attributes=trace_params)(
                validation_pixels_arr
            )
            u_classes = table["class"].unique().to_list()

            classes_arr_ohe = (
                sklearn.preprocessing.OneHotEncoder()
                .fit_transform(validation_classes_arr.reshape(-1, 1))
                .toarray()
            )
            classes_pred_ohe = (
                sklearn.preprocessing.OneHotEncoder()
                .fit_transform(classes_pred.reshape(-1, 1))
                .toarray()
            )

            avg_precision = [None for _ in u_classes]

            for iclass in u_classes:
                avg_precision[iclass] = sklearn.metrics.average_precision_score(
                    classes_arr_ohe[:, iclass], classes_pred_ohe[:, iclass]
                )

            score = numpy.mean(avg_precision)

            mlflow.log_metric("score", score)

            trial.set_user_attr("run_id", run.info.run_id)
            trial.set_user_attr("artifacts", run.info.artifact_uri)

        return score

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_id=mlflow_run_id,
    ), optuna_resource.study() as study:
        study.optimize(loss_fn, n_trials=optuna_resource.n_trials)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_score", study.best_value)

        model = mlflow.sklearn.load_model(f"{study.best_trial.user_attrs['artifacts']}/model")

    return model


@dagster.op(ins={"table": dagster.In(polars.DataFrame)})
def analyze_model(
    context: dagster.OpExecutionContext,
    iceberg: IcebergResource,
    model: sklearn.pipeline.Pipeline,
    table: polars.DataFrame | pyarrow.Table,
) -> None:
    """Analyze a model."""

    test_table = table.filter(polars.col("is_test")).to_arrow()

    pixels_arr = numpy.vstack(test_table["pixels"].to_numpy())
    classes_arr = test_table["class"].to_numpy()

    u_classes = table["class"].unique().to_numpy()

    classes_pred = model.predict(pixels_arr)

    classes_arr_ohe = (
        sklearn.preprocessing.OneHotEncoder().fit_transform(classes_arr.reshape(-1, 1)).toarray()
    )
    classes_pred_ohe = (
        sklearn.preprocessing.OneHotEncoder().fit_transform(classes_pred.reshape(-1, 1)).toarray()
    )

    precision, recall = [None for _ in u_classes], [None for _ in u_classes]
    avg_precision = [None for _ in u_classes]

    for iclass in u_classes:
        precision[iclass], recall[iclass], _ = sklearn.metrics.precision_recall_curve(
            classes_arr_ohe[:, iclass], classes_pred_ohe[:, iclass]
        )
        avg_precision[iclass] = sklearn.metrics.average_precision_score(
            classes_arr_ohe[:, iclass], classes_pred_ohe[:, iclass]
        )

    precision = numpy.vstack(precision)
    recall = numpy.vstack(recall)
    avg_precision = numpy.array(avg_precision)

    df_test = pandas.DataFrame.from_dict(
        metrics := {
            "precision": precision[:, 1],
            "recall": recall[:, 1],
            "avg_precision": avg_precision,
        }
    )
    df_test["index"] = list(u_classes)
    df_test = df_test.melt(id_vars="index")

    fig, ax = plt.subplots(figsize=(10, 8))
    seaborn.barplot(x="index", y="value", hue="variable", data=df_test, ax=ax)
    plt.hlines(y=1, xmin=0, xmax=u_classes.size)
    plt.legend(loc="lower right")

    experiment = mlflow.set_experiment(
        experiment_id=get_or_create_experiment(context.asset_key.to_user_string())
    )

    (run,) = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{persistent_run_id(context)}'",
        output_format="list",
    )

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run.info.run_id):
        for key, value in metrics.items():
            for iclass in u_classes:
                mlflow.log_metric(key=f"{key}_{iclass:02d}", value=value[iclass])

        mlflow.log_figure(fig, artifact_file="metrics.png")


@dagster.op(out=dagster.Out(io_manager_key="iceberg_io_manager"))
def test_op_dataframe_output() -> polars.DataFrame:
    """Return a `polars.DataFrame`."""

    return polars.DataFrame(data={"a": [0, 1, 2], "b": [1.2, 3.4, 5.6], "c": ["a", "b", "c"]})


@dagster.op(out=dagster.Out(io_manager_key="iceberg_io_manager"))
def test_op_dataframe_input(df: polars.DataFrame) -> polars.DataFrame:
    """Multiply column `a` times two."""

    return df.with_columns((polars.col("a") * 2).alias("a_doubled"))


@dagster.job()
def test_dataframes() -> None:
    """Execute some operations with dataframes."""

    test_op_dataframe_input(test_op_dataframe_output())
