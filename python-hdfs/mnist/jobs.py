from typing import Any

import daft
import dagster
import matplotlib.pyplot as plt
import mlflow
import numpy
import pandas
import pydantic
import seaborn
import sklearn
import sklearn.decomposition
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm

from .resources import IcebergResource


class TrainConfig(dagster.Config):
    """Model training configuration."""

    train_ratio: float = pydantic.Field(
        default=0.7, description="Ratio of rows used for training, the remained is used for testing."
    )
    validation_ratio: float = pydantic.Field(
        default=0.7, description="Ratio of rows used for validation, out of those used for training."
    )
    dataset_identifier: str = pydantic.Field(
        default="dagster.mnist_view",
        description=(
            "Full identifier of the table in the form `<schema>.<table>` of where to "
            "load the data from."
        ),
    )
    n_pca_components: int = pydantic.Field(default=10, description="Number of PCA components.")


class AnalyzeConfig(dagster.Config):
    """Model analysis configuration."""

    dataset_identifier: str = pydantic.Field(
        default="dagster.mnist_view",
        description=(
            "Full identifier of the table in the form `<schema>.<table>` of where to "
            "load the data from."
        ),
    )


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


@dagster.op()
def train_model(
    context: dagster.OpExecutionContext, config: TrainConfig, iceberg: IcebergResource, table: Any
) -> sklearn.pipeline.Pipeline:
    """Train a model to classify the dataset."""

    iceberg_table = iceberg.catalog.load_table(config.dataset_identifier)
    table = iceberg_table.scan().to_arrow()

    pixels_arr = numpy.vstack(table["pixels"].to_numpy())
    classes_arr = table["class"].to_numpy()

    model_pca = sklearn.decomposition.PCA(n_components=config.n_pca_components)

    model_svc = sklearn.svm.SVC()
    model = sklearn.pipeline.make_pipeline(model_pca, model_svc)

    if (run_id := context.run.parent_run_id) is not None:
        mlflow.set_experiment(experiment_name=run_id)
    else:
        mlflow.set_experiment(experiment_name=context.run_id)

    mlflow.autolog()

    with mlflow.start_run():
        model.fit(pixels_arr, classes_arr)

    return model


@dagster.op()
def analyze_model(
    context: dagster.OpExecutionContext,
    model: sklearn.pipeline.Pipeline,
    config: AnalyzeConfig,
    iceberg: IcebergResource,
) -> None:
    """Analyze a model."""

    iceberg_table = iceberg.catalog.load_table(config.dataset_identifier)
    table = iceberg_table.scan().to_arrow()

    pixels_arr = numpy.vstack(table["pixels"].to_numpy())
    classes_arr = table["class"].to_numpy()

    u_classes = numpy.unique(classes_arr)

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

    if (run_id := context.run.parent_run_id) is not None:
        mlflow.set_experiment(experiment_name=run_id)
    else:
        mlflow.set_experiment(experiment_name=context.run_id)

    with mlflow.start_run():
        for key, value in metrics.items():
            for iclass in u_classes:
                mlflow.log_metric(key=f"{key}_{iclass:02d}", value=value[iclass])

        mlflow.log_figure(fig, artifact_file="metrics.png")
