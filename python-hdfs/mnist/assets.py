from typing import Any, List

import dagster
import polars
import pyarrow
import pydantic
from sklearn.datasets import fetch_openml

from mnist.jobs import analyze_model, train_model

from .resources import IcebergResource


@dagster.asset(io_manager_key="iceberg_io_manager")
def mnist_dataset(
    context: dagster.AssetExecutionContext, iceberg: IcebergResource
) -> polars.DataFrame:
    """Load MNIST to the database."""

    fashion_mnist = fetch_openml(name="Fashion-MNIST", data_home="sklearn_cache")
    data = fashion_mnist["data"].to_numpy()
    target = fashion_mnist["target"].to_numpy()

    return polars.DataFrame(
        pyarrow.Table.from_pylist(
            mapping=[
                {"pixels": data[irow], "class": int(target[irow])} for irow in range(target.size)
            ],
        )
    )


class MnistViewConfig(dagster.Config):
    """Configures the created view."""

    classes: List[int] = pydantic.Field(
        default_factory=lambda: list(range(10)),
        description="Which classes are included in the view.",
    )


@dagster.asset(io_manager_key="iceberg_io_manager")
def mnist_view(
    context: dagster.AssetExecutionContext,
    config: MnistViewConfig,
    mnist_dataset: polars.DataFrame,
) -> polars.DataFrame:
    """Create a view on the full dataset."""

    return mnist_dataset.filter(polars.col("class").is_in(config.classes))


@dagster.graph_asset(ins={"mnist_view": dagster.AssetIn("mnist_view")})
def train_mnist_model(mnist_view: Any) -> None:
    """Trains a model on the MNIST fashion dataset."""

    return analyze_model(train_model(mnist_view))
