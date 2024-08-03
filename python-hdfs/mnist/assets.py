from typing import Any, List

import dagster
import polars
import pyarrow
import pydantic
from sklearn.datasets import fetch_openml

from .jobs import analyze_model, parallel_worker, take_one, train_model
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

    random_seed: int = pydantic.Field(default=0, description="Seed for the random number generator.")

    classes: List[int] = pydantic.Field(
        default_factory=lambda: list(range(10)),
        description="Which classes are included in the view.",
    )
    test_ratio: float = pydantic.Field(
        default=0.2, description="Ratio of rows used for test, the remainder is used for training."
    )
    validation_ratio: float = pydantic.Field(
        default=0.2, description="Ratio of rows used for validation, out of those used for training."
    )


@dagster.asset(io_manager_key="iceberg_io_manager")
def mnist_view(
    context: dagster.AssetExecutionContext,
    config: MnistViewConfig,
    mnist_dataset: polars.DataFrame,
) -> polars.DataFrame:
    """Create a view on the full dataset.

    Additionally, classes are optionally selected and rows are assigned to
    train, validation and test.
    """

    total_rows = mnist_dataset.shape[0]

    # Create a random column with a specified seed
    random_expr = polars.arange(0, total_rows, eager=True).shuffle(seed=config.random_seed)

    n_test = config.test_ratio * total_rows
    n_validation = config.validation_ratio * total_rows

    df = (
        mnist_dataset.with_columns([random_expr.alias("random")])
        .with_columns(
            [
                (polars.col("random") < n_test).alias("is_test"),
                (
                    (polars.col("random") >= n_test) & (polars.col("random") < n_test + n_validation)
                ).alias("is_validation"),
                (polars.col("random") >= n_test + n_validation).alias("is_train"),
            ]
        )
        .drop("random")
    ).filter(polars.col("class").is_in(config.classes))

    print(
        df.select(
            polars.col("is_test").sum().alias("test_count"),
            polars.col("is_validation").sum().alias("validation_count"),
            polars.col("is_train").sum().alias("train_count"),
        )
    )

    return df


@dagster.graph_asset(ins={"mnist_view": dagster.AssetIn("mnist_view")})
def train_mnist_model(mnist_view: Any) -> None:
    """Trains a model on the MNIST fashion dataset."""

    workers = parallel_worker()
    return analyze_model(
        take_one(workers.map(lambda run_id: train_model(mnist_view, run_id)).collect()), mnist_view
    )
