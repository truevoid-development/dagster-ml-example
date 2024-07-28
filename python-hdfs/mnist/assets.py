import contextlib
from typing import List

import dagster
import pyarrow
import pydantic
import pyiceberg.exceptions
from pyiceberg.io.pyarrow import schema_to_pyarrow
from sklearn.datasets import fetch_openml

from .resources import IcebergResource, TrinoResource


@dagster.asset()
def mnist_dataset(context: dagster.AssetExecutionContext, iceberg: IcebergResource) -> str:
    """Load MNIST to the database."""

    fashion_mnist = fetch_openml(name="Fashion-MNIST", data_home="sklearn_cache")
    data = fashion_mnist["data"].to_numpy()
    target = fashion_mnist["target"].to_numpy()

    from pyiceberg.schema import Schema
    from pyiceberg.types import IntegerType, ListType, NestedField

    schema = Schema(
        NestedField(
            required=True,
            field_id=1,
            name="pixels",
            field_type=ListType(element_id=0, element_required=True, element=IntegerType()),
        ),
        NestedField(
            required=True,
            field_id=2,
            name="class",
            field_type=IntegerType(),
        ),
    )

    schema_name = "mnist"
    table_name = "fashion"
    full_table_name = f"{schema_name}.{table_name}"

    with contextlib.suppress(pyiceberg.exceptions.NamespaceAlreadyExistsError):
        iceberg.catalog.create_namespace(schema_name, properties={"location": iceberg.location})

    with contextlib.suppress(pyiceberg.exceptions.TableAlreadyExistsError):
        iceberg.catalog.create_table(full_table_name, schema=schema)

    iceberg_table = iceberg.catalog.load_table(full_table_name)

    arrow_table = pyarrow.Table.from_pylist(
        mapping=[{"pixels": data[irow], "class": int(target[irow])} for irow in range(target.size)],
        schema=schema_to_pyarrow(iceberg_table.schema()),
    )

    iceberg_table.overwrite(arrow_table)

    return full_table_name


class MnistViewConfig(dagster.Config):
    """Configures the created view."""

    classes: List[int] = pydantic.Field(
        default_factory=lambda: list(range(10)),
        description="Which classes are included in the view.",
    )


@dagster.asset()
def mnist_view(
    context: dagster.AssetExecutionContext,
    config: MnistViewConfig,
    trino: TrinoResource,
    mnist_dataset: str,
) -> None:
    """Create a view on the full dataset."""

    sql = (
        "CREATE OR REPLACE VIEW fashion_view AS "
        f"(SELECT * FROM {mnist_dataset} WHERE class IN ({', '.join(map(str, config.classes))}))"
    )

    context.log.debug(sql)

    with trino.connection as session:
        session.cursor().execute(sql)
