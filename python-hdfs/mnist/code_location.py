import dagster

from .assets import mnist_dataset, mnist_view
from .resources import IcebergResource, TrinoResource

defs = dagster.Definitions(
    assets=[mnist_dataset, mnist_view],
    resources={
        "iceberg": IcebergResource(),
        "trino": TrinoResource(),
    },
)
