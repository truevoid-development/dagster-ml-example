import dagster

from .assets import mnist_dataset, mnist_view, train_mnist_model
from .jobs import test_dataframes
from .resources import DagsterResources

defs = dagster.Definitions(
    jobs=[test_dataframes],
    assets=[mnist_dataset, mnist_view, train_mnist_model],
    resources=DagsterResources.to_resources(),
)
