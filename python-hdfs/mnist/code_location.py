import dagster

from .assets import mnist_dataset, mnist_view, train_mnist_model
from .resources import DagsterResources

defs = dagster.Definitions(
    assets=[mnist_dataset, mnist_view, train_mnist_model],
    resources=DagsterResources.to_resources(),
)
