from typing import Any, Dict

import dagster
import pydantic
import pyiceberg
import pyiceberg.catalog
import trino


class IcebergResource(dagster.ConfigurableResource):
    """Configures an iceberg catalog."""

    catalog_uri: str = pydantic.Field(
        default=dagster.EnvVar("MNIST_CATALOG_URI"), description="Iceberg catalog URI."
    )
    location: str = pydantic.Field(
        default=dagster.EnvVar("MNIST_LOCATION"),
        description="Location of the schema in the database.",
    )

    @property
    def catalog(self) -> pyiceberg.catalog.Catalog:
        """Return the iceberg catalog."""

        return pyiceberg.catalog.load_catalog(
            uri=self.catalog_uri,
            properties={
                "s3": {
                    "endpoint": dagster.EnvVar("MNIST_S3_ENDPOINT"),
                    "access-key-id": dagster.EnvVar("AWS_ACCESS_KEY_ID"),
                    "secret-access-key": dagster.EnvVar("AWS_SECRET_ACCESS_KEY"),
                }
            },
        )


class TrinoResource(dagster.ConfigurableResource):
    """Settings related to Trino."""

    host: str = pydantic.Field(description="Hostname of the trino server.", default="trino")
    port: int = pydantic.Field(description="Port to connect to trino.", default=8080)
    user: str = pydantic.Field(description="Username to connect to trino.", default="admin")
    catalog: str = pydantic.Field(
        description="Name of the catalog used by default.", default="iceberg"
    )
    catalog_schema: str = pydantic.Field(
        description="Name of the schema used by default.", default="mnist"
    )
    additional_parameters: Dict[str, Any] = pydantic.Field(
        description="Additional configuration parameters passed to the connection.",
        default_factory=dict,
    )

    @property
    def connection(self) -> trino.dbapi.Connection:
        """Returns the `dbapi` connection to Trino."""

        return trino.dbapi.Connection(
            host=self.host,
            port=self.port,
            user=self.user,
            catalog=self.catalog,
            schema=self.catalog_schema,
            **self.additional_parameters,
        )
