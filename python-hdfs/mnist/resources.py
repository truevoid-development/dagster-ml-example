import contextlib
import math
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Sequence, Type, TypeVar

import dagster
import optuna
import polars
import pydantic
import pyiceberg
import pyiceberg.catalog
import pyiceberg.exceptions
import trino
from dagster._core.storage.db_io_manager import (
    DbClient,
    DbIOManager,
    DbTypeHandler,
    TableSlice,
)

from .utils import persistent_run_id


class IcebergResource(dagster.ConfigurableResource):
    """Configures an iceberg catalog."""

    catalog_uri: str = pydantic.Field(
        default=dagster.EnvVar("MNIST_CATALOG_URI"), description="Iceberg catalog URI."
    )
    location: str = pydantic.Field(
        default=dagster.EnvVar("MNIST_LOCATION"),
        description="Location of the schema in the database.",
    )
    catalog_schema: str = pydantic.Field(
        default="public", description="Name of the schema in which tables are created."
    )

    @property
    def catalog(self) -> pyiceberg.catalog.Catalog:
        """Return the iceberg catalog."""

        return pyiceberg.catalog.load_catalog(
            uri=self.catalog_uri.get_value()
            if isinstance(self.catalog_uri, dagster.EnvVar)
            else self.catalog_uri,
            properties={
                "s3": {
                    "endpoint": dagster.EnvVar("MNIST_S3_ENDPOINT"),
                    "access-key-id": dagster.EnvVar("AWS_ACCESS_KEY_ID"),
                    "secret-access-key": dagster.EnvVar("AWS_SECRET_ACCESS_KEY"),
                }
            },
        )

    def create_namespace(self, name: str) -> None:
        """Create a new namespace if it does not exist."""

        with contextlib.suppress(pyiceberg.exceptions.NamespaceAlreadyExistsError):
            self.catalog.create_namespace(name, properties={"location": self.location})


iceberg_resource = IcebergResource()


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
        """Return the `dbapi` connection to Trino."""

        return trino.dbapi.Connection(
            host=self.host,
            port=self.port,
            user=self.user,
            catalog=self.catalog,
            schema=self.catalog_schema,
            **self.additional_parameters,
        )


trino_resource = TrinoResource()


class IcebergIOManager(dagster.ConfigurableIOManagerFactory):
    """Stores outputs in Iceberg tables."""

    iceberg: IcebergResource

    @staticmethod
    @abstractmethod
    def type_handlers() -> Sequence[DbTypeHandler]:
        """Return a sequence of types handled by this IO manager."""

    @staticmethod
    def default_load_type() -> Type | None:
        """Return the default load type."""

        return None

    def create_io_manager(self, context) -> DbIOManager:
        """Return an initialized `DbIOManager`."""

        return DbIOManager(
            db_client=IcebergClient(),
            database=self.iceberg.catalog_uri,
            schema=self.iceberg.catalog_schema,
            type_handlers=self.type_handlers(),
            default_load_type=self.default_load_type(),
            io_manager_name="IcebergIOManager",
        )


IcebergConnection = TypeVar("IcebergConnection", bound=IcebergResource)


class IcebergClient(DbClient):
    """Client to read and write data in Iceberg tables."""

    @staticmethod
    def delete_table_slice(
        context: dagster.OutputContext, table_slice: TableSlice, connection: IcebergConnection
    ) -> None:
        """Delete a table slice."""

        if table_slice.partition_dimensions and len(table_slice.partition_dimensions) > 0:
            raise NotImplementedError

        else:
            with contextlib.suppress(pyiceberg.exceptions.NoSuchTableError):
                connection.catalog.drop_table(table_slice.schema)

    @staticmethod
    def ensure_schema_exists(
        context: dagster.OutputContext,
        table_slice: TableSlice,
        connection: IcebergConnection,
    ) -> None:
        """Create a schema if it does not exist."""

        connection.create_namespace(table_slice.schema)

    @staticmethod
    def get_select_statement(table_slice: TableSlice) -> str:
        """Return a select statement for the provided table slice."""

        return ""

    @staticmethod
    @contextlib.contextmanager
    def connect(
        context: dagster.InputContext | dagster.OutputContext, table_slice: TableSlice
    ) -> IcebergConnection:
        """Yield the Iceberg catalog."""

        yield DagsterResources.Iceberg.resource


class IcebergPolarsTypeHandler(DbTypeHandler[polars.DataFrame]):
    """Stores and loads Polars DataFrames in Iceberg."""

    def table_name(
        self, context: dagster.OutputContext | dagster.InputContext, table_slice: TableSlice
    ) -> str:
        """Return the table name for the input or output."""

        return f"{table_slice.schema}.{context.run_id}-{table_slice.table}"

    def handle_output(
        self,
        context: dagster.OutputContext,
        table_slice: TableSlice,
        obj: polars.DataFrame,
        connection: IcebergConnection,
    ):
        """Store the polars DataFrame in Iceberg."""

        obj_arrow = obj.to_arrow()

        with contextlib.suppress(pyiceberg.exceptions.NamespaceAlreadyExistsError):
            connection.create_namespace(table_slice.schema)

        table_name = self.table_name(context, table_slice)

        with contextlib.suppress(pyiceberg.exceptions.NoSuchTableError):
            connection.catalog.drop_table(table_name)

        with contextlib.suppress(pyiceberg.exceptions.TableAlreadyExistsError):
            connection.catalog.create_table(table_name, schema=obj_arrow.schema)

        iceberg_table = connection.catalog.load_table(table_name)
        iceberg_table.overwrite(obj_arrow)

        context.add_output_metadata(
            {
                "row_count": obj.shape[0],
                "dataframe_columns": dagster.MetadataValue.table_schema(
                    dagster.TableSchema(
                        columns=[
                            dagster.TableColumn(name=name, type=str(dtype))
                            for name, dtype in zip(obj.columns, obj.dtypes)
                        ]
                    )
                ),
            }
        )

    def load_input(
        self, context: dagster.InputContext, table_slice: TableSlice, connection: IcebergConnection
    ) -> polars.DataFrame:
        """Load the input as a Polars DataFrame."""

        if table_slice.partition_dimensions and len(context.asset_partition_keys) == 0:
            return polars.DataFrame()

        iceberg_table = connection.catalog.load_table(self.table_name(context, table_slice))

        if table_slice.partition_dimensions and len(table_slice.partition_dimensions) > 0:
            raise NotImplementedError

        else:
            return polars.DataFrame(
                iceberg_table.scan(
                    selected_fields=table_slice.columns if table_slice.columns else ("*",),
                ).to_arrow()
            )

    @property
    def supported_types(self) -> List[Type]:
        """Return the supported types."""

        return [polars.DataFrame]


class IcebergPolarsIOManager(IcebergIOManager):
    """Stores `polars` tables in Iceberg."""

    @classmethod
    def _is_dagster_maintained(cls) -> bool:
        return False

    @staticmethod
    def type_handlers() -> Sequence[DbTypeHandler]:
        """Return the types that this class handles."""

        return [IcebergPolarsTypeHandler()]

    @staticmethod
    def default_load_type() -> Type | None:
        """Return the default type that this class loads."""

        return polars.DataFrame


iceberg_polars_io_manager = IcebergPolarsIOManager(iceberg=IcebergResource(catalog_schema="dagster"))


class OptunaResource(dagster.ConfigurableResource):
    """Optuna-specific configuration."""

    study_name: str | None = pydantic.Field(default=None, description="Name of the study.")
    storage_url: str | None = pydantic.Field(
        default=dagster.EnvVar("OPTUNA_STORAGE_URL").get_value(),
        description="Indicates which database is used to store studies.",
    )
    direction: optuna.study.StudyDirection = pydantic.Field(
        default=optuna.study.StudyDirection.MAXIMIZE,
        description="Indicates whether loss is maximied or minimized.",
    )
    n_total_trials: int = pydantic.Field(
        default=16, description="Total number of trials. The actual number may be higher."
    )
    n_workers: int = pydantic.Field(default=4, description="Number of workers executing trials.")

    @property
    def n_trials(self) -> int:
        """Number of trials per worker."""

        return math.ceil(self.n_total_trials / self.n_workers)

    @contextlib.contextmanager
    def study(self) -> optuna.study.Study:
        """Create a context manager with a new or existing study."""

        yield optuna.create_study(
            storage=self.storage_url,
            study_name=self.study_name,
            direction=self.direction,
            load_if_exists=True,
        )


@dagster.resource
def build_optuna_resource(context: dagster.InitResourceContext) -> OptunaResource:
    """Build an Optuna resource."""

    return OptunaResource(study_name=persistent_run_id(context))


optuna_resource = build_optuna_resource


class DagsterResources(Enum):
    """Resources in the system."""

    Iceberg = "iceberg"
    Trino = "trino"
    IcebergPolarsIOManager = "iceberg_io_manager"
    Optuna = "optuna_resource"

    @property
    def resource(self) -> dagster.ConfigurableResource | dagster.ConfigurableIOManager:
        """Return the corresponding IO manager."""

        if self == self.Iceberg:
            return iceberg_resource

        if self == self.Trino:
            return trino_resource

        if self == self.IcebergPolarsIOManager:
            return iceberg_polars_io_manager

        if self == self.Optuna:
            return optuna_resource

    @classmethod
    def to_resources(cls) -> Dict[str, dagster.ConfigurableResource]:
        """Return resources in the code location."""

        return {k.value: k.resource for k in cls}
