import fsspec
import IPython

from pyiceberg.catalog import load_catalog

catalog = load_catalog("iceberg", uri="http://nessie:19120/iceberg")
fs = fsspec.filesystem(protocol="s3")

IPython.embed()
