from contextlib import contextmanager, closing
import duckdb
# from hogwarts.pyspark.sql.session import HogwartsSparkSession
from IPython import get_ipython
from IPython.display import HTML
import inspect as ins
import json
import os
import numba
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
# from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
# import pyspark.sql.functions as F
# import pyspark.sql.types as T
import scipy.sparse as sp
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
# import trino
from typing import *
import warnings
import zstandard as zstd

from . import sql
from . import bpe


_sql = sql
NL = "\n"
# Tokenizer = Callable[[str], Iterable[str]]
MAX_INT32 = np.iinfo(np.int32).max


# def connect_to_spark(name: str = "", num_cores: int = 128) -> SparkSession:
#     if not name:
#         if not os.environ.get("JUPYTERHUB_USER", ""):
#             raise RuntimeError("Cannot identify current user; set the Spark session name yourself.")
#         name = f"{os.environ['JUPYTERHUB_USER']}-cmdlines-embedding"
#     spark = (
#         HogwartsSparkSession(name)
#         .onRemote()
#         .withDriverMemory(16)
#         .withExecutorMemory(16)
#         .withExecutorCores(4)
#         .withCoresMax(num_cores)
#         .config("spark.driver.maxResultSize", "16g")
#         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
#         .getOrCreate()
#     )
#     return spark


T = TypeVar("T")
CollectorDataFrames = Callable[[Sequence[pd.DataFrame]], T]


class ConsistencyWarning(Warning):
    pass


def path_dataset(name: str):
    path = Path.home() / "share" / "users" / "bjhamel" / "datasets" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _join_segments(path: Path) -> Path:
    return path / "segments.parquet"


def prepare_dataset(
    path: Path,
    timestamps: Optional[Sequence[pd.Timestamp]] = None,
    spans: Union[pd.Timedelta, Sequence[pd.Timedelta], None] = None
) -> pd.DataFrame:
    path_segments = _join_segments(path)
    if path_segments.is_file():
        raise ValueError("This dataset already exists. Destroy it first, or use load_segments to recall its parameters from storage.")

    if timestamps is None:
        raise ValueError(f"Must specify at least the segment timestamps for a new dataset.")
    # Only Pandas DatetimeIndex has freq, and this accepts any sequence of timestamps.
    freq = timestamps.freq if hasattr(timestamps, "freq") else None
    col_timestamps = [pd.Timestamp(x) for x in timestamps]

    if spans is not None:
        # The spans argument may be either a single time delta, or a list of deltas. In the latter
        # case, we must match deltas and timestamps.
        if hasattr(spans, "__len__"):
            if len(spans) != len(timestamps):
                raise ValueError(
                    "Either provide no span, one single span, or exactly as many spans as timestamps."
                    f" [len(spans) == {len(spans)} != {len(timestamps)} == len(timestamps)]"
                )
            col_spans = spans
        else:
            col_spans = [pd.Timedelta(spans) for _ in timestamps]
    else:
        # No span given? Then bank on the freq attribute of the timestamps?
        if freq:
            col_spans = [freq.delta for _ in timestamps]
        else:
            # Let's assume that each segment of the study lasts up to the next timestamp.
            if len(timestamps) < 2:
                raise ValueError("If providing a single timestamp, a corresponding span is also needed.")

            # Are timestamps equally spaced? Then it's safe to leverage each timestamp as the start of one
            # segment span. Otherwise, with differently spaced timestamps, we can't possibly know how long
            # the span to associate to the last timestamp. In this case, let's drop this last timestamp as
            # the open boundary of the last span.
            spans_ns = np.array(np.diff(timestamps)).astype("timedelta64[ns]")
            if np.all(np.diff(spans_ns).astype(int) == 0):
                col_spans = [pd.Timedelta(spans_ns[0]) for _ in timestamps]
            else:
                warnings.warn(
                    (
                        "Since spans were not provided and the timestamps are not equally spaced, "
                        "the timestamp sequence is truncated by one, as the last timestamp is considered "
                        "the open boundary of the time interval."
                    ),
                    ConsistencyWarning,
                    stacklevel=2
                )
                col_timestamps = col_timestamps[:-1]
                col_spans = [pd.Timedelta(span) for span in spans_ns]

    segments = pd.DataFrame({"timestamp": col_timestamps, "span": col_spans})
    segments.to_parquet(path_segments, compression="zstd")
    return segments


def load_segments(path: Path) -> pd.DataFrame:
    return pd.read_parquet(_join_segments(path))


# def processes_all(spark: SparkSession, segments: pd.DataFrame, client_id: str = CLIENT_ID) -> SparkDataFrame:
#     assert len(segments) > 0
#     segments = segments[["timestamp", "span"]]
#     chain = processes(spark, *segments.iloc[0], client_id=client_id).select("*", F.lit(0).alias("segment"))
#     for i, (ts, span) in enumerate(segments.iloc[1:].itertuples(index=False), start=1):
#         chain = chain.unionAll(
#             processes(spark, ts, span, client_id=client_id)
#             .select("*", F.lit(i).alias("segment"))
#         )
#     return chain


# def spark_tokenizer(tokenize: Tokenizer) -> F.UserDefinedFunction:
#     return F.udf(
#         lambda cmdline: tokenize(cmdline) or None,
#         returnType=T.ArrayType(T.StringType())
#     )


def _concat_dataframes(dataframes: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dataframes, ignore_index=True)


# class Trino:

#     @classmethod
#     @contextmanager
#     def connection(cls) -> Iterable[trino.dbapi.Connection]:
#         shell = get_ipython()
#         trino_cell_magic_method = shell.magics_manager.magics.get("cell", {}).get("trino", None)
#         assert trino_cell_magic_method is not None
#         trino_cell_magic = trino_cell_magic_method.__self__
#         with trino.dbapi.connect(
#             host=trino_cell_magic.host,
#             port=trino_cell_magic.port,
#             auth=trino_cell_magic.auth,
#             user=trino_cell_magic.user,
#             catalog=trino_cell_magic.catalog,
#             schema=trino_cell_magic.schema,
#             http_scheme=trino_cell_magic.httpScheme,
#             verify=trino_cell_magic.verify
#         ) as connection:
#             yield connection

#     @classmethod
#     @contextmanager
#     def cursor(cls) -> Iterable[trino.dbapi.Connection]:
#         with cls.connection() as conn, closing(conn.cursor()) as cursor:
#             yield cursor

#     @classmethod
#     def sql(cls, sql: str, columns: List[str] = []) -> pd.DataFrame:
#         with cls.cursor() as cursor:
#             cursor.execute(sql)
#             return pd.DataFrame.from_records(cursor.fetchall(), columns=columns or None)

#     @classmethod
#     def sequence(
#         cls,
#         sqls: _sql.SQLS,
#         columns: List[str] = [],
#         collect: CollectorDataFrames[T] = _concat_dataframes
#     ) -> T:
#         results = []
#         for sql in tqdm(sqls):
#             results.append(cls.sql(sql, columns=columns))
#         return collect(results)

#     @classmethod
#     def parallel(
#         cls,
#         sqls: _sql.SQLS,
#         columns: List[str] = [],
#         collect: CollectorDataFrames[T] = _concat_dataframes,
#         max_workers: int = os.cpu_count()
#     ) -> T:
#         return collect(
#             thread_map(
#                 lambda sql: cls.sql(sql, columns=columns),
#                 sqls,
#                 max_workers=max_workers
#             )
#         )


def prefix_iceberg() -> str:
    return f"hogwarts_users_pb.bjhamel"


def zstd_save(path, x, serialize=pickle.dump):
    with zstd.open(path, 'wb') as file:
        if ins.isfunction(x):
            x(file)
        else:
            serialize(x, file)


def zstd_load(path, deserialize=pickle.load):
    with zstd.open(path, 'rb') as file:
        return deserialize(file)
