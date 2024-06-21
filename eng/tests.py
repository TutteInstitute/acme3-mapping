from . import connect_to_spark, path_dataset, prepare_dataset, load_segments, ConsistencyWarning#, name_private_table, PREFIX_PRIVATE_TABLE
from contextlib import contextmanager
import os
import pandas as pd
from pathlib import Path
import pytest
import shutil
from typing import *
from warnings import catch_warnings


DAY = pd.Timedelta(days=1)
HOUR = pd.Timedelta(hours=1)


@pytest.mark.slow
def test_connect_to_spark():
    spark = connect_to_spark(num_cores=4)
    try:
        name = spark.sparkContext.appName
        assert os.environ["JUPYTERHUB_USER"] in name
        assert "cmdline" in name
    finally:
        spark.stop()


@contextmanager
def volatile_dataset(*args, **kwargs) -> Iterator[pd.DataFrame]:
    path = Path(kwargs.pop("path", None) or path_dataset("_volatile_", shared=False))
    try:
        yield prepare_dataset(path, *args, **kwargs)
    finally:
        if os.path.isdir(path):
            shutil.rmtree(Path(path).expanduser())


def test_insufficient_parameters():
    with pytest.raises(ValueError):
        with volatile_dataset():
            pytest.fail("Should not get here")


def test_only_spans():
    with pytest.raises(ValueError):
        with volatile_dataset(spans=DAY):
            pytest.fail("Should not get here")


# def check_dataset(
#     dataset,
#     timestamps: Union[int, Sequence[pd.Timestamp]],
#     spans: Union[Tuple[int, pd.Timedelta], Sequence[pd.Timedelta]],
#     freq: Optional[pd.Timedelta] = None
# ):
#     for ds in [dataset, Dataset(dataset.path)]:
#         if isinstance(timestamps, int):
#             num = cast(int, timestamps)
#             assert len(ds.timestamps) == num
#         else:
#             assert list(ds.timestamps) == list(timestamps)

#         if isinstance(spans, tuple):
#             num, span = cast(tuple, spans)
#             assert len(ds.spans) == num
#             assert all(sp == span for sp in ds.spans)
#         else:
#             assert list(ds.spans) == list(spans)

#         assert dataset.timestamps.freq == freq


class TARGET:
    __year = 2024
    __month = 1
    __day_start = 1
    __num_days = 7

    @classmethod
    def range(klass):
        return range(klass.__day_start, klass.__day_start + klass.__num_days)

    @classmethod
    def date_range(klass):
        return pd.date_range(
            f'{klass.__year:04d}-{klass.__month:02d}-{klass.__day_start:02d}',
            f'{klass.__year:04d}-{klass.__month:02d}-{klass.__day_start + klass.__num_days:02d}',
            freq=DAY,
            inclusive="left"
        )

    @classmethod
    def days(klass):
        return [pd.Timestamp(f'{klass.__year:04d}-{klass.__month:02d}-{day:02d}') for day in klass.range()]

    @classmethod
    def growing_spans(klass, delta: pd.Timedelta):
        return [n * delta for n in klass.range()]


def make_segments(timestamps: Sequence[pd.Timestamp], spans: Sequence[pd.Timedelta]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": timestamps, "span": spans})


def check_segments(
    timestamps_expected: Sequence[pd.Timestamp],
    spans_expected: Sequence[pd.Timedelta],
    segments_actual: pd.DataFrame
) -> None:
    assert pd.DataFrame({"timestamp": timestamps_expected, "span": spans_expected}).equals(
        segments_actual
    )


def test_datetimeindex_no_span():
    with volatile_dataset(timestamps=TARGET.date_range()) as segments:
        check_segments(TARGET.days(), 7 * [DAY], segments)


def test_datetimeindex_singlespan():
    with volatile_dataset(timestamps=TARGET.date_range(), spans=HOUR) as segments:
        check_segments(TARGET.days(), 7 * [HOUR], segments)


def test_datetimeindex_spansvaried():
    with volatile_dataset(timestamps=TARGET.date_range(), spans=TARGET.growing_spans(HOUR)) as segments:
        check_segments(TARGET.days(), TARGET.growing_spans(HOUR), segments)


def test_daysequispaced_no_span():
    with volatile_dataset(timestamps=TARGET.days()) as segments:
        check_segments(TARGET.days(), 7 * [DAY], segments)


def test_daysequispaced_singlespan():
    with volatile_dataset(timestamps=TARGET.days(), spans=HOUR) as segments:
        check_segments(TARGET.days(), 7 * [HOUR], segments)


def test_daysequispaced_spansmultiple():
    with volatile_dataset(timestamps=TARGET.days(), spans=[HOUR for _ in TARGET.range()]) as segments:
        check_segments(TARGET.days(), 7 * [HOUR], segments)


def test_daysequispaced_spansvaried():
    with volatile_dataset(timestamps=TARGET.days(), spans=TARGET.growing_spans(HOUR)) as segments:
        check_segments(TARGET.days(), TARGET.growing_spans(HOUR), segments)


def test_daysvaried():
    subseq = [TARGET.days()[i] for i in [0, 1, 3, 6]]
    with catch_warnings(record=True) as warnings_issued:
        with volatile_dataset(timestamps=subseq) as segments:
            check_segments(subseq[:-1], [DAY, 2 * DAY, 3 * DAY], segments)
    assert warnings_issued
    assert any(w.category is ConsistencyWarning for w in warnings_issued)


def test_days_spans_unmatched():
    with pytest.raises(ValueError):
        with volatile_dataset(timestamps=TARGET.days(), spans=[DAY, DAY]):
            pytest.fail("Should not get here")


def test_expand_user():
    with volatile_dataset(path="~/_test_ds_", timestamps=TARGET.date_range()) as segments:
        assert (Path.home() / "_test_ds_" / "segments.parquet").is_file()


def test_load_segments_idempotent():
    with volatile_dataset(timestamps=TARGET.date_range()) as segments:
        assert load_segments(path_dataset("_volatile_", shared=False)).equals(segments)

# def test_private_table_no_prefix():
#     assert name_private_table("asdf").startswith(PREFIX_PRIVATE_TABLE)


# def test_private_table_idempotent():
#     name_private = name_private_table(name_private_table("asdf"))
#     assert name_private.startswith(PREFIX_PRIVATE_TABLE)
#     name_mangled = name_private[len(PREFIX_PRIVATE_TABLE):]
#     assert PREFIX_PRIVATE_TABLE not in name_mangled
#     assert name_mangled.endswith("asdf")
