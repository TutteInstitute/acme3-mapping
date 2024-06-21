from dataclasses import dataclass
import pandas as pd
from typing import *


# CLIENT_ID = '30'
# SQLS = Iterator[str]


# class SQLS(Protocol):

#     def __len__(self) -> int: ...
#     def __iter__(self) -> Iterator[str]: ...


# @dataclass
# class QueryEachSegment:
#     segments: pd.DataFrame
#     template: str

#     def __len__(self) -> int:
#         return len(self.segments)

#     def __iter__(self) -> Iterator[int]:
#         for segment, timestamp, span in self.segments.itertuples(index=True):
#             yield self.template.format(segment=segment, lower=timestamp, upper=timestamp + span)


# def processes(segments: pd.DataFrame, client_id: str = CLIENT_ID) -> SQLS:
#     return QueryEachSegment(
#         segments=segments,
#         template=f"""\
#             select
#                 *, {{segment}} as segment
#             from
#                 hogwarts_pb.hbs_base.kragle_process
#             where
#                     CommandLine is not null
#                 and length(trim(CommandLine)) > 0
#                 and timestamp >= timestamp '{{lower}}'
#                 and timestamp < timestamp '{{upper}}'
#                 {f"and hbs_client_id = '{client_id}'" if client_id and client_id != "*" else ""}\
#         """.strip()
#     )


# def frame(template: str, sqls: SQLS) -> SQLS:
#     return [template.format(sql=sql) for sql in sqls]


def process_interesting():
    return """
        select *
        from processs
        inner join process_path using (pid_hash)
        where process_path.ptree not like '%wintap%' and process_path.ptree not like '%amazon-ssm%'
    """