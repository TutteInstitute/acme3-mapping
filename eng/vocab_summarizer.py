from dataclasses import dataclass
import enum
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import thisnotthat as tnt
from typing import *


def mean_rows_sparse(ar: ArrayLike) -> np.ndarray:
    return np.array(np.mean(ar, axis=0)).squeeze()


class Aspect(enum.Enum):
    important = enum.auto()
    noise = enum.auto()


@dataclass
class VocabularySummarizer:
    top: int
    threshold_presence: float
    vocabulary: Mapping[int, str]
    vectors_raw: ArrayLike
    vectors_iwt: ArrayLike
    aspect: Aspect

    def summarize(self, selected: Sequence[int]) -> pd.DataFrame:
        sum_iwt = mean_rows_sparse(self.vectors_iwt[selected, :])
        presence = mean_rows_sparse((self.vectors_raw[selected, :] > 0).astype(int))
        i_important = np.asarray(np.argsort(-sum_iwt)[:self.top])

        match self.aspect:
            case Aspect.important:
                return pd.DataFrame(
                    [
                        (
                            self.vocabulary[i],
                            sum_iwt[i],
                            100.0 * presence[i]
                        )
                        for i in i_important
                        if presence[i] > self.threshold_presence
                    ],
                    columns=["Token", "Mean bits", "% present"],
                    index=pd.Index(
                        i_important[presence[i_important] > self.threshold_presence],
                        name=f"{len(selected)} command lines"
                    )
                )

            case Aspect.noise:
                tokens_noisy = [
                    vocabulary[i]
                    for i in i_important[presence[i_important] > 0]
                    if presence[i] <= self.threshold_presence
                ]
                return pd.DataFrame(
                    data=[(t,) for t in tokens_noisy[:self.top]],
                    columns=["unimportant tokens"],
                    index=pd.Index(
                        i_important[presence[i_important] > 0][:self.top],
                        name=f"{len(tokens_noisy)} low-presence tokens"
                    )
                )