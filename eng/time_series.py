import bokeh.models as bom
import bokeh.palettes as bpa
import bokeh.plotting as bpl
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import *


@dataclass
class HeatMapHover:
    hours: pd.DataFrame
    time_series: pd.DataFrame


@dataclass
class TimeSeriesHeatMap:
    counts_by_day: pd.DataFrame
    hover: Optional[HeatMapHover] = None

    def summarize(self, selected, width=800, height=800):
        time_series = np.vstack([
            np.hstack([
                np.zeros((24,)) if np.isnan(x).any() else x
                for x in list(row)
            ])
            for row in self.counts_by_day.iloc[selected].itertuples(index=False)
        ])

        fig = bpl.figure(
            title=f"Selected agents",
            width=width,
            height=height
        )
        fig.x_range.range_padding = fig.y_range.range_padding = 0
        fig.image(image=[time_series], x=0, y=0, dw=10*24, dh=len(selected), palette=bpa.mpl["Plasma"][4], level="underlay")

        if self.hover is not None:
            xx, yy = np.meshgrid(np.arange(24 * len(segments)), np.arange(len(selected)))
            src = bom.ColumnDataSource({
                "x": xx.ravel(),
                "y": yy.ravel(),
                "label": [
                    f"{self.hover.hours[x]}, {self.counts_by_day.index[selected[y]]}: {str(int(self.hover.time_series[y, x])) if self.hover.time_series[y, x] else '--'}"
                    for x, y in zip(xx.ravel(), yy.ravel())
                ]
            })
            fig.tooltips = [("Label", "@label")]
            fig.circle_dot(x="x", y="y", source=src, size=15, alpha=0, level="overlay")

        return fig