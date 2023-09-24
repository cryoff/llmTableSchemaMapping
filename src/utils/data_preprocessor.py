from typing import Tuple

import pandas as pd
from pandas import DataFrame

from const.constants import SAMPLE_SIZE, RANDOM_STATE


class Preprocessor:
    """
    Preprocessor class: gets a subsample of data and converts it to string
    """

    @classmethod
    def load_data(cls,
                  template_csv_path: str,
                  source_csv_path: str) -> tuple[DataFrame, DataFrame]:
        # read only first portion of the data and subsample directly, giving some randomness
        template_df: DataFrame = pd.read_csv(template_csv_path, nrows=SAMPLE_SIZE*5)
        template_df = template_df.sample(n=min(SAMPLE_SIZE, len(template_df)), random_state=RANDOM_STATE)

        source_df: DataFrame = pd.read_csv(source_csv_path, nrows=SAMPLE_SIZE*5)
        source_df = source_df.sample(n=min(SAMPLE_SIZE, len(source_df)), random_state=RANDOM_STATE)
        return template_df, source_df
