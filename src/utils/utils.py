from pandas import DataFrame

from const.constants import RANDOM_STATE, SAMPLE_SIZE


class Utils:
    @classmethod
    def get_data_sample(cls, df: DataFrame, col: str) -> DataFrame:
        # sample_size = min(SAMPLE_SIZE, len(df))
        # return df[col].sample(
        #     n=sample_size,
        #     random_state=RANDOM_STATE)
        return df[col]
