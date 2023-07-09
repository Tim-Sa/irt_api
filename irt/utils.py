from pandas import DataFrame
import pandas as pd


def open_xlsx(path: str) -> DataFrame:
    df = pd.read_excel(path, index_col=0)
    return df


def df_consist_only_of(df: DataFrame, values: set):
    df_values = df.stack().tolist()
    diff = set(df_values).difference(values)
    return len(diff) == 0
