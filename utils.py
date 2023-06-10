import pandas as pd


def open_xlsx(path):
    df = pd.read_excel(path, index_col=0)
    return df