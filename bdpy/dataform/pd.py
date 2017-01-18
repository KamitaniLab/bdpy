"""
Utilities for Pandas

This file is a part of BdPy
"""


__all__ = ['convert_dataframe']


import pandas as pd


def convert_dataframe(lst):
    """
    Converts `lst` to pandas dataframe

    Parameters
    ----------
    lst : list of dicts

    Returns
    -------
    df : pandas dataframe
    """

    df_lst = (pd.DataFrame([item.values()], columns=item.keys()) for item in lst)
    df = pd.concat(df_lst, axis=0, ignore_index=True)
    return df
