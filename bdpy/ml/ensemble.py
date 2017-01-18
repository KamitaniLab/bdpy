# coding: utf-8
"""
bdpy.ml.ensemble

Functions related to ensemble learning
"""

from collections import Counter

import numpy as np


__all__ = ['get_majority']


def get_majority(data, axis=0):
    """
    Returns a list of majority elements in each row or column.

    各行，または列で一番個数の多い要素のリストを返す．同数の要素があった場合，要素値の昇順で先に来る方の要素が返される．

    Parameters
    ----------
    data : 2D array
    axis : 0 (row) or 1 (column)

    Returns
    -------
    A list of majority elements
    """

    # 多数決結果を格納するリスト
    majority_list = []
    #
    if axis == 0: # 列ごとの多数決の場合，行列を転置する
        data = np.transpose(data) 
    #    
    # 行ごとに多数決を行う
    for i in range(data.shape[0]):
        target = data[i].tolist()
        c = Counter(target)
        majority = c.most_common(1) # 1番出現回数の多い要素を獲得 -> [(要素, 要素の個数)]
        majority_list.append(majority[0][0]) # 要素を抽出してappend

    return majority_list
