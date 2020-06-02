# -*- coding: utf-8 -*-
# Internal helper functions (not exported!)
def dup_idx(n):
    """
    Constructs index vector for transforming a vech vector
    into a vec vector to create an n*n symmetric matrix
    from the vech vector.
    tensor.index_select(0, idx).view(3,3)
    :param n: size of the resulting square matrix
    :return: array containing the indices
    """
    idx = []
    for row in range(n):
        for col in range(n):
            if row == col:
                idx.append(int(row * (2 * n - row + 1) / 2))
            if row < col:
                idx.append(int(row * (2 * n - row + 1) / 2) + col - row)
            if row > col:
                idx.append(int(col * (2 * n - col + 1) / 2) + row - col)
    return idx
