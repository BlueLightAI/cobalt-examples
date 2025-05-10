from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.sparse as sp


def arrays_equal(
    arr1: Union[np.ndarray, sp.csr_array],
    arr2: Union[np.ndarray, sp.csr_array],
) -> bool:
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        return np.array_equal(arr1, arr2, equal_nan=True)
    if (
        sp.issparse(arr1)
        and sp.issparse(arr2)
        and type(arr1) == type(arr2)
        and arr1.shape == arr2.shape
    ):
        # convert the arrays to a canonical format for comparison
        # for non-CSR arrays this is somewhat inefficient
        # CSR arrays will be the most common
        arr1_: sp.csr_array = arr1.tocsr(copy=True)
        arr2_: sp.csr_array = arr2.tocsr(copy=True)
        arr1_.sum_duplicates()
        arr2_.sum_duplicates()
        return all(
            [
                np.all(arr1_.data == arr2_.data),
                np.all(arr1_.indices == arr2_.indices),
                np.all(arr1_.indptr == arr2_.indptr),
            ]
        )
    return False


def array_dicts_equal(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray]) -> bool:
    if d1.keys() != d2.keys():
        return False
    return all(arrays_equal(d1[k], d2[k]) for k in d1)


def get_sorted_max_row_values_and_indices(
    X: sp.csr_array, i: int, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the values and indices of the k largest nonzero entries in X[i, :].

    If there are fewer than k nonzero entries, only returns the nonzero ones.

    Returns:
        top_values: the largest values in the row, in descending order.
        column_indices: the column indices in X corresponding to top_values.
    """
    row_data = X.data[X.indptr[i] : X.indptr[i + 1]]
    k = min(k, len(row_data))
    top_data_indices = np.argpartition(row_data, -k)[-k:]
    top_values = row_data[top_data_indices]
    top_col_indices = X.indices[top_data_indices + X.indptr[i]]

    sorted_data_indices = np.argsort(top_values)[::-1]
    sorted_top_values = top_values[sorted_data_indices]
    sorted_column_indices = top_col_indices[sorted_data_indices]
    return sorted_top_values, sorted_column_indices


def sum_sparse_array_rows(
    X: sp.csr_array, row_indices: List[np.ndarray]
) -> sp.csr_array:
    """Sum each of a collection of subsets of rows of X.

    For each set of indices in row_indices, sums the corresponding rows of X,
    and concatenates the resulting rows into the returned matrix.

    Equivalent code for a dense matrix would be something like
    np.concatenate([X[idxs, :].sum(keepdims=True) for idxs in row_indices]).
    """
    indices = np.concatenate(row_indices)
    if np.max(indices) >= X.shape[0]:
        # this will segfault otherwise!
        raise ValueError(f"Indices out of bounds for matrix with {X.shape[0]} rows.")
    indptr = np.concatenate(([0], np.cumsum([len(u) for u in row_indices])))
    data = np.ones(indices.shape, dtype=np.bool_)
    # this is a matrix where row i has ones in the columns given by row_indices[i]
    # multiplying it by X on the right sums those rows in X
    multiplier = sp.csr_array(
        (data, indices, indptr), shape=(len(row_indices), X.shape[0])
    )
    return multiplier @ X


def right_pad_array(arr: np.ndarray, length: int, extend_value) -> np.ndarray:
    if len(arr) > length:
        raise ValueError(
            f"array (length {len(arr)}) is longer than padding length {length}"
        )
    if len(arr) == length:
        return arr

    new_arr = np.concatenate(
        [arr, np.full_like(arr, shape=length - len(arr), fill_value=extend_value)]
    )
    return new_arr
