# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from scipy.stats import power_divergence, ttest_1samp, ttest_ind

# # ignore catastrophic cancellation scipy warnings
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="scipy.stats._axis_nan_policy"
)

# Defining this to match the return type of `ttest_ind()`
# Its availability is inconsistent across scipy versions
TtestResult = namedtuple(
    "TtestResult",
    ("statistic", "pvalue"),
)


def power_divergence_of_series(series_a: pd.Series, cmp: pd.Series):
    n = series_a.shape[0]
    N = cmp.shape[0]
    cmp = cmp.astype("category")
    series_a = series_a.astype("category")
    cmp_counts = cmp.value_counts(dropna=False)
    a_counts = series_a.value_counts(dropna=False)
    counts = pd.DataFrame({"group_a": a_counts, "group_b": cmp_counts})
    counts = counts.fillna(0)
    counts = counts[(counts["group_a"] != 0) | (counts["group_b"] != 0)]
    a_list = counts["group_a"]
    cmp_counts = counts["group_b"] * n / N

    with np.errstate(divide="ignore"):
        # ignore division by 0 scipy warnings
        result = power_divergence(a_list, cmp_counts, lambda_="log-likelihood")

    return result


def safe_ttest(
    data1: pd.Series, data2: pd.Series, alternative: str, eps: float = 1e-12
):
    """Runs a t-test on the two provided Series.

    Wraps scipy.stats.ttest_ind() with some basic checks to handle corner cases nicely.
    """
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]

    # if there is no data in either group, test is meaningless
    if len(data1) == 0 or len(data2) == 0:
        return TtestResult(np.nan, np.nan)

    # these values are now guaranteed not to be nan since the denominator
    # (length of the array) is positive
    mean1 = data1.mean(axis=0)
    mean2 = data2.mean(axis=0)
    std1 = np.std(data1)
    std2 = np.std(data2)

    # if both groups have very small variance, we will count the t-score as
    # infinite, unless the means are also very close, in which case the test
    # is inconclusive.
    if std1 <= eps and std2 <= eps:
        if abs(mean1 - mean2) <= eps:
            return TtestResult(np.nan, np.nan)
        ttest_value = -np.inf if mean1 < mean2 else np.inf
        return TtestResult(ttest_value, 0)

    # if only one group has very low variance, treat its mean as a known
    # value and run a 1-sample t-test
    # this covers the case where one sample has only a single data point
    # TODO: there is probably a better way to think about this case
    if std1 <= eps:
        new_alternative = alternative
        if alternative == "less":
            new_alternative = "greater"
        elif alternative == "greater":
            new_alternative = "less"
        res = ttest_1samp(data2, mean1, alternative=new_alternative)
        return TtestResult(-res.statistic, res.pvalue)
    elif std2 <= eps:
        return ttest_1samp(data1, mean2, alternative=alternative)

    return ttest_ind(data1, data2, equal_var=False, alternative=alternative)


def pd_is_any_real_numeric_dtype(arr_or_dtype):
    # pandas < 2.0 doesn't have is_any_real_numeric_dtype
    return (
        is_numeric_dtype(arr_or_dtype)
        and not is_complex_dtype(arr_or_dtype)
        and not is_bool_dtype(arr_or_dtype)
    )
