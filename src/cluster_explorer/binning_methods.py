"""
A collection of functions that implement different binning methods.
The binning methods used in this project are:
- Equal width binning: The range of the attribute is divided into equal width intervals.
- Equal frequency binning: The attribute values are divided into intervals such that each interval contains the same number of values.
- Decision tree based binning: A decision tree is trained to predict the class labels based on the attribute values. The decision tree splits are used as the bin boundaries.
- Multiclass optimal binning: An optimal binning algorithm is used to find the best bin boundaries for the attribute values.
"""

import numpy as np
from pandas import DataFrame, Series
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from optbinning import MulticlassOptimalBinning
from .utils import get_optimal_splits
from typing import List, Tuple


def bin_equal_width(df: DataFrame, labels: Series, numeric_attribute: str, conciseness_threshold: float, cluster_number: int=None) -> List[Tuple[float, float]]:
    """
    Perform equal width binning on a numeric attribute.

    The range of the attribute is divided into equal width intervals.

    :param df: DataFrame containing the data.
    :param labels: Labels corresponding to the data.
    :param numeric_attribute: The numeric attribute to be binned.
    :param conciseness_threshold: Threshold for conciseness (not used in this method).
    :param cluster_number: Cluster number (not used in this method).
    :return: A list of tuples representing the bin intervals.
    """
    # A set of possible number of bins
    num_bins = [2, 3, 4, 5, 6]
    bins_total = []
    min_value, max_value = df[numeric_attribute].min(), df[numeric_attribute].max()
    for num in num_bins:
        # Bin width is the range of the attribute divided by the number of bins
        width = (max_value - min_value) / num
        intervals = [(min_value + i * width, min_value + (i + 1) * width) for i in range(num)]
        bins_total.extend(intervals)
    return bins_total


def bin_equal_frequency(df: DataFrame, labels: Series, numeric_attribute: str, conciseness_threshold: float, cluster_number: int=None) -> List[Tuple[float, float]]:
    """
    Perform equal frequency binning on a numeric attribute.

    The attribute values are divided into intervals such that each interval contains the same number of values.

    :param df: DataFrame containing the data.
    :param labels: Labels corresponding to the data.
    :param numeric_attribute: The numeric attribute to be binned.
    :param conciseness_threshold: Threshold for conciseness (not used in this method).
    :param cluster_number: Cluster number (not used in this method).
    :return: A list of tuples representing the bin intervals.
    """
    num_bins = [2, 3, 4, 5, 6]
    bins_total = []
    values = df[numeric_attribute].sort_values()
    n = len(values)
    for num in num_bins:
        # Divide the sorted values into num equal parts, such that each part contains the same number of values
        bin_edges = [values.iloc[int(n * i / num)] for i in range(num)] + [values.iloc[-1]]
        intervals = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
        bins_total.extend(intervals)
    return bins_total


def bin_decision_tree_based(df: DataFrame, labels: Series, numeric_attribute: str, conciseness_threshold: float, cluster_number: int=None) -> List[Tuple[float, float]]:
    """
    Perform decision tree-based binning on a numeric attribute.

    This method uses a decision tree classifier to determine the bin boundaries for the numeric attribute.
    The decision tree is trained to predict the class labels based on the attribute values, and the splits
    of the decision tree are used as the bin boundaries.

    :param df: DataFrame containing the data.
    :param labels: Labels corresponding to the data.
    :param numeric_attribute: The numeric attribute to be binned.
    :param conciseness_threshold: Threshold for conciseness, used to determine the maximum depth of the decision tree as int(1 / conciseness_threshold).
    :param cluster_number: Cluster number (not used in this method).
    :return: A list of tuples representing the bin intervals.
    """
    # Create and train a decision tree classifier
    clf = DecisionTreeClassifier(max_depth=int(1 / conciseness_threshold))
    X = df[[numeric_attribute]]
    y = labels
    clf.fit(X, y)

    # Get the thresholds used for splitting the attribute values. The thresholds are the values at which the attribute is split.
    # The feature != -2 condition is used to filter out leaf nodes.
    thresholds = clf.tree_.threshold[clf.tree_.feature != -2]

    # Determine the optimal splits based on the thresholds, and return the bin intervals
    optimal_splits = get_optimal_splits(df, thresholds, X, y, numeric_attribute)
    optimal_splits = sorted(optimal_splits)
    optimal_splits = [df[numeric_attribute].min()] + optimal_splits + [df[numeric_attribute].max()]
    return [(optimal_splits[i], optimal_splits[i + 1]) for i in range(len(optimal_splits) - 1)]


def bin_decision_tree_reg_based(df, labels, numeric_attribute, conciseness_threshold, cluster_number=None):
    """
    Perform decision tree-based binning on a numeric attribute using a regressor.

    This method uses a decision tree regressor to determine the bin boundaries for the numeric attribute.
    The decision tree is trained to predict the class labels based on the attribute values, and the splits
    of the decision tree are used as the bin boundaries.

    :param df: DataFrame containing the data.
    :param labels: Labels corresponding to the data.
    :param numeric_attribute: The numeric attribute to be binned.
    :param conciseness_threshold: Threshold for conciseness, used to determine the maximum depth of the decision tree as int(1 / conciseness_threshold).
    :param cluster_number: Cluster number (not used in this method).
    :return: A list of tuples representing the bin intervals.
    """
    clf = DecisionTreeRegressor(max_depth=int(1 / conciseness_threshold))
    X = df[[numeric_attribute]]
    y = labels
    clf.fit(X, y)
    thresholds = clf.tree_.threshold[clf.tree_.feature != -2]
    optimal_splits = get_optimal_splits(df, thresholds, X, y, numeric_attribute)
    optimal_splits = sorted(optimal_splits)
    optimal_splits = [df[numeric_attribute].min()] + optimal_splits + [df[numeric_attribute].max()]
    return [(optimal_splits[i], optimal_splits[i + 1]) for i in range(len(optimal_splits) - 1)]


def bin_decision_tree_based_one_vs_all(df: DataFrame, labels: Series, numeric_attribute: str, conciseness_threshold: float, cluster_number: int) -> List[Tuple[float, float]]:
    """
    Perform decision tree-based binning on a numeric attribute using a one-vs-all approach.

    This method uses a decision tree classifier to determine the bin boundaries for the numeric attribute.
    The decision tree is trained to predict whether each data point belongs to the specified cluster or not,
    and the splits of the decision tree are used as the bin boundaries.

    :param df: DataFrame containing the data.
    :param labels: Labels corresponding to the data.
    :param numeric_attribute: The numeric attribute to be binned.
    :param conciseness_threshold: Threshold for conciseness, used to determine the maximum depth of the decision tree as int(1 / conciseness_threshold).
    :param cluster_number: Cluster number to be used for the one-vs-all classification.
    :return: A list of tuples representing the bin intervals.
    """
    clf = DecisionTreeClassifier(max_depth=int(1 / conciseness_threshold))
    X = df[[numeric_attribute]]
    # The main difference between this method and bin_decision_tree_based is that the labels are transformed to a binary classification problem,
    # where the target cluster is labeled as 1 and all other clusters are labeled as 0, while in bin_decision_tree_based, the labels are the cluster numbers.
    y = (labels == cluster_number) * 1
    clf.fit(X, y)
    thresholds = clf.tree_.threshold[clf.tree_.feature != -2]
    optimal_splits = get_optimal_splits(df, thresholds, X, y, numeric_attribute)
    optimal_splits = sorted(optimal_splits)
    optimal_splits = [df[numeric_attribute].min()] + optimal_splits + [df[numeric_attribute].max()]
    return [(optimal_splits[i], optimal_splits[i + 1]) for i in range(len(optimal_splits) - 1)]


def bin_tree_based_regressor(df: DataFrame, labels: Series, numeric_attribute: str, conciseness_threshold: float, cluster_number: int) -> List[Tuple[float, float]]:
    """
    Perform decision tree-based binning on a numeric attribute using a regressor.

    This method uses a decision tree regressor to determine the bin boundaries for the numeric attribute.
    The decision tree is trained to predict whether a sample belongs to the given cluster based on the attribute values, and the splits
    of the decision tree are used as the bin boundaries.

    :param df: DataFrame containing the data.
    :param labels: Labels corresponding to the data.
    :param numeric_attribute: The numeric attribute to be binned.
    :param conciseness_threshold: Threshold for conciseness, used to determine the maximum depth of the decision tree as int(1 / conciseness_threshold).
    :param cluster_number: Cluster number to be used for binning.
    :return: A list of tuples representing the bin intervals.
    """
    clf = DecisionTreeRegressor(max_depth=int(1 / conciseness_threshold))
    X = df[[numeric_attribute]]
    # The main difference from bin_decision_tree_reg_based is that in this case, the labels are binary - whether the data point belongs to the cluster or not,
    # whereas in bin_decision_tree_reg_based, the labels are the cluster numbers.
    y = (labels == cluster_number) * 1
    clf.fit(X, y)
    thresholds = clf.tree_.threshold[clf.tree_.feature != -2]
    optimal_splits = get_optimal_splits(df, thresholds, X, y, numeric_attribute)
    optimal_splits = sorted(optimal_splits)
    optimal_splits = [df[numeric_attribute].min()] + optimal_splits + [df[numeric_attribute].max()]
    return [(optimal_splits[i], optimal_splits[i + 1]) for i in range(len(optimal_splits) - 1)]


def bin_multiclass_optimal1(df, labels, numeric_attribute, conciseness_threshold, cluster_number):
    x = df[numeric_attribute].values
    y = labels
    optb = MulticlassOptimalBinning(name=numeric_attribute, solver="cp")
    optb.fit(x, y)
    bins = list(optb.splits)
    min_value = df[numeric_attribute].min()
    max_value = df[numeric_attribute].max()
    bins.insert(0, min_value)
    bins.append(max_value)
    return [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]


def bin_multiclass_optimal(df: DataFrame, labels: Series, numeric_attribute: str, conciseness_threshold: float, cluster_number: int) -> List[Tuple[float, float]]:
    """
    Perform multiclass optimal binning on a numeric attribute.

    This method uses an optimal binning algorithm from the optbinning library to find the best bin boundaries for the numeric attribute
    based on the class labels.

    :param df: DataFrame containing the data.
    :param labels: Labels corresponding to the data.
    :param numeric_attribute: The numeric attribute to be binned.
    :param conciseness_threshold: Threshold for conciseness (not used in this method).
    :param cluster_number: Cluster number (not used in this method).
    :return: A list of tuples representing the bin intervals.
    """
    x = df[numeric_attribute].values
    y = labels

    # Initialize MulticlassOptimalBinning object
    optb = MulticlassOptimalBinning(name=numeric_attribute, solver="cp")
    optb.fit(x, y)

    # Get the optimal split points
    bins = list(optb.splits)

    # Add minimum and maximum values to create intervals
    min_value = df[numeric_attribute].min()
    max_value = df[numeric_attribute].max()
    bins.insert(0, min_value)
    bins.append(max_value)

    # Create intervals
    intervals = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    return intervals
