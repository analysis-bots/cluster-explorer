import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from optbinning import MulticlassOptimalBinning
from utils import get_optimal_splits


def bin_equal_width(df, labels, numeric_attribute, conciseness_threshold, cluster_number=None):
    num_bins = [2, 3, 4, 5, 6]
    bins_total = []
    min_value, max_value = df[numeric_attribute].min(), df[numeric_attribute].max()
    for num in num_bins:
        width = (max_value - min_value) / num
        intervals = [(min_value + i * width, min_value + (i + 1) * width) for i in range(num)]
        bins_total.extend(intervals)
    return bins_total


def bin_equal_frequency(df, labels, numeric_attribute, conciseness_threshold, cluster_number=None):
    num_bins = [2, 3, 4, 5, 6]
    bins_total = []
    values = df[numeric_attribute].sort_values()
    n = len(values)
    for num in num_bins:
        bin_edges = [values.iloc[int(n * i / num)] for i in range(num)] + [values.iloc[-1]]
        intervals = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
        bins_total.extend(intervals)
    return bins_total


def bin_decision_tree_based(df, labels, numeric_attribute, conciseness_threshold, cluster_number=None):
    clf = DecisionTreeClassifier(max_depth=int(1 / conciseness_threshold))
    X = df[[numeric_attribute]]
    y = labels
    clf.fit(X, y)
    thresholds = clf.tree_.threshold[clf.tree_.feature != -2]
    optimal_splits = get_optimal_splits(df, thresholds, X, y, numeric_attribute)
    optimal_splits = sorted(optimal_splits)
    optimal_splits = [df[numeric_attribute].min()] + optimal_splits + [df[numeric_attribute].max()]
    return [(optimal_splits[i], optimal_splits[i + 1]) for i in range(len(optimal_splits) - 1)]


def bin_decision_tree_reg_based(df, labels, numeric_attribute, conciseness_threshold, cluster_number=None):
    clf = DecisionTreeRegressor(max_depth=int(1 / conciseness_threshold))
    X = df[[numeric_attribute]]
    y = labels
    clf.fit(X, y)
    thresholds = clf.tree_.threshold[clf.tree_.feature != -2]
    optimal_splits = get_optimal_splits(df, thresholds, X, y, numeric_attribute)
    optimal_splits = sorted(optimal_splits)
    optimal_splits = [df[numeric_attribute].min()] + optimal_splits + [df[numeric_attribute].max()]
    return [(optimal_splits[i], optimal_splits[i + 1]) for i in range(len(optimal_splits) - 1)]


def bin_decision_tree_based_one_vs_all(df, labels, numeric_attribute, conciseness_threshold, cluster_number):
    clf = DecisionTreeClassifier(max_depth=int(1 / conciseness_threshold))
    X = df[[numeric_attribute]]
    y = (labels == cluster_number) * 1
    clf.fit(X, y)
    thresholds = clf.tree_.threshold[clf.tree_.feature != -2]
    optimal_splits = get_optimal_splits(df, thresholds, X, y, numeric_attribute)
    optimal_splits = sorted(optimal_splits)
    optimal_splits = [df[numeric_attribute].min()] + optimal_splits + [df[numeric_attribute].max()]
    return [(optimal_splits[i], optimal_splits[i + 1]) for i in range(len(optimal_splits) - 1)]


def bin_tree_based_regressor(df, labels, numeric_attribute, conciseness_threshold, cluster_number):
    clf = DecisionTreeRegressor(max_depth=int(1 / conciseness_threshold))
    X = df[[numeric_attribute]]
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


def bin_multiclass_optimal(df, labels, numeric_attribute, conciseness_threshold, cluster_number):
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
