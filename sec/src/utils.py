import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def is_contained(small, large):
    return small[0] >= large[0] and small[1] <= large[1]


def chunkify(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def process_chunk(attr, chunk, intervals):
    chunk_result = {}
    for value in chunk:
        key = (attr, value)
        value_set = set()
        for interval in intervals:
            if is_contained((value, value), interval):
                value_set.add(interval)
        chunk_result[key] = value_set
    return chunk_result


def convert_interval_to_list(rule):
    var, value = rule
    if isinstance(value, tuple):
        left_num, right_num = value
        return [[var, '>=', left_num], [var, '<=', right_num]]
    return [[var, '==', value]]


def convert_itemset_to_rules(itemsets):
    rules = set()
    for itemset in itemsets:
        for items in itemsets[itemset]:
            explanation = []
            for item in items:
                explanation.extend(convert_interval_to_list(item))
            for i in range(len(explanation) - 1):
                explanation.insert((2 * i) + 1, ['and'])
            rules.add(tuple(tuple(e) for e in explanation))
    return [list(list(item) for item in rule) for rule in rules]


def convert_dataframe_to_transactions(df):
    dict_list = df.to_dict(orient='records')
    return [[(k, v) for k, v in record.items()] for record in dict_list]


def skyline_operator(df):
    skyline_points = [point for idx, point in df.iterrows() if not is_dominated(point, df)]
    return pd.DataFrame(skyline_points)


def is_dominated(point, df):
    x, y, z = point['coverage'], point['separation_err'], point['conciseness']
    for idx, row in df.iterrows():
        if row['coverage'] >= x and row['separation_err'] <= y and row['conciseness'] >= z and not row.equals(point):
            return True
    return False


def get_optimal_splits(df, tree_splits, X, y, c):
    def evaluate_split(split):
        bins = X[c] <= split
        bin_counts = np.bincount(y[bins], minlength=2)
        bin_size = bin_counts.sum()
        gini_impurity = 1.0 - np.sum((bin_counts / bin_size) ** 2)
        return gini_impurity

    split_scores = np.array([evaluate_split(split) for split in tree_splits])
    optimal_split_idx = np.argmin(split_scores)
    return sorted(tree_splits[:optimal_split_idx + 1])
