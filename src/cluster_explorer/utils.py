import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Generator, Any

from numpy import number
from pandas import DataFrame, Series


def is_contained(small: List | Tuple, large: List | Tuple) -> bool:
    """
    Check if a small interval is contained in a large interval.\n

    An interval is considered contained in another interval if:\n
    - The lower bound of the small interval is greater than or equal to the lower bound of the large interval\n
    - The upper bound of the small interval is less than or equal to the upper bound of the large interval\n
    It is expected that the intervals are of the form (lower_bound, upper_bound, ...).\n

    :param small: The small interval, to be checked if it is contained in the large interval
    :param large: The large interval, to be checked if it contains the small interval
    :return: True if the small interval is contained in the large interval, False otherwise
    """
    return small[0] >= large[0] and small[1] <= large[1]


def chunkify(lst: List, chunk_size: int) -> Generator[List, None, None]:
    """
    Splits a list into smaller chunks of a specified size.\n

    This function takes a list and divides it into smaller lists (chunks) of a given size.
    If the list size is not perfectly divisible by the chunk size, the last chunk will contain the remaining elements.

    :param lst: The list to be divided into chunks.
    :param chunk_size: The size of each chunk.
    :return: A generator that yields chunks of the specified size.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def process_chunk(attr: str, chunk: list, intervals: list) -> dict:
    """
    Process a chunk of data to determine which intervals each value belongs to.\n

    This function takes an attribute name, a chunk of values, and a list of intervals.
    It checks each value in the chunk to see if it falls within any of the given intervals.
    The result is a dictionary where the keys are tuples of the attribute name and value,
    and the values are sets of intervals that contain the value.

    :param attr: The name of the attribute being processed.
    :param chunk: A list of values to be checked against the intervals.
    :param intervals: A list of intervals to check the values against.
    :return: A dictionary where keys are (attribute, value) tuples and values are sets of intervals containing the value.
    """
    chunk_result = {}
    for value in chunk:
        key = (attr, value)
        value_set = set()
        # For each interval, if the value is contained in the interval, add the interval to the value set
        for interval in intervals:
            if is_contained((value, value), interval):
                value_set.add(interval)
        chunk_result[key] = value_set
    return chunk_result


def convert_interval_to_list(rule: Tuple[str, number | Tuple[number, number]]) -> List:
    """
    Convert a rule represented as a tuple into a list of conditions.\n

    This function takes a rule in the form of a tuple, where the first element is a variable name
    and the second element is either a single value or a tuple representing an interval. It converts
    this rule into a list of conditions that can be used for filtering or other logical operations.

    :param rule: A tuple where the first element is a variable name (str) and the second element is either
                 a single value (number) or a tuple of two numbers representing an interval.
    :return: A list of conditions representing the rule. If the second element is a single value, the list
             contains one equality condition. If the second element is a tuple, the list contains two inequality conditions over the interval.
    """
    var, value = rule
    if isinstance(value, tuple):
        left_num, right_num = value
        return [[var, '>=', left_num], [var, '<=', right_num]]
    return [[var, '==', value]]


def convert_itemset_to_rules(itemsets: dict) -> List[List[List]]:
    """
    Convert itemsets into a list of rules.\n

    This function takes a dictionary of itemsets and converts each itemset into a list of rules.
    Each rule is represented as a list of conditions, where each condition is a list containing
    a variable, an operator, and a value. The rules are then returned as a list of lists of lists.

    :param itemsets: A dictionary where keys are itemset identifiers and values are lists of itemsets.
    :return: A list of rules, where each rule is a list of conditions.
    """
    rules = set()
    for itemset in itemsets:
        for items in itemsets[itemset]:
            explanation = []
            # Each item in the itemset is a tuple of (attribute, value). We convert this to a list of conditions and add
            # that to the explanation list.
            for item in items:
                explanation.extend(convert_interval_to_list(item))
            # Add a conditional between each condition in the explanation
            for i in range(len(explanation) - 1):
                explanation.insert((2 * i) + 1, ['and'])
            # Add the explanation to the set of rules. We use a set to avoid duplicate rules.
            rules.add(tuple(tuple(e) for e in explanation))
    return [list(list(item) for item in rule) for rule in rules]


def convert_dataframe_to_transactions(df: DataFrame) -> List[List[Tuple[Any, Any]]]:
    """
    Convert a DataFrame into a list of transactions.\n

    This function takes a DataFrame and converts it into a list of transactions,
    where each transaction is represented as a list of tuples. Each tuple contains
    a column name and the corresponding value from the DataFrame.

    :param df: The DataFrame to be converted.
    :return: A list of transactions, where each transaction is a list of (column, value) tuples.
    """
    dict_list = df.to_dict(orient='records')
    return [[(k, v) for k, v in record.items()] for record in dict_list]


def skyline_operator(df: DataFrame) -> DataFrame:
    """
    Compute the skyline points from a DataFrame.\n

    The skyline operator filters out points that are dominated by others.
    A point is considered dominated if there exists another point that is
    better in all dimensions (coverage, separation_err, and conciseness).

    :param df: A DataFrame containing the points to be evaluated.
    :return: A DataFrame containing the skyline points.
    """
    skyline_points = [point for idx, point in df.iterrows() if not is_dominated(point, df)]
    return pd.DataFrame(skyline_points)


def is_dominated(point, df: DataFrame) -> bool:
    """
    Check if a point is dominated by any other point in a DataFrame.\n

    A point is considered dominated if there exists another point that is better in all dimensions.\n
    The dimensions are coverage, separation_err, and conciseness.\n
    :param point: The point to be evaluated. Must have 'coverage', 'separation_err', and 'conciseness' columns / keys.
    :param df: The DataFrame containing the points to compare against.
    :return: True if the point is dominated by any other point in the DataFrame, False otherwise.
    """
    x, y, z = point['coverage'], point['separation_err'], point['conciseness']
    for idx, row in df.iterrows():
        if row['coverage'] >= x and row['separation_err'] <= y and row['conciseness'] >= z and not row.equals(point):
            return True
    return False


def get_optimal_splits(df: DataFrame, tree_splits: np.ndarray, X: DataFrame, y: Series, c: str) -> List:
    """
    Determine the optimal splits for a given feature based on Gini impurity.

    This function evaluates potential splits for a specified feature and returns the optimal splits
    that minimize the Gini impurity. The Gini impurity is calculated for each split, and the splits
    with the lowest impurity are selected.

    :param df: The DataFrame containing the data.
    :param tree_splits: An array of potential split points for the feature.
    :param X: The DataFrame containing the feature data.
    :param y: The Series containing the target labels.
    :param c: The name of the feature to be split.
    :return: A list of optimal split points for the feature.
    """
    def evaluate_split(split):
        bins = X[c] <= split
        bin_counts = np.bincount(y[bins], minlength=2)
        bin_size = bin_counts.sum()
        gini_impurity = 1.0 - np.sum((bin_counts / bin_size) ** 2)
        return gini_impurity

    split_scores = np.array([evaluate_split(split) for split in tree_splits])
    optimal_split_idx = np.argmin(split_scores)
    return sorted(tree_splits[:optimal_split_idx + 1])

def str_rule_to_list(rule: str) -> List:
    """
    Convert a rule string to a list of conditions.

    This function takes a rule string and converts it into a list of conditions.
    Each condition is represented as a list containing a variable, an operator, and a value.

    :param rule: A string representing the rule.
    :return: A list of conditions representing the rule.
    """
    rule = rule.split(", [")
    for idx, r in enumerate(rule):
        r = r.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        r = r.split(",")
        if len(r) == 3:
            r[2] = np.float64(r[2].replace("np.float64(", "").replace(")", ""))
            r[1] = r[1].replace(" ", "")
        rule[idx] = r

    return rule

