import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Generator, Any, Set

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


def convert_itemset_to_rules(itemsets: dict, mode: str = 'conjunction') -> List[List[List]]:
    """
    Convert itemsets into a list of rules.\n

    This function takes a dictionary of itemsets and converts each itemset into a list of rules.
    Each rule is represented as a list of conditions, where each condition is a list containing
    a variable, an operator, and a value. The rules are then returned as a list of lists of lists.

    :param itemsets: A dictionary where keys are itemset identifiers and values are lists of itemsets.
    :param mode: The mode of the rules. If 'conjunction', 'and' is used between conditions. If 'disjunction', 'or' is used between conditions.
    :return: A list of rules, where each rule is a list of conditions.
    """
    rules = set()
    for itemset in itemsets:
        for items in itemsets[itemset]:
            explanation = []
            # First, we convert each interval in the itemset to a list of conditions
            for item in items:
                explanation.extend(convert_interval_to_list(item))
            # With conjunction mode, we need to add 'and' between each condition in the explanation
            if mode == 'conjunction':
                for i in range(len(explanation) - 1):
                    explanation.insert((2 * i) + 1, ['and'])

            # With disjunctions, we need to account for conditions on the same attribute that represent a range.
            # These conditions need to be grouped together with 'and' between them, and 'or' between conditions on different attributes
            # or different ranges on the same attribute.
            elif mode == 'disjunction':
                fixed_explanation = []
                flag = False
                for i in range(len(explanation) - 1):
                    current_cond = explanation[i]
                    next_cond = explanation[i + 1]
                    # If the current condition and the next condition are on the same attribute and represent a range, we group them together
                    if current_cond[0] == next_cond[0] and not flag and (current_cond[1] == ">=" or current_cond[1] == "<="):
                        fixed_explanation.append(['('])
                        fixed_explanation.append(current_cond)
                        fixed_explanation.append(['and'])
                        fixed_explanation.append(next_cond)
                        fixed_explanation.append([')'])
                        flag = True
                    else:
                        fixed_explanation.append(['or'])
                        if flag:
                            flag = False
                        else:
                            fixed_explanation.append(current_cond)

                explanation = fixed_explanation
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
            # r[2] is the value of the condition. We strip any leading or trailing whitespace, then try to convert it to a number.
            r[2] = r[2].strip()
            if r[2].startswith("np.float64("):
                r[2] = np.float64(r[2].replace("np.float64(", "").replace(")", ""))
            elif r[2].startswith("np.int64("):
                r[2] = np.int64(r[2].replace("np.int64(", "").replace(")", ""))
            else:
                try:
                    r[2] = int(r[2])
                except ValueError:
                    try:
                        r[2] = float(r[2])
                    except ValueError:
                        pass
            r[1] = r[1].replace(" ", "")
        rule[idx] = r

    return rule


def merge_ranges(ranges: List[Tuple[number, number]]) -> set[Tuple[number, number]]:
    """
    Merge overlapping ranges.

    This function takes a list of ranges and merges any overlapping ranges.
    If two ranges overlap, they are merged into a single range that spans both ranges.

    :param ranges: A list of ranges to be merged.
    :return: A list of merged ranges.
    """
    ranges = [r for r in ranges if len(r) == 2]
    ranges.sort()
    merged_ranges = set(ranges)
    # We use a separate set to keep track of ranges that need to be removed or added, to avoid modifying the set while iterating over it.
    to_remove = set()
    to_add = set()
    # For every range, check if it overlaps with any other range. If it does, merge the ranges.
    # We use a set to avoid duplicate ranges.
    for r in ranges:
        for other_r in ranges:
            if r != other_r:
                if r[1] >= other_r[0] and r[0] <= other_r[1]:
                    first_start, first_end = r
                    second_start, second_end = other_r
                    if first_start == -np.inf:
                        first_start = second_start
                    if first_end == np.inf:
                        first_end = second_end
                    if second_start == -np.inf:
                        second_start = first_start
                    if second_end == np.inf:
                        second_end = first_end
                    # Add the merged range and remove the original ranges
                    to_remove.add(r)
                    to_remove.add(other_r)
                    to_add.add((min(first_start, second_start), max(first_end, second_end)))

    merged_ranges -= to_remove
    merged_ranges |= to_add

    return merged_ranges



def rule_to_human_readable_conjunction(rule: List[List[List]], categorical_mapping: dict):
    """
    Convert a conjunctive rule to a human-readable string.

    This function takes a rule represented as a list of conditions and converts it into a human-readable string.
    Each condition is represented as a list containing a variable, an operator, and a value.

    :param rule: A list of conditions representing the rule.
    :param categorical_mapping: A dictionary mapping one-hot encoded categorical variables to their original names.
    :return: A human-readable string representing the rule.
    """
    attr_ranges = {}
    relation = ""
    # Go over each condition in the rule and extract ranges for each attribute
    for condition in rule:
        if len(condition) == 1:
            relation = condition[0]
        elif len(condition) == 3:
            attr, op, val = condition
            if attr not in attr_ranges.keys():
                attr_ranges[attr] = []
            if op == '==':
                attr_ranges[attr] = [(val, val)]
            elif op == '>=':
                attr_ranges[attr].append((val, np.inf))
            elif op == '<=':
                attr_ranges[attr].append((-np.inf, val))

    # Merge overlapping ranges for each attribute
    attr_ranges = {k: merge_ranges(v) for k, v in attr_ranges.items()}
    human_readable_rule = ""

    # Convert the ranges to a human-readable string
    for attr, ranges in attr_ranges.items():
        for r in ranges:
            start, end = r
            # If the start and end of the range are the same, we have an equality condition.
            if start == end:
                # We need to check if the attribute is categorical and if it is, we add a condition based on the original attribute name,
                # before the one-hot encoding.
                if attr in categorical_mapping:
                    attr_original = categorical_mapping[attr]
                    attr_split = attr.split("_", 1)
                    attr_value = attr_split[1]
                    if start == 0:
                        human_readable_rule += f"{attr_original} != {attr_value} "
                    elif start == 1:
                        human_readable_rule += f"{attr_original} == {attr_value} "
                else:
                    human_readable_rule += f"{attr} == {start} "
            else:
                # If the start is -inf, we have a less than or equal condition. If the end is inf, we have a greater than or equal condition.
                # Otherwise, we have a range condition.
                if start == -np.inf:
                    human_readable_rule += f"{attr} <= {end} "
                elif end == np.inf:
                    human_readable_rule += f"{attr} >= {start} "
                else:
                    human_readable_rule += f"{r[0]} <= {attr} <= {r[1]} "
            human_readable_rule += relation.upper() + " "

    return human_readable_rule



def rule_to_human_readable(rule: List[List[List]], categorical_mapping: dict, mode: str = 'conjunction') -> str:
    """
    Convert a rule to a human-readable string.

    This function takes a rule represented as a list of conditions and converts it into a human-readable string.
    Each condition is represented as a list containing a variable, an operator, and a value.

    :param rule: A list of conditions representing the rule.
    :param categorical_mapping: A dictionary mapping one-hot encoded categorical variables to their original names.
    :param mode: Whether the rule is conjunctive or disjunctive.
    :return: A human-readable string representing the rule.
    """
    if mode == "conjunction":
        human_readable_rule = rule_to_human_readable_conjunction(rule, categorical_mapping)

    # We return up to -4 or -5 to cut off the last "and" or "or" from the string
    if human_readable_rule.endswith("AND "):
        return human_readable_rule[:-5]
    elif human_readable_rule.endswith("OR "):
        return human_readable_rule[:-4]
    else:
        return human_readable_rule



