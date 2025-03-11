import math
import pandas as pd
import numpy as np
from typing import Iterable, List, Tuple

from pandas.core.interchange.dataframe_protocol import DataFrame


def conciseness(rules: Iterable[Iterable[Tuple | List]]) -> float | int:
    """
    Compute the conciseness of a set of rules.\n
    This measure is defined as:\n
    .. math:: Conciseness(E_c) = \\frac{1}{|\\{P \quad | \quad P \quad is \quad a \quad predicate \quad in \quad E_c\\}|}\n
    Where:\n
    - :math:`E_c` is a set of explanations for cluster c\n
    In words, the conciseness is the inverse of the number of predicates in the set of rules.\n
    :param rules: A set of rules
    :return: The conciseness of the rules
    """
    attributes = set()
    for rule in rules:
        for condition in rule:
            # All conditions should be of the form (attribute, operator, value)
            if len(condition) == 3:
                attributes.add(condition[0])  # Add the attribute name
    if len(attributes) == 0:
        return 0.01
    return len(attributes)


def condition_writer(data, rule, temp_condition, mode='conjunction'):
    """
    This function writes the condition to filter the data based on the rule provided.

    :param data: The data to be filtered
    :param rule: The rule to be applied
    :param temp_condition: The current condition
    :param mode: The mode of the rule (conjunction or disjunction)

    :return: The updated condition
    """
    attribute, operator, value = rule
    series = data[attribute].values
    if mode == 'conjunction':
        if operator == "==":
            temp_condition &= (series == value)
        elif operator == "<":
            temp_condition &= (series < value)
        elif operator == "<=":
            temp_condition &= (series <= value)
        elif operator == ">":
            temp_condition &= (series > value)
        elif operator == ">=":
            temp_condition &= (series >= value)
    elif mode == 'disjunction':
        if operator == "==":
            temp_condition |= (series == value)
        elif operator == "<":
            temp_condition |= (series < value)
        elif operator == "<=":
            temp_condition |= (series <= value)
        elif operator == ">":
            temp_condition |= (series > value)
        elif operator == ">=":
            temp_condition |= (series >= value)

    return temp_condition


def condition_generator(data: DataFrame, rules: Iterable[Iterable[Tuple | List]], mode: str = 'conjunction') -> np.ndarray:
    """
    Generate a condition based on a set of rules.\n
    :param data: The data the rules are applied to
    :param rules: A set of rules. Example of a rule: [('alcohol', '>', 10), ('and'), ('pH', '<', 3)]
    :param mode: Whether the rules are conjunctions or disjunctions.
    :return: A boolean array of the same length as the data, where True indicates that the data-point satisfies the rules
    and False indicates that the data-point does not satisfy the rules
    """
    # Initialize the condition array as all False
    condition = np.zeros(len(data), dtype=bool)
    for rule in rules:
        if rule == 'or':
            continue
        # Create a temporary condition array of all True or all False based on the mode
        if mode == 'conjunction':
            temp_condition = np.ones(len(data), dtype=bool)
        elif mode == 'disjunction':
            temp_condition = np.zeros(len(data), dtype=bool)
        and_flag = False
        for r in rule:
            if len(r) == 3 and and_flag:
                inner_condition = condition_writer(data, r, inner_condition, "conjunction")
            # The usual case. We write the condition based on the rule.
            # This case is always true when the mode is conjunction.
            elif len(r) == 3 and not and_flag:
                temp_condition = condition_writer(data, r, temp_condition, mode)
            # If the rule is a disjunction rule, we need to handle the brackets that are used to group ranges.
            # We raise a flag when we encounter an opening bracket and we keep track of the condition inside the brackets.
            elif r == ["("] and (mode == 'disjunction' or and_flag):
                and_flag = True
                inner_condition = np.ones(len(data), dtype=bool)
            # If we encounter a closing bracket, we apply the condition inside the brackets to the current condition.
            elif r == [")"] and (mode == 'disjunction' or and_flag):
                and_flag = False
                temp_condition |= inner_condition

        # Combine the temporary condition with the main condition
        condition |= temp_condition
    return condition


def support(data, class_number, rules):
    condition = condition_generator(data, rules)
    return (data.loc[condition, 'Cluster'] == class_number).sum()


def separation_err_and_coverage(data: DataFrame, class_number: int, rules: Iterable[Iterable[Tuple | List]],
                                other_classes: List[int], class_size: int, mode: str='conjunction') -> Tuple[float, float]:
    """
    Compute the separation error and coverage of a set of rules.\n
    The separation error is defined as:\n
    .. math:: SeperationErr(E_c) = (\\frac{1}{|\\{x \\in X | E(x) = True\\}|}) * |\\{x \in X \quad | \quad E_c(x) = True \ \wedge \ CL(x) \in  C - \\{c\\}\\}|\n
    Where:\n
    - :math:`E_c` is a set of explanations for cluster c\n
    - :math:`X` is the data\n
    - :math:`E(x)` is the application of the rules to the data-point x\n
    - :math:`CL` is the cluster label assignment function\n
    - :math:`C` is the set of all cluster labels\n
    - :math:`c` is the cluster label\n
    In words, the separation error is the ratio of points for which the explanation :math:`E_c` is true yet the cluster label is not c.\n
    .. math:: Coverage(E_c) = (\\frac{1}{|\\{x \in X | CL(x) = c\\}|}) * |\\{x \in X \quad | \quad E_c(x) = True \ \wedge \ CL(x) = c\\}|\n
    In words, the coverage is the ratio of points for which the explanation :math:`E_c` is true and the cluster label is c.\n
    :param data: The data to apply the rules to
    :param class_number: The cluster label :math:`c`
    :param rules: A set of rules
    :param other_classes: The set of all cluster labels without :math:`c` - :math:`C - \\{c\\\}`
    :param class_size: The size of the class :math:`|\\{x \in X | CL(x) = c\\}|`
    :param mode: Whether the rules should be applied as conjunctions or disjunctions.
    :return: The separation error and coverage of the rules
    """
    # Generate a condition based on the rules, and apply it to the data
    condition = condition_generator(data, rules, mode=mode)
    filter_data = data[condition]

    # If the rule does not cover any points, return 1 for separation error and 0 for coverage
    rule_support = len(filter_data)
    if rule_support == 0:
        return 1, 0

    # Count the number of points in the filtered data that belong to other classes
    miss_points = filter_data['Cluster'].isin(other_classes).sum()
    # Compute the separation error - by the above formula
    separation_error = miss_points / rule_support
    coverage = 0
    if class_size > 0:
        # Compute the coverage - by the above formula
        support = (len(filter_data)) - (int(miss_points))
        coverage = support / class_size
    return separation_error, coverage
