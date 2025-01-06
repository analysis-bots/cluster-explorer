import math
import pandas as pd
import numpy as np


def conciseness(rules):
    attributes = set()
    for rule in rules:
        for condition in rule:
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


def condition_generator(data, rules, mode='conjunction'):
    condition = np.zeros(len(data), dtype=bool)
    for rule in rules:
        if rule == 'or':
            continue
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


        condition |= temp_condition
    return condition


def support(data, class_number, rules):
    condition = condition_generator(data, rules)
    return (data.loc[condition, 'Cluster'] == class_number).sum()


def separation_err_and_coverage(data, class_number, rules, other_classes, class_size, mode='conjunction'):
    condition = condition_generator(data, rules, mode)
    filter_data = data[condition]

    rule_support = len(filter_data)
    if rule_support == 0:
        return 1, 0

    # Count the number of points in the filtered data that belong to other classes
    miss_points = filter_data['Cluster'].isin(other_classes).sum()
    ret1 = miss_points / rule_support
    ret2 = 0
    if class_size > 0:
        support = (len(filter_data)) - (int(miss_points))
        ret2 = support / class_size
    return ret1, ret2
