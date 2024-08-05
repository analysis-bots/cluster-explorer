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


def condition_generator(data, rules):
    condition = np.zeros(len(data), dtype=bool)
    for rule in rules:
        if rule == 'or':
            continue
        temp_condition = np.ones(len(data), dtype=bool)
        for r in rule:
            if len(r) == 3:
                attribute, operator, value = r
                series = data[attribute].values
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

        condition |= temp_condition
    return condition


def support(data, class_number, rules):
    condition = condition_generator(data, rules)
    return (data.loc[condition, 'Cluster'] == class_number).sum()


def separation_err_and_coverage(data, class_number, rules, other_classes, class_size):
    condition = condition_generator(data, rules)
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
