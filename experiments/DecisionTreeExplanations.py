from enum import unique
from sklearn.tree import _tree, DecisionTreeClassifier
import pandas as pd
import numpy as np
from AnalyzeItemsets import Analyze

def convert_to_rules(explanation):
    rules = []
    for i in range(len(explanation) - 1):
        explanation.insert((2 * i) + 1, ['and'])
    return explanation


def get_class_rules_for_analyze(tree: DecisionTreeClassifier, feature_names: list):
    instance = 0
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    def tree_dfs(node_id=0, current_rule=[], instance=0):
        split_feature = inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED:  # internal node
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            left_rule = current_rule.copy()
            left_rule.append([name, '<=', threshold])
            tree_dfs(inner_tree.children_left[node_id], left_rule, instance)
            right_rule = current_rule.copy()
            right_rule.append([name, '>', threshold])
            tree_dfs(inner_tree.children_right[node_id], right_rule, instance)
        else:  # leaf
            dist = inner_tree.value[node_id][0]
            instance += max(dist)
            dist = dist / dist.sum()
            max_idx = dist.argmax()
            if len(current_rule) == 0:
                rule_string = "ALL"
            else:
                rule_string = convert_to_rules(current_rule)
            if rule_string != "ALL" and max_idx == 1:  # Only capture rules for the positive class
                selected_class = classes[max_idx]
                class_probability = dist[max_idx]
                class_rules = class_rules_dict.get(selected_class, [])
                class_rules.append((rule_string, class_probability, instance))
                class_rules_dict[selected_class] = class_rules

    tree_dfs()
    return class_rules_dict

def analyze_rules(rules_dict, original_data, clusters):
    total_df = pd.DataFrame(columns=["coverage", "separation_err", "conciseness", "Cluster"])
    for cluster in rules_dict.keys():
        rules = rules_dict[cluster]
        analyze = Analyze()
        other_clusters = [i for i in clusters if i >= 0 and i != cluster]
        explanation_candidates = analyze.analyze_explanation(original_data, rules, int(cluster), other_clusters)
        explanation_candidates['Cluster'] = cluster
        total_df = total_df._append(explanation_candidates)
    total_df = total_df.sort_values(by=['Cluster'])
    return total_df.reset_index(names=['rule'])

def analyze_cluster_report(data: pd.DataFrame, clusters, theta_con, min_samples_leaf=50, pruning_level=0.01):
    rules_dict = {}
    max_depth = int(1 / theta_con)

    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        binary_target = (clusters == cluster).astype(int)
        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level, max_depth=max_depth)
        tree.fit(data, binary_target)

        feature_names = data.columns
        class_rule_dict = get_class_rules_for_analyze(tree, feature_names)
        positive_class_rules = class_rule_dict.get(1) # Only get rules for the positive class

        if positive_class_rules:
            for rule, probability in positive_class_rules:
                if rules_dict.get(cluster) is None:
                    rules_dict[cluster] = [rule]
                else:
                    rules_dict[cluster].append(rule)

    df = data.copy()
    df['Cluster'] = clusters
    return analyze_rules(rules_dict, df, unique_clusters)
