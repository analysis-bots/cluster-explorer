import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import ast
from anchor import anchor_tabular
import numpy as np
from AnalyzeItemsets import Analyze
import xgboost as xgb


def get_real_value_type(variable: str):
    # Check for boolean values
    if variable.lower() == "true":
        return True
    if variable.lower() == "false":
        return False

    # Check for numeric values
    try:
        num_value = ast.literal_eval(variable)
        if isinstance(num_value, (int, float, complex)):
            return num_value
    except (ValueError, SyntaxError):
        pass

def convert_to_rules(explantion):
    exp = []

    for i in range(len(explantion)):
        split_rule = explantion[i].split()
        if len(split_rule) == 3:
            exp.append([split_rule[0],split_rule[1],get_real_value_type(split_rule[2])])
        else:
            exp.append([split_rule[2], '>', get_real_value_type(split_rule[0])])
            exp.append([split_rule[2], '<=', get_real_value_type(split_rule[4])])

    for i in range(len(exp) - 1):
        exp.insert((2 * i) + 1, ['and'])
    return exp

def analyze_rules(dict, original_data):
    clusters = list(original_data['Cluster'].unique())
    total_df = pd.DataFrame(columns=["coverage", "separation_err", "conciseness", "Cluster"])
    for cluster in dict.keys():
        rules = dict[cluster]
        analyze = Analyze()
        other_cluster = [i for i in clusters if i != cluster]
        explanation_candidates = analyze.analyze_explanation(original_data, rules, int(cluster), other_cluster)
        explanation_candidates['Cluster'] = cluster
        total_df = pd.concat([total_df, explanation_candidates])
    total_df = total_df.sort_values(by=['Cluster'])
    return total_df.reset_index(names=['rule'])

def analyze_anchor_rules(dataset,labels, max_length, coverage_threshold):

    dataset.columns = [x.replace(' ', '_') for x in dataset.columns]
    uniqe_class = list(np.unique(labels))
    model = xgb.XGBClassifier()
    model.fit(dataset, labels)
    dict = {}
    for cluster_number in uniqe_class:
        for max_size in range(1,max_length):
            idxes = dataset.index[labels == cluster_number].tolist()

            local_explanations = []
            explainer = anchor_tabular.AnchorTabularExplainer(uniqe_class,dataset.columns,dataset.to_numpy())
            for i in idxes[:20]:
                explanation = explainer.explain_instance(np.array(dataset.iloc[[i], :]), model.predict, threshold=coverage_threshold, max_anchor_size=max_size)
                local_explanations.append(explanation)

            # Aggregate and analyze rules from local explanations for global insights
            # Calculate rule frequency
            rule_frequency = {}
            explanation_lst = []

            for explanation in local_explanations:
                if len(explanation.names()) > 0:
                    rule = ' '.join(explanation.names())
                    if rule in rule_frequency:
                        rule_frequency[rule] += 1
                    else:
                        rule_frequency[rule] = 1
                        explanation_lst.append(explanation.names())

            # Sort rules by frequency
            sorted_rules = sorted(rule_frequency.items(), key=lambda x: x[1], reverse=True)

            if sorted_rules:
                print(f"{cluster_number}: {sorted_rules[0][0]} ")
                index = list(rule_frequency).index(sorted_rules[0][0])
                rule = explanation_lst[index]
                if dict.get(cluster_number) is None:
                    dict[cluster_number] = [convert_to_rules(rule)]
                else:
                    dict[cluster_number].append(convert_to_rules(rule))
    data = dataset.copy()
    data['Cluster'] = labels
    return analyze_rules(dict, data)
