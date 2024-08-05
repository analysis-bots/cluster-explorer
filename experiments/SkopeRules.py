# Import skope-rules
import six
import sys
import numpy as np
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
from AnalyzeItemsets import Analyze
import pandas as pd

def convert_to_rules(explantion):
    rules = []
    for i in range(len(explantion) - 1):
        explantion.insert((2 * i) + 1, ['and'])
    return explantion


def analyze_rules(dict, original_data):
    clusters = list(original_data['Cluster'].unique())
    total_df = pd.DataFrame(columns=["coverage", "separation_err", "conciseness", "Cluster"])
    for cluster in dict.keys():
        rules = dict[cluster]
        analyze = Analyze()
        #other_cluster = [i for i in clusters if i >= 0 and i != cluster]
        other_cluster = [i for i in clusters if i != cluster]
        explanation_candidates = analyze.analyze_explanation(original_data, rules, int(cluster), other_cluster)
        explanation_candidates['Cluster'] = cluster
        total_df = total_df._append(explanation_candidates)
        # print(explanation_candidates)
    total_df = total_df.sort_values(by=['Cluster'])
    return total_df.reset_index(names=['rule'])

def analyze_skope_rules(dataset, labels, max_depth, precision_min, recall_min):
    i_cluster = 0
    uniqe_class = list(np.unique(labels))
    dict = {}
    for i_cluster in uniqe_class:
        y_train = (labels==i_cluster)*1
        for depth in range(1,max_depth+1):
            skope_rules_clf = SkopeRules(feature_names=dataset.columns, random_state=42, n_estimators=5,
                                           recall_min=recall_min, precision_min=precision_min, max_depth_duplication=0,
                                           max_samples=1., max_depth=depth)
            skope_rules_clf.fit(dataset, y_train)
            rules = skope_rules_clf.rules_
            for r in rules:
                splited = r[0].split(' and ')
                lst = []
                for split_rule in splited:
                    rule = split_rule.split()
                    lst.append([rule[0],rule[1],float(rule[2])])
                if dict.get(i_cluster) is None:
                    dict[i_cluster] = [convert_to_rules(lst)]
                else:
                    dict[i_cluster].append(convert_to_rules(lst))

    data = dataset.copy()
    data['Cluster'] = labels
    return analyze_rules(dict, data)
