import pandas as pd
import ScoreMetrics


class Analyze:
    def __init__(self):
        pass

    def analyze_explanation(self, dtf, rules, cluster_number, other_clusters):
        rules_rec = []
        class_size = (dtf['Cluster'] == cluster_number).sum()
        for r in rules:
            rule = [r]
            separation_err, coverage = ScoreMetrics.separation_err_and_coverage(dtf, cluster_number, rule,
                                                                                other_clusters, class_size)
            rdict = {
                "rule": str(rule),
                "coverage": round(coverage, 2),
                "separation_err": round(separation_err, 2),
                "conciseness": round(1 / ScoreMetrics.conciseness(rule), 2)
            }
            rules_rec.append(rdict)

        rules_df = pd.DataFrame(data=rules_rec, columns=['rule', 'coverage', 'separation_err', 'conciseness'])
        if len(rules_df) >= 1:
            rules_df.set_index('rule', inplace=True)
        return rules_df
