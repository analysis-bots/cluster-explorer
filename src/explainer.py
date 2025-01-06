from sklearn.preprocessing import OneHotEncoder
import gFIM
from src.AnalyzeItemsets import Analyze
from utils import *
from src.binning_methods import *


class Explainer:
    def __init__(self, df, labels):
        self.df = df
        self.data = df.copy()
        self.labels = labels
        self.numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        self.very_numerical = [nc for nc in self.numeric_columns if df[nc].nunique() > 6]
        self.taxonomy_tree_cache = {}
        self.coverage_threshold = 0
        self.conciseness_threshold = 0
        self.separation_threshold = 0

    def apply_binning_methods(self, numeric_attribute, binning_methods, cluster_number):
        intervals = set()

        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda method: method(self.df, self.labels, numeric_attribute, self.conciseness_threshold,
                                      cluster_number), binning_methods)
            for result in results:
                intervals.update(result)

        return sorted(intervals, key=lambda x: x[0])

    def process_single_attribute(self, attribute, binning_methods, cluster_number):
        if attribute in self.very_numerical:
            intervals = self.apply_binning_methods(attribute, binning_methods, cluster_number)
            self.taxonomy_tree_cache[attribute] = intervals
            return attribute, intervals
        return attribute, None

    def attribute_to_intervals(self, binning_methods, features, cluster_number):
        taxonomy = {}
        with ThreadPoolExecutor() as executor:
            future_to_attribute = {
                executor.submit(self.process_single_attribute, attribute, binning_methods, cluster_number): attribute
                for attribute in features
            }
            for future in as_completed(future_to_attribute):
                attribute, intervals = future.result()
                if intervals is not None:
                    taxonomy[attribute] = intervals
        return taxonomy

    def build_item_ancestors(self, df, taxonomy):
        item_ancestors_dict = {}

        with ThreadPoolExecutor() as executor:
            futures = []
            for attr, intervals in taxonomy.items():
                unique_values = df[attr].unique()
                chunks = chunkify(unique_values, chunk_size=100000)
                for chunk in chunks:
                    futures.append(executor.submit(process_chunk, attr, chunk, intervals))

            for future in as_completed(futures):
                chunk_result = future.result()
                item_ancestors_dict.update(chunk_result)

        return item_ancestors_dict

    def model_feature_importance(self, df, labels, p=5):
        feature_importance = {}
        feature_names = df.columns
        class_labels = np.unique(labels)

        for class_of_interest in set(labels):
            aggregate_importance = {name: [] for name in feature_names}
            for other_class in class_labels:
                if other_class == class_of_interest:
                    continue

                mask = (labels == class_of_interest) | (labels == other_class)
                x_pair = df[mask]
                y_pair = labels[mask]
                y_pair = np.where(y_pair == class_of_interest, 0, 1)

                tree = DecisionTreeClassifier(random_state=42, max_depth=int(1 / self.conciseness_threshold))
                tree.fit(x_pair, y_pair)

                for name, importance in zip(feature_names, tree.feature_importances_):
                    aggregate_importance[name].append(importance)

            average_importance = {name: np.mean(importance) for name, importance in aggregate_importance.items()}
            feature_importance[class_of_interest] = sorted(average_importance.items(), key=lambda x: x[1],
                                                           reverse=True)[:p]

        return feature_importance

    def one_hot(self):
        categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        numeric_columns = self.df.dtypes[(self.df.dtypes == "float64") | (self.df.dtypes == "int64")].index.tolist()
        ordinals = [nc for nc in numeric_columns if self.df[nc].nunique() < 6]
        categorical_features = categorical_features + ordinals
        for col in categorical_features:
            self.df[col] = self.df[col].astype(str)

        enc = OneHotEncoder(min_frequency=0.1, handle_unknown="infrequent_if_exist", sparse_output=False)

        encoded_categories = enc.fit_transform(self.df.loc[:, categorical_features])
        encoded_df = pd.DataFrame(encoded_categories, columns=enc.get_feature_names_out(categorical_features))
        self.df = pd.concat([self.df.drop(categorical_features, axis=1), encoded_df], axis=1)

    def generate_explanations(self, coverage_threshold=0.6, conciseness_threshold=0.33, separation_threshold=0.5,
                              p_value=0, mode='conjunction'):
        # Initialize the dataframe to store explanations for all clusters
        explanation_for_all_clusters = pd.DataFrame(columns=["coverage", "separation_err", "conciseness", "Cluster"])

        # Set thresholds
        self.coverage_threshold = coverage_threshold
        self.conciseness_threshold = conciseness_threshold
        self.separation_threshold = separation_threshold

        # Define the binning methods
        binning_methods = [
            bin_equal_width,
            bin_equal_frequency,
            bin_multiclass_optimal,
            bin_decision_tree_based,
            bin_decision_tree_based_one_vs_all,
            bin_tree_based_regressor,
            bin_decision_tree_reg_based,
        ]

        # one-hot encoding
        self.one_hot()

        # Determine the number of top features to consider based on conciseness threshold
        #p_value = int((1 / conciseness_threshold))
        feature_importance = self.model_feature_importance(self.df.copy(), self.labels, p_value)

        for cluster_number in self.labels.unique():
            # Filter data for the current cluster
            filtered_data = self.df[self.labels == cluster_number]
            top_features = [f[0] for f in feature_importance[cluster_number]]
            filtered_data = filtered_data[top_features]

            # Generate taxonomy based on binning methods
            taxonomy = self.attribute_to_intervals(binning_methods, top_features, cluster_number)

            # Convert data to transactions and build item ancestors
            transactions = convert_dataframe_to_transactions(filtered_data)
            item_ancestors_dict = self.build_item_ancestors(filtered_data, taxonomy)

            max_length = int(1 / conciseness_threshold)
            # Generate frequent itemsets
            frequent_itemsets, _ = gFIM.itemsets_from_transactions(transactions, item_ancestors_dict,
                                                                       coverage_threshold, max_length, mode=mode)

            # Convert itemsets to rules
            rules = convert_itemset_to_rules(frequent_itemsets, mode=mode)

            # Initialize analyzer
            analyze = Analyze()
            original_df = self.df.copy()
            original_df['Cluster'] = self.labels

            # Filter original data for the top features and analyze explanations
            filtered_original_df = original_df[top_features]
            filtered_original_df['Cluster'] = self.labels
            explanation_candidates = analyze.analyze_explanation(filtered_original_df, rules, cluster_number,
                                                                 [i for i in self.labels.unique() if
                                                                  i != cluster_number], mode=mode)

            # Filter and refine explanations based on separation threshold and skyline operator
            explanation = explanation_candidates[explanation_candidates['separation_err'] <= separation_threshold]
            explanation = skyline_operator(explanation)
            explanation['Cluster'] = cluster_number
            explanation_for_all_clusters = explanation_for_all_clusters._append(explanation)

        explanation_for_all_clusters = explanation_for_all_clusters.reset_index(names=['rule'])
        explanation_for_all_clusters = explanation_for_all_clusters.sort_values(by=['Cluster'])

        return explanation_for_all_clusters
