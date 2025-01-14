from sklearn.preprocessing import OneHotEncoder
from . import gFIM
from .AnalyzeItemsets import Analyze
from .utils import *
from .binning_methods import *
from typing import List, Callable


class Explainer:
    """
    The Explainer class provides an interface for generating explanations for clusters in a dataset.\n
    This class acts as the main entry point for generating explanations for clusters in a dataset.\n
    Generates rule based explanations for clusters in a dataset, using a model-agnostic algorithm.\n
    """

    def __init__(self, df: DataFrame, labels: Series):
        """
        :param df: The dataframe containing the data and features.
        :param labels: A series containing the class labels.
        """
        self.df = df
        self.data = df.copy()
        self.labels = labels
        self.numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        self.very_numerical = [nc for nc in self.numeric_columns if df[nc].nunique() > 6]
        self.taxonomy_tree_cache = {}
        self.coverage_threshold = 0
        self.conciseness_threshold = 0
        self.separation_threshold = 0

    def apply_binning_methods(self, numeric_attribute: str,
                              binning_methods: List[Callable[[DataFrame, Series, str, float, int], List[Tuple[float, float]]]],
                              cluster_number: int) -> List[Tuple[float, float]]:
        """
        Apply multiple binning methods to a numeric attribute and return the combined intervals.

        This method applies a list of binning methods to a specified numeric attribute for a given cluster.
        The results from each binning method are combined and sorted to produce a final list of intervals.

        :param numeric_attribute: The name of the numeric attribute to be binned.
        :param binning_methods: A list of binning methods to apply. Each method should take a DataFrame,
                                a Series of labels, the attribute name, a conciseness threshold, and a cluster number,
                                and return a list of intervals.
        :param cluster_number: The cluster number for which the binning methods are applied.
        :return: A sorted list of intervals resulting from the application of the binning methods.
        """
        intervals = set()

        # Apply binning methods in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda method: method(self.df, self.labels, numeric_attribute, self.conciseness_threshold,
                                      cluster_number), binning_methods)
            for result in results:
                intervals.update(result)

        # Sort the intervals based on the lower bound
        return sorted(intervals, key=lambda x: x[0])

    def process_single_attribute(self, attribute: str,
                                 binning_methods: List[Callable[[DataFrame, Series, str, float, int], List[Tuple[float, float]]]],
                                 cluster_number: int):
        """
        Process a single attribute by applying binning methods if it is highly numerical.

        This function checks if the given attribute is highly numerical (i.e., has more than 6 unique values).
        If so, it applies the specified binning methods to the attribute for the given cluster and stores the
        resulting intervals in the taxonomy tree cache.

        :param attribute: The name of the attribute to process.
        :param binning_methods: A list of binning methods to apply. Each method should take a DataFrame,
                                a Series of labels, the attribute name, a conciseness threshold, and a cluster number,
                                and return a list of intervals.
        :param cluster_number: The cluster number for which the binning methods are applied.
        :return: A tuple containing the attribute name and the list of intervals if the attribute is highly numerical,
                 otherwise the attribute name and None.
        """
        if attribute in self.very_numerical:
            intervals = self.apply_binning_methods(attribute, binning_methods, cluster_number)
            self.taxonomy_tree_cache[attribute] = intervals
            return attribute, intervals
        return attribute, None

    def attribute_to_intervals(self, binning_methods: List[Callable[[DataFrame, Series, str, float, int], List[Tuple[float, float]]]],
                               features: List[str], cluster_number: int) -> dict:
        """
        Create a list of intervals for each feature in the dataset.\n

        This method applies the specified binning methods to each feature in the dataset and stores the resulting
        intervals in a dictionary. The intervals are computed for the specified cluster number.
        :param binning_methods: A list of binning methods to apply. Each method should take a DataFrame,
                                a Series of labels, the attribute name, a conciseness threshold, and a cluster number,
                                and return a list of intervals.
        :param features: A list of feature names for which to compute intervals.
        :param cluster_number: The cluster number for which the binning methods are applied.
        :return: A dictionary where the keys are feature names and the values are lists of intervals.
        """
        taxonomy = {}
        with ThreadPoolExecutor() as executor:
            # Apply the process_single_attribute method to each feature in parallel, then store the results in the taxonomy
            future_to_attribute = {
                executor.submit(self.process_single_attribute, attribute, binning_methods, cluster_number): attribute
                for attribute in features
            }
            for future in as_completed(future_to_attribute):
                attribute, intervals = future.result()
                if intervals is not None:
                    taxonomy[attribute] = intervals
        return taxonomy

    def build_item_ancestors(self, df: DataFrame, taxonomy: dict) -> dict:
        """
        Build the item ancestors dictionary for the given dataframe and taxonomy.\n

        This function goes over the attributes and intervals in the taxonomy, and uses them to build a dictionary
        where the keys are tuples of the attribute name and a value, and the values are sets of intervals that contain
        the value.
        :param df: The dataframe containing the data and features.
        :param taxonomy: A dictionary where the keys are feature names and the values are lists of intervals (the taxonomy computed by attribute_to_intervals).
        :return: A dictionary where the keys are tuples of the attribute name and value, and the values are sets of intervals that contain the value.
        """
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

    def model_feature_importance(self, df: DataFrame, labels: Series, p: int=5) -> dict:
        """
        Compute the feature importance for each class.\n

        This method computes the feature importance for each class based on the decision tree classifier.\n
        It trains a decision tree classifier for each pair of classes and uses the feature importance of the classifier
        to determine the importance of each feature for the class of interest based on the average importance across
        all other classes.\n
        :param df: The dataframe containing the data and features.
        :param labels: A series containing the class labels.
        :param p: The number of top features to consider.
        :return: A dictionary where the key is the class label and the value is a list of tuples containing the top
                    features and their importance.
        """
        feature_importance = {}
        feature_names = df.columns
        class_labels = np.unique(labels)

        for class_of_interest in set(labels):
            aggregate_importance = {name: [] for name in feature_names}
            for other_class in class_labels:
                if other_class == class_of_interest:
                    continue

                # Take only the data for the class of interest and the other class
                mask = (labels == class_of_interest) | (labels == other_class)
                x_pair = df[mask]
                y_pair = labels[mask]
                y_pair = np.where(y_pair == class_of_interest, 0, 1)

                # Train a decision tree classifier on the pair of classes, then store its feature importance
                tree = DecisionTreeClassifier(random_state=42, max_depth=int(1 / self.conciseness_threshold))
                tree.fit(x_pair, y_pair)

                for name, importance in zip(feature_names, tree.feature_importances_):
                    aggregate_importance[name].append(importance)

            # Compute the average importance for each feature across comparisons with all other classes, as the feature importance
            # for the class of interest
            average_importance = {name: np.mean(importance) for name, importance in aggregate_importance.items()}
            feature_importance[class_of_interest] = sorted(average_importance.items(), key=lambda x: x[1],
                                                           reverse=True)[:p]

        return feature_importance

    def one_hot(self) -> List[str]:
        """
        Perform one-hot encoding on the categorical features of the dataframe.

        This method identifies categorical features in the dataframe, including
        ordinal features with fewer than 6 unique values. It then converts these
        features to string type and applies one-hot encoding using the
        `OneHotEncoder` from scikit-learn. The resulting encoded features are
        concatenated back to the dataframe, replacing the original categorical
        features.

        :return: The list of categorical features that were one-hot encoded.
        """
        categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        numeric_columns = self.df.dtypes[(self.df.dtypes == "float64") | (self.df.dtypes == "int64")].index.tolist()

        # If a numeric column has less than 6 unique values, it is considered ordinal and treated as categorical
        ordinals = [nc for nc in numeric_columns if self.df[nc].nunique() < 6]
        categorical_features = categorical_features + ordinals
        for col in categorical_features:
            self.df[col] = self.df[col].astype(str)

        enc = OneHotEncoder(min_frequency=0.1, handle_unknown="infrequent_if_exist", sparse_output=False)

        encoded_categories = enc.fit_transform(self.df.loc[:, categorical_features])
        encoded_df = pd.DataFrame(encoded_categories, columns=enc.get_feature_names_out(categorical_features))
        self.df = pd.concat([self.df.drop(categorical_features, axis=1), encoded_df], axis=1)

        # Return the list of categorical features that were one-hot encoded. This is not used in cluster-explorer itself,
        # but it useful for outside users who may want to know which features were one-hot encoded.
        return categorical_features

    def generate_explanations(self, coverage_threshold=0.6, conciseness_threshold=0.33, separation_threshold=0.5, p_value=0,
                              mode='conjunctive'):
        """
        Generate explanations for all clusters in the dataset.\n

        This function is an implementation of the algorithm detailed in the paper "Explaining Black-Box Clustering Pipelines With
        Cluster-Explorer" by Sariel Ofek and Amit Somech. It generates rule-based explanations for each cluster in the dataset
        using a model-agnostic approach. The explanations are generated based on the specified thresholds for coverage, conciseness,
        and separation error, as well as the p-value for feature importance.
        :param coverage_threshold: The minimum coverage threshold for an explanation rule to be considered.
        :param conciseness_threshold: The minimum conciseness threshold for an explanation rule to be considered.
                                      This value is the inverse of the maximum number of predicates in a rule.
        :param separation_threshold: The maximum separation error threshold for an explanation rule to be considered.
        :param p_value: The p-value for feature importance. This value is used to determine the number of top features to consider.
                        Currently, it should be set to the number of top features to consider, and should be the inverse of the conciseness threshold.
        :return: A dataframe containing the explanations for all clusters.
        """
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
                                                                   coverage_threshold, max_length)

            # Convert itemsets to rules
            rules = convert_itemset_to_rules(frequent_itemsets)

            # Initialize analyzer
            analyze = Analyze()
            original_df = self.df.copy()
            original_df['Cluster'] = self.labels

            # Filter original data for the top features and analyze explanations
            filtered_original_df = original_df[[*top_features, 'Cluster']]
            # filtered_original_df['Cluster'] = self.labels
            explanation_candidates = analyze.analyze_explanation(filtered_original_df, rules, cluster_number,
                                                                 [i for i in self.labels.unique() if
                                                                  i != cluster_number])

            # Filter and refine explanations based on separation threshold and skyline operator
            explanation = explanation_candidates[explanation_candidates['separation_err'] <= separation_threshold]
            explanation = skyline_operator(explanation)
            explanation['Cluster'] = cluster_number
            explanation_for_all_clusters = explanation_for_all_clusters._append(explanation)

        explanation_for_all_clusters = explanation_for_all_clusters.reset_index(names=['rule'])
        explanation_for_all_clusters = explanation_for_all_clusters.sort_values(by=['Cluster'])

        return explanation_for_all_clusters
