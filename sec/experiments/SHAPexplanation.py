import xgboost as xgb
import re
import numpy as np
import pandas as pd
import ast
#!pip install shap
import shap

def binning(col, bin_num):
    """Perform data binning to specific col
           Args:
               col: col to binning
               bin_num: intervals number for binning
           Returns:
               new binning col
           """

    # Binning using cut function of pandas
    col_bin = pd.cut(col, bin_num, include_lowest=True, precision=0)

    # Rename the categories after the interval
    col_bin = col_bin.cat.rename_categories([f'{col_bin.left}_{col_bin.right}' for col_bin in col_bin.cat.categories])

    return col_bin


def binning_all_columns(df, bin_number, numeric_variables):
    """Perform data binning to the numeric variables
               Args:
                   df: data frame of the data
                   bin_number: intervals number for binning
                   numeric_variables: numeric variables names

               Returns:
                   new binning data frame
               """

    numeric_variables_binned = []
    for variable in numeric_variables:
        binned_output_column = variable + "_binned"
        df[binned_output_column] = binning(df[variable],bin_number)
        numeric_variables_binned.append(binned_output_column)
    return numeric_variables_binned


def one_hot(df, numeric_variables_binned,categorical_variables):
    """Perform one hot encoding
                   Args:
                       df: data frame of the data
                       numeric_variables_binned: numeric variables binned names
                       categorical_variables: categorical variables names

                   Returns:
                       new one hot encoding data frame
                   """
    oh_df = pd.DataFrame()

    n_samples = len(df)
    min_frequency = 0.1

    # now adding the one hot encoded data
    for variable in numeric_variables_binned+categorical_variables:
        unique_values = df[variable].nunique()
        if unique_values <= min_frequency * n_samples:
            onehot_col = pd.get_dummies(df[variable],prefix=variable)
            oh_df = pd.concat([oh_df,onehot_col],axis=1)
    return oh_df


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

    # If not boolean or numeric, it must be a string
    return variable



def analyze_clustering_explainer_with_shap_values(original_data, labels, numeric_variables, categorical_variables,
                                                  max_length, sample_size=2000):
    """Explainer for clustering result by SHAP values.

    Args:
        original_data: The original data before one-hot encoding.
        labels: Cluster labels.
        numeric_variables: List of numeric variables.
        categorical_variables: List of categorical variables.
        max_length: Maximum length of SHAP value descriptions.
        sample_size: Maximum number of samples to use for SHAP calculation per cluster.

    Returns:
        A DataFrame containing the rule list for each cluster.
    """
    df = original_data.copy()
    numeric_variables_binned = binning_all_columns(df, 4, numeric_variables)

    # One hot encoding
    oh_df = one_hot(df, numeric_variables_binned, categorical_variables)

    unique_labels = np.unique(labels)
    dict = {}
    report_class_list = []

    for cluster in unique_labels:  # Loop over clusters
        # Sample data for the cluster
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) > sample_size:
            cluster_sample_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)
        else:
            cluster_sample_indices = cluster_indices

        oh_df_cluster_sample = oh_df.iloc[cluster_sample_indices]
        labels_cluster_sample = [labels[i] for i in cluster_sample_indices]

        # Fit model on the cluster's data
        clf = xgb.XGBClassifier(max_depth=10, n_jobs=-1)
        clf.fit(oh_df_cluster_sample, labels_cluster_sample)

        # SHAP
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(oh_df_cluster_sample)
        columns_name = oh_df_cluster_sample.columns
        variables = numeric_variables + categorical_variables

        combined_string = ""
        # Convert cluster SHAP values to DataFrame
        shap_values_df = pd.DataFrame(shap_values, columns=columns_name)
        total_mean_var = pd.DataFrame(index=['mean', 'var'])

        df_bool = oh_df_cluster_sample == 1  # After one hot encoding, consider only '1' values

        for col in variables:  # Loop over all the original columns
            mean_var = pd.DataFrame(index=['mean', 'var'])
            escaped_col = re.escape(col)
            shap_values_df_col = shap_values_df.filter(regex='^' + escaped_col + '_')  # Get all col_binned

            for bin_col in shap_values_df_col.columns:
                filtered_values = shap_values_df.loc[
                    (np.array(df_bool.loc[:, bin_col])) & (np.array(labels_cluster_sample) == 1), bin_col]
                if len(filtered_values) > 1:
                    mean_var[bin_col] = [filtered_values.loc[filtered_values > 0.0001].mean(),
                                         filtered_values.loc[filtered_values > 0.0001].var(ddof=0)]

            if not mean_var.empty:
                max_elements = mean_var.idxmax(axis=1)
                max_mean_col_name = max_elements['mean']
                if str(max_mean_col_name) != "nan":
                    total_mean_var[max_mean_col_name] = mean_var[max_mean_col_name]

        for length in range(1, max_length + 1):
            df_topk = total_mean_var.T.nlargest(length, 'mean')
            lst = []

            for index_name in df_topk.index.values:
                if "binned" in index_name:
                    split_name = index_name.split('_binned_')
                    item_a = split_name[1].split('_')[0]
                    item_b = split_name[1].split('_')[1]
                    feature_name = split_name[0]
                    combined_string += f"{feature_name} between:  {item_a} - {item_b}\n"
                    combined_string += f"SHAP values mean: {df_topk.loc[index_name]['mean']}\n"
                    combined_string += f"SHAP values var: {df_topk.loc[index_name]['var']}\n\n"
                    lst.extend([[split_name[0], ">=", float(item_a)], [split_name[0], "<=", float(item_b)]])
                else:
                    split_name = index_name.split('_')
                    item = get_real_value_type(split_name[1])
                    feature_name = split_name[0]
                    combined_string += f"{feature_name} =  {item}\n"
                    combined_string += f"SHAP values mean: {df_topk.loc[index_name]['mean']}\n"
                    combined_string += f"SHAP values var: {df_topk.loc[index_name]['var']}\n\n"
                    lst.extend([[feature_name, "==", item]])

            if cluster in dict.keys():
                dict[cluster].append(lst)
            else:
                dict[cluster] = [lst]

        report_class_list.append((cluster, combined_string))

    report_df = pd.DataFrame(report_class_list, columns=['class_name', 'rule_list'])

    return report_df
