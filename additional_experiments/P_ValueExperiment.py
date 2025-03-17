import os
import traceback
import pandas as pd
import numpy as np
import ast
import time
import warnings
from csv import writer
from sklearn.preprocessing import OneHotEncoder

from experiments import Experiments
from cluster_explorer import Explainer
from cluster_explorer.AnalyzeItemsets import Analyze


warnings.filterwarnings('ignore')

# File paths and global variables
FOLDER_PATH = 'p_results'
DATASETS_FILE = 'Datasets.csv'
PIPELINE_SCORES_FILE = 'Pipeline_Scores.csv'


def add_dataset_to_csv(dataset, dataset_name, file_name=DATASETS_FILE):
    """Adds dataset information to the CSV file."""
    data_types = dataset.dtypes
    unique_data_types_list = list(set(data_types))

    datasets = pd.read_csv(file_name)
    dataset_id = len(datasets)

    dataset_information = {
        'Id': dataset_id,
        'Dataset Name': dataset_name,
        'Number of Rows': dataset.shape[0],
        'Number of Columns': dataset.shape[1],
        'Data Types': unique_data_types_list,
    }

    with open(file_name, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(dataset_information.values())

    return dataset_information


def write_to_csv(data_dict, path):
    """Writes a dictionary to a CSV file."""
    with open(path, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(data_dict.values())


def analyze_rules(rules_dict, original_data):
    """Analyzes rules and returns a DataFrame with rule metrics."""
    clusters = original_data['Cluster'].unique()
    total_df = pd.DataFrame(columns=["coverage", "separation_err", "conciseness", "Cluster"])

    for cluster in rules_dict.keys():
        rules = rules_dict[cluster]
        analyzer = Analyze()
        other_clusters = [i for i in clusters if i != cluster]
        explanation_candidates = analyzer.analyze_explanation(original_data, rules, int(cluster), other_clusters)
        explanation_candidates['Cluster'] = cluster
        total_df = total_df._append(explanation_candidates)

    total_df = total_df.sort_values(by=['Cluster'])
    return total_df.reset_index(names=['rule'])


def process_explainer_p_exact(train_df, labels, coverage_threshold, conciseness_threshold, separation_threshold):
    """Processes ClusterExplore explanations."""
    try:
        explainer = Explainer(train_df, labels)
        start_time = time.time()
        df_cluster_explore = explainer.generate_explanations(
            coverage_threshold=coverage_threshold,
            conciseness_threshold=conciseness_threshold,
            separation_threshold=separation_threshold,
            p_value= len(train_df.columns)
        )
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        df_cluster_explore['p'] = 0
        return df_cluster_explore, execution_time
    except Exception as e:
        print(f"Error processing ClusterExplore p=Exact: {str(e)}")
        traceback.print_exc()
        return None, None
    #int((1 / conciseness_threshold))

def process_explainer_p_1(train_df, labels, coverage_threshold, conciseness_threshold, separation_threshold):
    """Processes ClusterExplore explanations."""
    try:
        explainer = Explainer(train_df, labels)
        start_time = time.time()
        df_cluster_explore = explainer.generate_explanations(
            coverage_threshold=coverage_threshold,
            conciseness_threshold=conciseness_threshold,
            separation_threshold=separation_threshold,
            p_value= int((1 / conciseness_threshold))
        )
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        df_cluster_explore['p'] = 0
        return df_cluster_explore, execution_time
    except Exception as e:
        print(f"Error processing ClusterExplore p=1: {str(e)}")
        traceback.print_exc()
        return None, None

def process_explainer_p_1_5(train_df, labels, coverage_threshold, conciseness_threshold, separation_threshold):
    """Processes ClusterExplore explanations."""
    try:
        explainer = Explainer(train_df, labels)
        start_time = time.time()
        df_cluster_explore = explainer.generate_explanations(
            coverage_threshold=coverage_threshold,
            conciseness_threshold=conciseness_threshold,
            separation_threshold=separation_threshold,
            p_value= int(int((1 / conciseness_threshold))*1.5)
        )
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        df_cluster_explore['p'] = 0
        return df_cluster_explore, execution_time
    except Exception as e:
        print(f"Error processing ClusterExplore p=1.5: {str(e)}")
        traceback.print_exc()
        return None, None

def process_explainer_p_2(train_df, labels, coverage_threshold, conciseness_threshold, separation_threshold):
    """Processes ClusterExplore explanations."""
    try:
        explainer = Explainer(train_df, labels)
        start_time = time.time()
        df_cluster_explore = explainer.generate_explanations(
            coverage_threshold=coverage_threshold,
            conciseness_threshold=conciseness_threshold,
            separation_threshold=separation_threshold,
            p_value= int(int((1 / conciseness_threshold))*2)
        )
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        df_cluster_explore['p'] = 0
        return df_cluster_explore, execution_time
    except Exception as e:
        print(f"Error processing ClusterExplore p=2: {str(e)}")
        traceback.print_exc()
        return None, None


def process_files(files):
    """Main processing function for files."""
    pipeline_data = pd.read_csv(PIPELINE_SCORES_FILE)

    for file_name in files:
        if file_name.endswith('.csv'):
            file_path = os.path.join(FOLDER_PATH, file_name)
            df = pd.read_csv(file_path)
            dataset_name = os.path.splitext(os.path.basename(file_name))[0]
            dataset_information = add_dataset_to_csv(df, dataset_name)

            cluster_column_index = next((i for i, col in enumerate(df.columns) if col.startswith("Cluster")), None)

            if cluster_column_index is not None:
                data_df = df.iloc[:, :cluster_column_index]
                process_clusters(df, data_df, cluster_column_index, dataset_information, dataset_name)


def process_clusters(df, data_df, cluster_column_index, dataset_information, dataset_name):
    """Processes clusters for each file."""
    for index in range(cluster_column_index, len(df.columns)):
        data = data_df.copy()
        clusters = df.iloc[:, [index]]
        column_name = clusters.columns[0]
        parts = column_name.split('_', 1)
        pipeline_steps = ast.literal_eval(parts[1])
        unique_labels, label_counts = np.unique(clusters.values, return_counts=True)
        labels = pd.Series(data=clusters.iloc[:, 0].values)

        data_df.columns = [x.replace(' ', '_') for x in data_df.columns]
        data_df.columns = [x.replace('_', '') for x in data_df.columns]

        time_dict = {}
        dataframes = []

        categorical_features, numeric_features = get_features(data_df)

        enc = OneHotEncoder(min_frequency=0.1, handle_unknown="infrequent_if_exist", sparse_output=False)
        encoded_df = pd.DataFrame(enc.fit_transform(data_df[categorical_features]),
                                  columns=enc.get_feature_names_out(categorical_features))
        train_df = pd.concat([data_df.drop(categorical_features, axis=1), encoded_df], axis=1)

        max_length = int(1 / 0.2)  # conciseness_threshold

        p_values = {
            'Exact': lambda: process_explainer_p_exact(train_df, labels, 0.8, 0.2, 0.3),
            '1': lambda: process_explainer_p_1(train_df, labels, 0.8, 0.2, 0.3),
            '1.5': lambda: process_explainer_p_1_5(train_df, labels, 0.8, 0.2, 0.3),
            '2': lambda: process_explainer_p_2(train_df, labels, 0.8, 0.2, 0.3),
        }

        for p_name, p_function in p_values.items():
            try:
                result_df, exec_time = p_function()
                if result_df is not None:
                    time_dict[p_name] = exec_time
                    dataframes.append(result_df)
            except Exception as e:
                print(f"Error processing {p_name}: {str(e)}")
                traceback.print_exc()

        aggregate_results(dataframes, dataset_information, dataset_name, pipeline_steps, unique_labels, time_dict)


def get_features(data_df):
    """Gets categorical and numeric features from the DataFrame."""
    categorical_features = data_df.select_dtypes(include=['object']).columns.tolist()
    numeric_columns = data_df.dtypes[(data_df.dtypes == "float64") | (data_df.dtypes == "int64")].index.tolist()
    ordinals = [nc for nc in numeric_columns if data_df[nc].nunique() < 6]
    categorical_features += ordinals
    return categorical_features, numeric_columns


def aggregate_results(dataframes, dataset_information, dataset_name, pipeline_steps, unique_labels, time_dict):
    """Aggregates results and writes to the CSV file."""
    try:
        df_experiments = pd.concat(dataframes, ignore_index=True)
        ex_df = Experiments.calculation(df_experiments)

        for p_value in ex_df.index:
            QSE_mean = ex_df.loc[p_value, 'QSE_mean']
            results = {
                'Dataset': dataset_name,
                'Pipeline Steps': pipeline_steps,
                'Number of Clusters': len(unique_labels),
                'p': p_value,
                'QSE_mean': QSE_mean,
                'time(minutes)': time_dict[p_value],
            }
            merged_dict = {**dataset_information, **results}
            write_to_csv(merged_dict, 'Experiments results.csv')
    except Exception as e:
        print(f"Error processing: {str(e)}")
        traceback.print_exc()


if __name__ == '__main__':
    files = os.listdir(FOLDER_PATH)
    process_files(files)
