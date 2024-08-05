import pandas as pd


def fill_empty_clusters(df):
    # List of unique clusters and frameworks
    unique_clusters = df['Cluster'].unique()
    unique_frameworks = df['framework'].unique()
    # Loop through clusters and frameworks
    for cluster in unique_clusters:
        for framework in unique_frameworks:
            # Check if a row exists for the cluster in the framework
            if not ((df['Cluster'] == cluster) & (df['framework'] == framework)).any():
                # Add a new row with default values
                new_row = {
                    'rule': '',
                    'coverage': 0,
                    'separation_err': 1,
                    'conciseness': 0,
                    'Cluster': cluster,
                    'framework': framework
                }
                df = df._append(new_row, ignore_index=True)
    df.sort_values(['framework', 'Cluster'], inplace=True)
    return df

def sum_calculation(data):
    df = data.copy()
    df = df[df['Cluster'] != -1]

    # Create a new DataFrame with the maximum sum for each framework within each cluster
    final_table = df.groupby(['Cluster', 'framework'])['Mean'].max().reset_index()

    # Calculate the sum of each framework separately
    framework_sums = final_table.groupby('framework')['Mean'].sum().reset_index()

    # Calculate the number of unique values in the 'Cluster' column
    cluster_counts = df['Cluster'].nunique()

    # Divide the sum of each framework by the number of unique values in the 'framework' column
    framework_sums['Mean'] = framework_sums['Mean'] / cluster_counts

    return framework_sums


def calculation(data):
    data['Mean'] = (data['coverage'] + (1 - data['separation_err']) + data['conciseness']) / 3
    data = fill_empty_clusters(data)
    sum_df = sum_calculation(data).set_index("framework")
    frameworks = data['framework'].unique()
    total_expiraments = pd.DataFrame(index=frameworks)

    for framework in frameworks:
        total_expiraments.loc[framework, 'sum_mean'] = sum_df.loc[framework, 'Mean']
    return total_expiraments



