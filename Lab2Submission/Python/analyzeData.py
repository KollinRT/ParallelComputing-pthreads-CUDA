import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# MATRIX_SIZE = 10

# def print_matrix_init(MATRIX_SIZE):
#     for i in range(MATRIX_SIZE):
#         for j in range(MATRIX_SIZE):
#             print(f"A[{i}][{j}] = 1;", end=' ')
#         print()  # Move to the next line after printing each row



#def plot_graphs(df):
#    """
#    Function to plot the graphs
#    :param df: dataframe from which the graphs will be plotted
#    :return:
#    """
#    # Filter data to include only 'Power' operations
#    # power_data = df[df['Operation'] == 'Power']
#
#    # Create a plot for each matrix size
#    matrix_sizes = df['Matrix Size'].unique()
#
#    # Determine the smallest non-zero time to set as the lower limit for the log scale
#    min_non_zero_time = df[df['Time (microseconds)'] > 0]['Time (microseconds)'].min()
#
#    # Create a full-sized plot for each matrix size with a logarithmic time scale, starting from the minimum non-zero time
#    for size in sorted(matrix_sizes):
#        plt.figure(figsize=(10, 6))
#        subset = df[df['Matrix Size'] == size]
#        ax = sns.lineplot(data=subset, x='Power', y='Time (microseconds)', hue='Method', marker='o')
#        plt.title(f'Performance Analysis for Matrix Size {size} (Log Scale)')
#        plt.xlabel('Power')
#        plt.ylabel('Time (microseconds)')
#        ax.set_yscale('log')
#        ax.set_ylim(bottom=min_non_zero_time)  # Set the lower limit to the smallest non-zero time
#        plt.grid(True)
#        plt.legend(title='Method')
#        plt.savefig(f"figures/plot_for_{size}.png")
#        # plt.show()

def plot_graphs_withERRORBars(df):
    """
    Function to plot the graphs with error bars showing the standard deviation.
    :param df: DataFrame from which the graphs will be plotted.
    """
    df.columns = df.columns.str.strip()

    matrix_sizes = df['Matrix Size'].unique()

    # Calculate mean and standard deviation for each group
    grouped = df.groupby(['Method', 'Matrix Size', 'Power'])['Time (milliseconds)'].agg(['mean', 'std']).reset_index()

    for size in sorted(matrix_sizes):
        plt.figure(figsize=(10, 6))
        subset = grouped[grouped['Matrix Size'] == size]

        # Loop through each method to plot individually so we can add error bars
        for method in subset['Method'].unique():
            method_data = subset[subset['Method'] == method]
            plt.errorbar(method_data['Power'], method_data['mean'], yerr=method_data['std'], fmt='-o', capsize=5,
                         label=method)

        plt.title(f'Performance Analysis for Matrix Size {size} (Log Scale)')
        plt.xlabel('Power')
        plt.ylabel('Time (microseconds)')
        plt.yscale('log')

        # Set a reasonable lower limit for the y-axis
        valid_times = subset[subset['mean'] > 0]['mean']
        if not valid_times.empty:
            min_non_zero_time = max(valid_times.min() * 0.1, 1e-3)
            plt.ylim(bottom=min_non_zero_time)

        plt.grid(True)
        plt.legend(title='Method')
        plt.savefig(f"figures/plot_for_{size}.png")
        plt.show()

def makeIntoFigures(dir):
    """
    Helps to ease making figures from plots.
    :param dir: make figures into the dir directory. This will just automate the basic figure creation for LaTeX
    :return:
    """
    filenames = os.listdir(dir)
    # Sorting filenames based on the numerical part of the filename
    filenames.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

    for filename in filenames:
        if filename.endswith(".png"):  # Adjust file extension as needed
            # Extracting the number from the filename
            M = filename.split('_')[2].split('.')[0]

            print(
fr"""
\begin{{figure}}[h!]
    \centering
    \includegraphics[width=\textwidth]{{{dir}/{filename}}}
    \caption{{Time scaling for $M = {M}$}}
    \label{{fig:benchmark-for-M={M}}}
\end{{figure}}
"""
            )

df = pd.read_csv("./data/results.csv")
plot_graphs_withERRORBars(df)

# makeIntoFigures("./figures")
