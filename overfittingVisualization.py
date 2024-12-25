"""
Code to visualize trends and fit models for relationships between loss values and our 
phi-values and phi-composites.
"""

import warnings
warnings.filterwarnings('ignore')

import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
tf.config.run_functions_eagerly(True)

import os

os.environ["PYPHI_WELCOME_OFF"] = "yes"

import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

import pyphi
pyphi.config.PROGRESS_BARS = False # may need to comment this one out for bigger networks, but this is fine for now
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

import pickle

import dcor
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score


def capped_relu(x):
    return tf.keras.activations.relu(x, max_value=1)

tf.keras.utils.get_custom_objects().update({'capped_relu': capped_relu})

folder_path = "overfittingAnalysisPaper/overfittingAnalysisPickles"

columns = [
    "Filename",
    "Train",
    "Val",
    "Variance",
    "Skewness",
    "Kurtosis",
    "L2 Norm",
    "Gini Coef",
    "Entropy",
    "Phi"
]

df = pd.DataFrame(columns=columns)

def add_pickle_file(filename):
    if filename.endswith(".pickle"):  # Check if the file is a pickle file
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            
            train_loss = data[0]
            val_loss = data[1]
            sia_metrics = data[3]
            phi_vals = data[2]

            X = data[6]
            y = data[7]

            for i in range(len(train_loss)):
                entry = [
                    filename.split("Analysis")[-1].split(".")[0],
                    train_loss[i].numpy(),
                    val_loss[i].numpy(),
                    sia_metrics[i]["Variance"],
                    sia_metrics[i]["Skewness"],
                    sia_metrics[i]["Kurtosis"],
                    sia_metrics[i]["L2 Norm"],
                    sia_metrics[i]["Gini Coef"],
                    sia_metrics[i]["Entropy"],
                    phi_vals[i]
                ]
                df.loc[len(df)] = entry

for filename in os.listdir(folder_path):
    add_pickle_file(filename)

# add_pickle_file("overfittingAnalysis2.pickle")

# print(df.values.shape) # 15F entries where F = num files
# print(df.head())

df.to_csv("overfittingAnalysisPaper/compilation.csv")

#### CORRELATION MATRIX!!!
def calculate_distance_correlation(df):
    distance_corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            distance_corr_matrix.loc[col1, col2] = dcor.distance_correlation(df[col1], df[col2])
    return distance_corr_matrix

def calculate_mutual_information(df):
    mi_corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            mi_corr_matrix.loc[col1, col2] = mutual_info_score(df[col1], df[col2])
    return mi_corr_matrix

def calculate_cosine_similarity(df):
    cosine_corr_matrix = pd.DataFrame(cosine_similarity(df.T), index=df.columns, columns=df.columns)
    return cosine_corr_matrix


# corrPearson = df.corr(method="pearson")
# corrKendall = df.corr(method="kendall")
# corrSpearman = df.corr(method="spearman")
# corrDCor = calculate_distance_correlation(df).apply(pd.to_numeric, errors='coerce')
# corrMI = calculate_mutual_information(df).apply(pd.to_numeric, errors='coerce')
# corrCosine = calculate_cosine_similarity(df).apply(pd.to_numeric, errors='coerce')

# plt.figure(figsize=(8, 6))  # Adjust the figure size
# sns.heatmap(corrPearson, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

# # Add title
# plt.title("Correlation Matrix Heatmap for Loss vs. Phi-composites")

# Show the plot
# plt.show()

X_VAL = "Phi"
Y_VAL = "Train"

valuesDf = df[[X_VAL, Y_VAL]]

### Graph phi average for each epoch vs. validation error for each epoch

# Add a modulo column for grouping
valuesDf['Group'] = (valuesDf.index % 15) + 1

# Calculate the averages for each group
grouped = valuesDf.groupby('Group').mean()

grouped_scaled = grouped.copy()
for column in [X_VAL, Y_VAL]:
    min_val = grouped[column].min()
    max_val = grouped[column].max()
    grouped_scaled[column] = (grouped[column] - min_val) / (max_val - min_val)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(grouped_scaled.index, grouped_scaled[X_VAL], marker='o', label=X_VAL)
plt.plot(grouped_scaled.index, grouped_scaled[Y_VAL], marker='o', label=Y_VAL)
plt.xticks(range(1, 16))  # Set x-axis ticks from 1 to 15
plt.xlabel('Epoch')
plt.ylabel('Average')
plt.title(f'Average value per Epoch for Loss and {X_VAL}')
plt.legend()
plt.grid(True)
plt.show()

### MACHINE LEARNING REGRESSION MODEL FOR FITTING!

def get_sublists(lst):
    sublists = []
    for r in range(1, len(lst) + 1):
        sublists.extend(list(itertools.combinations(lst, r)))
    return sublists

transformDf = valuesDf

# Function to scale every 15 values in a column
def scale_in_chunks(column):
    scaled_column = np.zeros_like(column)  # Array to store scaled values
    for i in range(0, len(column), 15):  # Process in chunks of 15
        chunk = column[i:i + 15].values.reshape(-1, 1)  # Reshape for scaler
        scaled_chunk = MinMaxScaler((0.01, 1)).fit_transform(chunk)  # Apply MinMaxScaler
        scaled_column[i:i + 15] = scaled_chunk.flatten()  # Store scaled values
    return scaled_column

# Apply the function to both columns
transformDf[X_VAL] = scale_in_chunks(transformDf[X_VAL])
transformDf[Y_VAL] = scale_in_chunks(transformDf[Y_VAL])

# transformDf[X_VAL] = np.log(transformDf[X_VAL])
# transformDf[Y_VAL] = np.log(transformDf[Y_VAL])

transformDf = transformDf[(transformDf[Y_VAL] > 0.1) & (transformDf[Y_VAL] < 0.95)]

X = transformDf[[X_VAL]].values
y = transformDf[[Y_VAL]].values

model = LinearRegression()

model.fit(X, y)
y_preds = model.predict(X)
score = r2_score(y, y_preds)

print(score)

# Plot the data
plt.figure(figsize=(8, 6))

# Define a list of 15 distinct colors
colors = plt.cm.tab20.colors # Use 15 distinct colors from the colormap

for i in range(len(X)):
    try:
        color_index = i // 15 # Cycle through 15 colors
        plt.scatter(X[i], y[i], color=colors[color_index])
    except IndexError:
        break

plt.scatter(X, y)

# Generate points for the line of best fit
line_X = np.linspace(transformDf[X_VAL].min(), transformDf[X_VAL].max(), 100).reshape(-1, 1)
line_y = model.predict(line_X)

# Plot the line of best fit
plt.plot(line_X, line_y, color='green', label='Line of Best Fit')

# Add labels and title
plt.xlabel(X_VAL, fontsize=12)
plt.ylabel(Y_VAL, fontsize=12)
plt.title(f'Graph of {Y_VAL} vs {X_VAL}', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()