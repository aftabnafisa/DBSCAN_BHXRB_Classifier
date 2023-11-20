import pandas as pd
# import numpy as np
from sklearn.cluster import DBSCAN ## DBSCAN clusters even nonlinear data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


X = pd.read_csv(r"ip_feature_shuffle_1.csv")

print(X.shape)

# Specifying the number of features
n_features = 6


# Feature scaling to standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creating a DBSCAN model
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Fitting the model to the scaled data
dbscan.fit(X_scaled)

# Getting the cluster labels for each data point
cluster_labels = dbscan.labels_

# Printing the cluster labels
print(cluster_labels)


# Creating a DataFrame (op_df) with features and cluster labels
op_df = X.copy()
op_df['cluster_label'] = cluster_labels

# Saving the dataframe (op_df) into a csv file (op_feature_labels_GB.csv)
op_df.to_csv('op_feature_labels_GB_1.csv', mode = 'w', index=False)

# Creating a pairplot with different colors for each cluster
pairplot = sns.pairplot(op_df, hue='cluster_label')

# Displaying the plot
plt.show()

# Saving the pairplot as an image (pairplot_synthetic_GB.png)
pairplot.savefig("pairplot_GB.png")

# Evaluation of the model to classify the Black hole states
# By calculating Silhouette Score
# 'cluster_labels' is the cluster labels assigned by DBSCAN
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")


# ==========================================================
# TO CHECK HOW MUCH IS THE CLASSIFICATION OF THE BLACK HOLE STATE IS CORRECT BY THE MODEL
# By two dataframe comparision, 
# One dataframe is standard and has true labels, another is the output dataframe obtained with this code
# ============================================================

# ip_df is the standard dataframe which has all the six features and the label of BH state
ip_df = pd.read_csv(r'ip_feature_label_shuffle_1.csv')

label_column_ip = ip_df.columns[-1]
label_column_op = op_df.columns[-1]

# Comparing the last (label) columns
column_comparison_result = (ip_df[label_column_ip] == op_df[label_column_op])

# Checking if all values are equal
are_columns_equal = column_comparison_result.all()

# Printing the result
if are_columns_equal:
    print(f"The last columns ({label_column_ip}) are equal, i.e. the label classification is 100% correct")
else:
    print(f"The last columns ({label_column_ip}) are not equal")


# ============================================================
# To get confusion matrix, precision, f1 score and accuracy
# By comparing the true label (in last column of the standard file) and the predicted label (in last column of the output file)
# ============================================================

# Extracting the true labels and predicted labels
true_labels = ip_df['label']
predicted_labels = op_df['cluster_label']

# Creating a confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Calculating precision and F1 score
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
accuracy = accuracy_score(true_labels, predicted_labels)

# Printing the results
print("Confusion Matrix:")
print(conf_matrix)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

# =====================THE END OF CODE =======================================




