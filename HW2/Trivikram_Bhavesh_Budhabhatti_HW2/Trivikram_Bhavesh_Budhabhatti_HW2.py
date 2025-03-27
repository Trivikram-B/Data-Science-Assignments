import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np

##########################################
##Hyperparameter ranges (to tune)
##########################################
max_depth_values = [3, 5]
min_samples_split_values = [5, 10]
min_samples_leaf_values = [3, 5]
min_impurity_decrease_values = [0.01, 0.001]
ccp_alpha_values = [0.001, 0.0001]


def input_data(file_name, label_name):
    data = pd.read_csv(file_name)
    labels = data.loc[:, data.columns == label_name]
    feats = data.loc[:, data.columns != label_name]
    return feats, labels


# Load the Diabetes dataset
features, labels = input_data(file_name="diabetes.csv", label_name="Outcome")

# Split the data into train (72%), validation (8%), and test (20%) sets
train_feat, temp_feat, train_label, temp_label = train_test_split(
    features, labels, test_size=0.28, random_state=42
)
val_feat, test_feat, val_label, test_label = train_test_split(
    temp_feat, temp_label, test_size=0.2857, random_state=42
)

# Initialize variables to store the best hyperparameters and accuracy
best_accuracy = 0
best_params = {}

# Hyperparameter tuning using validation set
for max_depth in max_depth_values:
    for min_samples_split in min_samples_split_values:
        for min_samples_leaf in min_samples_leaf_values:
            for min_impurity_decrease in min_impurity_decrease_values:
                for ccp_alpha in ccp_alpha_values:

                    # Create a model with the current hyperparameters
                    treemodel = tree.DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_impurity_decrease=min_impurity_decrease,
                        ccp_alpha=ccp_alpha,
                    )

                    # Train the model
                    treemodel.fit(train_feat, train_label)

                    # Evaluate the model on the validation set
                    val_pred_label = treemodel.predict(val_feat)
                    accuracy = accuracy_score(val_label, val_pred_label)

                    # Update best model if we get a better accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                            "min_impurity_decrease": min_impurity_decrease,
                            "ccp_alpha": ccp_alpha,
                        }

# Print the best hyperparameters
print("Best Hyperparameters based on validation accuracy:")
print(best_params)

# Create and train the model using the best hyperparameters
treemodel = tree.DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    min_impurity_decrease=best_params["min_impurity_decrease"],
    ccp_alpha=best_params["ccp_alpha"],
)

treemodel.fit(train_feat, train_label)

# Accuracy on test data
test_pred_label = treemodel.predict(test_feat)
testing_accuracy = accuracy_score(test_label, test_pred_label)
print(f"Testing accuracy: {testing_accuracy}")

# Confusion matrix
conf_matrix = confusion_matrix(test_label, test_pred_label)
ConfusionMatrixDisplay(conf_matrix, display_labels=["No Diabetes", "Diabetes"]).plot()
plt.show()

# Visualizing the tree
plt.figure(figsize=(12, 12))
tree.plot_tree(
    treemodel,
    feature_names=train_feat.columns,
    class_names=["No Diabetes", "Diabetes"],
    filled=True,
)
plt.show()

# rules extracted
print("Some rules I Noticed: ")
print("1. If BMI <= 29.35 and Age <= 49.5 and Glucose > 136.5 then diabetes")
print("2. If BMI > 29.35 and BMI <= 31.35 and Glucose <= 146.5, then Diabetes")
print("3. If BMI > 41.8 AND Glucose <= 111.5, then Diabetes ")
