import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Function to load the dataset into a DataFrame (Replace 'your_dataset.csv' with your actual dataset file)
def load_data():
    df = pd.read_csv('SampleCSVFile_53000kb.csv', encoding='unicode_escape')
    return df

def main():
    st.title("Random Forest Classifier for predictive analysis")
    df = load_data()
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    target_column= st.selectbox("Select the column that u need to predict ",  numerical_features, key="target_column")
    if target_column not in df.columns.values:
        st.error("Wrong input. column name not found.")

    # Remove rows with missing values in the target column
    df_cleaned = df.dropna()

    # Convert continuous target to discrete classes using binning
    num_classes = 5  # Define the number of classes
    bin_labels = range(num_classes)  # Labels for each class
    y = pd.cut(df_cleaned[target_column], bins=num_classes, labels=bin_labels)

    # Select the numerical features and target
    numerical_features = df_cleaned.select_dtypes(include=[np.number]).drop(target_column, axis=1)
    X = numerical_features.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    minority_class_samples = np.sum(y_train == np.min(y_train))
    k_neighbors = min(3, minority_class_samples - 1)

    if k_neighbors > 0:
        oversampler = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_scaled, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train

    rf = RandomForestClassifier(n_estimators=100, random_state=5)

    # Fit the classifier to your data
    rf.fit(X_train_resampled, y_train_resampled)
    importances = rf.feature_importances_

    sorted_indices = importances.argsort()[::-1]

    st.subheader("Feature Importances:")
    for i, feature_index in enumerate(sorted_indices):
        feature_name = numerical_features.columns[feature_index]
        importance = importances[feature_index]
        st.write(f"Feature #{i+1}: {feature_name} - Importance: {importance}")

    # Make predictions on the test set
    y_pred = rf.predict(X_test_scaled)

    # Evaluate the model
    accuracy = metrics.accuracy_score(y_test, y_pred)
    classification_report = metrics.classification_report(y_test, y_pred)

    st.subheader("Model Evaluation Metrics:")
    st.write("Accuracy:", accuracy)

    report_dict = metrics.classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.subheader("Classification Report:")
    st.dataframe(report_df.style.format("{:.2f}"))

    # Diagnostic analysis
    diagnostic_result = ""

    # Analyze accuracy
    if accuracy >= 0.8:
        diagnostic_result += "The model has achieved a satisfactory accuracy of {:.2f}.\n".format(accuracy)
    else:
        diagnostic_result += "The model accuracy is below the desired threshold.\n"

    # Analyze recall and precision (averaged for multiclass problems)
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    precision = metrics.precision_score(y_test, y_pred, average='weighted')

    if recall >= 0.8:
        diagnostic_result += "The model has a high recall of {:.2f}, indicating it captures a large portion of positive instances.\n".format(recall)
    else:
        diagnostic_result += "The model recall is below the desired threshold.\n"

    if precision >= 0.8:
        diagnostic_result += "The model has a high precision of {:.2f}, suggesting it performs well in predicting positive instances.\n".format(precision)
    else:
        diagnostic_result += "The model precision is below the desired threshold.\n"

    st.subheader("Diagnostic Analysis:")
    st.text(diagnostic_result)

    y_pred_full = rf.predict(X)

    # Add the predicted classes as a new column in the DataFrame
    df_cleaned['Predicted_Class'] = y_pred_full

    analysis_result = ""

    # Analyze the distribution of predicted classes
    class_counts = df_cleaned['Predicted_Class'].value_counts()
    class_percentage = class_counts / len(df_cleaned) * 100

    analysis_result += "Predicted Class Distribution:\n"
    analysis_result += str(class_counts) + "\n"
    analysis_result += str(class_percentage.round(2)) + "%\n\n"

    # Analyze the relationship between the target, features, and the predicted class
    analysis_result += "Diagnostic Analysis:\n"
    analysis_result += "----------------------\n"

    # Analyze the relationship for each predicted class
    predicted_classes = df_cleaned['Predicted_Class'].unique()
    for predicted_class in predicted_classes:
        class_data = df_cleaned[df_cleaned['Predicted_Class'] == predicted_class]
        avg_target = class_data[target_column].mean()

        analysis_result += "For Predicted Class {}: \n".format(predicted_class)
        analysis_result += "- Average {}: {:.2f}\n\n".format(target_column, avg_target)

    st.subheader("Analysis Result:")
    st.text(analysis_result)

    recommendations = ""

    correlations = {}
    for feature in numerical_features.columns:
        correlation = df_cleaned[feature].corr(df_cleaned['Predicted_Class'])
        correlations[feature] = correlation

    st.subheader("Business Recommendations:")
    for feature, correlation in correlations.items():
        if abs(correlation) >= 0.8:
            recommendation = f"There is a strong correlation between {feature} and the predicted class. Increasing {feature} may result in higher predicted class values."
        else:
            recommendation = f"The correlation between {feature} and the predicted class is weak or moderate. Other factors may influence the predicted class values."
        st.text(recommendation)
    predictions_table = pd.DataFrame({'Original_Value': y_test, 'Predicted_Value': y_pred})

    # Add a column to indicate if the prediction is correct
    predictions_table['Prediction_Correct'] = predictions_table['Original_Value'] == predictions_table[
        'Predicted_Value']
    st.write(predictions_table['Prediction_Correct'])

    # Select a specific tree to plot (change tree_idx to the desired index)
    # tree_idx = 0
    # selected_tree = rf.estimators_[tree_idx]

    # Plot the selected tree
    # plt.figure(figsize=(12, 8))
    # plot_tree(selected_tree, feature_names=numerical_features.columns, filled=True)
    # st.pyplot()

if __name__ == "__main__":
    main()
