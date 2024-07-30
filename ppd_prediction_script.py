# Importing the necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Create output directory if it doesn't exist
output_dir = 'HPC_Output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Create a DataFrame with the data
data = {
    'Province': ['Newfoundland and Labrador', 'Prince Edward Island', 'Nova Scotia', 'New Brunswick', 
                 'Quebec', 'Ontario', 'Manitoba', 'Saskatchewan', 'Alberta', 'British Columbia'],
    'Percentage': [28, 26, 31, 24, 23, 23, 24, 16, 22, 23],
    'Compared_to_National_Average': ['higher', 'similar', 'higher', 'similar', 'similar', 'similar', 
                                     'similar', 'lower', 'similar', 'similar'],
    'Age_17_24': [10.4] * 10,
    'Age_25_29': [27.9] * 10,
    'Age_30_34': [37.0] * 10,
    'Age_35_39': [19.3] * 10,
    'Age_40_plus': [5.5] * 10,
    'Married_Common_Law': [83.1] * 10,
    'Not_Married': [16.9] * 10,
    'Less_than_High_School': [9.2] * 10,
    'High_School_Graduate': [18.4] * 10,
    'Post_Secondary_Graduate': [72.4] * 10,
    'History_Depression_Yes': [37.1] * 10,
    'History_Depression_No': [62.9] * 10,
    'No_Treatment': [40.7] * 10,
    'Medication': [23.7] * 10,
    'Counselling': [17.6] * 10,
    'Medication_and_Counselling': [18.0] * 10
}

df = pd.DataFrame(data)

# Convert categorical feature to numerical values
df['Compared_to_National_Average'] = df['Compared_to_National_Average'].map({'higher': 1, 'similar': 0, 'lower': -1})

# Define features and target
X = df[['Compared_to_National_Average', 'Age_17_24', 'Age_25_29', 'Age_30_34', 'Age_35_39', 'Age_40_plus',
        'Married_Common_Law', 'Not_Married', 'Less_than_High_School', 'High_School_Graduate', 
        'Post_Secondary_Graduate', 'History_Depression_Yes', 'History_Depression_No', 'No_Treatment', 
        'Medication', 'Counselling', 'Medication_and_Counselling']]
y = df['Percentage']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the RandomForestRegressor model with parallel computing
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}\n")

# Save the mean squared error to a file
with open(os.path.join(output_dir, 'mse.txt'), 'w') as f:
    f.write(f"Mean Squared Error: {mse}\n")

# Perform Grid Search for hyperparameter tuning with parallel processing
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# Use KFold with a sufficient number of splits to avoid small fold issues
kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=RandomForestRegressor(n_jobs=-1, random_state=42), param_grid=param_grid, cv=kf, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Model Parameters: {grid_search.best_params_}")

# Save the best model parameters to a file
with open(os.path.join(output_dir, 'best_model_params.txt'), 'w') as f:
    f.write(f"Best Model Parameters: {grid_search.best_params_}\n")

# Cross-validation with parallel processing
scores = cross_val_score(best_model, X, y, cv=kf, n_jobs=-1)
print(f"Cross-Validation Scores: {scores}")

# Save the cross-validation scores to a file
with open(os.path.join(output_dir, 'cv_scores.txt'), 'w') as f:
    f.write(f"Cross-Validation Scores: {scores}\n")

# Predict the percentage for each province
df['Predicted_Percentage'] = best_model.predict(X)
print(df[['Province', 'Percentage', 'Predicted_Percentage']])

# Save the dataframe with predictions to a CSV file
df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

# Visualization
fig, ax = plt.subplots(figsize=(14, 6))

# Line Plot
df_sorted = df.sort_values(by='Province')
ax.plot(df_sorted['Province'], df_sorted['Percentage'], marker='o', label='Actual Percentage', color='blue')
ax.plot(df_sorted['Province'], df_sorted['Predicted_Percentage'], marker='o', linestyle='--', label='Predicted Percentage', color='green')
ax.set_xlabel('Province', fontsize=12)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('\nPPD Predictions (Actual vs Predicted Percentages)', fontsize=16, fontweight='bold')
ax.legend(loc='best')
ax.tick_params(axis='x', rotation=45)

# Save the plot to a file
plt.savefig(os.path.join(output_dir, 'prediction_plot.png'))

## Pie Chart per province
# Prepare the data for plotting
pie_data = df.melt(id_vars=['Province'], 
                   value_vars=['Percentage', 'Predicted_Percentage'], 
                   var_name='Type', 
                   value_name='Value')

# Number of provinces
num_provinces = len(df['Province'])

# Define colors and explode settings
colors = ['blue', 'green']
explode = (0.1, 0.1)  # Explode both slices slightly

# Create subplots (2 rows and 5 columns)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))

# Flatten axes array for easy iteration
axes = axes.flatten()

# Plot each province in a separate subplot
for i, province in enumerate(df['Province']):
    province_data = pie_data[pie_data['Province'] == province]
    
    # Create a pie chart with exploded slices and customized text properties
    wedges, texts, autotexts = axes[i].pie(province_data['Value'], 
                                           colors=colors, 
                                           explode=explode, 
                                           autopct='%1.1f%%',
                                           textprops=dict(fontweight='bold', fontsize=12))
    
    # Change the color of the percentages for blue slices to white
    for autotext, wedge in zip(autotexts, wedges):
        if wedge.get_facecolor() == (0.0, 0.0, 1.0, 1.0):  # Check if the color is blue
            autotext.set_color('white')
    
    # Set the title of the subplot with increased font size and bold text
    axes[i].set_title(province, fontsize=14, fontweight='bold')

# Add a single legend for the entire figure
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
labels = ['Actual Percentage', 'Predicted Percentage']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=12, frameon=False)

# Set the overall title with increased font size and bold text
plt.suptitle('PPD Predictions per Province (Actual vs Predicted Percentages)', fontsize=20, fontweight='bold', y=1.05)
plt.tight_layout()

# Save the plot to a file
output_plot = os.path.join(output_dir, 'ppd_predictions_pie.png')
plt.savefig(output_plot)