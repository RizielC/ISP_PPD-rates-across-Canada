##Predicting Post-Partum Depression (PPD) Rates in Canadian Provinces

**Project Overview**
This project explores and predicts Post-Partum Depression (PPD) rates across Canadian provinces using a RandomForestRegressor model. The analysis leverages demographic and medical data from historical datasets, with model optimization achieved through Grid Search and cross-validation. The results are visualized to compare actual and predicted PPD rates, providing insights into maternal mental health trends across the country.

**Table of Contents**
1. Introduction
2. Background
3. Dataset
4. Data Preparation
5. Analysis Method
6. High-Performance Computing (HPC)
7. Results
8. Discussion
9. Conclusion
10. Future Directions

**Introduction**
Post-Partum Depression (PPD) is a significant maternal mental health issue that can profoundly affect both mothers and their infants. This project aims to accurately predict PPD prevalence across Canadian provinces using demographic, medical, and psychological data. By identifying patterns and predictors of PPD, healthcare providers can better allocate resources and tailor interventions to those most in need.

**Background**
PPD is characterized by prolonged feelings of sadness, anxiety, and fatigue that can last for several months and require professional treatment. Early identification and treatment are crucial to improving maternal and child health outcomes. This project uses data from the Survey on Maternal Health (SMH) and the Public Health Agency of Canada, which provide comprehensive information on new mothers.

**Dataset**
The dataset includes detailed demographic, medical, and psychological data on new mothers across Canadian provinces:

Demographics: Age groups, marital status, education levels.
Medical History: History of depression or mood disorders.
Psychological Assessments: Self-reported mental and physical health status.
Support Systems: Availability of social support, participation in parenting support programs.

Sources:
Mean age of mothers at Time of Delivery (Live Births)
Maternal Mental Health
Mental Health and Access to Care Survey

**Data Preparation**
Data Cleaning: Handled missing values and corrected discrepancies.
Feature Encoding: Converted categorical variables to numerical values.
Normalization: Standardized numerical features to improve model performance.

**Analysis Method**
Model Selection: Chose RandomForestRegressor for its ability to handle complex, non-linear relationships and robustness with various feature types and missing values.
Hyperparameter Tuning: Used Grid Search with cross-validation to optimize model hyperparameters (n_estimators, max_depth, min_samples_split).

**High-Performance Computing (HPC)**
Job Script Execution: Created a bash script to automate environment setup, package installation, and script execution on an HPC cluster.
Parallel Processing: Leveraged HPC resources to parallelize Grid Search and cross-validation, reducing computation time.
Output Management: Directed output files (predictions, plots, model parameters) to a specific directory for organized storage and easy access.

**Results**
The RandomForestRegressor model’s predictions were reasonably accurate for some provinces but less accurate for others. The model's performance is evaluated using Mean Squared Error (MSE): 4.377376191043089. Visualizations highlight discrepancies and areas for improvement.

**Discussion**
Despite some challenges, the project demonstrated the feasibility of using machine learning to predict PPD prevalence. Limitations include limited data specifically for mothers, time constraints, and potential unmeasured factors influencing PPD.

**Conclusion**
The RandomForestRegressor model effectively predicted the prevalence of PPD across Canadian provinces. The project shows the potential of machine learning in predicting PPD prevalence based on various factors, aiding in early identification and targeted intervention.

**Future Directions**
Expand Dataset: Incorporate additional variables such as socioeconomic status, cultural background, and access to healthcare services.
Increase Sample Size: Collaborate with more regions and collect longitudinal data.
Explore Other Models: Experiment with different machine learning algorithms or ensemble methods to improve predictive accuracy and interpretability.

