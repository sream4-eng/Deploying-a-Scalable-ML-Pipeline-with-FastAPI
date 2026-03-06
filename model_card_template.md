# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
In this project,  I created a supervised machine learning classification model that predicts whether an individual's income is above $50K per year based on demographic and employment-related attributes, utilizing data from the U.S. Census dataset. I used the RandomForestClassifier from the scikit-learn library to implement this model. The project includes functionalities for data preprocessing, model training, evaluation, slice-based performance measurement, unit testing, and deployment of the model through a FastAPI REST API.

## Intended Use
The intended use of this model is educational. I designed it to demonstrate best practices for building, testing, and deploying machine learning models using modern MLOps workflows. The model should not be used for real-world decision making such as employment screening, lending decisions, insurance evaluation, or other high-stakes applications.

## Training Data
I trained the model using the provided Census Income dataset (`census.csv`). The dataset included the  following demographic and employment variables:

- age
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- hours-per-week
- native-country

The target variable was `salary`. This variable classified whether income was `<=50K` or `>50K`.

The data was split into training and test datasets using a train-test split before training the model.

## Evaluation Data
For the evaluation, I used the test section of the train-test-split dataset. This section was not utilized during model training; it was held back to test the model's performance.


## Metrics
The following classification metrics were used:

- Precision
- Recall
- F1 Score

The test set model performance was:

- Precision: **0.7419**
- Recall: **0.6384**
- F1 Score: **0.6863**

Model performance was evaluated across categorical slices of the dataset. The slice-based results were added to `slice_output.txt`.

## Ethical Considerations
This dataset was based upon sensitive demographic traits such as race, sex, marital status, and native country. Therefore, it may gain bias from historical data. When used in real-world applications, they might lead to unfair results for some populations.

This model is intended for demonstration and it should not be used in production situations without adequate bias analysis and fairness assessment.

## Caveats and Recommendations
This model was created for learning and has a few limitations:
	- A single dataset was employed to train the model and it may not be general to other populations.
	- The data sample may not reflect current conditions
	- Model performance may differ across categories which could reflect concern about fairness.

Another possible option would be to adjust hyperparameter settings, fairness analysis, and cross-validation.

Future improvements that could be considered are expansion of the dataset, feature engineering, and fairness-aware machine learning.
