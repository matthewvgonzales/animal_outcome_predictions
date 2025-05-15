# Predicting Euthanasia Outcomes at the Austin Animal Center

This project builds on my earlier work in the [Austin Animal Shelter]([(https://github.com/matthewvgonzales/austin_animal_shelter)]) repository, where I cleaned and prepared raw data from [data.gov](https://catalog.data.gov/dataset/austin-animal-center-outcomes) for analysis and modeling. This follow-up focuses on developing a machine learning model to predict whether an animal is likely to be euthanized.

## Project Goal

The primary objective of this project is to develop a machine learning classifier that can identify animals at high risk of euthanasia. The goal is not to replace human judgment, but to assist animal shelter staff in early intervention and decision-making.

To accomplish this, I prioritized **recall** — ensuring that the model captures as many euthanasia cases as possible, even if it means accepting some false positives.

---

## Data Source

The original dataset was obtained from [data.gov](https://catalog.data.gov/dataset/austin-animal-center-outcomes), which contains detailed records from the Austin Animal Center. I performed data cleaning and preprocessing in my related project: [Austin Animal Shelter](https://github.com/matthewvgonzales/austin_animal_shelter).

The dataset includes animal characteristics (e.g., breed, color, intake type) and outcome details.

---

## Summary of Results

- **Recall (Class 1 - Euthanized):** 0.64  
- **Precision:** 0.58  
- **F1 Score:** 0.61  
- **Accuracy:** 95% (heavily influenced by the majority class)

The final model correctly identified 914 euthanasia cases, missed 511, and returned 653 false positives.

---

## Key Features and Methods

- **Pipeline-based model** using `ColumnTransformer` and `OneHotEncoder`
- **XGBoost classifier** selected via `GridSearchCV` with scoring optimized for recall
- **Categorical feature handling** with `handle_unknown='ignore'` to manage high-cardinality and unseen test values
- Emphasis on **model interpretability** and **reproducibility**

---

## Handling High Cardinality

Features such as `Breed` and `Color` contain hundreds of unique values, many of which appear only once. These were handled using `OneHotEncoder(handle_unknown='ignore')` to prevent model errors when unseen categories appear in test data. This approach avoids overfitting and supports generalization across new data.

---

## Practical Use

This model is intended to be used as a **decision-support tool** within animal shelters. It can help:

- Flag high-risk animals for further review
- Prioritize behavioral interventions or medical assessments
- Inform foster or adoption outreach strategies

Because the model favors **recall**, it is designed to catch as many euthanasia-risk cases as possible, even if it occasionally raises false alarms.

---

## Future Work

- Explore advanced feature engineering (e.g., grouping rare breeds)
- Apply sampling techniques (e.g., SMOTE, class weighting)
- Tune classification threshold to improve balance between recall and precision
- Deploy a prototype API or dashboard for use by shelter staff

---

## Author

Matthew V. Gonzales  


---

## License

This project is open-source under the MIT License.
