# =============================
# 1. IMPORT LIBRARIES & READ DATA
# =============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn and ensemble libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel

# Boosting libraries
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Optional: for stacking using mlens (if you prefer)
# from mlens.ensemble import StackingCVClassifier

# Read data (update file paths as needed)
train_data = pd.read_csv('data/train.csv')
test_data  = pd.read_csv('data/test.csv')
submission_data = pd.read_csv('data/sample_submission.csv')

# =============================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================

df = train_data.copy()

# Uncomment to see summary and info:
# print(df.describe())
# print(df.info())

# Plot target distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Transported', palette='viridis')
plt.title('Distribution of Target Variable: Transported')
plt.xlabel('Transported')
plt.ylabel('Count')
plt.show()

# Missing values analysis (optional)
missing_percent = (df.isnull().sum() / len(df)) * 100
plt.figure(figsize=(10, 6))
missing_percent.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Percentage of Missing Values by Column')
plt.ylabel('Percentage (%)')
plt.xlabel('Columns')
plt.xticks(rotation=45)
plt.show()

# (Other EDA plots such as histograms, boxplots, and pairplots can be inserted here)
# =============================
# 3. DATA PREPROCESSING & FEATURE ENGINEERING
# =============================

# (a) Fill missing numerical values with median:
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for feature in num_features:
    median_val = df[feature].median()
    df[feature].fillna(median_val, inplace=True)

# (b) Fill missing categorical values with mode:
cat_features = ['HomePlanet', 'Destination', 'Cabin']
for feature in cat_features:
    mode_val = df[feature].mode()[0]
    df[feature].fillna(mode_val, inplace=True)

# For 'VIP', fill missing with False:
df['VIP'].fillna(False, inplace=True)

# (c) Feature Engineering for CryoSleep based on expenses:
expense_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalExpenses'] = df[expense_features].sum(axis=1)
df.loc[(df['TotalExpenses'] == 0) & (df['CryoSleep'].isnull()), 'CryoSleep'] = True
df.loc[(df['TotalExpenses'] > 0) & (df['CryoSleep'].isnull()), 'CryoSleep'] = False
df.drop(columns=['TotalExpenses'], inplace=True)

# (d) Drop non‐informative columns:
df.drop(columns=['Name'], inplace=True)

# (e) Split 'Cabin' into Deck, Num, and Side:
df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
df.drop('Cabin', axis=1, inplace=True)
df["Num"] = pd.to_numeric(df["Num"], errors="coerce")  # Convert Num to numeric

# (f) Convert categorical values to numeric (mapping):
conv_dict = {
    'HomePlanet': {'Europa': 0, 'Earth': 1, 'Mars': 2},
    'CryoSleep': {False: 0, True: 1},
    'Destination': {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2},
    'VIP': {True: 1, False: 0},
    'Deck': {j: i for i, j in enumerate(sorted(df['Deck'].unique()))},
    'Side': {'P': 0, 'S': 1},
    'Transported': {False: 0, True: 1}
}

df = df.replace(conv_dict)

# (g) Move the target column to the end:
target = 'Transported'
cols = [col for col in df.columns if col != target] + [target]
df = df[cols]

# =============================
# 4. SPLIT DATA INTO TRAINING & TESTING SETS
# =============================
X = df.drop(target, axis=1)
y = df[target]

# If PassengerId is not informative, we can remove it from training features
# (but we will use it later for test submissions)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# For modeling, remove PassengerId from features if it exists:
X_train_model = X_train.drop('PassengerId', axis=1, errors='ignore')
X_test_model  = X_test.drop('PassengerId', axis=1, errors='ignore')

# =============================
# 5. BASELINE MODELS (for reference)
# =============================

# Logistic Regression (baseline)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_model, y_train)
y_pred_lr = lr_model.predict(X_test_model)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))

# Random Forest (baseline)
rf_model = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
rf_model.fit(X_train_model, y_train)
y_pred_rf = rf_model.predict(X_test_model)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# =============================
# 6. ENSEMBLE METHODS: THREE OPTIONS
# =============================

# Prepare training data for ensemble models (exclude PassengerId if exists)
X_train_ens = X_train.drop('PassengerId', axis=1, errors='ignore')
X_test_ens  = X_test.drop('PassengerId', axis=1, errors='ignore')

# Define individual models with (tuned) hyperparameters:
rf_ens = RandomForestClassifier(class_weight=None, n_estimators=82, max_depth=15,
                                min_samples_split=5, min_samples_leaf=5, random_state=42)

lgb_ens = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=12,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    class_weight='balanced'
)

xgb_ens = XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=12,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

models = [
    ('rf', rf_ens),
    ('lgb', lgb_ens),
    ('xgb', xgb_ens)
]

# ---------
# Option 1: Weighted Averaging of Probabilities
# ---------
print("\n--- Option 1: Weighted Averaging Ensemble ---")
pred_probas = []
for name, model in models:
    print(f"Training {name}...")
    model.fit(X_train_ens, y_train)
    proba = model.predict_proba(X_test_ens)
    pred_probas.append(proba)

# Ensure that the weights list matches the number of models (3 in this case).
weights = [0.3, 0.3, 0.4]  # You can adjust these based on performance
weighted_pred_proba = sum(w * p for w, p in zip(weights, pred_probas))
ensemble_preds_weighted = (weighted_pred_proba[:, 1] >= 0.5).astype(int)

print('Weighted Ensemble Accuracy:', accuracy_score(y_test, ensemble_preds_weighted))
print('Weighted Ensemble Classification Report:\n', classification_report(y_test, ensemble_preds_weighted))

# ---------
# Option 2: VotingClassifier (Soft Voting)
# ---------
print("\n--- Option 2: VotingClassifier Ensemble ---")
voting_ensemble = VotingClassifier(estimators=models, voting='soft', weights=[0.3, 0.3, 0.4])
voting_ensemble.fit(X_train_ens, y_train)
voting_preds = voting_ensemble.predict(X_test_ens)

print('Voting Classifier Accuracy:', accuracy_score(y_test, voting_preds))
print('Voting Classifier Classification Report:\n', classification_report(y_test, voting_preds))

# ---------
# Option 3: StackingClassifier
# ---------
print("\n--- Option 3: StackingClassifier Ensemble ---")
stacking_ensemble = StackingClassifier(
    estimators=models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=True  # optionally include original features for the meta-model
)
stacking_ensemble.fit(X_train_ens, y_train)
stacking_preds = stacking_ensemble.predict(X_test_ens)

print('Stacking Ensemble Accuracy:', accuracy_score(y_test, stacking_preds))
print('Stacking Ensemble Classification Report:\n', classification_report(y_test, stacking_preds))

# =============================
# 7. PIPELINE INTEGRATION (Optional)
# =============================

# Define numerical and categorical columns for the pipeline:
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'CryoSleep', 'VIP']

# Pipeline for numerical features: impute and scale.
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features: impute and one-hot encode.
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformations:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ]
)

# Build a full pipeline with the VotingClassifier as the final estimator.
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', VotingClassifier(estimators=models, voting='soft', weights=[0.3, 0.3, 0.4]))
])

# Fit pipeline on the full training set (using X_train with PassengerId if present)
full_pipeline.fit(X_train, y_train)
pipeline_preds = full_pipeline.predict(X_test)

print("\n--- Pipeline Ensemble Performance ---")
print("Pipeline Accuracy:", accuracy_score(y_test, pipeline_preds))
print("Pipeline Classification Report:\n", classification_report(y_test, pipeline_preds))

# # =============================
# # 8. PREPARING TEST SET FOR SUBMISSION (if needed)
# # =============================
#
# # Process the test set in the same way as the train set:
# # (This is a simplified version; make sure to mirror the preprocessing done on the training data)
#
# test_df = test_data.copy()
#
# # Impute missing numerical values:
# for feature in num_features:
#     median_val = test_df[feature].median()
#     test_df[feature].fillna(median_val, inplace=True)
#
# # Impute missing categorical values:
# for feature in cat_features:
#     mode_val = test_df[feature].mode()[0]
#     test_df[feature].fillna(mode_val, inplace=True)
#
# # For 'VIP'
# test_df['VIP'].fillna(False, inplace=True)
#
# # Feature Engineering for CryoSleep:
# test_df['TotalExpenses'] = test_df[expense_features].sum(axis=1)
# test_df.loc[(test_df['TotalExpenses'] == 0) & (test_df['CryoSleep'].isnull()), 'CryoSleep'] = True
# test_df.loc[(test_df['TotalExpenses'] > 0) & (test_df['CryoSleep'].isnull()), 'CryoSleep'] = False
# test_df.drop(columns=['TotalExpenses'], inplace=True)
#
# # Drop Name
# test_df.drop(columns=['Name'], inplace=True)
#
# # Split Cabin into Deck, Num, Side:
# test_df[['Deck', 'Num', 'Side']] = test_df['Cabin'].str.split('/', expand=True)
# test_df.drop('Cabin', axis=1, inplace=True)
# test_df["Num"] = pd.to_numeric(test_df["Num"], errors="coerce")
#
# # Replace categorical values with numeric mapping (similar to training data):
# test_conv_dict = {
#     'HomePlanet': {'Europa': 0, 'Earth': 1, 'Mars': 2},
#     'CryoSleep': {False: 0, True: 1},
#     'Destination': {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2},
#     'VIP': {True: 1, False: 0},
#     'Deck': {j: i for i, j in enumerate(sorted(test_df['Deck'].unique()))},
#     'Side': {'P': 0, 'S': 1}
# }
#
# test_df = test_df.replace(test_conv_dict)
#
# # Use the pipeline to predict:
# submission_preds = full_pipeline.predict(test_df)
# submission = pd.DataFrame({
#     'PassengerId': test_df['PassengerId'],
#     'Transported': submission_preds
# })
# submission.replace({'Transported': {0: False, 1: True}}, inplace=True)
#
# # Save the submission file:
# submission.to_csv('submission.csv', index=False)
# print("\nSubmission file saved as 'submission.csv'.")
