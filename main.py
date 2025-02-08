import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV,
                                     StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, recall_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

import optuna

# Optional: if you have skrebate installed for ReliefF
try:
    from skrebate import ReliefF
except ImportError:
    ReliefF = None


class SpaceshipTitanic:
    def __init__(self, train_path, test_path=None, submission_path=None,
                 random_state=42, display_plots=True):
        """
        Initialize the pipeline with file paths and parameters.

        Parameters:
            train_path (str): Path to the training CSV file.
            test_path (str): (Optional) Path to the test CSV file.
            submission_path (str): (Optional) Path to a sample submission file.
            random_state (int): Random state for reproducibility.
            display_plots (bool): Whether to display plots during EDA.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.submission_path = submission_path
        self.random_state = random_state
        self.display_plots = display_plots

        # Data containers
        self.train_data = None
        self.test_data = None
        self.submission_data = None
        self.df = None  # will hold the processed training dataframe

        # After splitting
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Models & scaler
        self.scaler = None
        self.lr_model = None
        self.rf_model = None
        self.final_model = None

    def load_data(self):
        """Load the CSV files and create a working dataframe."""
        self.train_data = pd.read_csv(self.train_path)
        if self.test_path:
            self.test_data = pd.read_csv(self.test_path)
        if self.submission_path:
            self.submission_data = pd.read_csv(self.submission_path)

        # Work on a copy of the train data
        self.df = self.train_data.copy()
        print("Data loaded successfully.")
        return self.df

    def run_eda(self):
        """Run Exploratory Data Analysis (EDA) with plots and printouts."""
        print("----- Data Description -----")
        print(self.df.describe())
        print("\n----- Data Info -----")
        self.df.info()

        # Plot target variable distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x='Transported', palette='viridis')
        plt.title('Distribution of Target Variable: Transported')
        plt.xlabel('Transported')
        plt.ylabel('Count')
        if self.display_plots:
            plt.show()

        # Missing values analysis
        missing_values = self.df.isnull().sum()
        print("Missing Values:\n", missing_values)
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
        plt.figure(figsize=(10, 6))
        missing_percent.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Percentage of Missing Values by Column')
        plt.ylabel('Percentage (%)')
        plt.xlabel('Columns')
        plt.xticks(rotation=45)
        if self.display_plots:
            plt.show()

        # Numerical features analysis
        numerical_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        self.df[numerical_features].hist(bins=30, figsize=(10, 8), layout=(2, 3),
                                         color='skyblue', edgecolor='black')
        plt.suptitle('Distribution of Numerical Features')
        plt.tight_layout()
        if self.display_plots:
            plt.show()

        # Boxplots for outlier analysis
        plt.figure(figsize=(12, 6))
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(data=self.df, x=feature, palette='coolwarm')
            plt.title(f'{feature} Outlier Analysis')
        plt.tight_layout()
        if self.display_plots:
            plt.show()

        # Categorical features analysis
        categorical_features = ['HomePlanet', 'Destination', 'VIP']
        for feature in categorical_features:
            print(f"Unique values in {feature}: {self.df[feature].unique()}")
        plt.figure(figsize=(12, 8))
        for i, feature in enumerate(categorical_features, 1):
            plt.subplot(2, 2, i)
            sns.countplot(data=self.df, x=feature, palette='viridis',
                          order=self.df[feature].value_counts().index)
            plt.title(f'Distribution of {feature}')
            plt.xticks(rotation=45)
        plt.tight_layout()
        if self.display_plots:
            plt.show()

        # Correlation heatmap for numerical features
        correlation_matrix = self.df[numerical_features].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                    square=True, cbar=True)
        plt.title('Correlation Heatmap for Numerical Features')
        if self.display_plots:
            plt.show()

        # Pairplot with target
        sns.pairplot(self.df, vars=numerical_features, hue='Transported',
                     palette='viridis', diag_kind='kde', corner=True)
        plt.suptitle('Pair Plot of Numerical Features with Transported', y=1.02)
        if self.display_plots:
            plt.show()

        # Feature interaction: RoomService by CryoSleep status
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.df, x='CryoSleep', y='RoomService', palette='viridis')
        plt.title('RoomService Expenses by CryoSleep Status')
        plt.xlabel('CryoSleep')
        plt.ylabel('RoomService Expense')
        if self.display_plots:
            plt.show()

        # Total expenses by CryoSleep status
        self.df['TotalExpenses'] = self.df[['RoomService', 'FoodCourt',
                                            'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.df, x='CryoSleep', y='TotalExpenses', palette='coolwarm')
        plt.title('Total Expenses by CryoSleep Status')
        plt.xlabel('CryoSleep')
        plt.ylabel('Total Expenses')
        if self.display_plots:
            plt.show()

        # Clean up if necessary
        if 'TotalExpenses' in self.df.columns:
            self.df.drop(columns=['TotalExpenses'], inplace=True)

    def preprocess_data(self):
        """
        Perform missing value imputation, feature engineering, and type conversion.
        This includes:
          - Imputing numerical features with median values.
          - Imputing categorical features with mode.
          - Imputing the 'VIP' column with False where missing.
          - Imputing the 'CryoSleep' column based on expense patterns.
          - Extracting 'Group' and 'Group_id' from 'PassengerId'.
          - Splitting the 'Cabin' column into 'Deck', 'Num', and 'Side'.
          - Converting categorical values using a mapping dictionary.
        """
        # Impute numerical features with median values.
        num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for feature in num_features:
            median_value = self.df[feature].median()
            self.df[feature].fillna(median_value, inplace=True)

        # Impute categorical features with mode.
        cat_features = ['HomePlanet', 'Destination', 'Cabin']
        for feature in cat_features:
            mode_value = self.df[feature].mode()[0]
            self.df[feature].fillna(mode_value, inplace=True)

        # For 'VIP' column, fill missing values with False.
        self.df['VIP'].fillna(False, inplace=True)

        # Feature Engineering: Impute CryoSleep based on expense patterns.
        expense_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        self.df['TotalExpenses'] = self.df[expense_features].sum(axis=1)
        # If TotalExpenses is zero and CryoSleep is missing, assume CryoSleep = True.
        self.df.loc[(self.df['TotalExpenses'] == 0) & (self.df['CryoSleep'].isnull()), 'CryoSleep'] = True
        # If TotalExpenses > 0 and CryoSleep is missing, assume CryoSleep = False.
        self.df.loc[(self.df['TotalExpenses'] > 0) & (self.df['CryoSleep'].isnull()), 'CryoSleep'] = False
        # Drop the temporary TotalExpenses column.
        self.df.drop(columns=['TotalExpenses'], inplace=True)

        # Process PassengerId to extract Group and Group_id (if available)
        if 'PassengerId' in self.df.columns:
            self.df["Group"] = self.df['PassengerId'].map(lambda x: x.split('_')[0])
            self.df["Group_id"] = self.df['PassengerId'].map(lambda x: x.split('_')[1])
            self.df.drop(columns=['PassengerId'], inplace=True)

        # Split the Cabin column into 'Deck', 'Num', and 'Side'
        self.df[['Deck', 'Num', 'Side']] = self.df['Cabin'].str.split('/', expand=True)
        self.df.drop('Cabin', axis=1, inplace=True)

        # Convert categorical/text values using a conversion dictionary.
        conv_dict = {
            'HomePlanet': {'Europa': 0, 'Earth': 1, 'Mars': 2},
            'CryoSleep': {False: 0, True: 1},
            'Destination': {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2},
            'VIP': {True: 1, False: 0},
            'Deck': {j: i for i, j in enumerate(sorted(self.df['Deck'].unique()))},
            'Side': {'P': 0, 'S': 1},
            'Transported': {False: 0, True: 1}
        }
        self.df.replace(conv_dict, inplace=True)

        # Move the target column 'Transported' to the end.
        cols = [col for col in self.df.columns if col != 'Transported'] + ['Transported']
        self.df = self.df[cols]

        # Convert 'Group', 'Group_id', and 'Num' to integers.
        self.df['Group'] = self.df['Group'].astype(int)
        self.df['Group_id'] = self.df['Group_id'].astype(int)
        self.df['Num'] = self.df['Num'].astype(int)

        print("Preprocessing completed. Missing values after imputation:")
        print(self.df.isnull().sum())
        return self.df

    def feature_selection_reliefF(self, n_features_to_select=3, n_neighbors=10):
        """
        Use ReliefF (from skrebate, if available) to select the top features.
        Alternatively, you could use the manual implementation.
        """
        if ReliefF is None:
            print("skrebate is not installed. Skipping ReliefF feature selection.")
            return None

        X_array = self.df.drop('Transported', axis=1).values
        y_array = self.df['Transported'].values

        relieff = ReliefF(n_neighbors=n_neighbors, n_features_to_select=n_features_to_select)
        relieff.fit(X_array, y_array)

        selected_features = self.df.drop('Transported', axis=1).columns[relieff.top_features_]
        print("Selected Features based on ReliefF:")
        print(selected_features)
        return selected_features

    @staticmethod
    def manual_reliefF(X, y, k=10):
        """
        A manual implementation of ReliefF feature importance.
        Returns an array of feature weights.
        """
        n_samples, n_features = X.shape
        feature_weights = np.zeros(n_features)
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        for i in range(n_samples):
            distances, indices = knn.kneighbors([X.iloc[i]])
            hit_indices = [idx for idx in indices[0] if y.iloc[idx] == y.iloc[i]]
            miss_indices = [idx for idx in indices[0] if y.iloc[idx] != y.iloc[i]]
            for feature in range(n_features):
                hit_diff = np.sum((X.iloc[hit_indices, feature] - X.iloc[i, feature]) ** 2)
                miss_diff = np.sum((X.iloc[miss_indices, feature] - X.iloc[i, feature]) ** 2)
                feature_weights[feature] += hit_diff - miss_diff
        feature_weights /= n_samples
        return feature_weights

    def split_data(self, test_size=0.33, scale=False):
        """
        Split the processed dataframe into training and testing sets.
        Optionally scale the features using MinMaxScaler.
        """
        self.X = self.df.drop('Transported', axis=1)
        self.y = self.df['Transported']

        if scale:
            self.scaler = MinMaxScaler()
            self.X = self.scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state)
        print("Data splitting completed.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_logistic_regression(self, **kwargs):
        """
        Train a Logistic Regression model using the training set.
        Additional keyword arguments are passed to the scikit-learn model.
        """
        self.lr_model = LogisticRegression(random_state=self.random_state, **kwargs)
        self.lr_model.fit(self.X_train, self.y_train)
        y_pred = self.lr_model.predict(self.X_test)
        print("----- Logistic Regression -----")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        return self.lr_model

    def train_random_forest(self, **kwargs):
        """
        Train a Random Forest Classifier using the training set.
        Additional keyword arguments are passed to the scikit-learn model.
        """
        self.rf_model = RandomForestClassifier(random_state=self.random_state, **kwargs)
        self.rf_model.fit(self.X_train, self.y_train)
        y_pred = self.rf_model.predict(self.X_test)
        print("----- Random Forest -----")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        return self.rf_model

    def plot_roc_curve(self, model=None):
        """
        Plot the ROC curve and display the AUC.
        By default, uses the Random Forest model if available.
        """
        if model is None:
            model = self.rf_model
        if not hasattr(model, 'predict_proba'):
            print("The provided model does not have predict_proba.")
            return

        y_probs = model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_probs)
        auc_score = roc_auc_score(self.y_test, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        if self.display_plots:
            plt.show()

    def cross_validation(self, model, cv=5, scoring='accuracy'):
        """
        Perform cross-validation and print the scores.
        """
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring)
        print(f"Cross-Validation scores ({scoring}): {scores}")
        print(f"Mean {scoring}: {scores.mean():.4f}")
        return scores

    def grid_search_cv(self, param_grid, cv=5):
        """
        Perform Grid Search Cross-Validation on a Random Forest Classifier.

        Parameters:
            param_grid (dict): The hyperparameter grid.
            cv (int): Number of cross-validation folds.

        Returns:
            The best estimator found.
        """
        grid_search = GridSearchCV(RandomForestClassifier(random_state=self.random_state),
                                   param_grid, cv=cv, verbose=10)
        grid_search.fit(self.X_train, self.y_train)
        self.best_model = grid_search.best_estimator_
        print("Best Parameters from Grid Search:", grid_search.best_params_)
        return self.best_model

    def optuna_optimization(self, n_trials=100):
        """
        Use Optuna to optimize hyperparameters for a Random Forest.

        Parameters:
            n_trials (int): Number of trials to run.

        Returns:
            Best hyperparameters found.
        """

        def objective(trial):
            class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                random_state=self.random_state
            )
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            return accuracy_score(self.y_test, y_pred)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        print("Best hyperparameters from Optuna:", study.best_params)
        print("Best accuracy:", study.best_value)
        return study.best_params

    def final_model_training(self, model_params, scale=False):
        """
        Retrain a final Random Forest model using the chosen parameters.
        This method re-splits the data (and optionally scales it) before training.

        Parameters:
            model_params (dict): Hyperparameters for the final model.
            scale (bool): Whether to scale features with MinMaxScaler.
        """
        self.X = self.df.drop('Transported', axis=1)
        self.y = self.df['Transported']
        if scale:
            self.scaler = MinMaxScaler()
            self.X = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=self.random_state)
        self.final_model = RandomForestClassifier(random_state=self.random_state, **model_params)
        self.final_model.fit(self.X_train, self.y_train)
        y_pred = self.final_model.predict(self.X_test)
        print("----- Final Random Forest Model -----")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        return self.final_model

    def predict(self, X_new):
        """
        Predict on new data using the final model.
        If scaling was used, the same scaler is applied.

        Parameters:
            X_new (DataFrame or array): New input data.

        Returns:
            Predictions from the final model.
        """
        if self.scaler is not None:
            X_new = self.scaler.transform(X_new)
        return self.final_model.predict(X_new)

    def run_pipeline(self, scale=False, grid_search_params=None,
                     optuna_trials=100, run_models=True):
        """
        A convenience method to run all steps of the pipeline.
        Adjust this method to call only the parts you need.

        Parameters:
            scale (bool): Whether to scale features before splitting.
            grid_search_params (dict): Parameter grid for GridSearchCV (if any).
            optuna_trials (int): Number of trials for Optuna optimization.
            run_models (bool): If True, run the logistic regression and random forest training.
        """
        self.load_data()
        self.run_eda()
        self.preprocess_data()
        self.split_data(scale=scale)

        if run_models:
            # Train basic models.
            self.train_logistic_regression()
            self.train_random_forest()

            # (Optional) Grid search tuning.
            if grid_search_params is not None:
                self.grid_search_cv(grid_search_params)

            # Hyperparameter optimization with Optuna.
            best_params = self.optuna_optimization(n_trials=optuna_trials)
            print("You can now use 'best_params' to retrain your final model.")

        print("Pipeline run complete.")


if __name__ == "__main__":
    # Set paths to your CSV files (update as needed)
    train_csv = 'data/train.csv'
    test_csv = 'data/test.csv'
    submission_csv = 'data/sample_submission.csv'

    # Initialize the pipeline with the desired parameters
    pipeline = SpaceshipTitanic(train_path=train_csv,
                                          test_path=test_csv,
                                          submission_path=submission_csv,
                                          random_state=42,
                                          display_plots=True)

    # Run the complete pipeline (this will load data, perform EDA, preprocess, and train models)
    pipeline.run_pipeline(scale=True, grid_search_params={
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'class_weight': ['balanced', 'balanced_subsample', None],
    }, optuna_trials=50, run_models=True)

    # For final model training, for example, using best parameters from grid search or optuna:
    final_params = {'class_weight': None, 'n_estimators': 78,
                    'max_depth': 13, 'min_samples_split': 10, 'min_samples_leaf': 4}
    final_model = pipeline.final_model_training(final_params, scale=True)

    # Plot ROC curve for the final model (or any other model)
    pipeline.plot_roc_curve(model=final_model)

    # (Optional) Predict on new data (make sure new data is preprocessed the same way)
    # new_data = pd.read_csv('data/new_data.csv')
    # predictions = pipeline.predict(new_data)
    # print(predictions)
