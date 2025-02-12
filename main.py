import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import MinMaxScaler
from skrebate import ReliefF
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from typing import Optional, Tuple
from dataclasses import dataclass

matplotlib.use('Agg')


@dataclass
class PipelineConfig:
    train_path: str
    test_path: Optional[str] = None
    submission_path: Optional[str] = None
    random_state: int = 42
    display_plots: bool = True
    test_size: float = 0.2
    scale: bool = False


class DataProcessor:
    """Handles data loading, EDA, and preprocessing"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.train_data = None
        self.test_data = None
        self.scaler = MinMaxScaler() if config.scale else None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and return raw datasets"""
        self.train_data = pd.read_csv(self.config.train_path)
        if self.config.test_path:
            self.test_data = pd.read_csv(self.config.test_path)
        return self.train_data, self.test_data

    def run_eda(self, df: pd.DataFrame) -> None:
        """Perform exploratory data analysis with visualizations"""
        eda_df = self._create_eda_features(df.copy())

        self._plot_target_distribution(eda_df)
        self._plot_missing_values(eda_df)
        self._plot_numerical_distributions(eda_df)
        self._plot_categorical_distributions(eda_df)
        self._plot_correlation_matrix(eda_df)
        self._plot_expense_analysis(eda_df)

    @staticmethod
    def _create_eda_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create temporary features for EDA"""
        expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df['TotalExpenses'] = df[expense_cols].sum(axis=1)
        return df

    def preprocess_data(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Clean and transform data with feature engineering"""
        df = df.copy()
        df = self._extract_features(df)
        df = self._encode_categoricals(df)
        df = self._impute_missing_values(df, is_train)
        df = self._feature_augmentation(df)

        if self.config.scale:
            df = self._scale_features(df, is_train)

        return df.drop(columns=['PassengerId', 'Cabin', 'Name'], errors='ignore')

    @staticmethod
    def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing columns"""
        # Split Cabin into components
        df[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = (
            df['Cabin'].str.split('/', expand=True)
        )

        # Split PassengerId into components
        df[['Group', 'Group_Id']] = df['PassengerId'].str.split('_', expand=True)

        # Create TotalExpenses feature
        expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df['TotalExpenses'] = df[expense_cols].sum(axis=1)

        return df

    @staticmethod
    def _impute_missing_values(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Handle missing data"""

        # Convert data types
        df['Cabin_Num'] = df['Cabin_Num'].astype(float)
        df['Group_Id'] = df['Group_Id'].astype(float)
        df['Group'] = df['Group'].astype(float)

        # Numerical columns
        num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
                    'Spa', 'VRDeck', 'Cabin_Num', 'Group_Id']
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # Categorical columns
        cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Group']
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # for training data drop 500 data points that we don't need
        if is_train:
            df_temp = df[df['CryoSleep'] == 1.0]
            df = df.drop(df_temp[df_temp['Transported'] == 0].index)

        # Adjust CryoSleep based on spending
        spending_cols = ['FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']

        # For passengers marked as cryosleep, ensure missing spending values become 0.
        cryo_mask = df['CryoSleep'] == 1.0
        df.loc[cryo_mask, spending_cols] = df.loc[cryo_mask, spending_cols].fillna(0)

        # Compute total spending across the five columns.
        spending_sum = df[spending_cols].sum(axis=1)

        # Reassign CryoSleep based on spending:
        # If total spending is 0, set CryoSleep to 1.0; otherwise, set it to 0.0.
        df.loc[spending_sum == 0, 'CryoSleep'] = 1.0
        df.loc[spending_sum != 0, 'CryoSleep'] = 0.0

        return df

    @staticmethod
    def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical features to numerical"""
        mappings = {
            'HomePlanet': {'Europa': 0, 'Earth': 1, 'Mars': 2},
            'CryoSleep': {False: 0, True: 1},
            'Destination': {'TRAPPIST-1e': 0, "55 Cancri e": 1, 'PSO J318.5-22': 2},
            'VIP': {False: 0, True: 1},
            'Cabin_Side': {'P': 0, 'S': 1},
            'Cabin_Deck': {'A': 0, 'B': 1, 'C': 2,
                           'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}
        }
        return df.replace(mappings)

    def _scale_features(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Scale numerical features"""
        num_cols = ['Age', 'RoomService', 'FoodCourt',
                    'ShoppingMall', 'Spa', 'VRDeck',
                    'Cabin_Num', 'Group', 'TotalExpenses']

        if is_train:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])

        return df

    @staticmethod
    def _feature_augmentation(df):
        df['HasPaid'] = (df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']) > 0
        df['Paid'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
        df['HasPaid_RoomService'] = (df['RoomService']) > 0
        df['HasPaid_FoodCourt'] = (df['FoodCourt']) > 0
        df['HasPaid_ShoppingMall'] = (df['ShoppingMall']) > 0
        df['HasPaid_Spa'] = (df['Spa']) > 0
        df['HasPaid_VRDeck'] = (df['VRDeck']) > 0
        df['IsAdult'] = (df['Age']) > 17

        df['HasPaid'] = df['HasPaid'].astype(int)
        df['HasPaid_RoomService'] = df['HasPaid_RoomService'].astype(int)
        df['HasPaid_FoodCourt'] = df['HasPaid_FoodCourt'].astype(int)
        df['HasPaid_ShoppingMall'] = df['HasPaid_ShoppingMall'].astype(int)
        df['HasPaid_Spa'] = df['HasPaid_Spa'].astype(int)
        df['HasPaid_VRDeck'] = df['HasPaid_VRDeck'].astype(int)
        df['IsAdult'] = df['IsAdult'].astype(int)

        return df

    # EDA visualization methods
    def _plot_target_distribution(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='Transported', hue='Transported',
                      palette='viridis', legend=False)
        plt.title('Target Variable Distribution')
        if self.config.display_plots:
            plt.show()

    def _plot_missing_values(self, df: pd.DataFrame) -> None:
        """Plot percentage of missing values per column"""
        missing = (df.isnull().sum() / len(df)) * 100
        missing = missing[missing > 0].sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing.values, y=missing.index, hue=missing.values, palette='viridis', legend=False)
        plt.title('Percentage of Missing Values by Column')
        plt.xlabel('Percentage Missing')
        plt.ylabel('Features')
        if self.config.display_plots:
            plt.show()

    def _plot_numerical_distributions(self, df: pd.DataFrame) -> None:
        """Plot histograms for numerical features"""
        numerical = ['Age', 'RoomService', 'FoodCourt',
                     'ShoppingMall', 'Spa', 'VRDeck']

        df[numerical].hist(bins=30, figsize=(12, 10), layout=(3, 2),
                           color='skyblue', edgecolor='black')
        plt.suptitle('Numerical Features Distribution')
        plt.tight_layout()
        if self.config.display_plots:
            plt.show()

    def _plot_categorical_distributions(self, df: pd.DataFrame) -> None:
        """Plot count plots for categorical features"""
        categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(categorical, 1):
            plt.subplot(2, 2, i)
            sns.countplot(data=df, x=feature, hue=feature, palette='viridis',
                          order=df[feature].value_counts().index, legend=False)
            plt.title(f'{feature} Distribution')
            plt.xticks(rotation=45)
        plt.tight_layout()
        if self.config.display_plots:
            plt.show()

    def _plot_correlation_matrix(self, df: pd.DataFrame) -> None:
        """Plot correlation heatmap for numerical features"""
        numerical = ['Age', 'RoomService', 'FoodCourt',
                     'ShoppingMall', 'Spa', 'VRDeck', 'TotalExpenses']

        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numerical].corr(), annot=True, cmap='coolwarm',
                    fmt=".2f", square=True, cbar=True)
        plt.title('Numerical Features Correlation Matrix')
        if self.config.display_plots:
            plt.show()

    def _plot_expense_analysis(self, df: pd.DataFrame) -> None:
        """Plot expense analysis by CryoSleep status"""
        expense_cols = ['RoomService', 'FoodCourt',
                        'ShoppingMall', 'Spa', 'VRDeck']

        plt.figure(figsize=(12, 6))
        for i, col in enumerate(expense_cols, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(data=df, x='CryoSleep', hue='CryoSleep', y=col, palette='coolwarm', legend=False)
            plt.title(f'{col} by CryoSleep')
        plt.tight_layout()
        if self.config.display_plots:
            plt.show()


class ModelTrainer:
    """Handles feature selection, model training, and optimization"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            'catboost': CatBoostClassifier(verbose=0),
            'lightgdm': LGBMClassifier(n_estimators=40, learning_rate=0.1, max_depth=15),
        }
        self.feature_selector = None
        self.selected_features = None
        self.best_model = None

    def feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Perform feature selection using ReliefF"""
        # if ReliefF is None:
        #     print("skrebate not available. Skipping feature selection.")
        #     return X

        # self.feature_selector = ReliefF(n_neighbors=10, n_features_to_select=10)
        # self.feature_selector.fit(X.values, y.values)
        # self.selected_features = X.columns[self.feature_selector.top_features_]
        # return X[self.selected_features]

        return X

    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train specified model with cross-validation"""
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        model.fit(X, y)
        return {
            'model': model,
            'cv_scores': scores,
            'mean_score': np.mean(scores)
        }

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Optimize Random Forest parameters using Optuna"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
            }
            model = RandomForestClassifier(**params, random_state=self.config.random_state)
            return cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        self.best_model = RandomForestClassifier(**study.best_params)
        return study.best_params

    @staticmethod
    def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Generate comprehensive evaluation metrics"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }


class MLPipeline:
    """Orchestrates end-to-end machine learning workflow"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.model_trainer = ModelTrainer(config)
        self.best_params = None
        self.full_model = None

    def run(self, full_train: bool = False):
        """Execute complete pipeline with option for full training"""
        # Load and preprocess data
        train_df, test_df = self.data_processor.load_data()
        self.data_processor.run_eda(train_df)

        if full_train:
            # Use entire dataset for final training
            processed_full = self.data_processor.preprocess_data(train_df)
            X_full = processed_full.drop('Transported', axis=1)
            y_full = processed_full['Transported'].astype(int)

            # Train on full dataset with best params
            if self.best_params is None:
                self._optimize_hyperparameters(X_full, y_full)

            self.full_model = RandomForestClassifier(**self.best_params)
            self.full_model.fit(X_full, y_full)

            return {'final_model': self.full_model}
        else:
            # Original split-based training for evaluation
            processed_df = self.data_processor.preprocess_data(train_df)
            self._prepare_splits(processed_df)
            self._train_and_evaluate(processed_df)  # Pass processed data here

            model_results = {}
            for model_name in ['logistic_regression', 'random_forest', 'catboost', 'lightgdm']:
                result = self.model_trainer.train_model(model_name, self.X_train, self.y_train)
                evaluation = self.model_trainer.evaluate_model(result['model'], self.X_test, self.y_test)
                model_results[model_name] = {**result, **evaluation}

            # Hyperparameter optimization
            best_params = self.model_trainer.optimize_hyperparameters(self.X_train, self.y_train)
            print(f"Best parameters from optimization: {best_params}")

            return model_results


    def create_submission(self, output_path: str = 'submission.csv'):
        """Create submission using full training data and all features"""
        if self.full_model is None:
            self.run(full_train=True)

        # Load and preprocess test data
        _, test_df = self.data_processor.load_data()
        processed_test = self.data_processor.preprocess_data(test_df, is_train=False)

        # Generate predictions (using all features)
        predictions = self.full_model.predict(processed_test)

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Transported': predictions.astype(bool)
        })

        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission_df

    def _train_and_evaluate(self, processed_df: pd.DataFrame):
        """Original training logic with splits"""
        # Prepare data splits (already done in run())
        # Feature selection
        X_train_selected = self.model_trainer.feature_selection(self.X_train, self.y_train)

        # Hyperparameter optimization
        self.best_params = self.model_trainer.optimize_hyperparameters(X_train_selected, self.y_train)

        # Store best params for later use
        self.model_trainer.best_model = RandomForestClassifier(**self.best_params)
        self.model_trainer.best_model.fit(X_train_selected, self.y_train)

    def _prepare_splits(self, df: pd.DataFrame):
        """Create train/test splits"""
        X = df.drop('Transported', axis=1)
        y = df['Transported'].astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series):
        """Optimize on full dataset"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
            }
            model = RandomForestClassifier(**params)
            return cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        self.best_params = study.best_params


if __name__ == "__main__":
    # Configuration
    config = PipelineConfig(
        train_path='data/train.csv',
        test_path='data/test.csv',
        submission_path='data/sample_submission.csv',
        display_plots=False,
        scale=True
    )

    # Execute pipeline
    pipeline = MLPipeline(config)
    results = pipeline.run()

    # Display results
    for model_name, metrics in results.items():
        print(f"\n=== {model_name.upper()} ===")
        print(f"Mean CV Accuracy: {metrics['mean_score']:.4f}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(metrics['classification_report'])

    # Access best model
    final_model = pipeline.model_trainer.best_model
    print("Best model trained:", final_model)

    # Create submission using full data
    # print("\nCreating submission with full training data...")
    # pipeline.create_submission()
