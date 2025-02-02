# Spaceship Titanic Data Analysis Project

## Project Overview

This project analyzes the [Spaceship Titanic dataset](https://www.kaggle.com/competitions/spaceship-titanic) from Kaggle, aiming to predict which passengers were transported to an alternate dimension during the spaceship's collision with a spacetime anomaly. The analysis includes exploratory data analysis (EDA), data preprocessing, feature engineering, and model building to achieve accurate predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Dataset

The dataset consists of passenger information with the following features:

- `PassengerId`: Unique identifier for each passenger.
- `HomePlanet`: The planet the passenger departed from.
- `CryoSleep`: Indicates if the passenger elected for cryogenic sleep during the voyage.
- `Cabin`: Cabin number where the passenger is staying.
- `Destination`: The intended destination of the passenger.
- `Age`: The age of the passenger.
- `VIP`: Whether the passenger has VIP status.
- `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`: Expenses incurred by the passenger in various amenities.
- `Name`: The full name of the passenger.
- `Transported`: Target variable indicating if the passenger was transported to an alternate dimension.

For more details, refer to the [Spaceship Titanic competition page](https://www.kaggle.com/competitions/spaceship-titanic).

## Installation

To replicate this analysis, ensure you have Python installed. You can install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, the key libraries used in this project include:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Exploratory Data Analysis

Initial exploratory analysis includes:

- Checking for missing values
- Visualizing feature distributions
- Identifying correlations between features and the target variable
- Handling categorical variables and outliers

## Data Preprocessing and Feature Engineering

The preprocessing steps include:

- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Feature selection and engineering

## Modeling

The model-building process includes:

- Splitting the data into training and testing sets
- Trying different classification models (e.g., Random Forest, Logistic Regression, XGBoost)
- Hyperparameter tuning using Grid Search
- Evaluating models using accuracy, recall, precision, and AUC scores

## Results

Key evaluation metrics used to measure model performance:

- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix
- ROC-AUC Score

## Conclusion

The final model aims to achieve the best balance between precision and recall to accurately classify transported passengers. Future improvements may include advanced feature engineering and trying deep learning approaches.

## References

- [Spaceship Titanic Kaggle Competition](https://www.kaggle.com/competitions/spaceship-titanic)
- scikit-learn documentation
- seaborn and matplotlib documentation

