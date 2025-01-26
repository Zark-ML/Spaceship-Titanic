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

To replicate this analysis, ensure you have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn