# Machine Learning project for the estimation of cellphone values
## Introduction
In an increasingly competitive mobile phone market, pricing strategies are crucial for manufacturers and retailers to attract consumers while ensuring profitability. Accurately predicting the appropriate price range for mobile phones based on their specifications can significantly enhance market competitiveness and customer satisfaction. This project aims to develop a predictive model that can determine the price range of mobile phones using various device specifications.
## Problem Statement
Determining the optimal price range for mobile phones is challenging due to the vast array of features and specifications that can influence a consumer's purchasing decision. Incorrect pricing can either lead to lost sales opportunities or diminished profit margins. 
## Objectives
1. To analyze the correlation between mobile phone specifications and their market price ranges.
1. To develop a machine learning model that predicts the price range of mobile phones based on their features.
1. To provide a tool that aids manufacturers and retailers in setting competitive and profitable price points for their mobile phone offerings.
## Dataset Overview
The dataset, sourced from Kaggle[1], consists of over 2000 rows, each representing a mobile phone model with 22 features. It includes 6 categorical (boolean) features, each with 2 levels (0 and 1), indicating the presence or absence of features such as Bluetooth, dual SIM support, 4G, 3G, touch screen, and WiFi. The remaining 15 numerical features provide detailed specifications, including battery power, processor speed, internal memory, camera megapixels, pixel resolution, RAM, screen dimensions, and weight. The target variable categorizes mobile phones into one of four price ranges: low, medium-low, medium-high, or high, making this a multi-class classification problem. This dataset offers a comprehensive view of the factors that may influence mobile phone pricing, making it ideal for developing a predictive model for price range classification.
## Methodology
The project will employ supervised machine learning classification techniques to predict the price range of mobile phones. Preliminary analysis will include exploratory data analysis (EDA) to understand the distribution of variables and identify patterns. Data preprocessing steps will include handling missing values, encoding categorical variables, feature scaling etc. Several classification algorithms, such as Decision Trees, Random Forest, Gradient Boosting, SVM, Catboost, Xgboost etc, will be evaluated and fine tuned to determine the most effective model based on accuracy, precision, recall and other metrics.The machine learning model will be integrated into a web-based interface.
## Significance
By accurately predicting the price range of mobile phones based on specifications, this project will enable companies to adopt data-driven pricing strategies. This approach can lead to increased sales, customer satisfaction, and competitive advantage in the mobile phone market.
## Running the code 
To generate the pickle file to store the model you first need to setup your python environment. Please run 

``` sh
pip install -r requirements.txt
```

Now to generate the pickle you run 

``` sh
python src/train_models.py
```
This will train all 7 models and pick the best one to store. 

When using the pickle make sure to import the `train_models.py` file to access all the methods of the `CellPhoneModel`. 
An example of reading the pickle file is provided in the notebook `read_pickle.py`
## Running the Deployment
1.    ```sh
      pip install -r requirements.txt 
   found in the Mobile Price Deployment folder

2. Run the Flask application in the Mobile Price Deployment folder :
   ```sh
    flask run app.py 
Authors: Iyoha Peace Osamuyi and  Gabriel Octavio Lozano Pinzón
