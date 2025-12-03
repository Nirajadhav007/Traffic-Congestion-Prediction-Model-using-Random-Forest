# Traffic-Congestion-Prediction-Model-using-Random-Forest

Machine Learning model using Random Forest to predict traffic volume and classify congestion levels (Low/Medium/High).

## Overview  
Predict hourly traffic volume and classify congestion levels (Low / Medium / High) using machine learning — based on time, weather, and recent traffic history.

## Motivation  
- Traffic congestion is a major issue during rush hour and festivals (e.g. Diwali).  
- With limited infrastructure and growing population, forecasting traffic helps in better traffic management, route planning, and alert systems.  
- This project uses historical highway data (hourly) to build a predictive model — no complex external sensors or real-time feeds required.  

## Dataset  
- Source: Metro Interstate Traffic Volume dataset.  
- Key features: `date_time`, `traffic_volume`, weather data (`temp`, `rain_1h`, `clouds_all`, ...), plus derived features.  
- Preprocessing: datetime parsing, sorting, missing value imputation, lag & rolling features, holiday / weekend flags.
- Model: RandomForestRegressor
- Evaluation: MAE, RMSE, R², confusion matrix, accuracy

## What the Model Does  
- **Regression**: Predicts `traffic_volume` (vehicles/hour) for a given hour.  
- **Classification**: Converts predicted volume to congestion category (Low / Medium / High) for easier interpretation.  


Output Images of the Regression Model

- ![Actual vs Predicted Traffic](https://github.com/user-attachments/assets/5f8a4310-4dff-4252-b20f-618f7b428662)
- ![Classification Report including Accuray + Confusion Matrix ](https://github.com/user-attachments/assets/078a5b01-4c75-42ec-a970-cc6c2e5229b2)

  

