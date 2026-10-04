# furniture_prediction

Furniture Prediction is a machine learning project that preprocesses furniture data, trains and evaluates multiple regression models, and predicts prices for new furniture inputs.

## Features
- Data preprocessing with imputation, scaling, and one-hot encoding
- Regression training with model selection (`LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`)
- Evaluation using RMSE, MAE, and R2
- Model persistence with `joblib`
- Simple CLI deployment interface for training and prediction

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train a model
```bash
python app.py train --data furniture.csv --target price --model-output model.joblib
```

## Predict new inputs
Single record:
```bash
python app.py predict --model model.joblib --input-json '{"material":"wood","width":70,"height":45,"depth":40,"brand":"A"}'
```

Multiple records:
```bash
python app.py predict --model model.joblib --input-json '[{"material":"wood","width":70,"height":45,"depth":40,"brand":"A"},{"material":"metal","width":60,"height":40,"depth":35,"brand":"B"}]'
```
