import argparse
import pandas as pd
from xgboost import XGBRegressor
import joblib

def load_data(data_path):
    """
    Load Boston Housing dataset from CSV file.
    """
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def predict(X, model_path, predictions_path):
    """
    Load trained model and make predictions on the input data.
    """
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    df = pd.DataFrame(y_pred, columns=['Predicted Price'])
    df.to_csv(predictions_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the input data')
    parser.add_argument('--model_path', type=str, help='Path to the trained model')
    parser.add_argument('--predictions_path', type=str, help='Path to save the predictions')
    args = parser.parse_args()

    X, y = load_data(args.data_path)
    predict(X, args.model_path, args.predictions_path)
