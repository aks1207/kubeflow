import argparse
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
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

def train(X, y, model_path):
    """
    Train XGBoost model on Boston Housing dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the input data')
    parser.add_argument('--model_path', type=str, help='Path to save the trained model')
    args = parser.parse_args()

    X, y = load_data(args.data_path)
    train(X, y, args.model_path)
