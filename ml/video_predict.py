import pandas as pd
import joblib

def predict_video_fertility():

    model = joblib.load(
        "backend/models/fertility_xgboost_model.pkl"
    )

    df = pd.read_csv(
        "outputs/sperm_metrics.csv"
    )

    prediction = model.predict(df)[0]

    return prediction