import json
import pandas as pd
import traceback
from azureml.core.model import Model
import joblib

model = None

def init():
    global model
    try:
        model_path = Model.get_model_path("houses-price-predictor-model")
        try:
            from pycaret.regression import load_model as py_load_model
            model = py_load_model(model_path)
            return
        except Exception:
            # fallback to joblib/pickle for plain pickles
            pass
        model = joblib.load(model_path)
    except Exception:
        print("Model load failed:")
        traceback.print_exc()
        raise


def run(raw_data):
    try:
        data = pd.DataFrame(json.loads(raw_data))
        # If model is a PyCaret pipeline/estimator, use predict_model
        try:
            from pycaret.regression import predict_model as py_predict
            preds_df = py_predict(model, data=data)
            # convert prediction column(s) to JSON
            return preds_df.to_json(orient="records")
        except Exception:
            # fallback to scikit-learn style predict
            preds = model.predict(data)
            return json.dumps({"predictions": preds.tolist()})
    except Exception:
        print("Scoring failed:")
        traceback.print_exc()
        return json.dumps({"error": "scoring failed, check logs"})
