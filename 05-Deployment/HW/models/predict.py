import pickle
import os



record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}


def load_pipeline():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(BASE_DIR, 'pipeline_v1.bin')

    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    return pipeline


def predict_lead(record):
    pipeline = load_pipeline()
    prob = pipeline.predict_proba(record)[0, 1]
    return round(prob, 4)


print(predict_lead(record))