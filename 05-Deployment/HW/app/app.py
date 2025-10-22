from fastapi import FastAPI
from pydantic import BaseModel, Field
from models.predict import predict_lead

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float = Field(..., ge=0)

app = FastAPI()

@app.get("/")
async def greet():
    return "Welcome"


@app.post("/predict")
async def predict(client: Client):
    prob = predict_lead(client.model_dump())

    return {"Subscription Probability": prob}


