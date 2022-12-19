from fastapi import FastAPI, Request
from src.process import process

app = FastAPI()


@app.get("/")
async def root():
    """
    Return status on root GET for healthchecks
    """
    return {"status": "running"}


@app.post("/predict")
async def predict(request: Request):
    """
    Get transcript highlight prediction with timestamps
    :param Request request: POST request
    :return: dictionary with highlight text and timestamps
    """
    payload_json = await request.json()
    return process(payload_json)
