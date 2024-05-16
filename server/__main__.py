import uvicorn
from utils import CLASSES_DICT
from fastapi import FastAPI, HTTPException
from llm.sentiment_analyzer import predict_sentiment


app = FastAPI()


@app.get("/predict/")
async def predict(text: str, mode: str = 'claude'):
    try:
        result = predict_sentiment(text, mode)
        result = CLASSES_DICT[result]
        return {"sentiment": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server.__main__:app", host="0.0.0.0", port=8000, reload=True)
