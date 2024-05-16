import uvicorn
from utils import CLASSES_DICT
from fastapi import FastAPI, HTTPException
from llm.sentiment_analyzer import predict_sentiment
from embeddings_classifier.inference import predict_sentiment as predict_sentiment_linear
from embeddings_classifier.embeddings_generator import cohere_embeddings


app = FastAPI()


@app.get("/predict/")
async def predict(text: str, mode: str = 'linear'):
    try:
        if mode == 'linear':
            embeddings = cohere_embeddings([text])
            result = predict_sentiment_linear(embeddings[0])
            result = CLASSES_DICT[result]
            return {"sentiment": result}
        else:
            result = predict_sentiment(text, mode)
            result = CLASSES_DICT[result]
            return {"sentiment": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server.__main__:app", host="0.0.0.0", port=8000, reload=True)
