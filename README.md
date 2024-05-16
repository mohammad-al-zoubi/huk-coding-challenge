# Sentiment Analysis Coding Challenge - HUK Coburg

To build the docker image:

`docker build -t sentiment_analysis:latest`


To start the sentiment analysis server:

`docker run -p 8000:8000 sentiment_analysis`


To call the service:

```python3
from server.client import call_predict_endpoint

text = "This is a great movie!"
mode = "claude"
sentiment = call_predict_endpoint(text, mode)
print(f"Sentiment: {sentiment}")
```