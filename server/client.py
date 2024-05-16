import requests


def call_predict_endpoint(text, mode='claude', host='localhost', port=8000):
    url = f"http://{host}:{port}/predict/"
    params = {'text': text, 'mode': mode}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()['sentiment']
    else:
        raise ValueError(f"Failed to call predict endpoint: {response.text}")


if __name__ == "__main__":
    text = "This is a great movie!"
    mode = "claude"
    sentiment = call_predict_endpoint(text, mode)
    print(f"Sentiment: {sentiment}")
