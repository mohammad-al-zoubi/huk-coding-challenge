import cohere
from configuration import config

co = cohere.Client(config['embeddings_classifier']['COHERE_API_KEY'])


def cohere_embeddings(passages, input_type="classification"):
    embeds = co.embed(texts=passages, model=config['embeddings_classifier']['cohere_embeddings_model'],
                      input_type=input_type).embeddings
    return embeds


# TODO: Embed data and save it to desk then train the classifier

if __name__ == "__main__":
    passages = ['hi there', 'how are you']
    embeddings = cohere_embeddings(passages)
    print(embeddings)
