#import openai
import tiktoken

from json import loads
from scipy.spatial import distance
import pandas as pd

class EmbeddingParameters:
    def __init__(self, embedding_model, embedding_dimensions):
        self.model = embedding_model
        self.dimensions = embedding_dimensions

        if embedding_model == "text-embedding-ada-002":
            self.threshold = 0.15
            self.threshold_definitions = 0.20
            self.dimensions = 1536 # this model does not 
        elif embedding_model == "text-embedding-3-large":
            if embedding_dimensions == 1024:
                self.threshold = 0.38
                self.threshold_definitions = 0.45
            elif embedding_dimensions == 3072:
                self.threshold = 0.40
                self.threshold_definitions = 0.45
        else:
            raise ValueError("Unknown Embedding model or embedding dimension")


def num_tokens_from_string(string: str, encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")) -> int:
    if pd.isna(string):
        return 0
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
# see https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        #print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        #print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        #print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        #print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_ada_embedding(openai_client, text, model="text-embedding-ada-002", dimensions = 1024):
    if model == "text-embedding-ada-002":
        return openai_client.embeddings.create(input = [text], model=model).data[0].embedding

    return openai_client.embeddings.create(input = [text], model=model, dimensions=dimensions).data[0].embedding

# def get_ada_embedding_old(text, model="text-embedding-ada-002"):
#    return openai.embeddings.create(input = [text], model=model).data[0].embedding

def get_closest_nodes(df, embedding_column_name, content_embedding, threshold = 0.15):
      #df['cosine_distance'] = df[embedding_column_name].apply(lambda x: distance.cosine(x, content_embedding))
      df.loc[:, 'cosine_distance'] = df[embedding_column_name].apply(lambda x: distance.cosine(x, content_embedding))
      closest_nodes = df[df['cosine_distance'] < threshold].sort_values(by='cosine_distance', ascending=True)
      return closest_nodes