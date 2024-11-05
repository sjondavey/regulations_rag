import os
import pandas as pd
from openai import OpenAI
from regulations_rag.corpus_index import DataFrameCorpusIndex
from .navigating_corpus import NavigatingCorpus
from regulations_rag.embeddings import get_ada_embedding

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

embedding_path = "./test/inputs/"
output_files = ["navigation_dfns.parquet", "navigation_index.parquet", "navigation_workflow.parquet"]

# Definitions DataFrame
columns_in_dfns = ["document", "section_reference", "text", "definition"]
dfns_as_list = [
    ["Plett", "A.1(A)", "What is the gym?", "The Gym: The Health and Fitness Center on Piesang Valley Road"],
    ["Plett", "A.1(A)", "What is the Robberg Nature Reserve?", "The Robberg Nature Reserve: The Cape Nature park at the end of the Robberg Peninsula"],
]
df_dfns = pd.DataFrame(dfns_as_list, columns=columns_in_dfns)

# Index DataFrame
columns_in_index = ["document", "section_reference", "source", "text"]
index_as_list = [
    ["WRR", "1.1", "question", "How do I get to West Gate?"],
    ["WRR", "1.2", "question", "How do I get to the Main Gate?"],
    ["WRR", "1.3", "question", "How do I get to South Gate?"],
    ["Plett", "A.2(A)", "question", "How do I get to the gym?"],
    ["Plett", "A.2(B)", "question", "How do I get to Robberg Nature Reserve?"],
]
df_index = pd.DataFrame(index_as_list, columns=columns_in_index)

# Workflow DataFrame
columns_in_workflow = ["workflow", "text"]
workflow_as_list = [
    ["map", "Can you show this on a map?"]
]
df_workflow = pd.DataFrame(workflow_as_list, columns=columns_in_workflow)

def create_embeddings_and_save_files():
    """
    Creates embeddings for definitions, index, and workflow data and saves them as parquet files.
    """
    model = "text-embedding-3-large"
    dimensions = 1024
    increment = 10
    all_data = [df_dfns, df_index, df_workflow]
    
    for j in range(len(all_data)):
        df = all_data[j]
        df['embedding'] = pd.NA  # Initialize the column to hold NA values
        df['embedding'] = df['embedding'].astype(object)  # Ensure the column type is object
        
        for i in range(0, len(df), increment):
            chunk = df.iloc[i:i+increment].copy()
            chunk["embedding"] = chunk["text"].apply(lambda x: get_ada_embedding(openai_client, x, model, dimensions))
            df.loc[chunk.index, "embedding"] = chunk["embedding"]
        
        print(f'Writing {embedding_path + output_files[j]}')
        df.to_parquet(embedding_path + output_files[j], engine='pyarrow')

class NavigatingIndex(DataFrameCorpusIndex):
    """
    A class for handling the navigating corpus index.
    """
    def __init__(self):
        corpus = NavigatingCorpus()
        user_type = "a Visitor"
        corpus_description = "the Simplest way to Navigate Plett"

        definitions = pd.read_parquet(embedding_path + output_files[0], engine='pyarrow')
        index = pd.read_parquet(embedding_path + output_files[1], engine='pyarrow')
        workflow = pd.read_parquet(embedding_path + output_files[2], engine='pyarrow')

        super().__init__(user_type, corpus_description, corpus, definitions, index, workflow)

# Uncomment these methods if needed
# def get_relevant_definitions(self, user_content, user_content_embedding, threshold):
# def cap_rag_section_token_length(self, relevant_sections, capped_number_of_tokens):
# def get_relevant_sections(self, user_content, user_content_embedding, threshold, rerank_algo=RerankAlgos.NONE):
# def get_relevant_workflow(self, user_content_embedding, threshold):
