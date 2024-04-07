import logging
import os
import pandas as pd
from cryptography.fernet import Fernet

from regulations_rag.data import Data

from regulations_rag.section_reference_checker import SectionReferenceChecker
from regulations_rag.embeddings import get_closest_nodes, num_tokens_from_string
from regulations_rag.reg_tools import get_regulation_detail
from regulations_rag.rerank import RerankAlgos, rerank

# Create a logger for this module
logger = logging.getLogger(__name__)
DEV_LEVEL = 15
logging.addLevelName(DEV_LEVEL, 'DEV')       

class DataInDataFrames(Data):
    """
    A class to handle and provide relevant sections, definitions, and workflow from the Currency and Exchange Manuals
    based on the user's content using embeddings and thresholds for similarity.
    """    
    def __init__(self, user_type, regulation_name, section_reference_checker, df_regulations, df_definitions, df_index, df_workflow):
        """
        Parameters:
        -----------
        user_type : str
            Used in the LLM system prompt to tell the model who they are assisting
        regulation_name : str
            Used in the LLM system prompt to tell the model what document they will be answering questions on
        section_reference_checker : Section_Reference_Checker
            Used to check and extract section references that will be used to request the regulation extracts
        df_regulations : DataFrame
            DataFrame containing regulation content so we can find sections.
        df_definitions : DataFrame
            DataFrame containing definitions from the manual. It must have columns ["definition", "embedding"]
        df_index : DataFrame
            DataFrame containing index information from the manual. If must have columns ["section_reference", "embedding"]
        df_workflow : DataFrame
            DataFrame containing workflow information for the chat so we can change from RAG mode into something else
        """
        self.regulations = df_regulations
        self.definitions = df_definitions
        self.workflow = df_workflow
        self.index = df_index
        super().__init__(user_type = user_type, regulation_name = regulation_name, section_reference_checker = section_reference_checker)

    def _set_required_column_headings(self):
        '''
        These are the minimum required column headings and should be used in the _check*_column_headings() methods. 
        Feel free to override them if you need to have additional information in any step
        '''
        super()._set_required_column_headings()
        self.index_columns = ["section_reference", "embedding", "text", "source"] # I want more info for logging / debugging



    def _get_definitions_column_headings(self):
        if not self.definitions.empty:
            return self.definitions.columns.to_list()
        return []

    def _get_index_column_headings(self):
        if not self.index.empty:
            return self.index.columns.to_list()
        return []

    def _get_workflow_column_headings(self):
        if not self.workflow.empty:
            return self.index.workflow.to_list()
        return []

    def _get_regulations_column_headings(self):
        if not self.regulations.empty:
            return self.regulations.columns.to_list()
        return []

    def get_regulation_detail(self, section_index):
        raise NotImplementedError()
    


    def get_relevant_definitions(self, user_content_embedding, threshold):
        """
        Retrieves definitions close to the given user content embedding.

        Parameters:
        -----------
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant definitions.

        Returns:
        --------
        DataFrame
            A DataFrame with definitions close to the user content embedding.
        """
        relevant_definitions = get_closest_nodes(self.definitions, embedding_column_name = "embedding", content_embedding = user_content_embedding, threshold = threshold)

        if not relevant_definitions.empty:
            logger.log(DEV_LEVEL, "--   Relevant Definitions")
            for index, row in relevant_definitions.iterrows():
                logger.log(DEV_LEVEL, f'{row["cosine_distance"]:.4f}: {row["definition"]}')
        else:
            logger.log(DEV_LEVEL, "--   No relevant definitions found")

        return relevant_definitions

    # def filter_relevant_sections_boilerplate(self, relevant_sections):
    #     # # Check it is not empty before you get here
    #     # columns_to_check = ["section_reference", "text", "source", "cosine_distance"]
    #     # if not all(column in relevant_sections.columns for column in columns_to_check):
    #     #     raise AttributeError("The data received in the relevant_sections input does not have the correct columns")

    #     max_search_items = 15
    #     relevant_sections = relevant_sections.nsmallest(max_search_items, 'cosine_distance')

    #     logger.log(DEV_LEVEL, "--   top results from simple search")
    #     for index, row in relevant_sections.iterrows():
    #         logger.log(DEV_LEVEL, f'{row["cosine_distance"]:.4f}: {row["section_reference"]:>20}: {row["source"]:>15}: {row["text"]}')

    #     return relevant_sections


#     def filter_relevant_sections_llm(self, openai_client, model_to_use, user_question, relevant_sections):
#         if relevant_sections.empty:
#             return pd.DataFrame([], columns = ["section_reference", "text", "source", "cosine_distance", "count"])

#         relevant_sections = self.filter_relevant_sections_boilerplate(relevant_sections=relevant_sections)

#         list_of_sections_and_text = []
#         for index, row in relevant_sections.iterrows():
#             list_of_sections_and_text.append(f"{row['section_reference']}: {row['text']}")
#         string_of_sections_and_text = "\n".join(list_of_sections_and_text)

#         short_pattern = r"[A-Z]\.\d{0,2}\([A-Z]\)\(\b(?:i|ii|iii|iv|v|vi)\b\)\([a-z]\)\([a-z]{2}\)\(\d+\)"
#         system_content = "You are assisting and Autorised Dealer to answer questions from the Currency and Exchange Control Manual for Authorised Dealers (CEMAD). \
# You will be given the users question followed by a list of sections from CEMAD along with the section's heading, summary or a questions which is answered in section. \
# Your job is to choose which sections are likely to contain an answer and output that list as a pipe delimited list. Do not respond with any other text. Just the pipe \
# delimited list paying attention to ensure the format of the section reference is {short_pattern} with no following text."
#         user_content = f"### Question: {user_question}### CEMAD Sections with heading, summary or a question that is answered in the section: \n{string_of_sections_and_text}"

#         messages = []
#         messages.append({"role": "system", "content": system_content})
#         messages.append({"role": "user", "content": user_content})

#         try:
#             response = openai_client.chat.completions.create(
#                                         model=model_to_use,
#                                         temperature=0,
#                                         max_tokens=500,
#                                         messages=messages
#                                     )
#             response_text = response.choices[0].message.content
#         except APIConnectionError as e:
#             print("API connection error")


#         relevant_section_references = [item.strip() for item in response_text.split('|')]    

#         checked_list_of_strings = match_strings_to_reference_list(list_of_strings = relevant_section_references, reference_list_of_strings = relevant_sections['section_reference'].to_list())

#         logger.log(DEV_LEVEL, "--   results requested by LLM filter")
#         for section in checked_list_of_strings:
#             logger.log(DEV_LEVEL, f'{section}')

#         # Initialize a list to hold the rows for the subset
#         subset_rows = []

#         # Iterate over the list of strings
#         for string in checked_list_of_strings:
#             # Filter the DataFrame for rows matching the current string
#             matches = relevant_sections[relevant_sections['section_reference'].eq(string)]
            
#             # If there are matches, find the one with the minimum 'cosine_distance'
#             if not matches.empty:
#                 best_match = matches.loc[matches['cosine_distance'].idxmin()]
#                 # Add the best match row to the list
#                 subset_rows.append(best_match)


#         # Create a DataFrame from the list of best match rows
#         subset_df = pd.DataFrame(subset_rows).reset_index(drop=True)

#         if len(checked_list_of_strings) != len(subset_df):
#             logger.log(DEV_LEVEL, f'Not all requested sections could be matched. The ones that could are:')
#             for index, row in subset_df.iterrows():
#                 logger.log(DEV_LEVEL,f'{row["section_reference"]}')


#         return subset_df



    def cap_rag_section_token_length(self, relevant_sections, capped_number_of_tokens):
        #relevant_sections["section_reference"] = relevant_sections["section_reference"]
        relevant_sections["regulation_text"] = relevant_sections["section_reference"].apply(
            lambda x: get_regulation_detail(x, self.regulations, self.section_reference_checker)
        )        
        relevant_sections["token_count"] = relevant_sections["regulation_text"].apply(num_tokens_from_string)

        # Initialize the cumulative sum and the counter 'n'
        cumulative_sum = 0
        counter = 0
        n = 0

        # Correct loop to apply the specific logic
        for index, row in relevant_sections.iterrows():
            next_cumulative_sum = cumulative_sum + row["token_count"]
            # Condition to check before exceeding the cap
            if next_cumulative_sum > capped_number_of_tokens:
                n = counter  # Correct 'n' for 1-based index as per user specification
                break
            else:
                cumulative_sum = next_cumulative_sum
            counter += 1

        # Apply boundary conditions
        if n == 0:  # If 'n' has not been updated, check the boundary conditions
            if relevant_sections["token_count"].iloc[0] > capped_number_of_tokens:
                n = 1
            else:
                n = len(relevant_sections)  # Set 'n' to the total length if cap never exceeded

        if n != len(relevant_sections):
            logger.log(DEV_LEVEL, f"--   Token capping reduced the number of reference sections from {len(relevant_sections)} to {n}")

        final_row = min(n, 5)
        top_subset_df = relevant_sections.nsmallest(final_row, 'cosine_distance').reset_index(drop=True)

        return top_subset_df



    def get_relevant_sections(self, user_content, user_content_embedding, threshold, rerank_algo = RerankAlgos.NONE):
        """
        Retrieves sections close to the given user content embedding.

        Parameters:
        -----------
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant sections.

        Returns:
        --------
        DataFrame
            A DataFrame with sections close to the user content embedding. This method also adds the content of the manual
            to the DataFrame in the column "regulation_text"
        """
        relevant_sections = get_closest_nodes(self.index, embedding_column_name = "embedding", content_embedding = user_content_embedding, threshold = threshold) 
        n = rerank_algo.params["initial_section_number_cap"]
        relevant_sections = relevant_sections.nsmallest(n, 'cosine_distance')      
        logger.log(DEV_LEVEL, f"Selecting the top {n} items based on cosine-similarity score")
        for index, row in relevant_sections.iterrows():
            logger.log(DEV_LEVEL, f'{row["cosine_distance"]:.4f}: {row["section_reference"]:>20}: {row["source"]:>15}: {row["text"]}')

        if not relevant_sections.empty:
            logger.log(DEV_LEVEL, "--   Relevant sections found")
            rerank_algo.params["user_question"] = user_content
            rerank(relevant_sections=relevant_sections, rerank_algo=rerank_algo)        
            relevant_sections = self.cap_rag_section_token_length(relevant_sections, rerank_algo.params["final_token_cap"])

        else:
            logger.log(DEV_LEVEL, "--   No relevant sections found")
            relevant_sections = pd.DataFrame([], columns = ["section_reference", "cosine_distance", "count", "regulation_text", "token_count"])

        return relevant_sections


    def get_relevant_workflow(self, user_content_embedding, threshold):
        """
        Retrieves workflow steps close to the given user content embedding if available.

        Parameters:
        -----------
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant workflow steps.

        Returns:
        --------
        DataFrame
            A DataFrame with workflow steps close to the user content embedding.
            Returns an empty DataFrame if no workflow information is available.
        """
        if len(self.workflow) > 0:
            return get_closest_nodes(self.workflow, embedding_column_name = "embedding", content_embedding = user_content_embedding, threshold = threshold)
        else:
            return pd.DataFrame([], columns = ["workflow", "text", "embedding"])

def load_csv_data(path_to_file):
    """
    Loads data from a CSV file, ensuring no NaN values are present.

    Parameters:
    -----------
    path_to_file : str
        The path to the CSV file to be loaded.

    Returns:
    --------
    df : DataFrame
        The loaded DataFrame if the file exists and contains no NaN values.

    Raises:
    -------
    FileNotFoundError:
        If the specified file does not exist.
    ValueError:
        If the loaded DataFrame contains NaN values.
    """
    if not os.path.exists(path_to_file):
        msg = f"Could not find the file {path_to_file}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(path_to_file, sep="|", encoding="utf-8", na_filter=False)  

    # Check for NaN values in the DataFrame
    if df.isna().any().any():
        msg = f'Encountered NaN values while loading {path_to_file}. This will cause ugly issues with the get_regulation_detail method'
        logger.error(msg)
        raise ValueError(msg)
    return df

def append_csv_data(path_to_file, original_df):
    if path_to_file == "":
        return original_df

    tmp = load_csv_data(path_to_file)
    # data in the "_plus.csv" file contains an additional column "sections_referenced" which is only used to identify the rows that need to be updated when the manual changes
    if "sections_referenced" in tmp.columns:
        tmp.drop("sections_referenced", axis=1, inplace=True)

    return pd.concat([original_df, tmp], ignore_index = True)


def load_parquet_data(path_to_file, decryption_key = ""):
    if not os.path.exists(path_to_file):
        msg = f"Could not find the file {path_to_file}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_parquet(path_to_file, engine='pyarrow')
    if decryption_key:
        fernet = Fernet(decryption_key)
        df['text'] = df['text'].apply(lambda x: fernet.decrypt(x.encode()).decode())
    return df

def save_parquet_data(df, path_to_file, decryption_key = ""):
    if not os.path.exists(path_to_file):
        msg = f"Could not find the file {path_to_file}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if decryption_key:
        fernet = Fernet(decryption_key)
        df['text'] = df['text'].apply(lambda x: fernet.encrypt(x.encode()).decode())
    df.to_parquet(path_to_file, engine = 'pyarrow')
    # but leave the column unchanged in the input df so the user can continue to use it
    if decryption_key:
        df['text'] = df['text'].apply(lambda x: fernet.decrypt(x.encode()).decode())


def append_parquet_data(path_to_file, original_df, decryption_key = ""):
    if path_to_file == "":
        return original_df

    tmp = load_parquet_data(path_to_file, decryption_key = "")

    return pd.concat([original_df, tmp], ignore_index = True)


def load_data_from_files(
              path_to_manual_as_csv_file, path_to_additional_manual_as_csv_file, 
              path_to_definitions_as_parquet_file, path_to_additional_definitions_as_parquet_file,
              path_to_index_as_parquet_file, path_to_additional_index_as_parquet_file,
              workflow_as_parquet_file = "",
              decryption_key = ""):

    df_regulations = load_csv_data(path_to_manual_as_csv_file)
    df_regulations = append_csv_data(path_to_additional_manual_as_csv_file, df_regulations)

    if path_to_definitions_as_parquet_file == "":
        df_definitions = pd.DataFrame([],columns = ["definition", "embedding", "source"])    
    df_definitions = load_parquet_data(path_to_definitions_as_parquet_file)
    df_definitions = append_parquet_data(path_to_additional_definitions_as_parquet_file, df_definitions)

    # index is the data that is encrypted
    df_index = load_parquet_data(path_to_index_as_parquet_file, decryption_key)
    df_index = append_parquet_data(path_to_additional_index_as_parquet_file, df_index, decryption_key)

    if workflow_as_parquet_file == "":
        df_workflow = pd.DataFrame([],columns = ["section_reference", "text", "source", "embedding"])    
    else:
        df_workflow = load_parquet_data(workflow_as_parquet_file)

    return df_regulations, df_definitions, df_index, df_workflow


class EmbeddingParameters:
    def __init__(self, embedding_model, embedding_dimensions):
        self.model = embedding_model
        self.dimensions = embedding_dimensions

        if embedding_model == "text-embedding-ada-002":
            self.threshold = 0.15
            self.dimensions = 1536 # this model does not 
            self.folder = "/ada_v2" # The folder variable is used for testing 
        elif embedding_model == "text-embedding-3-large":
            if embedding_dimensions == 1024:
                self.threshold = 0.38
                self.folder = "/v3_large/1024"
            elif embedding_dimensions == 3072:
                self.threshold = 0.40
                self.folder = "/v3_large/3072"
        else:
            raise ValueError("Unknown Embedding model or embedding dimension")


class ChatParameters:
    def __init__(self, chat_model, temperature, max_tokens):
        self.model = chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.tested_models = ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]
        untested_models = ["gpt-4-1106-preview", "gpt-4-0125-preview"]
        if self.model not in self.tested_models:
            if self.model not in untested_models:
                raise ValueError("You are attempting to use a model that does not seem to exist")
            else: 
                logger.info("You are attempting to use a model that has not been tested")

def load_embedding_parameters(embedding_model, embedding_dimensions):
    return EmbeddingParameters(embedding_model, embedding_dimensions)

def load_chat_parameters(model_to_use, temperature, max_tokens):
    return ChatParameters(model_to_use, temperature, max_tokens)

