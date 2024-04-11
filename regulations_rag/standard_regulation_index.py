import logging
import os
import pandas as pd
from cryptography.fernet import Fernet

from regulations_rag.regulation_index import RegulationIndex

from regulations_rag.reference_checker import ReferenceChecker
from regulations_rag.embeddings import get_closest_nodes, num_tokens_from_string
from regulations_rag.regulation_reader import RegulationReader
from regulations_rag.rerank import RerankAlgos, rerank

# Create a logger for this module
logger = logging.getLogger(__name__)
DEV_LEVEL = 15
logging.addLevelName(DEV_LEVEL, 'DEV')       

# NOTE: There required columns include columns that are only used for logging, There variables are in the class. The are stored as:
# self.index_columns = required_columns_index 
# self.definition_columns = required_columns_definition
# self.workflow_columns = required_columns_workflow
# self.regulation_columns =  required_columns_regulation

required_columns_definition = ["definition", "embedding", "source"]
required_columns_index = ["section_reference", "text", "source", "embedding"]
required_columns_workflow = ["workflow", "text", "embedding"]

# These are the outputs from the 
required_columns_section_lookup = ["section_reference", "cosine_distance", "count", "regulation_text", "token_count"]

class StandardRegulationIndex(RegulationIndex):
    """
    A class to handle and provide relevant sections, definitions, and workflow from the Currency and Exchange Manuals
    based on the user's content using embeddings and thresholds for similarity.
    """    
    def __init__(self, user_type, regulation_name, regulation_reader, df_definitions, df_index, df_workflow):
        """
        Parameters:
        -----------
        user_type : str
            Used in the system prompt to tell the LLM who is asking the question
        regulation_name : str
            Used in the system prompt to tell the LLM what document they are going to be using to answer questions
        regulation_reader : RegulationReader
            For the method get_regulation_detail and get_regulation_heading
        df_definitions : DataFrame
            DataFrame containing definitions from the manual. It must have columns in "required_columns_definition"
        df_index : DataFrame
            DataFrame containing index information from the manual. If must have columns "required_columns_index"
        df_workflow : DataFrame
            DataFrame containing workflow information for the chat so we can change from RAG mode into something else.
            It must have columns "required_columns_workflow"
        """
        self.definitions = df_definitions
        self.index = df_index
        self.workflow = df_workflow
        super().__init__(user_type = user_type, regulation_name = regulation_name, regulation_reader = regulation_reader)
        self._set_required_column_headings()
        self._check_definitions_column_headings()
        self._check_index_column_headings()
        self._check_workflow_column_headings()



    def _set_required_column_headings(self):
        '''
        These are the minimum required column headings and should be used in the _check*_column_headings() methods. 
        Feel free to override them if you need to have additional information in any step
        '''
        # These include columns that are only used for logging and debugging
        self.index_columns = required_columns_index 
        self.definition_columns = required_columns_definition
        self.workflow_columns = required_columns_workflow


    def _check_df_columns(self, actual_column_headings, expected_columns_as_list, data_source_name):
        for column in expected_columns_as_list:
            if column not in expected_columns_as_list:
                raise AttributeError(f"Column {column} should be in {data_source_name} but it is not")


    def _get_definitions_column_headings(self):
        if not self.definitions.empty:
            return self.definitions.columns.to_list()
        return []

    def _check_definitions_column_headings(self):
        actual_column_headings = self._get_definitions_column_headings()
        if actual_column_headings != []:
            self._check_df_columns(actual_column_headings, self.definition_columns, "definitions")

    def _get_index_column_headings(self):
        if not self.index.empty:
            return self.index.columns.to_list()
        return []

    def _check_index_column_headings(self):
        actual_column_headings = self._get_index_column_headings()
        if actual_column_headings != []:
            self._check_df_columns(actual_column_headings, self.index_columns, "index")

    def _get_workflow_column_headings(self):
        if not self.workflow.empty:
            return self.workflow.columns.to_list()
        return []

    def _check_workflow_column_headings(self):
        actual_column_headings = self._get_workflow_column_headings()
        if actual_column_headings != []:
            self._check_df_columns(actual_column_headings, self.workflow_columns, "workflow")



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


    def cap_rag_section_token_length(self, relevant_sections, capped_number_of_tokens):
        #relevant_sections["section_reference"] = relevant_sections["section_reference"]
        relevant_sections["regulation_text"] = relevant_sections["section_reference"].apply(
            lambda x: self.regulation_reader.get_regulation_detail(x)
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
            relevant_sections = pd.DataFrame([], columns = required_columns_section_lookup)

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
            return pd.DataFrame([], columns = required_columns_workflow)



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

    tmp = load_parquet_data(path_to_file, decryption_key)

    return pd.concat([original_df, tmp], ignore_index = True)


def load_index_data_from_files(
              path_to_definitions_as_parquet_file, path_to_additional_definitions_as_parquet_file,
              path_to_index_as_parquet_file, path_to_additional_index_as_parquet_file,
              workflow_as_parquet_file = "",
              decryption_key = ""):

    if path_to_definitions_as_parquet_file == "":
        df_definitions = pd.DataFrame([],columns = required_columns_definition)    
    df_definitions = load_parquet_data(path_to_definitions_as_parquet_file)
    df_definitions = append_parquet_data(path_to_additional_definitions_as_parquet_file, df_definitions)

    # index is the data that is encrypted
    df_index = load_parquet_data(path_to_index_as_parquet_file, decryption_key)
    df_index = append_parquet_data(path_to_additional_index_as_parquet_file, df_index, decryption_key)

    if workflow_as_parquet_file == "":
        df_workflow = pd.DataFrame([],columns = required_columns_workflow)    
    else:
        df_workflow = load_parquet_data(workflow_as_parquet_file)

    return df_definitions, df_index, df_workflow

def create_test_data():
    """
    Creates and returns a ReferenceChecker instance used for testing.
    """
    exclusion_list = ['Legal context', 'Introduction']
    index_patterns = [
        r'^[A-Z]\.\d{0,2}',             # Matches capital letter followed by a period and up to two digits.
        r'^\([A-Z]\)',                  # Matches single capital letters within parentheses.
        r'^\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx|xxi|xxii|xxiii|xxiv|xxv|xxvi|xxvii)\)', # Matches Roman numerals within parentheses.
        r'^\([a-z]\)',                  # Matches single lowercase letters within parentheses.
        r'^\([a-z]{2}\)',               # Matches two lowercase letters within parentheses.
        r'^\((?:[1-9]|[1-9][0-9])\)',   # Matches numbers within parentheses, excluding leading zeros.
    ]    
    text_pattern = r"[A-Z]\.\d{0,2}\([A-Z]\)\((?:i|ii|iii|iv|v|vi)\)\([a-z]\)\([a-z]{2}\)\(\d+\)"

    reference_checker = ReferenceChecker(regex_list_of_indices=index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)

    path_to_manual_as_csv_file = "./test/inputs/manual.csv"
    path_to_definitions_as_parquet_file = "./test/inputs/definitions.parquet"
    path_to_index_as_parquet_file = "./test/inputs/index.parquet"
    path_to_additional_manual_as_csv_file = ""
    path_to_additional_definitions_as_parquet_file = ""
    path_to_additional_index_as_parquet_file = "./test/inputs/index_plus.parquet"
    path_to_workflow_as_parquet = "./test/inputs/workflow.parquet"

    decryption_key = os.getenv('excon_encryption_key')

    df_definitions, df_index, df_workflow = load_index_data_from_files(
                                path_to_definitions_as_parquet_file, path_to_additional_definitions_as_parquet_file,
                                path_to_index_as_parquet_file, path_to_additional_index_as_parquet_file,
                                path_to_workflow_as_parquet,
                                decryption_key=decryption_key)
    df_regulations = load_regulation_data_from_files(path_to_manual_as_csv_file, path_to_additional_manual_as_csv_file)

    user_type = "Authorised Dealer (AD)" 
    regulation_name = "\'Currency and Exchange Manual for Authorised Dealers\' (Manual or CEMAD)"

    data = StandardRegulationIndex(user_type = user_type, 
                            regulation_name = regulation_name, 
                            reference_checker = reference_checker, 
                            df_regulations = df_regulations, 
                            df_definitions = df_definitions, 
                            df_index = df_index, 
                            df_workflow = df_workflow)
    return data


