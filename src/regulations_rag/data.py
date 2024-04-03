from abc import ABC, abstractmethod

from src.regulations_rag.section_reference_checker import SectionReferenceChecker


class Data(ABC):
    """
    A class to handle and provide relevant sections, definitions, and workflow.
    """    
    def __init__(self, user_type, regulation_name, section_reference_checker):
        """
        Parameters:
        -----------
        user_type : str
            Used in the LLM system prompt to tell the model who they are assisting
        regulation_name : str
            Used in the LLM system prompt to tell the model what document they will be answering questions on
        section_reference_checker : Section_Reference_Checker
            Used to check and extract section references that will be used to request the regulation extracts
        """
        self.user_type = user_type 
        self.regulation_name = regulation_name
        self.section_reference_checker = section_reference_checker

        self._set_required_column_headings()

        self._check_definitions_column_headings()
        self._check_index_column_headings()
        self._check_regulations_column_headings()

    def _set_required_column_headings(self):
        '''
        These are the minimum required column headings and should be used in the _check*_column_headings() methods. 
        Feel free to override them if you need to have additional information in any step
        '''
        self.index_columns = ["section_reference", "embedding"]
        self.definition_columns = ["definition", "embedding"]
        self.workflow_columns = ["workflow", "embedding"]

        ''' 
        'indent'            : number of indents before the line starts - to help interpret it (i) is the letter or the Roman numeral (for example)
        'reference'         : the part of the section_reference at the start of the line. Can be blank
        'text'              : the text on the line excluding the 'reference' and any special text (identifying headings, page number etc)
        'heading'           : boolean identifying the text as as (sub-) section heading
        'section_reference' : the full reference. Starting at the root node and ending with the value in 'reference'
        '''
        self.regulation_columns =  ['indent', 'reference', 'text', 'heading', 'section_reference']

    def _check_df_columns(self, actual_column_headings, expected_columns_as_list, data_source_name):
        for column in expected_columns_as_list:
            if column not in expected_columns_as_list:
                raise AttributeError(f"Column {column} should be in {data_source_name} but it is not")

    @abstractmethod
    def _get_definitions_column_headings(self):
        '''
        Returns a list of the actual column headings for the definitions table or DataFrame. 
        If there are no definitions, this should return the empty list []
        '''
        pass

    def _check_definitions_column_headings(self):
        actual_column_headings = self._get_definitions_column_headings()
        if actual_column_headings != []:
            self._check_df_columns(actual_column_headings, self.definition_columns, "definitions")

    @abstractmethod
    def _get_index_column_headings(self):
        pass

    def _check_index_column_headings(self):
        actual_column_headings = self._get_index_column_headings()
        if actual_column_headings != []:
            self._check_df_columns(actual_column_headings, self.index_columns, "index")

    @abstractmethod
    def _get_workflow_column_headings(self):
        pass

    def _check_workflow_column_headings(self):
        actual_column_headings = self._get_workflow_column_headings()
        if actual_column_headings != []:
            self._check_df_columns(actual_column_headings, self.workflow_columns, "workflow")

    @abstractmethod
    def _get_regulations_column_headings(self):
        pass

    def _check_regulations_column_headings(self):
        actual_column_headings = self._get_regulations_column_headings()
        if actual_column_headings != []:
            self._check_df_columns(actual_column_headings, self.regulation_columns, "regulation")


    @abstractmethod
    def get_regulation_detail(self, section_index):
        """
        Retrieves the formatted text from the document

        Parameters:
        -----------
        section_index : str
            An section index which should be checked before being used

        Returns:
        --------
        str
            The formatted text from the regulation
        """
        pass

    @abstractmethod
    def get_relevant_definitions(self, user_content, user_content_embedding, threshold):
        """
        Retrieves definitions close to the given user content embedding.

        Parameters:
        -----------
        user_content : str
            The users question
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant definitions.

        Returns:
        --------
        DataFrame
            A DataFrame with with a column 'definition' containing the text of the close definitions.
        """
        pass

    @abstractmethod
    def get_relevant_sections(self, user_content, user_content_embedding, threshold):
        """
        Retrieves sections close to the given user content embedding. This should also filter the sections so that the
        the returned chunks don't contain 'too many' tokens

        Parameters:
        -----------
        user_content : str
            The users question
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant sections.

        Returns:
        --------
        DataFrame
            A DataFrame with the columns 'section_reference' and 'section_text' which contain a valid index for the document and
            the full test relating to that index as returned by self.get_regulation_detail(section_index)
        """
        pass

    @abstractmethod
    def get_relevant_workflow(self, user_content, user_content_embedding, threshold):
        """
        Retrieves workflow steps close to the given user content embedding if available.

        Parameters:
        -----------
        user_content : str
            The users question
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant workflow steps.

        Returns:
        --------
        DataFrame
            A DataFrame with a column 'workflow' and one row which represents the one most likely workflow. This can return an 
            empty DataFrame if there are no 'close' workflows
        """
        pass

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


def append_parquet_data(path_to_file, original_df, decryption_key = ""):
    if path_to_file == "":
        return original_df

    tmp = load_parquet_data(path_to_file)
    if decryption_key:
        tmp['text'] = tmp['text'].apply(lambda x: fernet.decrypt(x.encode()).decode())

    return pd.concat([original_df, tmp], ignore_index = True)


def load_data_from_files(chat_for_ad,
              path_to_manual_as_csv_file, path_to_additional_manual_as_csv_file, 
              path_to_definitions_as_parquet_file, path_to_additional_definitions_as_parquet_file,
              path_to_index_as_parquet_file, path_to_additional_index_as_parquet_file,
              workflow_as_parquet_file = ""):
    df_manual = load_csv_data(path_to_manual_as_csv_file)
    df_manual = append_csv_data(path_to_additional_manual_as_csv_file, df_manual)

    df_definitions = load_parquet_data(path_to_definitions_as_parquet_file)
    df_definitions = append_parquet_data(path_to_additional_definitions_as_parquet_file, df_definitions)

    df_index = load_parquet_data(path_to_index_as_parquet_file)
    df_index = append_parquet_data(path_to_additional_index_as_parquet_file, df_index)

    if workflow_as_parquet_file == "":
        df_workflow = pd.DataFrame([],columns = ["section", "text", "source", "embedding"])    
    else:
        df_workflow = load_parquet_data(workflow_as_parquet_file)

    return Data(chat_for_ad, df_manual, df_definitions, df_index, df_workflow)

def load_data_from_folders(chat_for_ad, base_directory, embeddings_directory):
    # base_directory = "." or ".." typically
    # embeddings_directory = "ada_v2" or "/v3_large/1024" or "/v3_large/3072"
    if chat_for_ad:
        logger.info("Loaded Authorised Dealer Manual")
        path_to_manual_as_csv_file = f"{base_directory}/inputs/ad_manual.csv"
        path_to_manual_as_csv_file_plus = f"{base_directory}/inputs/ad_manual_plus.csv"

        path_to_definitions_as_parquet_file = f"{base_directory}/inputs{embeddings_directory}/ad_definitions.parquet"
        path_to_definitions_as_parquet_file_plus = f"{base_directory}/inputs{embeddings_directory}/ad_definitions_plus.parquet"

        path_to_index_as_parquet_file = f"{base_directory}/inputs{embeddings_directory}/ad_index.parquet"
        path_to_index_as_parquet_file_plus = f"{base_directory}/inputs{embeddings_directory}/ad_index_plus.parquet"

        workflow_as_parquet_file = f"{base_directory}/inputs{embeddings_directory}/workflow.parquet"
    else:
        logger.info("Loaded ADLA manual")
        path_to_manual_as_csv_file = f"{base_directory}/inputs/adla_manual.csv"
        path_to_manual_as_csv_file_plus = ""

        path_to_definitions_as_parquet_file = f"{base_directory}/inputs{embeddings_directory}/adla_definitions.parquet"
        path_to_definitions_as_parquet_file_plus = ""

        path_to_index_as_parquet_file = f"{base_directory}/inputs{embeddings_directory}/adla_index.parquet"
        path_to_index_as_parquet_file_plus = ""

        workflow_as_parquet_file = f"{base_directory}/inputs{embeddings_directory}/workflow.parquet"

    return load_data_from_files(chat_for_ad,
                            path_to_manual_as_csv_file, path_to_manual_as_csv_file_plus, 
                            path_to_definitions_as_parquet_file, path_to_definitions_as_parquet_file_plus,
                            path_to_index_as_parquet_file, path_to_index_as_parquet_file_plus,
                            workflow_as_parquet_file)


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

