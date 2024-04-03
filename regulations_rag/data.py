from abc import ABC, abstractmethod

from regulations_rag.section_reference_checker import SectionReferenceChecker


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

