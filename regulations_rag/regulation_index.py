import logging

from abc import ABC, abstractmethod

from regulations_rag.regulation_reader import RegulationReader
from regulations_rag.rerank import RerankAlgos

# Create a logger for this module
logger = logging.getLogger(__name__)
DEV_LEVEL = 15
logging.addLevelName(DEV_LEVEL, 'DEV')       

class EmbeddingParameters:
    def __init__(self, embedding_model, embedding_dimensions):
        self.model = embedding_model
        self.dimensions = embedding_dimensions

        if embedding_model == "text-embedding-ada-002":
            self.threshold = 0.15
            self.dimensions = 1536 # this model does not 
        elif embedding_model == "text-embedding-3-large":
            if embedding_dimensions == 1024:
                self.threshold = 0.38
            elif embedding_dimensions == 3072:
                self.threshold = 0.40
        else:
            raise ValueError("Unknown Embedding model or embedding dimension")


class RegulationIndex(ABC):
    """
    A class to handle and provide relevant sections, definitions, and workflow.
    """    
    def __init__(self, user_type, regulation_name, regulation_reader):
        """
        Parameters:
        -----------
        user_type : str
            Used in the LLM system prompt to tell the model who they are assisting
        regulation_name : str
            Used in the LLM system prompt to tell the model what document they will be answering questions on
        reference_checker : ReferenceChecker
        regulation_reader : RegulationReader
            Used to check and extract section references that will be used to request the regulation extracts
        """
        self.user_type = user_type
        self.regulation_name = regulation_name
        self.regulation_reader = regulation_reader


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
    def get_relevant_sections(self, user_content, user_content_embedding, threshold, RerankAlgos = RerankAlgos.NONE):
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


