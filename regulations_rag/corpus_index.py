import logging

from abc import ABC, abstractmethod

class CorpusIndex(ABC):
    """
    A class to handle and provide relevant sections, definitions, and workflow. This could wrap a database or a DataFrame
    """    
    def __init__(self, user_type, corpus_description, corpus):
        """
        Parameters:
        -----------
        user_type : str
            Used in the LLM system prompt to tell the model who they are assisting
        corpus_description : str
            Used in the LLM system prompt to tell the model what document they will be answering questions on
        corpus : Corpus - a collection of Documents
            Used to check and extract section references that will be used to request the regulation extracts
        """
        self.user_type = user_type
        self.corpus_description = corpus_description
        self.corpus = corpus


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
            A DataFrame with with a column 'document' and 'definition' containing the text of the close definitions.
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
            A DataFrame with sections close to the user content embedding. This method also adds the content of the manual
            to the DataFrame in the column "document", "section_reference", "regulation_text"
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

