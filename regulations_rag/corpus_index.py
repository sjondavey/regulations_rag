import logging
from abc import ABC, abstractmethod
import pandas as pd
from regulations_rag.rerank import RerankAlgos, rerank
from regulations_rag.embeddings import get_closest_nodes, num_tokens_from_string

# Create a logger for this module
logger = logging.getLogger(__name__)
DEV_LEVEL = 15
logging.addLevelName(DEV_LEVEL, 'DEV')       

class CorpusIndex(ABC):
    """
    A class to handle and provide relevant sections, definitions, and workflow.
    This could wrap a database or a DataFrame.
    """    
    def __init__(self, user_type, corpus_description, corpus):
        """
        Parameters:
        -----------
        user_type : str
            Used in the LLM system prompt to tell the model who they are assisting.
        corpus_description : str
            Used in the LLM system prompt to tell the model what document they will be answering questions on.
        corpus : Corpus
            A collection of Documents used to check and extract section references that will be used to request the regulation extracts.
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
            The user's question.
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant definitions.

        Returns:
        --------
        DataFrame
            A DataFrame with columns 'document' and 'definition' containing the text of the close definitions.
        """
        pass

    @abstractmethod
    def get_relevant_sections(self, user_content, user_content_embedding, threshold, RerankAlgos=RerankAlgos.NONE):
        """
        Retrieves sections close to the given user content embedding. This should also filter the sections so that the
        returned chunks don't contain 'too many' tokens.

        Parameters:
        -----------
        user_content : str
            The user's question.
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant sections.

        Returns:
        --------
        DataFrame
            A DataFrame with sections close to the user content embedding. This method also adds the content of the manual
            to the DataFrame in the columns "document", "section_reference", "regulation_text".
        """
        pass

    @abstractmethod
    def get_relevant_workflow(self, user_content, user_content_embedding, threshold):
        """
        Retrieves workflow steps close to the given user content embedding if available.

        Parameters:
        -----------
        user_content : str
            The user's question.
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant workflow steps.

        Returns:
        --------
        DataFrame
            A DataFrame with a column 'workflow' and one row which represents the most likely workflow. This can return an 
            empty DataFrame if there are no 'close' workflows.
        """
        pass

class DataFrameCorpusIndex(CorpusIndex):
    """
    An instance of the Corpus Index if the data is contained in DataFrames rather than Databases.
    """
    def __init__(self, user_type, corpus_description, corpus, definitions, index, workflow):
        columns_in_dfns = ["embedding", "document", "section_reference", "text", "definition"]
        for column in columns_in_dfns:
            assert column in definitions.columns.to_list()
        columns_in_sections = ["embedding", "document", "section_reference", "source", "text"]
        for column in columns_in_sections:
            assert column in index.columns.to_list()
        if not workflow.empty and len(workflow) > 0:
            columns_in_workflow = ["embedding"]
            for column in columns_in_workflow:
                assert column in workflow.columns.to_list()
        
        self.required_columns_workflow = ["workflow", "text", "embedding"]
        super().__init__(user_type, corpus_description, corpus)
        self.definitions = definitions
        self.index = index
        self.workflow = workflow

    def get_relevant_definitions(self, user_content, user_content_embedding, threshold):
        relevant_definitions = get_closest_nodes(self.definitions, embedding_column_name="embedding", content_embedding=user_content_embedding, threshold=threshold)

        if not relevant_definitions.empty:
            logger.log(DEV_LEVEL, "--   Relevant Definitions")
            for index, row in relevant_definitions.iterrows():
                logger.log(DEV_LEVEL, f'{row["cosine_distance"]:.4f}: {row["text"]}')
        else:
            logger.log(DEV_LEVEL, "--   No relevant definitions found")

        return relevant_definitions

    def cap_rag_section_token_length(self, relevant_sections, capped_number_of_tokens):
        relevant_sections["regulation_text"] = ""
        relevant_sections["token_count"] = 0
        for index, row in relevant_sections.iterrows():
            text = self.corpus.get_text(row["document"], row["section_reference"])
            relevant_sections.loc[index, "regulation_text"] = text
            relevant_sections.loc[index, "token_count"] = num_tokens_from_string(text)

        cumulative_sum = 0
        counter = 0
        n = 0

        for index, row in relevant_sections.iterrows():
            next_cumulative_sum = cumulative_sum + row["token_count"]
            if next_cumulative_sum > capped_number_of_tokens:
                n = counter
                break
            else:
                cumulative_sum = next_cumulative_sum
            counter += 1

        if n == 0:
            if relevant_sections["token_count"].iloc[0] > capped_number_of_tokens:
                n = 1
            else:
                n = len(relevant_sections)

        if n != len(relevant_sections):
            logger.log(DEV_LEVEL, f"--   Token capping reduced the number of reference sections from {len(relevant_sections)} to {n}")

        final_row = min(n, 5)
        top_subset_df = relevant_sections.nsmallest(final_row, 'cosine_distance').reset_index(drop=True)

        return top_subset_df

    def get_relevant_sections(self, user_content, user_content_embedding, threshold, rerank_algo=RerankAlgos.NONE):
        """
        Retrieves sections close to the given user content embedding.

        Parameters:
        -----------
        user_content : str
            The user's question.
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant sections.

        Returns:
        --------
        DataFrame
            A DataFrame with sections close to the user content embedding. This method also adds the content of the manual
            to the DataFrame in the columns "document", "section_reference", "regulation_text".
        """
        relevant_sections = get_closest_nodes(self.index, embedding_column_name="embedding", content_embedding=user_content_embedding, threshold=threshold)         
        n = rerank_algo.params["initial_section_number_cap"]
        relevant_sections = relevant_sections.nsmallest(n, 'cosine_distance')      
        logger.log(DEV_LEVEL, f"Selecting the top {n} items based on cosine-similarity score")
        for index, row in relevant_sections.iterrows():
            logger.log(DEV_LEVEL, f'{row["cosine_distance"]:.4f}: {row["document"]:>20}: {row["section_reference"]:>20}: {row["source"]:>15}: {row["text"]}')

        if not relevant_sections.empty:
            logger.log(DEV_LEVEL, "--   Relevant sections found")
            rerank_algo.params["user_question"] = user_content
            reranked_sections = rerank(relevant_sections=relevant_sections, rerank_algo=rerank_algo).copy(deep=True)     
            if reranked_sections.empty:
                logger.log(DEV_LEVEL, "--   Re-ranking concluded there were no relevant sections")
                columns = self.index.columns.to_list()
                columns.append("regulation_text")
                empty_sections = pd.DataFrame([], columns=columns)
                return empty_sections

            capped_sections = self.cap_rag_section_token_length(reranked_sections, rerank_algo.params["final_token_cap"])
            relevant_sections = capped_sections
            relevant_sections["regulation_text"] = relevant_sections.apply(lambda row: self.corpus.get_text(row["document"], row["section_reference"], add_markdown_decorators=False), axis=1)
        else:
            logger.log(DEV_LEVEL, "--   No relevant sections found")
            columns = self.index.columns.to_list()
            columns.append("regulation_text")
            relevant_sections = pd.DataFrame([], columns=columns)
            
        return relevant_sections

    def get_relevant_workflow(self, user_content, user_content_embedding, threshold):
        """
        Retrieves workflow steps close to the given user content embedding if available.

        Parameters:
        -----------
        user_content : str
            The user's question.
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
            return get_closest_nodes(self.workflow, embedding_column_name="embedding", content_embedding=user_content_embedding, threshold=threshold)
        else:
            return pd.DataFrame([], columns=self.required_columns_workflow)
