import logging
from regulations_rag.corpus_index import CorpusIndex
from regulations_rag.embeddings import get_ada_embedding
from regulations_rag.corpus_chat_tools import ChatParameters
from regulations_rag.embeddings import EmbeddingParameters
from regulations_rag.rerank import RerankAlgos

logger = logging.getLogger(__name__)
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')

class PathSearch:   

    def __init__(self, corpus_index: CorpusIndex, chat_parameters: ChatParameters, embedding_parameters: EmbeddingParameters, rerank_algo: RerankAlgos):
        self.corpus_index = corpus_index

        self.chat_parameters = chat_parameters

        self.embedding_parameters = embedding_parameters
        self.rerank_algo = rerank_algo

        self.execution_path = [] # used to track the execution path for testing and analysis

    def _track_path(self, step):
        self.execution_path.append(step)

    def similarity_search(self, user_question):
        """
        Finds the index values, definitions and workflows that are most similar to the provided user content. A workflow is 
        triggered if the lowest cosine similarity score for the closest workflow is lower that the lowest cosine similarity
        score for the closest definition and the closest index

        Parameters:
        - user_content (str): The content provided by the user to conduct the similarity search against.

        Returns:
        - tuple: Contains the most relevant workflow triggered (if any), a DataFrame of relevant definitions, and
                a DataFrame of the top relevant sections sorted by their relevance.
        """
        logger.log(DEV_LEVEL, "similarity_search called")
        self._track_path("PathSearch.similarity_search")

        question_embedding = get_ada_embedding(self.chat_parameters.openai_client, 
                                               user_question, 
                                               self.embedding_parameters.model, 
                                               self.embedding_parameters.dimensions)      

        if len(self.corpus_index.workflow) > 0:
            relevant_workflows = self.corpus_index.get_relevant_workflow(user_content = user_question, 
                                                                        user_content_embedding = question_embedding, 
                                                                        threshold = self.embedding_parameters.threshold)
            if not relevant_workflows.empty > 0:
                # relevant_workflows is sorted in the get_closest_nodes method
                most_relevant_workflow_score = relevant_workflows.iloc[0]['cosine_distance']
                workflow_triggered = relevant_workflows.iloc[0]['workflow']
                logger.info(f"similarity_search: Found a potentially relevant workflow: {workflow_triggered}")
            else:
                most_relevant_workflow_score = 1.0
                workflow_triggered = "none"
        else:
            most_relevant_workflow_score = 1.0
            workflow_triggered = "none"

        relevant_definitions = self.corpus_index.get_relevant_definitions(user_content = user_question, 
                                                                                        user_content_embedding = question_embedding, 
                                                                                        threshold = self.embedding_parameters.threshold_definitions)
        if not relevant_definitions.empty:

            most_relevant_definition_score = relevant_definitions.iloc[0]['cosine_distance']

            if most_relevant_definition_score < most_relevant_workflow_score: # there is something more relevant than a workflow
                logger.log(DEV_LEVEL, f"similarity_search: Found a definition that was more relevant than the workflow: {workflow_triggered}")
                workflow_triggered = "none"        


        relevant_sections = self.corpus_index.get_relevant_sections(user_content = user_question, 
                                                                                    user_content_embedding = question_embedding, 
                                                                                    threshold = self.embedding_parameters.threshold, 
                                                                                    rerank_algo = self.rerank_algo)
        if not relevant_sections.empty:    
        
            if not relevant_sections.empty:    
                most_relevant_section_score = relevant_sections.iloc[0]['cosine_distance']

                if most_relevant_section_score < most_relevant_workflow_score and workflow_triggered != "none": # there is something more relevant than a workflow
                    logger.log(DEV_LEVEL, f"similarity_search:  Found a section that was more relevant than the workflow: {workflow_triggered}")
                    workflow_triggered = "none"

        return workflow_triggered, relevant_definitions, relevant_sections
