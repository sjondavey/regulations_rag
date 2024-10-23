import logging
from regulations_rag.embeddings import get_ada_embedding

logger = logging.getLogger(__name__)
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')

class ChatDataForSearch:
    def __init__(self, corpus_index, chat_parameters, embedding_parameters, rerank_algo):
        self.corpus_index = corpus_index
        self.corpus = corpus_index.corpus
        self.chat_parameters = chat_parameters
        self.openai_client = chat_parameters.openai_client
        self.embedding_parameters = embedding_parameters
        self.rerank_algo = rerank_algo



def similarity_search(chat_data_for_search, user_content):
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
    question_embedding = get_ada_embedding(chat_data_for_search.openai_client, 
                                           user_content, 
                                           chat_data_for_search.embedding_parameters.model, 
                                           chat_data_for_search.embedding_parameters.dimensions)      

    if len(chat_data_for_search.corpus_index.workflow) > 0:
        relevant_workflows = chat_data_for_search.corpus_index.get_relevant_workflow(user_content = user_content, 
                                                                                     user_content_embedding = question_embedding, 
                                                                                     threshold = chat_data_for_search.embedding_parameters.threshold)
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

    relevant_definitions = chat_data_for_search.corpus_index.get_relevant_definitions(user_content = user_content, 
                                                                                       user_content_embedding = question_embedding, 
                                                                                       threshold = chat_data_for_search.embedding_parameters.threshold_definitions)
    if not relevant_definitions.empty:

        most_relevant_definition_score = relevant_definitions.iloc[0]['cosine_distance']

        if most_relevant_definition_score < most_relevant_workflow_score: # there is something more relevant than a workflow
            logger.log(DEV_LEVEL, f"similarity_search: Found a definition that was more relevant than the workflow: {workflow_triggered}")
            workflow_triggered = "none"        


    relevant_sections = chat_data_for_search.corpus_index.get_relevant_sections(user_content = user_content, 
                                                                                user_content_embedding = question_embedding, 
                                                                                threshold = chat_data_for_search.embedding_parameters.threshold, 
                                                                                rerank_algo = chat_data_for_search.rerank_algo)
    if not relevant_sections.empty:    
    
        if not relevant_sections.empty:    
            most_relevant_section_score = relevant_sections.iloc[0]['cosine_distance']

            if most_relevant_section_score < most_relevant_workflow_score and workflow_triggered != "none": # there is something more relevant than a workflow
                logger.log(DEV_LEVEL, f"similarity_search:  Found a section that was more relevant than the workflow: {workflow_triggered}")
                workflow_triggered = "none"

    return workflow_triggered, relevant_definitions, relevant_sections
