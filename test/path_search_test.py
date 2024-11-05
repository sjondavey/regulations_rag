import os
from regulations_rag.path_search import PathSearch
from regulations_rag.corpus_chat_tools import ChatParameters
from regulations_rag.embeddings import EmbeddingParameters
from regulations_rag.rerank import RerankAlgos
from .navigating_index import NavigatingIndex


def test_similarity_search():


    api_key=os.environ.get("OPENAI_API_KEY")
    chat_parameters = ChatParameters(chat_model = "gpt-4o",  
                                     api_key=api_key, 
                                     temperature = 0, 
                                     max_tokens = 500, 
                                     token_limit_when_truncating_message_queue = 3500)
    embedding_parameters = EmbeddingParameters("text-embedding-3-large", 1024)
    corpus_index = NavigatingIndex()
    rerank_algo  = RerankAlgos.NONE 

    # Check that random chit-chat to the main dataset does not return any hits from the embeddings
    text = "Hi"

    path_search = PathSearch(corpus_index=corpus_index, 
                             chat_parameters=chat_parameters, 
                             embedding_parameters=embedding_parameters, 
                             rerank_algo=rerank_algo)

    workflow_triggered, df_definitions, df_search_sections = path_search.similarity_search(text)
    assert len(df_definitions) == 0
    assert len(df_search_sections) == 0 

    # now move to the testing dataset for fine grained tests
    user_content = "How do I get to South Gate?"
    workflow_triggered, relevant_definitions, relevant_sections = path_search.similarity_search(user_content)
    assert len(relevant_definitions) == 0
    assert len(relevant_sections) == 3
    assert relevant_sections.iloc[0]["section_reference"] == '1.3' # this is the answer we want
    assert relevant_sections.iloc[1]["section_reference"] == '1.1' # order of these is not important
    assert relevant_sections.iloc[2]["section_reference"] == '1.2' # order of these is not important

    # test workflows are found
    user_content = "Can I see this on a map?"
    workflow_triggered, relevant_definitions, relevant_sections = path_search.similarity_search(user_content)
    assert workflow_triggered == 'map'
    assert len(relevant_definitions) == 0
    assert len(relevant_sections) == 0

