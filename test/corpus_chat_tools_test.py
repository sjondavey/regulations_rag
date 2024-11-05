import os
import pytest

from scipy.spatial import distance

from regulations_rag.embeddings import EmbeddingParameters, get_ada_embedding
import regulations_rag.corpus_chat_tools as cc_tools


def test_truncate_message_list():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable is not set. Skipping this test.")
    
    chat_parameters = cc_tools.ChatParameters(chat_model="gpt-4o-mini", api_key=api_key, temperature=0.0, max_tokens=200, token_limit_when_truncating_message_queue = 3500)

    # Test case where system message is not empty
    system_message = [{"role": "system", "content": "You are a helpful assistant."}]
    message_list = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."},
        {"role": "user", "content": "What is the population of Paris?"},
        {"role": "assistant", "content": "Paris has a population of 2.1 million people."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "Paris has a temperate climate with mild winters and cool summers."},
        {"role": "user", "content": "Why is the Seine so polluted?"}]

    truncated_messages = chat_parameters.truncate_message_list(system_message = system_message, message_list = message_list)
    assert len(truncated_messages) == 8 # with a lot of tokens, we should not loose messages
    assert truncated_messages[0] == system_message[0]
    assert truncated_messages[-1]['role'] == 'user'
    assert truncated_messages[-1]["content"] == 'Why is the Seine so polluted?'

    # make the token limit lower than the number of tokens in the last message
    chat_parameters = cc_tools.ChatParameters(chat_model="gpt-4o-mini", 
                                              api_key=api_key, 
                                              temperature=0.0, 
                                              max_tokens=200, 
                                              token_limit_when_truncating_message_queue = 50)
    truncated_messages = chat_parameters.truncate_message_list(system_message, message_list)
    assert len(truncated_messages) ==  4 
    assert truncated_messages[0] == system_message[0]
    assert truncated_messages[-1]['role'] == 'user'
    assert truncated_messages[-1]["content"] == 'Why is the Seine so polluted?'

    # Test case where system message is empty
    chat_parameters = cc_tools.ChatParameters(chat_model="gpt-4o-mini", 
                                              api_key=api_key, 
                                              temperature=0.0, 
                                              max_tokens=200, 
                                              token_limit_when_truncating_message_queue = 3500)
    empty_system_message = []
    truncated_messages = chat_parameters.truncate_message_list(system_message=empty_system_message, message_list=message_list)
    assert len(truncated_messages) == 7  # All messages from message_list should be included
    assert truncated_messages[0]['role'] == 'user'  # First message should be from user
    assert truncated_messages[-1]['role'] == 'user'
    assert truncated_messages[-1]["content"] == 'Why is the Seine so polluted?'

    # Test with very low token limit and empty system message
    chat_parameters = cc_tools.ChatParameters(chat_model="gpt-4o-mini", 
                                              api_key=api_key, 
                                              temperature=0.0, 
                                              max_tokens=200, 
                                              token_limit_when_truncating_message_queue = 50)
    truncated_messages = chat_parameters.truncate_message_list(empty_system_message, message_list)
    assert len(truncated_messages) == 4 
    assert truncated_messages[-1]['role'] == 'user'
    assert truncated_messages[-1]["content"] == 'Why is the Seine so polluted?'


def test_get_api_response():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable is not set. Skipping this test.")
    
    
    chat_parameters = cc_tools.ChatParameters(chat_model="gpt-4o-mini", api_key=api_key, temperature=0.0, max_tokens=200, token_limit_when_truncating_message_queue = 3500)
    
    embedding_parameters = EmbeddingParameters("text-embedding-3-large", 1024)
    messages = [{"role": "user", "content": "What is the capital of France? Please answer 'xxx is the capital of France.'"}]
    llm_answer = chat_parameters.get_api_response(system_message=[], message_list=messages)
    embedding_llm_answer = get_ada_embedding(chat_parameters.openai_client, llm_answer, embedding_parameters.model, embedding_parameters.dimensions)
    
    answer = "Paris is the capital of France."
    embedding_answer = get_ada_embedding(chat_parameters.openai_client, answer, embedding_parameters.model, embedding_parameters.dimensions)
    # I am not sure this will always pass. I would like to set the threshold lower but I expect that may cause the test to fail spuriously
    assert distance.cosine(embedding_answer, embedding_llm_answer) < 0.05


