import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from .navigating_index import NavigatingIndex

from regulations_rag.data_classes import AnswerWithRAGResponse, AnswerWithoutRAGResponse, \
    NoAnswerClassification, NoAnswerResponse, ErrorClassification, ErrorResponse

from regulations_rag.corpus_chat_tools import ChatParameters, get_caveat_for_no_rag_response

from regulations_rag.path_no_rag_data import PathNoRAGData

from regulations_rag.embeddings import EmbeddingParameters
from regulations_rag.rerank import RerankAlgos

from openai import OpenAI

def test_is_user_content_relevant():
    corpus_index = NavigatingIndex()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable is not set. Skipping this test.")
    chat_parameters = ChatParameters(chat_model="gpt-4o-mini", api_key=api_key, temperature=0.0, max_tokens=200, token_limit_when_truncating_message_queue = 3500)
    path = PathNoRAGData(corpus_index=corpus_index, chat_parameters=chat_parameters)

    messages = []
    user_message = {"role": "user", "content": "Some user input"}



    mock_get_api_response = "Relevant"
    with patch.object(chat_parameters, 'get_api_response', return_value = mock_get_api_response): 
        relevant, text = path.is_user_content_relevant(message_history=messages, current_user_message=user_message)
        assert relevant
    # Only "relevant" with capitialisation and white spaces should be true
    mock_get_api_response = "Not Relevant\n\nLong Story Here"
    with patch.object(chat_parameters, 'get_api_response', return_value = mock_get_api_response):
        relevant, text = path.is_user_content_relevant(message_history=messages, current_user_message=user_message)
        assert not relevant
        assert text.strip() == "Long Story Here"


def test_create_system_content_no_rag_data():
    corpus_index = NavigatingIndex()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable is not set. Skipping this test.")
    chat_parameters = ChatParameters(chat_model="gpt-4o-mini", api_key=api_key, temperature=0.0, max_tokens=200, token_limit_when_truncating_message_queue = 3500)
    path = PathNoRAGData(corpus_index=corpus_index, chat_parameters=chat_parameters)

    
    corpus_description = "the Simplest way to Navigate Plett"
    user_type = "a Visitor"
    tap_out_phrase = "No Answer"
    expected_message = f"You are answering questions about {corpus_description} for {user_type}. Based on an initial search of the relevant document database, no reference documents could be found to assist in answering the users question. Please review the user question. If you are able to answer the question, please do so. If you are not able to answer the question, respond with the words {tap_out_phrase} without punctuation or any other text."
    assert path.create_system_content_no_rag_data(corpus_description, user_type, tap_out_phrase) == expected_message

def test_query_no_rag_data():
    corpus_index = NavigatingIndex()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable is not set. Skipping this test.")
    chat_parameters = ChatParameters(chat_model="gpt-4o-mini", api_key=api_key, temperature=0.0, max_tokens=200, token_limit_when_truncating_message_queue = 3500)
    messages = []
    user_message = {"role": "user", "content": "Test question"}
    path = PathNoRAGData(corpus_index=corpus_index, chat_parameters=chat_parameters)
    mock_get_api_response = MagicMock(return_value="Not Relevant. This does not have anything to do with the topic.")    
    with patch.object(chat_parameters, 'get_api_response', mock_get_api_response):
        response = path.query_no_rag_data(message_history=messages, current_user_message=user_message) 
    assert response["role"] == "assistant"
    assert response["content"] == "This does not have anything to do with the topic."
    assert isinstance(response["assistant_response"], NoAnswerResponse)
    assert response["assistant_response"].classification == NoAnswerClassification.QUESTION_NOT_RELEVANT

    # Create a mock for get_api_response
    mock_get_api_response.side_effect = [
        "Relevant",
        "Just have a look at the map you muppet!"
    ] 
    # Use patch as a context manager
    with patch.object(chat_parameters, 'get_api_response', mock_get_api_response):
        result = path.query_no_rag_data(message_history=messages, current_user_message=user_message)
    assert result["role"] == "assistant"
    caveat = get_caveat_for_no_rag_response()
    assert result["content"] == f"{caveat}\n\nJust have a look at the map you muppet!"
    assert isinstance(result["assistant_response"], AnswerWithoutRAGResponse)
    assert result["assistant_response"].answer == "Just have a look at the map you muppet!"
    assert result["assistant_response"].caveat == caveat



