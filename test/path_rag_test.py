import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from .navigating_index import NavigatingIndex

from regulations_rag.corpus_chat_tools import ChatParameters, get_caveat_for_no_rag_response    

from regulations_rag.data_classes import AnswerWithRAGResponse, AnswerWithoutRAGResponse, \
    NoAnswerClassification, NoAnswerResponse, ErrorClassification, ErrorResponse
                                         

from regulations_rag.path_rag import PathRAG

from regulations_rag.path_search import  PathSearch
from regulations_rag.embeddings import EmbeddingParameters
from regulations_rag.rerank import RerankAlgos

from openai import OpenAI

# test data is in conftest.py
def test_format_user_question(dummy_definitions, dummy_search_sections):
    corpus_index = NavigatingIndex()
    api_key = os.environ.get("OPENAI_API_KEY")
    chat_parameters = ChatParameters(chat_model="gpt-4o-mini", 
                                     api_key=api_key, 
                                     temperature=0.0, 
                                     max_tokens=200, 
                                     token_limit_when_truncating_message_queue = 3500)

    path_rag = PathRAG(corpus_index= corpus_index, chat_parameters=chat_parameters)

    question = "user asks question"
    output_string = path_rag.format_user_question(question, dummy_definitions, dummy_search_sections)
    expected_text = f'Question: user asks question\n\nExtract 1:\nMy definition from WRR\nExtract 2:\nMy definition from Plett\nExtract 3:\nMy Section 1.2 from WRR\nExtract 4:\nMy Section 1.3 from WRR\nExtract 5:\nMy section A.2(A)(i) from Plett\n'
    assert output_string == expected_text

def test_create_system_message():
    corpus_index = NavigatingIndex()
    api_key = os.environ.get("OPENAI_API_KEY")
    chat_parameters = ChatParameters(chat_model="gpt-4o-mini", 
                                     api_key=api_key, 
                                     temperature=0.0, 
                                     max_tokens=200, 
                                     token_limit_when_truncating_message_queue = 3500)

    path_rag = PathRAG(corpus_index= corpus_index, chat_parameters=chat_parameters)
    ref_string = r"[1-9](\.[1-9]){0,2}"  # the text_pattern from SimpleReferenceChecker which is the reference checker for the main document in the "navigating" Corpus (WRR)
    user_type = "a Visitor"
    corpus_description = "the Simplest way to Navigate Plett"

    expected_message = f"You are answering questions about {corpus_description} for {user_type} based only on the reference extracts provided. You have 3 options:\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference:'. Do not include the word Extract, only provide the number(s).\n2) Request additional documentation. If, in the body of the extract(s) provided, there is a reference to another section that is directly relevant and not already provided, respond with the word 'SECTION:' followed by 'Extract extract_number, Reference: section_reference' - for example SECTION: Extract 1, Reference: {ref_string}.\n3) State 'NONE:' and nothing else in all other cases\n"
    assert path_rag.create_system_message_RAG(number_of_options=3, review = False) == expected_message

    expected_message = f"You are answering questions about {corpus_description} for {user_type} based only on the reference extracts provided. You have 2 options:\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference:'. Do not include the word Extract, only provide the number(s).\n2) State 'NONE:' and nothing else in all other cases\n"
    assert path_rag.create_system_message_RAG(number_of_options=2, review = False) == expected_message

    expected_message = f"Please review your answer. You were asked to assist the user by responding to their question in 1 of 3 ways but your response does not follow the expected format. Please reformat your response so that it follows the requested format.\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference:'. Do not include the word Extract, only provide the number(s).\n2) Request additional documentation. If, in the body of the extract(s) provided, there is a reference to another section that is directly relevant and not already provided, respond with the word 'SECTION:' followed by 'Extract extract_number, Reference: section_reference' - for example SECTION: Extract 1, Reference: {ref_string}.\n3) State 'NONE:' and nothing else in all other cases\n"
    assert path_rag.create_system_message_RAG(number_of_options=3, review = True) == expected_message

    expected_message = f"Please review your answer. You were asked to assist the user by responding to their question in 1 of 2 ways but your response does not follow the expected format. Please reformat your response so that it follows the requested format.\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference:'. Do not include the word Extract, only provide the number(s).\n2) State 'NONE:' and nothing else in all other cases\n"
    assert path_rag.create_system_message_RAG(number_of_options=2, review = True) == expected_message

def test_check_response_RAG(dummy_definitions, dummy_search_sections):
    # Setup
    corpus_index = NavigatingIndex()
    corpus_index = NavigatingIndex()
    api_key = os.environ.get("OPENAI_API_KEY")
    chat_parameters = ChatParameters(chat_model="gpt-4o-mini", 
                                     api_key=api_key, 
                                     temperature=0.0, 
                                     max_tokens=200, 
                                     token_limit_when_truncating_message_queue = 3500)

    path_rag = PathRAG(corpus_index= corpus_index, chat_parameters=chat_parameters)

    reference_key_word = "Reference:"
    # Test Case where the reference_key_word appears more than once
    
    llm_response_text = f"{PathRAG.LLMPrefix.ANSWER.value} This is part of an answer. Reference: 1, 2. \n\nThen here is more of the answer. Reference: 3, 4"
    llm_message_response = {"role": "assistant", "content": llm_response_text}
    result = path_rag.check_response_RAG(llm_message_response  = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert result["RAGPath"] == PathRAG.RAGPath.FOLLOWUP.value
    assert result["llm_followup_instruction"] == f"When answering the question, you used the keyword 'Reference:' more than once. It is vitally important that this keyword is only used once in your answer and then only at the end of the answer followed only by an integer, comma separated list of the extracts used. Please reformat your response so that there is only one instance of the keyword 'Reference:' and it is at the end of the answer."

    # Test case for a valid ANSWER response without references
    llm_response = f"{PathRAG.LLMPrefix.ANSWER.value} This is a valid answer without references."
    llm_message_response = {"role": "assistant", "content": llm_response}
    result = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert "assistant_response" in result
    assert isinstance(result["assistant_response"], AnswerWithoutRAGResponse)        
    assert result["assistant_response"].answer == "This is a valid answer without references."
    assert result["assistant_response"].caveat == get_caveat_for_no_rag_response()

    # Test case for a valid ANSWER response with references
    llm_response = f"{PathRAG.LLMPrefix.ANSWER.value} This is a valid answer. {reference_key_word} 1, 2"
    llm_message_response = {"role": "assistant", "content": llm_response}
    result = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert "assistant_response" in result
    assert isinstance(result["assistant_response"], AnswerWithRAGResponse)        
    assert result["assistant_response"].answer == "This is a valid answer."
    assert result["assistant_response"].references.shape[0] == 2

    # Test case for an ANSWER but with non integer references 
    llm_response = f"{PathRAG.LLMPrefix.ANSWER.value} This is a valid answer. {reference_key_word} A"
    llm_message_response = {"role": "assistant", "content": llm_response}
    result = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert result["RAGPath"] == PathRAG.RAGPath.FOLLOWUP.value
    assert result["llm_followup_instruction"] == "When answering the question, you have made reference to an extract but I am unable to extract the number from your reference. Please re-write your answer using integer extract number(s)"

    # Test case for an ANSWER but with something that is not an integer but from which an integer can be extracted useing regex 
    llm_response = f"{PathRAG.LLMPrefix.ANSWER.value} This is a valid answer. {reference_key_word} 10.0"
    llm_message_response = {"role": "assistant", "content": llm_response}
    result = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert result["RAGPath"] == PathRAG.RAGPath.FOLLOWUP.value
    assert result["llm_followup_instruction"] == "When answering the question, you have made reference to an extract number that was not provided. Please re-write your answer and only refer to the extracts provided by their number"

    # test case for SECTION response
    llm_response = f"{PathRAG.LLMPrefix.SECTION.value} Extract 1, Reference 1.1"
    llm_message_response = {"role": "assistant", "content": llm_response}
    result = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert result["RAGPath"] == PathRAG.RAGPath.SECTION.value
    assert result["extract"] == 1
    assert result["document"] == "WRR"
    assert result["section"] == "1.1"

    # test case for SECTION response when references are not specified correctly - NOTE the ":" after the word Reference
    llm_response = f"{PathRAG.LLMPrefix.SECTION.value} Extract 1, Reference: 1.1.2"
    llm_message_response = {"role": "assistant", "content": llm_response}   
    result = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert result["RAGPath"] == PathRAG.RAGPath.SECTION.value
    assert result["extract"] == 1
    assert result["document"] == "WRR"
    assert result["section"] == "1.1.2"


    # test case for SECTION response when references are not specified correctly 
    llm_response = f"{PathRAG.LLMPrefix.SECTION.value} Reference 1, Section 1.1.2"
    llm_message_response = {"role": "assistant", "content": llm_response}   
    result = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert result["RAGPath"] == PathRAG.RAGPath.FOLLOWUP.value
    assert result["llm_followup_instruction"] == r'When requesting an additional section, you did not use the format "Extract (\d+), Reference (.+)" or you included additional text. Please re-write your response using this format'

    # test the NONE response
    llm_response = f"{PathRAG.LLMPrefix.NONE.value} This is a valid answer. {reference_key_word} 1, 2"
    llm_message_response = {"role": "assistant", "content": llm_response}
    result = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert "assistant_response" in result
    assert isinstance(result["assistant_response"], NoAnswerResponse)
    assert result["assistant_response"].classification == NoAnswerClassification.NO_RELEVANT_DATA

    # NOTE: No ERROR path for LLM responses

    # test the case where the LLM response does not start with one of the valid prefixes
    llm_response = "This is not a valid response"
    llm_message_response = {"role": "assistant", "content": llm_response}
    result = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert result["RAGPath"] == PathRAG.RAGPath.FOLLOWUP.value
    assert result["llm_followup_instruction"] == f"Your response, did not begin with one of the keywords, '{PathRAG.LLMPrefix.ANSWER.value}', '{PathRAG.LLMPrefix.SECTION.value}' or '{PathRAG.LLMPrefix.NONE.value}'. Please review the question and provide an answer in the required format. Also make sure the referenced extracts are quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword '{reference_key_word}'. Do not include the word Extract, only provide the number(s).\n"


def test_add_section_to_resource(dummy_definitions, dummy_search_sections):
    # A valid "result" dictionary looks like this:
    # {"success": True, "RAGPath": RAGPath.SECTION.value, "extract": extract_number, "document": document_name, "section": section_reference}

    corpus_index = NavigatingIndex()
    api_key = os.environ.get("OPENAI_API_KEY")
    chat_parameters = ChatParameters(chat_model="gpt-4o-mini", 
                                     api_key=api_key, 
                                     temperature=0.0, 
                                     max_tokens=200, 
                                     token_limit_when_truncating_message_queue = 3500)

    path_rag = PathRAG(corpus_index= corpus_index, chat_parameters=chat_parameters)

    # check if the section string passes validation but does not refer to something in the document
    result = {"success": True, "RAGPath": PathRAG.RAGPath.SECTION.value, "extract": 1, "document": 'WRR', "section": "9.1"}
    success, df_updated = path_rag.add_section_to_resource(result = result, df_definitions = dummy_definitions, df_search_sections = dummy_search_sections)
    assert success == False
    assert len(df_updated) == 3
    assert df_updated.iloc[0]['section_reference'] == "1.2"
    assert df_updated.iloc[1]['section_reference'] == '1.3'
    assert df_updated.iloc[2]['section_reference'] == 'A.2(A)(i)'

    # Adding a valid string
    result = {"success": True, "RAGPath": PathRAG.RAGPath.SECTION.value, "extract": 5, "document": 'WRR', "section": "1.1"}
    success, df_updated = path_rag.add_section_to_resource(result = result, df_definitions = dummy_definitions, df_search_sections = dummy_search_sections)
    assert success == True
    assert len(df_updated) == 4
    assert df_updated.iloc[0]['section_reference'] == "1.2"
    assert df_updated.iloc[1]['section_reference'] == '1.3'
    assert df_updated.iloc[2]['section_reference'] == 'A.2(A)(i)'
    assert df_updated.iloc[3]['section_reference'] == '1.1'

    # check if the section string only comes from the definitions
    result = {"success": True, "RAGPath": PathRAG.RAGPath.SECTION.value, "extract": 1, "document": 'WRR', "section": "1.1"}
    success, df_updated = path_rag.add_section_to_resource(result = result, df_definitions = dummy_definitions, df_search_sections = dummy_search_sections)
    assert len(df_updated) == 4
    assert success == True
    assert df_updated.iloc[0]['section_reference'] == "1.2"
    assert df_updated.iloc[1]['section_reference'] == '1.3'
    assert df_updated.iloc[2]['section_reference'] == 'A.2(A)(i)'
    assert df_updated.iloc[3]['section_reference'] == '1.1'

def test_process_llm_response(dummy_definitions, dummy_search_sections):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable is not set. Skipping this test.")
    chat_parameters = ChatParameters(chat_model="gpt-4o-mini", 
                                     api_key=api_key, 
                                     temperature=0.0, 
                                     max_tokens=200, 
                                     token_limit_when_truncating_message_queue = 3500)

    corpus_index = NavigatingIndex()
    user_message = {"role": "user", "content": "Test question", "reference_material": {"definitions": dummy_definitions, "sections": dummy_search_sections}}
    path_rag = PathRAG(corpus_index= corpus_index, chat_parameters=chat_parameters)

    # Test case 1: Invalid response (should trigger FOLLOWUP)
    reference_key_word_RAG = "Reference:"
    invalid_response = "This is an invalid response without a proper prefix"
    llm_message_response = {"role": "assistant", "content": invalid_response}
    check_response = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    assert check_response["RAGPath"] == PathRAG.RAGPath.FOLLOWUP.value
    assert check_response["llm_followup_instruction"] == f"Your response, did not begin with one of the keywords, '{PathRAG.LLMPrefix.ANSWER.value}', '{PathRAG.LLMPrefix.SECTION.value}' or '{PathRAG.LLMPrefix.NONE.value}'. Please review the question and provide an answer in the required format. Also make sure the referenced extracts are quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword '{reference_key_word_RAG}'. Do not include the word Extract, only provide the number(s).\n"
    # Create a mock for the get_api_response method
    mock_get_api_response = MagicMock(return_value=f"{PathRAG.LLMPrefix.ANSWER.value} But the followup call to the API then fixes it: This is a test answer. Reference: 1, 2")
    # Patch the get_api_response method of the ChatParameters instance
    with patch.object(path_rag.chat_parameters, 'get_api_response', mock_get_api_response):
        result = path_rag.process_llm_response(llm_checked_response=check_response, message_history=[], current_user_message=user_message)
    assert "assistant_response" in result
    assert isinstance(result["assistant_response"], AnswerWithRAGResponse)
    assert result["content"] == 'But the followup call to the API then fixes it: This is a test answer. \n\nReference: \n\nDefinition 1 from Navigating Whale Rock Ridge: \n\nMy definition from WRR  \n\nDefinition A.1 from Navigating Plett: \n\nMy definition from Plett  \n\n'

    # Test case 2: RAGPath.SECTION which then works
    section_response = f"{PathRAG.LLMPrefix.SECTION.value} Extract 1, Reference 1.1"
    llm_message_response = {"role": "assistant", "content": section_response}
    checked_response = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)


    mock_resource_augmented_query = MagicMock(return_value={"role": "assistant", "content": f"{PathRAG.LLMPrefix.ANSWER.value} OK, I was able to answer with the new reference. Reference: 1, 2"})    
    # Use patch as a context manager
    with patch.object(path_rag, 'resource_augmented_query', mock_resource_augmented_query):
        result = path_rag.process_llm_response(llm_checked_response=checked_response, message_history=[], current_user_message=user_message)
    assert isinstance(result["assistant_response"], AnswerWithRAGResponse)

    # Test case 3: RAGPath.SECTION which asks for yet another section
    section_response = f"{PathRAG.LLMPrefix.SECTION.value} Extract 1, Reference 1.1"
    llm_message_response = {"role": "assistant", "content": section_response}
    checked_response = path_rag.check_response_RAG(llm_message_response = llm_message_response, df_definitions = dummy_definitions, df_sections = dummy_search_sections)
    mock_resource_augmented_query = MagicMock(return_value={"role": "assistant", "content": f"{PathRAG.LLMPrefix.SECTION.value} Extract 1, Reference 2.1"})
    # Use patch as a context manager
    with patch.object(path_rag, 'resource_augmented_query', mock_resource_augmented_query):
        result = path_rag.process_llm_response(llm_checked_response=checked_response, message_history=[], current_user_message=user_message)
    
    assert isinstance(result["assistant_response"], ErrorResponse)
    assert result["assistant_response"].classification == ErrorClassification.NOT_FOLLOWING_INSTRUCTIONS

    # # Test case 7: Invalid RAGPath (this should never happen in practice, but good to test)
    # # Currently this will trigger RAGPath.FOLLOWUP with the instruction to reformulate the answer    
    # # You might need to modify your implementation to test this case
    # # For example, you could add a special test prefix that triggers this case
    # # invalid_path_response = "TEST_INVALID_PATH: This should trigger an error"
    # # result = process_llm_response(chat_data, invalid_path_response)
    # # assert result["RAGPath"] == RAGPath.ERROR.value
    # # assert "The RAGPath is not valid" in result["error_message"]    


def test_extract_used_references(dummy_definitions, dummy_search_sections):
    corpus_index = NavigatingIndex()
    corpus_index = NavigatingIndex()
    api_key = os.environ.get("OPENAI_API_KEY")
    chat_parameters = ChatParameters(chat_model="gpt-4o-mini", 
                                     api_key=api_key, 
                                     temperature=0.0, 
                                     max_tokens=200, 
                                     token_limit_when_truncating_message_queue = 3500)

    path = PathRAG(corpus_index= corpus_index, chat_parameters=chat_parameters)



    # Test case 1: Valid result with both definition and section references
    result = {
        "reference": [1, 3]  # 1 is a definition, 3 is a section
    }
    df_result = path.extract_used_references(integer_list_of_used_references = result["reference"], df_definitions = dummy_definitions, df_search_sections = dummy_search_sections)

    assert len(df_result) == 2
    assert df_result.iloc[0]['is_definition'] == True
    assert df_result.iloc[0]['text'] == "My definition from WRR"
    assert df_result.iloc[1]['is_definition'] == False
    assert df_result.iloc[1]['section_reference'] == "1.2"

    # Test case 2: Valid result with only definition references
    result = {
        "reference": [1, 2]  # Both are definitions
    }
    df_result = path.extract_used_references(integer_list_of_used_references = result["reference"], df_definitions = dummy_definitions, df_search_sections = dummy_search_sections)
    
    assert len(df_result) == 2
    assert all(df_result['is_definition'] == True)
    assert df_result.iloc[0]['text'] == "My definition from WRR"
    assert df_result.iloc[1]['text'] == "My definition from Plett"

    # Test case 3: Valid result with only section references
    result = {
        "reference": [3, 4, 5]  # All are sections
    }
    df_result = path.extract_used_references(integer_list_of_used_references = result["reference"], df_definitions = dummy_definitions, df_search_sections = dummy_search_sections)
    
    assert len(df_result) == 3
    assert all(df_result['is_definition'] == False)
    assert df_result.iloc[0]['section_reference'] == "1.2"
    assert df_result.iloc[1]['section_reference'] == "1.3"
    assert df_result.iloc[2]['section_reference'] == "A.2(A)(i)"

    # Test case 4: Empty reference list
    result = {
        "reference": []
    }
    df_result = path.extract_used_references(integer_list_of_used_references = result["reference"], df_definitions = dummy_definitions, df_search_sections = dummy_search_sections)
    
    assert len(df_result) == 0

    # Test case 5: Reference number out of range
    result = {
        "reference": [1, 6]  # 6 is out of range
    }
    with pytest.raises(IndexError):
        path.extract_used_references(integer_list_of_used_references = result["reference"], df_definitions = dummy_definitions, df_search_sections = dummy_search_sections)


def test_perform_RAG_path(dummy_definitions, dummy_search_sections):
    api_key=os.environ.get("OPENAI_API_KEY")
    chat_parameters = ChatParameters(chat_model = "gpt-4o",  
                                     api_key=api_key, 
                                     temperature = 0, 
                                     max_tokens = 500, 
                                     token_limit_when_truncating_message_queue = 3500)
    embedding_parameters = EmbeddingParameters("text-embedding-3-large", 1024)
    corpus_index = NavigatingIndex()
    rerank_algo  = RerankAlgos.NONE
    user_content = "How do I get to the gym?"
    path_search = PathSearch(corpus_index = corpus_index, 
                             chat_parameters = chat_parameters, 
                             embedding_parameters = embedding_parameters, 
                             rerank_algo = rerank_algo)
    workflow_triggered, relevant_definitions, relevant_sections = path_search.similarity_search(user_content)
    user_message = {"content": user_content, "reference_material": {"definitions": relevant_definitions, "sections": relevant_sections}}

    path_rag = PathRAG(corpus_index = corpus_index, chat_parameters = chat_parameters)

    # Check that if the api returns an answer: 
    mock_get_api_response = MagicMock(return_value="ANSWER: test to see what happens when if the API believes it successfully answered the question with the resources provided")    
    # Use patch as a context manager
    with patch.object(path_rag.chat_parameters, 'get_api_response', mock_get_api_response):
        result = path_rag.perform_RAG_path(message_history=[], current_user_message=user_message)

    assert "assistant_response" in result
    assert result["role"] == "assistant"
    assert result["content"] == 'NOTE: The following answer is provided without references and should therefore be treated with caution. \n\ntest to see what happens when if the API believes it successfully answered the question with the resources provided'
    assert isinstance(result["assistant_response"], AnswerWithoutRAGResponse)

    # Check when the api returns NONE: 
    mock_get_api_response = MagicMock(return_value="NONE:")
    with patch.object(path_rag.chat_parameters, 'get_api_response', mock_get_api_response):
        result = path_rag.perform_RAG_path(message_history=[], current_user_message=user_message)

    assert "assistant_response" in result
    assert isinstance(result["assistant_response"], NoAnswerResponse)
    assert result["assistant_response"].classification == NoAnswerClassification.NO_RELEVANT_DATA
    assert result["role"] == "assistant"
    
    
    # Check the "section_reference" branch
    mock_get_api_response = MagicMock()
    mock_get_api_response.side_effect = [
        "SECTION: Extract 1, Reference 1.1",
        "ANSWER: test to see what happens when if the API believes it successfully answered the question with the resources provided"
    ]    
    with patch.object(path_rag.chat_parameters, 'get_api_response', mock_get_api_response):
        result = path_rag.perform_RAG_path(message_history=[], current_user_message=user_message)

    assert "assistant_response" in result
    assert isinstance(result["assistant_response"], AnswerWithoutRAGResponse)
    assert result["role"] == "assistant"
    assert result["content"] == 'NOTE: The following answer is provided without references and should therefore be treated with caution. \n\ntest to see what happens when if the API believes it successfully answered the question with the resources provided'

    # If the fist call is does not follow instrucitons, there is only one followup call. If this creates 
    # a valid message that needs ANOTHER followup call, the method will still fail - I may want to fix this
    # but the test here is to make sure that the current implementation works as expected
    mock_get_api_response = MagicMock()
    mock_get_api_response.side_effect = [
        "Test to see what happens when if the API cannot listen to instructions", 
        "SECTION: Extract 1, Reference 1.1"]
    with patch.object(path_rag.chat_parameters, 'get_api_response', mock_get_api_response):
        result = path_rag.perform_RAG_path(message_history=[], current_user_message=user_message)

    assert "assistant_response" in result
    assert isinstance(result["assistant_response"], ErrorResponse)
    assert result["role"] == "assistant"
    assert result["content"] == "SECTION: Extract 1, Reference 1.1"
    assert result["assistant_response"].classification == ErrorClassification.NOT_FOLLOWING_INSTRUCTIONS
    # Check that the mock was called twice
    assert mock_get_api_response.call_count == 2

    # Check the stubbornly disobedient branch
    mock_get_api_response = MagicMock()
    mock_get_api_response.side_effect = [
        "Test to see what happens when if the API cannot listen to instructions", 
        "But even after marking its own homework it cannot listen to instructions"
    ]
    with patch.object(path_rag.chat_parameters, 'get_api_response', mock_get_api_response):
        result = path_rag.perform_RAG_path(message_history=[], current_user_message=user_message)

    assert "assistant_response" in result
    assert isinstance(result["assistant_response"], ErrorResponse)
    assert result["role"] == "assistant"
    assert result["content"] == "But even after marking its own homework it cannot listen to instructions"
    assert result["assistant_response"].classification == ErrorClassification.NOT_FOLLOWING_INSTRUCTIONS



