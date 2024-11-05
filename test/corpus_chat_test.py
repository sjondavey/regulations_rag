import os
import pandas as pd
from openai import OpenAI
import pytest
from unittest.mock import patch, MagicMock

from regulations_rag.embeddings import  EmbeddingParameters
from regulations_rag.rerank import RerankAlgos
from regulations_rag.corpus_chat import CorpusChat
from regulations_rag.data_classes import AnswerWithRAGResponse, AnswerWithoutRAGResponse, \
    NoAnswerClassification, NoAnswerResponse, ErrorClassification, ErrorResponse

from regulations_rag.corpus_chat_tools import ChatParameters, get_caveat_for_no_rag_response

#from .navigating_corpus import NavigatingCorpus
from .navigating_index import NavigatingIndex


class TestCorpusChat:
    api_key=os.environ.get("OPENAI_API_KEY")
    # openai_client = OpenAI(api_key=api_key,) # moved to chat_parameters

    chat_parameters = ChatParameters(chat_model = "gpt-4o",  
                                     api_key=api_key, 
                                     temperature = 0, 
                                     max_tokens = 500, 
                                     token_limit_when_truncating_message_queue = 3500)

    embedding_parameters = EmbeddingParameters("text-embedding-3-large", 1024)

    #key = os.getenv('encryption_key_gdpr')
    corpus_index = NavigatingIndex()
    rerank_algo  = RerankAlgos.NONE
    chat = CorpusChat(
                        embedding_parameters = embedding_parameters, 
                        chat_parameters = chat_parameters, 
                        corpus_index = corpus_index,
                        rerank_algo = rerank_algo,   
                        user_name_for_logging = 'test_user')
    regression_test_enabled = False

    def test_construction(self):
        assert True

    def test_reset_conversation_history(self):
        self.chat.messages_intermediate.append({"role": "user", "content": "test"})
        self.chat.reset_conversation_history()
        assert len(self.chat.messages_intermediate) == 0
        assert self.chat.system_state == CorpusChat.State.RAG

    def test_append_content(self):
        self.chat.reset_conversation_history()
        # check that the same message is not added twice
        self.chat.append_content(message = {"role": "user", "content": "test"})
        assert len(self.chat.messages_intermediate) == 1
        assert self.chat.messages_intermediate[-1]["content"] == "test"
        assert self.chat.messages_intermediate[-1]["role"] == "user"
        self.chat.append_content(message = {"role": "user", "content": "test"})
        assert len(self.chat.messages_intermediate) == 1

        # check that a system message is added
        self.chat.append_content(message = {"role": "system", "content": "test"})
        assert len(self.chat.messages_intermediate) == 2
        assert self.chat.messages_intermediate[-1]["content"] == "test"
        assert self.chat.messages_intermediate[-1]["role"] == "system"

        # check that an assistant message is added
        self.chat.append_content(message = {"role": "assistant", "content": "test"})
        assert len(self.chat.messages_intermediate) == 3
        assert self.chat.messages_intermediate[-1]["content"] == "test"
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        self.chat.reset_conversation_history()

    def test_place_in_stuck_state(self):        
        self.chat.place_in_stuck_state(ErrorClassification.STUCK)
        assert self.chat.system_state == CorpusChat.State.STUCK
        assert len(self.chat.messages_intermediate) == 1
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], ErrorResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == ErrorClassification.STUCK
        assert self.chat.messages_intermediate[-1]["content"] == ErrorClassification.STUCK.value
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        # check that the same message is not added twice
        self.chat.place_in_stuck_state()
        assert self.chat.system_state == CorpusChat.State.STUCK
        assert len(self.chat.messages_intermediate) == 1

        self.chat.reset_conversation_history()
        assert self.chat.system_state == CorpusChat.State.RAG
        assert len(self.chat.messages_intermediate) == 0

        self.chat.place_in_stuck_state(error_classification = ErrorClassification.CALL_FOR_MORE_DOCUMENTS_FAILED)
        assert self.chat.system_state == CorpusChat.State.STUCK
        assert len(self.chat.messages_intermediate) == 1
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], ErrorResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == ErrorClassification.CALL_FOR_MORE_DOCUMENTS_FAILED
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"



    def test_user_provides_input(self):
        self.chat.reset_conversation_history()
        # check the empty input
        self.chat.user_provides_input(None)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == ErrorClassification.ERROR.value
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], ErrorResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == ErrorClassification.ERROR

        # check the response if the system is stuck
        self.chat.reset_conversation_history()
        self.chat.system_state = self.chat.State.STUCK
        user_content = "How do I get to the Gym?"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == ErrorClassification.ERROR.value
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], ErrorResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == ErrorClassification.ERROR
        assert self.chat.system_state == self.chat.State.STUCK
        # try again and make sure no more messages are added
        self.chat.user_provides_input(user_content)
        assert len(self.chat.messages_intermediate) == 1

        # check the response if the system is in an unknown state
        self.chat.reset_conversation_history()
        self.chat.system_state = "random state not in list"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == ErrorClassification.ERROR.value
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], ErrorResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == ErrorClassification.ERROR
        assert self.chat.system_state == self.chat.State.STUCK

        # test that a workflow triggers as expected
        self.chat.reset_conversation_history()
        self.chat.system_state = self.chat.State.RAG
        self.chat.user_provides_input("Can I see this on a map?")
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == ErrorClassification.WORKFLOW_NOT_IMPLEMENTED.value
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], ErrorResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == ErrorClassification.WORKFLOW_NOT_IMPLEMENTED
        assert self.chat.system_state == self.chat.State.STUCK

        # test the path for execute_path_no_retrieval_no_conversation_history
        
        self.chat.reset_conversation_history()
        self.chat.system_state = self.chat.State.RAG
        self.chat.user_provides_input("Hi")
        self.chat.strict_rag = True
        # test the path if the system answers the question as hoped
        corpus_index = NavigatingIndex()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY environment variable is not set. Skipping this test.")
        chat_parameters = self.chat.chat_parameters
        self.chat.reset_conversation_history()
        self.chat.system_state = self.chat.State.RAG
        self.chat.strict_rag = True
        flag = "ANSWER: "
        input_response = 'Drive to West Gate. Reference: 2'
        output_response = 'Drive to West Gate.  \nReference:  \nSection A.2(A) from Navigating Plett'
        mock_get_api_response = flag + input_response
        with patch.object(chat_parameters, 'get_api_response', return_value = mock_get_api_response):
            self.chat.user_provides_input(user_content)

        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], AnswerWithRAGResponse)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == 'Drive to West Gate. \n\nReference: \n\nSection A.2(A) from Navigating Plett: \n\n# A.2 Directions\n\n## A.2(A) To the Gym\n\n### A.2(A)(i) From West Gate (see 1.1)\n\nTurn left into Longships Drive and right at the T-junction into Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\n\n### A.2(A)(ii) From Main Gate (see 1.2)\n\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\n\n### A.2(A)(iii) From South Gate (see 1.3)\n\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym  \n\n'
        assert self.chat.system_state == self.chat.State.RAG # rag
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"] == 'How do I get to the Gym?'
        assert "reference_material" in self.chat.messages_intermediate[-2]


        # test the workflow if the system cannot find useful content in the supplied data
        self.chat.system_state = self.chat.State.RAG
        flag = "NONE: "
        response = ""
        mock_get_api_response = flag 
        with patch.object(chat_parameters, 'get_api_response', return_value = mock_get_api_response):
            self.chat.user_provides_input(user_content)
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], NoAnswerResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == NoAnswerClassification.NO_RELEVANT_DATA
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"].strip() == 'How do I get to the Gym?'
        assert self.chat.system_state == self.chat.State.RAG

        # test the workflow if the system needs additional content
        self.chat.system_state = self.chat.State.RAG
        self.chat.reset_conversation_history()
        flag = "SECTION: "
        response = "Extract 2, Reference 1.1"
        first_response = flag + response
        # now the response once it has received the additional data
        flag = "ANSWER: "
        input_response = 'Drive to the West Gate. Reference: 2'
        # NOTE this output_response differs to the previous one because we have changed the search_sections while leaving the input_response to refer to extract 2 (which is now different)
        output_response = 'Drive to the West Gate.  \nReference:  \nSection A.2(A) from Navigating Plett'
        second_response = flag + input_response

        mock_get_api_response = MagicMock()
        mock_get_api_response.side_effect = [first_response, second_response]
        with patch.object(chat_parameters, 'get_api_response', mock_get_api_response):
            self.chat.user_provides_input(user_content)

        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], AnswerWithRAGResponse)        
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == 'Drive to the West Gate. \n\nReference: \n\nSection A.2(A) from Navigating Plett: \n\n# A.2 Directions\n\n## A.2(A) To the Gym\n\n### A.2(A)(i) From West Gate (see 1.1)\n\nTurn left into Longships Drive and right at the T-junction into Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\n\n### A.2(A)(ii) From Main Gate (see 1.2)\n\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\n\n### A.2(A)(iii) From South Gate (see 1.3)\n\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym  \n\n'
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"] == 'How do I get to the Gym?'
        assert self.chat.system_state == self.chat.State.RAG


        # Test what happens if it calls for a section that it already has
        self.chat.reset_conversation_history()
        self.chat.system_state = self.chat.State.RAG
        flag = "SECTION: "
        response = "Extract 2, Reference A.2(A)(i)"
        first_response = flag + " " + response
        # now the response once it has received the additional data
        flag = "ANSWER: "
        input_response = 'Drive to the West Gate. Reference: 2'
        # NOTE this output_response differs to the previous one because we have changed the search_sections while leaving the input_response to refer to extract 2 (which is now different)
        second_response = flag + input_response        
        mock_get_api_response.side_effect = [first_response, second_response]
        with patch.object(chat_parameters, 'get_api_response', mock_get_api_response):
            self.chat.user_provides_input(user_content)
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], AnswerWithRAGResponse)        
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == 'Drive to the West Gate. \n\nReference: \n\nSection A.2(A) from Navigating Plett: \n\n# A.2 Directions\n\n## A.2(A) To the Gym\n\n### A.2(A)(i) From West Gate (see 1.1)\n\nTurn left into Longships Drive and right at the T-junction into Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\n\n### A.2(A)(ii) From Main Gate (see 1.2)\n\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\n\n### A.2(A)(iii) From South Gate (see 1.3)\n\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym  \n\n'
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"] == 'How do I get to the Gym?'
        assert self.chat.system_state == self.chat.State.RAG

        #test what happens if the LLM does not listen to instructions and returns something random
        self.chat.reset_conversation_history()
        self.chat.system_state = self.chat.State.RAG
        response = "None of the supplied documentation was relevant"
        first_response = response
        second_response = response
        mock_get_api_response.side_effect = [first_response, second_response]
        with patch.object(chat_parameters, 'get_api_response', mock_get_api_response):
            self.chat.user_provides_input(user_content)
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], ErrorResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == ErrorClassification.NOT_FOLLOWING_INSTRUCTIONS
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"].strip() == 'How do I get to the Gym?'




    def test_execute_path_no_retrieval_no_conversation_history(self):
        self.chat.reset_conversation_history()
        self.chat.strict_rag = True
        user_content = "What is an exchange rate?"
        self.chat.execute_path_no_retrieval_no_conversation_history(user_content)
        
        assert self.chat.system_state == self.chat.State.RAG         
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"] == user_content
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], NoAnswerResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == NoAnswerClassification.NO_DATA
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == NoAnswerResponse(NoAnswerClassification.NO_DATA).create_openai_content()

        self.chat.strict_rag = False
        self.chat.reset_conversation_history()
        mock_get_api_response = "Not Relevant. This does not have anything to do with the topic."
        with patch.object(self.chat.chat_parameters, 'get_api_response', return_value = mock_get_api_response):
            self.chat.execute_path_no_retrieval_no_conversation_history(user_content)        

        assert self.chat.system_state == self.chat.State.RAG         
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"] == user_content
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], NoAnswerResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == NoAnswerClassification.QUESTION_NOT_RELEVANT
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == "This does not have anything to do with the topic."

        self.chat.strict_rag = False
        self.chat.reset_conversation_history()
        llm_response = "Really?! You don't know what an exchange rate is?"
        mock_get_api_response = MagicMock()
        mock_get_api_response.side_effect = [
            "Relevant",
            llm_response
        ]
        with patch.object(self.chat.chat_parameters, 'get_api_response', mock_get_api_response):
            self.chat.execute_path_no_retrieval_no_conversation_history(user_content)        
        assert self.chat.system_state == self.chat.State.RAG         
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"] == user_content
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], AnswerWithoutRAGResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].answer == llm_response
        assert self.chat.messages_intermediate[-1]["assistant_response"].caveat == get_caveat_for_no_rag_response()
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == get_caveat_for_no_rag_response() + "\n\n" + llm_response

        self.chat.strict_rag = True # reset the value



    def test_execute_path_no_retrieval_with_conversation_history(self):
        # execute_path_no_retrieval_with_conversation_history returns the same as execute_path_no_retrieval_no_conversation_history for now so no need to test
        assert True



    def test_execute_path_answer_question_with_no_data(self):
        self.chat.strict_rag = True
        self.chat.reset_conversation_history()
        # with strict rag on, this should return a NoAnswerResponse with classification QUESTION_NOT_RELEVANT
        user_content = "What is an exchange rate?"
        self.chat.execute_path_answer_question_with_no_data(user_content)
        assert self.chat.system_state == self.chat.State.RAG
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], NoAnswerResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == NoAnswerClassification.NO_DATA
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == NoAnswerClassification.NO_DATA.value


        self.chat.strict_rag = False
        self.chat.reset_conversation_history()
        # with strict rag off and an irrelevant answer, this should return a NoAnswerResponse with classification QUESTION_NOT_RELEVANT
        user_content = "What is an exchange rate?"
        self.chat.execute_path_answer_question_with_no_data(user_content)
        assert self.chat.system_state == self.chat.State.RAG
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], NoAnswerResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == NoAnswerClassification.QUESTION_NOT_RELEVANT
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        #assert self.chat.messages_intermediate[-1]["content"] == NoAnswerClassification.NO_DATA.value

        self.chat.strict_rag = True


    def test_regression(self):
        if not self.regression_test_enabled:
            pytest.skip("Skipping regression test")
            return

        #Because these will make calls to the LLM, they may fail for statistical reasons
        self.chat.reset_conversation_history()
        self.chat.strict_rag = True
        self.chat.system_state = self.chat.State.RAG

        # Small talk: similarity_search --> no search results --> execute_path_no_retrieval_no_conversation_history (strict rag) --> NO_DATA 
        self.chat.user_provides_input("Hi")
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == NoAnswerClassification.NO_DATA
        expected_path = [
            "CorpusChat.user_provides_input",
            "PathSearch.similarity_search",
            "CorpusChat.run_base_rag_path",
            "CorpusChat.execute_path_no_retrieval_no_conversation_history. Strict RAG"
        ]
        assert self.chat.execution_path == expected_path

        # Small talk: 
        self.chat.strict_rag = False
        self.chat._reset_execution_path()
        # similarity_search --> no results --> execute_path_no_retrieval_no_conversation_history (permissive rag) --> is_user_content_relevant  --> (no) UNABLE_TO_ANSWER 
        self.chat.user_provides_input("Hi")
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == NoAnswerClassification.QUESTION_NOT_RELEVANT
        expected_path = [
            "CorpusChat.user_provides_input",
            "PathSearch.similarity_search",
            "CorpusChat.run_base_rag_path",
            "CorpusChat.execute_path_no_retrieval_with_conversation_history",
            "CorpusChat.execute_path_no_retrieval_no_conversation_history. Permissive RAG",
            "PathNoRAGData.query_no_rag_data. Not relevant"
        ]
        assert self.chat.execution_path == expected_path

        # # Normal question with RAG
        # similarity_search --> has search results --> perform_RAG_path --> ANSWER --> AnswerWithRAGResponse
        self.chat.strict_rag = False
        self.chat._reset_execution_path()
        user_content = "How do I get to the Gym?"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], AnswerWithRAGResponse)
        self.chat.strict_rag = True
        expected_path = [
            "CorpusChat.user_provides_input",
            "PathSearch.similarity_search",
            "CorpusChat.run_base_rag_path",
            "PathRAG.perform_RAG_path",
            'PathRAG.resource_augmented_query',
            'PathRAG.check_response_RAG',
            'PathRAG.extract_used_references'
        ]
        assert self.chat.execution_path == expected_path

        # A question that it should not be able to answer
        # similarity_search --> no results --> execute_path_no_retrieval_no_conversation_history (permissive rag) --> is_user_content_relevant  --> (yes) UNABLE_TO_ANSWER 
        self.chat._reset_execution_path()
        user_content = "How do I get to the Woolworths?"
        self.chat.strict_rag = False
        self.chat.user_provides_input(user_content)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], NoAnswerResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == NoAnswerClassification.UNABLE_TO_ANSWER
        expected_path = [
            "CorpusChat.user_provides_input",
            "PathSearch.similarity_search",
            "CorpusChat.run_base_rag_path",
            "CorpusChat.execute_path_no_retrieval_with_conversation_history",
            "CorpusChat.execute_path_no_retrieval_no_conversation_history. Permissive RAG",
            "PathNoRAGData.query_no_rag_data. Relevant",
            "PathNoRAGData.query_no_rag_data. Relevant. No answer",
        ]
        assert self.chat.execution_path == expected_path


        # A question that should find sections but that it should not be able to answer
        # similarity_search --> has results --> perform_RAG_path --> NONE
        user_content = "How do I get to the west gate? But this MUST be via the the tennis courts."
        self.chat._reset_execution_path()
        self.chat.strict_rag = False
        self.chat.user_provides_input(user_content)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], NoAnswerResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == NoAnswerClassification.UNABLE_TO_ANSWER
        expected_path = [
            "CorpusChat.user_provides_input",
            "PathSearch.similarity_search",
            "CorpusChat.run_base_rag_path",
            "PathRAG.perform_RAG_path",
            'PathRAG.resource_augmented_query',
            'PathRAG.check_response_RAG',
            "CorpusChat.execute_path_answer_question_with_no_data. Permissive RAG"
        ]
        assert self.chat.execution_path == expected_path

        
        self.chat.strict_rag = True

