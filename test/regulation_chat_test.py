import pandas as pd
from openai import OpenAI
import os
from cryptography.fernet import Fernet

from regulations_rag.section_reference_checker import SectionReferenceChecker
from regulations_rag.regulation_chat import RegulationChat
from regulations_rag.data import  EmbeddingParameters, load_embedding_parameters, ChatParameters, load_chat_parameters
from regulations_rag.data_in_dataframes import load_data_from_files, DataInDataFrames, load_parquet_data, save_parquet_data


class TestRegulationChat:
    include_calls_to_api = True
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
    embedding_parameters = EmbeddingParameters(embedding_model = "text-embedding-ada-002", embedding_dimensions = 1536)
    chat_parameters = ChatParameters(chat_model = "gpt-3.5-turbo", temperature = 0, max_tokens = 500)

    #embedding_parameters = EmbeddingParameters(embedding_model = "text-embedding-ada-002", embedding_dimensions = 1536)
    # path_to_manual_as_csv_file = "./test/inputs/manual.csv"
    # path_to_additional_manual_as_csv_file = "./test/inputs/manual_plus.csv"
    # path_to_definitions_as_parquet_file = "./test/inputs/definitions.parquet"
    # path_to_additional_definitions_as_parquet_file = "./test/inputs/definitions_plus.parquet"
    # path_to_index_as_parquet_file = "./test/inputs/index.parquet"
    # path_to_additional_index_as_parquet_file = ""
    # workflow_as_parquet_file = "./test/inputs/workflow.parquet"


    embedding_parameters = load_embedding_parameters("text-embedding-3-large", 1024)
    path_to_manual_as_csv_file = "E:/Code/chat/cemad_rag/inputs/ad_manual.csv"
    path_to_additional_manual_as_csv_file = "E:/Code/chat/cemad_rag/inputs/ad_manual_plus.csv"
    path_to_definitions_as_parquet_file = "E:/Code/chat/cemad_rag/inputs/ad_definitions.parquet"
    path_to_additional_definitions_as_parquet_file = "E:/Code/chat/cemad_rag/inputs/ad_definitions_plus.parquet"
    path_to_index_as_parquet_file = "E:/Code/chat/cemad_rag/inputs/ad_index.parquet"
    path_to_additional_index_as_parquet_file = "E:/Code/chat/cemad_rag/inputs/ad_index_plus.parquet"
    workflow_as_parquet_file = "E:/Code/chat/cemad_rag/inputs/workflow.parquet"
    

    decryption_key = os.getenv('excon_encryption_key')

    df_regulations, df_definitions, df_index, df_workflow = load_data_from_files(
                                                                path_to_manual_as_csv_file = path_to_manual_as_csv_file, 
                                                                path_to_additional_manual_as_csv_file = path_to_additional_manual_as_csv_file, 
                                                                path_to_definitions_as_parquet_file = path_to_definitions_as_parquet_file, 
                                                                path_to_additional_definitions_as_parquet_file = path_to_additional_definitions_as_parquet_file, 
                                                                path_to_index_as_parquet_file = path_to_index_as_parquet_file, 
                                                                path_to_additional_index_as_parquet_file = path_to_additional_index_as_parquet_file,
                                                                workflow_as_parquet_file = workflow_as_parquet_file,
                                                                decryption_key=decryption_key )

    user_type = "Authorised Dealer (AD)" 
    regulation_name = "\'Currency and Exchange Manual for Authorised Dealers\' (Manual or CEMAD)"

    """
    Creates and returns a SectionReferenceChecker instance used for testing.
    """
    exclusion_list = ['Legal context', 'Introduction']
    index_patterns = [
        r'^[A-Z]\.\d{0,2}',             # Matches capital letter followed by a period and up to two digits.
        r'^\([A-Z]\)',                  # Matches single capital letters within parentheses.
        r'^\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx|xxi|xxii|xxiii|xxiv|xxv|xxvi|xxvii)\)', # Matches Roman numerals within parentheses.
        r'^\([a-z]\)',                  # Matches single lowercase letters within parentheses.
        r'^\([a-z]{2}\)',               # Matches two lowercase letters within parentheses.
        r'^\((?:[1-9]|[1-9][0-9])\)',   # Matches numbers within parentheses, excluding leading zeros.
    ]    
    text_pattern = r"[A-Z]\.\d{0,2}\([A-Z]\)\((?:i|ii|iii|iv|v|vi)\)\([a-z]\)\([a-z]{2}\)\(\d+\)"
    section_reference_checker = SectionReferenceChecker(regex_list_of_indices=index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)


    data = DataInDataFrames(user_type = user_type, 
                            regulation_name = regulation_name, 
                            section_reference_checker = section_reference_checker, 
                            df_regulations = df_regulations, 
                            df_definitions = df_definitions, 
                            df_index = df_index, 
                            df_workflow = df_workflow)

    chat = RegulationChat(openai_client, embedding_parameters, chat_parameters, data)

    # path_to_manual_as_csv_file_regulation_chat_test = "./inputs_test/manual.csv"
    # path_to_definitions_as_parquet_file_regulation_chat_test = "./inputs_test/definitions.parquet"
    # path_to_index_as_parquet_file_regulation_chat_test = "./inputs_test/index.parquet"
    # data_test = load_data_from_files(
    #                                path_to_manual_as_csv_file_regulation_chat_test, "", 
    #                                path_to_definitions_as_parquet_file_regulation_chat_test, "",
    #                                path_to_index_as_parquet_file_regulation_chat_test, "")    
    # regulation_chat_test = RegulationChat(openai_client, embedding_parameters, chat_parameters,  data_test)
    # include_calls_to_api = True


    def test_construction(self):
        assert True

    def test_reformat_assistant_answer(self):
        # if there are no sections in the rag, return the raw response 
        sections_in_rag = []
        raw_response = "Some random text here. Reference: E.(A)"
        formatted_response = self.chat.reformat_assistant_answer(raw_response, sections_in_rag)
        assert formatted_response == raw_response

        sections_in_rag = ["E.(A)", "B.12(A)"]
        raw_response = "Hi"
        formatted_response = self.chat.reformat_assistant_answer(raw_response, sections_in_rag)
        assert raw_response == formatted_response

        raw_response = "Some random text here. Reference: E.(A)"
        used_references, formatted_response = self.chat.reformat_assistant_answer(raw_response, sections_in_rag)
        assert formatted_response == "Some random text here.  \nReference:  \nE.(A): E. Non-resident Rand account, Customer Foreign Currency accounts, foreign currency accounts and foreign bank accounts. (A) Non-resident Rand accounts."
        assert len(used_references) == 1
        assert used_references[0] == 'E.(A)' 

        raw_response = "Some random text here. Reference: E.(A), B.12(A)"
        used_references, formatted_response = self.chat.reformat_assistant_answer(raw_response, sections_in_rag)
        assert formatted_response == "Some random text here.  \nReference:  \nE.(A): E. Non-resident Rand account, Customer Foreign Currency accounts, foreign currency accounts and foreign bank accounts. (A) Non-resident Rand accounts.  \nB.12(A): B.12 Merchanting, barter and counter trade. (A) Merchanting trade."
        assert len(used_references) == 2
        assert used_references[0] == 'E.(A)' 
        assert used_references[1] == 'B.12(A)' 


    def test_append_content(self):
        self.chat.append_content('user', 'Question: What documents are required')
        assert len(self.chat.messages) == 1
        assert self.chat.messages[-1]['content'] == 'Question: What documents are required'
        assert self.chat.messages[-1]['role'] == 'user'

        assert len(self.chat.messages_without_rag) == 1
        assert self.chat.messages_without_rag[-1]['role'] == 'user'
        assert self.chat.messages_without_rag[-1]['content'] == 'What documents are required'

        # Try to add content for a role that does not exist
        self.chat.reset_conversation_history()
        self.chat.append_content('other_role', 'Question: What documents are required')
        assert len(self.chat.messages) == 0
        assert len(self.chat.messages_without_rag) == 0

        self.chat.reset_conversation_history()
        self.chat.append_content('assistant', 'Answer here')
        assert len(self.chat.messages) == 1
        assert self.chat.messages[-1]['content'] == 'Answer here'
        assert self.chat.messages[-1]['role'] == 'assistant'

        assert len(self.chat.messages_without_rag) == 1
        assert self.chat.messages_without_rag[-1]['role'] == 'assistant'
        assert self.chat.messages_without_rag[-1]['content'] == 'Answer here'


    def test_user_provides_input(self):
        # check the response if the system is stuck
        self.chat.system_state = self.chat.State.STUCK
        user_content = "How much money can an individual take offshore in any year?"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"] == self.chat.Errors.STUCK.value
        assert self.chat.system_state == self.chat.State.STUCK

        # check the response if the system is in an unknown state
        self.chat.system_state = "random state not in list"
        user_content = "How much money can an individual take offshore in any year?"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"] == self.chat.Errors.UNKNOWN_STATE.value

        # check the response if there are no relevant documents        
        # self.chat.system_state = self.chat.system_states[0] # rag
        # user_content = "How much money can an individual take offshore in any year?"
        # self.chat.user_provides_input(user_content, 
        #                                threshold = 0.15, 
        #                                model_to_use="gpt-3.5-turbo", 
        #                                temperature = 0, 
        #                                max_tokens = 200)
        # assert self.chat.messages[-1]["role"] == "assistant"
        # assert self.chat.messages[-1]["content"] == self.chat.assistant_msg_no_data
        # #assert self.chat.system_state == "no_relevant_embeddings"
        # assert self.chat.system_state == self.chat.system_states[0] # rag

        # test the workflow if the system answers the question as hoped
        self.chat.system_state = self.chat.State.RAG
        user_content = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        flag = "ANSWER:"
        input_response = 'The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1).  \nReference:  \nC.(C)'
        output_response = 'The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1).  \nReference:  \nC.(C): C. Gold. (C) Acquisition of gold for trade purposes.'
        manual_responses_for_testing = [flag + input_response]
        self.chat.user_provides_input(user_content,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == output_response
        assert self.chat.system_state == self.chat.State.RAG # rag

        # test the workflow if the system cannot find useful content in the supplied data
        self.chat.system_state = self.chat.State.RAG
        user_content = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        flag = "NONE:"
        response = "None of the supplied documentation was relevant"
        manual_responses_for_testing = []
        manual_responses_for_testing.append(flag + response)
        self.chat.user_provides_input(user_content,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == self.chat.Errors.NO_RELEVANT_DATA.value
        assert self.chat.system_state == self.chat.State.RAG

        # test the workflow if the system needs additional content
        self.chat.system_state = self.chat.State.RAG
        user_content = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        flag = "SECTION:"
        #response = "C.(C)"
        response = "A.3(A)(i)"
        manual_responses_for_testing = []
        manual_responses_for_testing.append(flag + response)

        # now the response once it has received the additional data
        flag = "ANSWER:"
        response = "The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1).  \nReference:  \nC.(G): C. Gold."
        manual_responses_for_testing.append(flag + response)

        self.chat.user_provides_input(user_content,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-2]["role"] == "user"
        assert self.chat.messages[-2]["content"] == "Question: Who can trade gold?\n\nSections from the Manual\nC. Gold\n    (C) Acquisition of gold for trade purposes\n        (i) The acquisition of gold for legitimate trade purposes by e.g. manufacturing jewellers, dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator.\n        (ii) After receiving such approval, a permit must be obtained from SARS which will entitle the permit holder to approach Rand Refinery Limited for an allocation of gold.\n        (iii) The holders of gold, having received the approvals outlined above, are exempt from the provisions of Regulation 5(1).\nC. Gold\n    (B) Other exports of gold\n        (i) All applications for permission to export gold in any form should be referred to the South African Diamond and Precious Metals Regulator.\nA.3 Duties and responsibilities of Authorised Dealers\n    (A) Introduction\n        (i) Authorised Dealers should note that when approving requests in terms of the Authorised Dealer Manual, they are in terms of the Regulations, not allowed to grant permission to clients and must refrain from using wording that approval/permission is granted in correspondence with their clients. Instead reference should be made to the specific section of the Authorised Dealer Manual in terms of which the client is permitted to transact.\n"

        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == response
        assert self.chat.system_state == self.chat.State.RAG

        # Test what happens if it calls for a section that it already has
        self.chat.reset_conversation_history()
        response = "C.(C)"
        flag = "SECTION:"
        manual_responses_for_testing = []
        manual_responses_for_testing.append(flag + response)

        flag = "ANSWER:"
        response = "The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1).  \nReference:  \nC.(G): C. Gold."
        manual_responses_for_testing.append(flag + response)

        self.chat.user_provides_input(user_content,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-2]["role"] == "user"
        assert self.chat.messages[-2]["content"] == 'Question: Who can trade gold?\n\nSections from the Manual\nC. Gold\n    (C) Acquisition of gold for trade purposes\n        (i) The acquisition of gold for legitimate trade purposes by e.g. manufacturing jewellers, dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator.\n        (ii) After receiving such approval, a permit must be obtained from SARS which will entitle the permit holder to approach Rand Refinery Limited for an allocation of gold.\n        (iii) The holders of gold, having received the approvals outlined above, are exempt from the provisions of Regulation 5(1).\nC. Gold\n    (B) Other exports of gold\n        (i) All applications for permission to export gold in any form should be referred to the South African Diamond and Precious Metals Regulator.\n'

        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == response
        assert self.chat.system_state == self.chat.State.RAG


        # test what happens if the LLM does not listen to instructions and returns something random
        self.chat.system_state = self.chat.State.RAG
        user_content = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        response = "None of the supplied documentation was relevant"
        manual_responses_for_testing = []
        manual_responses_for_testing.append(response)
        manual_responses_for_testing.append(response) # need to add it twice when checking this branch
        self.chat.user_provides_input(user_content, 
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == self.chat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value
        assert self.chat.system_state == self.chat.State.STUCK

        # Now all the test with additional sections requested

    def test__add_rag_data_to_question(self):
        dfns = []
        dfns.append("def1")
        dfns.append("def2")
        df_definitions = pd.DataFrame(dfns, columns = ["Definition"])
        sections = []
        sections.append("A.1(A)(i)(aa)")
        sections.append("B.2(B)(ii)(bb)")
        df_search_sections = pd.DataFrame(sections, columns = ["regulation_text"])
        question = "user asks question"
        output_string = self.chat._add_rag_data_to_question(question, df_definitions, df_search_sections)

        expected_text = f"Question: {question}\n\nDefinitions from the Manual\ndef1\ndef2\n\
Sections from the Manual\nA.1(A)(i)(aa)\nB.2(B)(ii)(bb)\n"

        assert output_string == expected_text

    def test__create_system_message(self):
        short_pattern = r"[A-Z]\.\d{0,2}\([A-Z]\)\(\b(?:i|ii|iii|iv|v|vi)\b\)\([a-z]\)\([a-z]{2}\)\(\d+\)"
        expected_message = f"You are answering questions for an Authorised Dealer (AD) based only on the sections from the South African Exchange Control Manual that are provided. Please use the manual's index pattern when referring to sections: {short_pattern}. You have three options:\n\
1) Answer the question. Preface an answer with the tag 'ANSWER:'. End the answer with 'Reference: ' and a comma separated list of the section you used to answer the question if you used any.\n\
2) Request additional documentation. If, in the body of the sections provided, there is a reference to another section of the Manual that is directly relevant and not already provided, respond with the word 'SECTION:' followed by the full section reference.\n\
3) State 'NONE:' and nothing else in all other cases\n\n\""
        assert self.chat._create_system_message() == expected_message

        expected_message = f"You are answering questions for an Authorised Dealer (AD) based only on the sections from the South African Exchange Control Manual that are provided. Please use the manual's index pattern when referring to sections: {short_pattern}. You have two options:\n\
1) Answer the question. Preface an answer with the tag 'ANSWER:'. End the answer with 'Reference: ' and a comma separated list of the section you used to answer the question if you used any.\n\
2) State 'NONE:' and nothing else if you cannot answer the question with the resources provided\n\n\""
        assert self.chat._create_system_message(number_of_options=2) == expected_message



    def test_resource_augmented_query(self):
        user_content = "Who can trade gold?"
        workflow_triggered, relevant_definitions, relevant_sections = self.chat.similarity_search(user_content)
        # If there are no messages in the queue, we should get an error
        flag, response = self.chat.resource_augmented_query(df_definitions = relevant_definitions, 
                                                                  df_search_sections = relevant_sections)
        assert flag == self.chat.State.STUCK
        assert response == self.chat.State.STUCK

        # Add a message to the queue and give the LLM relevant data from which to answer the question
        self.chat.system_state = self.chat.State.RAG
        self.chat.messages = [{"role": "user", "content": user_content}]
        # NOTE: I am not going to test the openai api call. I am going to use 'testing' mode with canned answers
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("ANSWER: test to see what happens when if the API believes it successfully answered the question with the resources provided")
        flag, response = self.chat.resource_augmented_query(df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections,
                                                                testing = testing,
                                                                manual_responses_for_testing = manual_responses_for_testing)
        assert flag == self.chat.Prefix.ANSWER
        assert len(self.chat.messages) == 1
        assert self.chat.messages[-1]["role"] == "user"
        assert self.chat.messages[-1]["content"] == 'Question: Who can trade gold?\n\nSections from the Manual\nC. Gold\n    (C) Acquisition of gold for trade purposes\n        (i) The acquisition of gold for legitimate trade purposes by e.g. manufacturing jewellers, dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator.\n        (ii) After receiving such approval, a permit must be obtained from SARS which will entitle the permit holder to approach Rand Refinery Limited for an allocation of gold.\n        (iii) The holders of gold, having received the approvals outlined above, are exempt from the provisions of Regulation 5(1).\nC. Gold\n    (B) Other exports of gold\n        (i) All applications for permission to export gold in any form should be referred to the South African Diamond and Precious Metals Regulator.\n'

        if self.include_calls_to_api: # also test it with a call to the API
            flag, response = self.chat.resource_augmented_query(df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections)
            assert flag == self.chat.Prefix.ANSWER # "ANSWER:"
            # Also check that nothing happened to the internal message stack
            # assert len(self.chat.messages) == 1

        # Check that if the question and reference data mismatch, the system returns a NONE: value
        self.chat.messages = [{"role": "user", "content": "How much money can an individual take offshore in any year?"}]
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("NONE: test to see what happens when if the API believes it cannot answer the question with the resources provided")
        flag, response = self.chat.resource_augmented_query(df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections,
                                                                testing = testing,
                                                                manual_responses_for_testing = manual_responses_for_testing)
        assert flag == self.chat.Prefix.NONE
        # Also check that nothing happened to the internal message stack
        # assert len(self.chat.messages) == 1
        
        
        # Check the "section_reference" branch
        question = "Can you list the dispensations necessary for a rand facility to a non-resident exceed 6 months?"
        self.chat.messages = [{"role": "user", "content": question}]
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("SECTION: test to see what happens when if the API believes it needs information from another section")

        flag, response = self.chat.resource_augmented_query(df_definitions = relevant_definitions, 
                                                        df_search_sections = relevant_sections,
                                                        testing = testing,
                                                        manual_responses_for_testing = manual_responses_for_testing)
        assert flag == self.chat.Prefix.SECTION
        # Also check that nothing happened to the internal message stack
        # assert len(self.chat.messages) == 1

        # Check the despondent user branch
        self.chat.messages = [{"role": "user", "content": "How much money can an individual take offshore in any year?"}]
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("Test to see what happens when if the API cannot listen to instructions")
        manual_responses_for_testing.append("SECTION: but after marking its own homework it behaves")

        flag, response = self.chat.resource_augmented_query(df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections,
                                                                testing = testing,
                                                                manual_responses_for_testing = manual_responses_for_testing)
        assert flag == self.chat.Prefix.SECTION
        # Also check that nothing happened to the internal message stack
        # assert len(self.chat.messages) == 1

        # Check the stubbornly disobedient branch
        self.chat.messages = [{"role": "user", "content": "How much money can an individual take offshore in any year?"}]
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("Test to see what happens when if the API cannot listen to instructions")
        manual_responses_for_testing.append("But even after marking its own homework it cannot listen to instructions")

        flag, response = self.chat.resource_augmented_query(df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections,
                                                                testing = testing,
                                                                manual_responses_for_testing = manual_responses_for_testing)
        assert flag == self.chat.Prefix.FAIL
        # Also check that nothing happened to the internal message stack
        #assert len(self.chat.messages) == 1

        if self.include_calls_to_api:
            # Manually force the first API response to get to the second loop, then test the second API call
            self.chat.system_state = self.chat.State.RAG
            self.chat.messages = [{"role": "user", "content": user_content}]
            # NOTE: I am not going to test the openai api call. I am going to use 'testing' mode with canned answers
            testing = True
            manual_responses_for_testing = []
            manual_responses_for_testing.append("Hello")
            flag, response = self.chat.resource_augmented_query(df_definitions = relevant_definitions, 
                                                                    df_search_sections = relevant_sections,
                                                                    testing = testing,
                                                                    manual_responses_for_testing = manual_responses_for_testing)
            assert flag == self.chat.Prefix.ANSWER
            # Also check that nothing happened to the internal message stack
            # assert len(self.chat.messages) == 1


    def test_similarity_search(self):
        if self.include_calls_to_api:
            # Check that random chit-chat to the main dataset does not return any hits from the embeddings
            text = "Hi"
            workflow_triggered, df_definitions, df_search_sections = self.chat.similarity_search(text)
            assert len(df_definitions) == 0
            assert len(df_search_sections) == 0 
            # now move to the testing dataset for fine grained tests
            user_content = "Who can trade gold?"
            workflow_triggered, relevant_definitions, relevant_sections = self.chat.similarity_search(user_content)
            assert len(relevant_definitions) == 0
            assert len(relevant_sections) == 2
            assert relevant_sections.iloc[0]["section_reference"] == 'C.(C)'
            assert relevant_sections.iloc[1]["section_reference"] == 'C.(B)'



    def test_get_regulation_detail(self):
        section_reference = 'B.18(B)(i)(b)'
        expected_text = "B.18 Control of exports - general\n\
    (B) Regulations in respect of goods exported for sale abroad\n\
        (i) Authorised Dealers must ensure that all exporters are aware of their legal obligation in terms of the provisions of Regulations 6, 10 and 11 to:\n\
            (b) receive the full foreign currency proceeds not later than six months from the date of shipment. Authorised Dealers may authorise South African exporters to grant credit of up to 12 months to foreign importers, provided that the Authorised Dealer granting the authority is satisfied that the credit is necessary in the particular trade or that it is needed to protect an existing export market or to capture a new export market. In this regard, Authorised Dealers are requested to specifically draw the attention of exporters to the provisions of Regulation 6(1) and (5);"

        retrieved_text = self.chat.get_regulation_detail(section_reference)
        assert retrieved_text == expected_text
        
        # A problem case that should now be fixed
        retrieved_text = self.chat.get_regulation_detail('B.10(C)(i)(c)')
        expected_text = "B.10 Insurance and pensions\n    (C) Foreign currency payments in respect of short-term insurance premiums or reinsurance premiums\n        (i) In respect of insurance and reinsurance premiums placed abroad, Authorised Dealers may approve the following:\n            (c) Insurance (excluding reinsurance) through Lloyd's correspondents approved by Lloyd's of London\n            Applications by Lloyd's correspondents approved by Lloyd's of London to remit insurance premiums, excluding insurance premiums in respect of currency risks, in respect of:\n                (aa) cover placed in its entirety with Lloyd's underwriters through a broker at Lloyd's, which request must be accompanied by a letter signed by two senior officials of the Lloyd's correspondent concerned incorporating:\n                    (1) a declaration that the Lloyd's correspondent is authorised to carry on such insurance business under the Short-term Insurance Act; and\n                    (2) a declaration that the transaction was entered into with an underwriter at Lloyd's through a broker at Lloyd's.\n                (bb) cover placed through a broker at Lloyd's which is not in its entirety underwritten by an underwriter at Lloyd's which request must be accompanied by:\n                    (1) a letter signed by two senior officials of the Lloyd's correspondent declaring that the Lloyd's correspondent is authorised to carry on such insurance business under the Short-term Insurance Act; and\n                    (2) a copy of a letter  issued by the Registrar of Short-term Insurance, granting approval  in terms of section 8(2)(d) of the Short-term Insurance Act to the intermediary/ Lloyd's correspondent to render services in relation to that short-term policy."
        assert retrieved_text == expected_text

    def test__find_reference_that_calls_for(self):
        text = "I.2 Local facilities to non-residents\n\
                    (cc) The overall finance period, including any initial credit granted by the exporter, may not exceed six months from date of shipment of the underlying goods from South Africa unless the dispensation outlined in section B.18(B)(i)(b) of the Authorised Dealer Manual has been granted, when the overall finance period, including any initial credit granted by the exporter, may not exceed 12 months from date of shipment of the underlying goods from South Africa. An export finance facility may be extended in the event of the overseas importer requiring an extension of the original credit period, provided that the overall finance periods set out above are not exceeded."
        manual_data = []
        manual_data.append(["I.2(A)(i)(a)(cc)", 0.15,  1, text, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
        referring_sections = self.chat._find_reference_that_calls_for("B.18(B)(i)(b)", df_manual_data)
        assert len(referring_sections) == 1
        assert referring_sections[0] == "I.2(A)(i)(a)(cc)"

        # Add a second reference to the RAG data
        text_2 = "I.2 Local facilities to non-residents\n\
                    (dd) Random Text with no reference"
        manual_data.append(["I.2(A)(i)(a)(dd)", 0.14, 1, text_2, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
        referring_sections = self.chat._find_reference_that_calls_for("B.18(B)(i)(b)", df_manual_data)
        assert len(referring_sections) == 1
        assert referring_sections[0] == "I.2(A)(i)(a)(cc)"

        # Add a third reference to the RAG data
        text_3 = "I.2 Local facilities to non-residents\n\
                    (ee) Random Text with another reference to B.18(B) (i) (b) but with some random spaces"
        manual_data.append(["I.2(A)(i)(a)(ee)", 0.15, 1, text_3, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
        referring_sections = self.chat._find_reference_that_calls_for("B.18(B)(i)(b)", df_manual_data)
        assert len(referring_sections) == 2
        assert referring_sections[0] == "I.2(A)(i)(a)(cc)"
        assert referring_sections[1] == "I.2(A)(i)(a)(ee)"


    def test_add_section_to_resource(self):
        # Note that I need to use references that appear in the test data
        text = "A.3 Duties and responsibilities of Authorised Dealers\n\
                    (A) Introduction\n\
                        (i) Fake reference to B.4(B)(iv)(f)"
        manual_data = []
        manual_data.append(["A.3(A)(i)", 0.15, 1, text, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
        df_updated = self.chat.add_section_to_resource('B.4(B)(iv)(f)', df_manual_data)
        assert len(df_updated) == 2
        assert df_updated.iloc[0]['section_reference'] == "A.3(A)(i)"
        assert df_updated.iloc[1]['section_reference'] == 'B.4(B)(iv)(f)'

        # Add a second reference to the RAG data
        text_2 = "A.3 Duties and responsibilities of Authorised Dealers\n\
                    (A) Introduction\n\
                        (ii) No references to be found here"
        manual_data.append(["A.3(A)(ii)", 0.14, 1, text_2, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
        df_updated = self.chat.add_section_to_resource('B.4(B)(iv)(f)', df_manual_data)
        assert len(df_updated) == 2
        assert df_updated.iloc[0]['section_reference'] == "A.3(A)(i)"
        assert df_updated.iloc[1]['section_reference'] == 'B.4(B)(iv)(f)'

        # Add a third reference to the RAG data
        text_3 = "A.3 Local facilities to non-residents\n\
                    (B) Random Text with another reference to B.4(B) (iv) (f) but with some random spaces"
        manual_data.append(["A.3(B)", 0.13, 1, text_3, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
        df_updated = self.chat.add_section_to_resource('B.4(B)(iv)(f)', df_manual_data)
        assert len(df_updated) == 3
        assert df_updated.iloc[0]['section_reference'] == "A.3(A)(i)"
        assert df_updated.iloc[1]['section_reference'] == "A.3(B)"
        assert df_updated.iloc[2]['section_reference'] == 'B.4(B)(iv)(f)'


    def test__find_fuzzy_reference(self):
        text = "(cc) unless the dispensation outlined in section B.18(B) (i) (b) of the Authorised Dealer Manual has been granted"        
        section = "B.18(B)(i)(b)"
        match = self.chat._find_fuzzy_reference(text, section)
        assert match is not None

    def test__truncate_message_list(self):
        l = [{"content": "1"}, 
            {"content": "2"},
            {"content": "3"},
            {"content": "4"},
            {"content": "5"},
            {"content": "6"},
            {"content": "7"},
            {"content": "8"},
            {"content": "9"},
            {"content": "10"}]
        system_message = [{"content" : "s"}]
        truncated =self.chat._truncate_message_list(system_message, l, 2)
        assert len(truncated) == 2
        assert truncated[0]["content"] == "s"
        assert truncated[1]["content"] == "10"

        truncated =self.chat._truncate_message_list(system_message, l, 6)
        assert len(truncated) == 5
        assert truncated[0]["content"] == "s"
        assert truncated[1]["content"] == "7"
        assert truncated[4]["content"] == "10"

    def test_enrich_user_request_for_documentation(self):
        messages_without_rag = [{'role': 'user', 'content': 'Can foreign nationals send money home?'},
                                {'role': 'assistant', 'content': 'Yes, foreign nationals can send money abroad if they meet certain conditions. Foreign nationals temporarily in South Africa are required to declare whether they are in possession of foreign assets upon arrival. If they complete the necessary declarations and undertakings, they may be permitted to conduct their banking on a resident basis, dispose of or invest their foreign assets, conduct non-resident or foreign currency accounts, and transfer funds abroad. However, they must be able to substantiate the source of the funds and the value of the funds should be reasonable in relation to their income generating activities in South Africa. The completed declarations and undertakings must be retained by the Authorised Dealers for a period of five years. There are also exemptions for single remittance transactions up to R5,000 and transactions where a business relationship has been established. (B.5(A)(i)(d), B.5(A)(i)(e))'}]
        user_content = 'Is there any documentation required?'
        model_to_use = "gpt-3.5-turbo"
        response = self.chat.enrich_user_request_for_documentation(user_content, messages_without_rag, model_to_use)
        print(response)
        assert(response.startswith('What documentation is required as evidence for'))

