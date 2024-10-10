import os
import pandas as pd
from openai import OpenAI
import pytest
from unittest.mock import patch

from regulations_rag.embeddings import  EmbeddingParameters
from regulations_rag.rerank import RerankAlgos
from regulations_rag.corpus_chat import ChatParameters, CorpusChat

#from .navigating_corpus import NavigatingCorpus
from .navigating_index import NavigatingIndex


class TestCorpusChat:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
    chat_parameters = ChatParameters(chat_model = "gpt-4o", temperature = 0, max_tokens = 500)
    #chat_parameters = ChatParameters(chat_model = "gpt-4-turbo", temperature = 0, max_tokens = 500)

    embedding_parameters = EmbeddingParameters("text-embedding-3-large", 1024)
    key = os.getenv('encryption_key_gdpr')

    corpus_index = NavigatingIndex()

    rerank_algo  = RerankAlgos.NONE

    chat = CorpusChat(openai_client = openai_client, 
                        embedding_parameters = embedding_parameters, 
                        chat_parameters = chat_parameters, 
                        corpus_index = corpus_index,
                        rerank_algo = rerank_algo,   
                        user_name_for_logging = 'test_user')

    def test_construction(self):
        assert True

    def create_dummy_definitions_and_search_data(self):
        dfns = []
        dfns.append(["WRR", "1", "My definition from WRR"])
        dfns.append(["Plett", "A.1", "My definition from Plett"])
        df_definitions = pd.DataFrame(dfns, columns = ["document", "section_reference", "definition"])
        sections = []
        sections.append(["WRR", "1.2", "My Section 1.2 from WRR"])
        sections.append(["WRR", "1.3", "My Section 1.3 from WRR"])
        sections.append(["Plett", "A.2(A)(i)", "My section A.2(A)(i) from Plett"])
        df_search_sections = pd.DataFrame(sections, columns = ["document", "section_reference", "regulation_text"])
        return df_definitions, df_search_sections

    def test_create_openai_system_message(self):
        role = 'system'
        content = "Your system prompt here"
        df_definitions, df_search_sections = self.create_dummy_definitions_and_search_data()
        other_text = {"case_1": "explicit additional instructions in case_1", "case_2": "explicit additional instructions in case_2"}
        system_dict = {"role": role, "content": content, "definitions": df_definitions, "sections": df_search_sections, "other_text": other_text}
        
        openai_system_dict = self.chat.create_openai_system_message(system_dict)
        keys_list = list(openai_system_dict.keys())
        assert len(keys_list) == 2
        assert 'role' in keys_list
        assert 'content' in keys_list
        assert openai_system_dict['role'] == role
        assert openai_system_dict['content'] == content

    def test__add_rag_data_to_question(self):
        df_definitions, df_search_sections = self.create_dummy_definitions_and_search_data()

        question = "user asks question"
        output_string = self.chat._add_rag_data_to_question(question, df_definitions, df_search_sections)
        expected_text = f'Question: user asks question\n\nExtract 1:\nMy definition from WRR\nExtract 2:\nMy definition from Plett\nExtract 3:\nMy Section 1.2 from WRR\nExtract 4:\nMy Section 1.3 from WRR\nExtract 5:\nMy section A.2(A)(i) from Plett\n'
        assert output_string == expected_text

    def test_create_openai_user_message(self):
        df_definitions, df_search_sections = self.create_dummy_definitions_and_search_data()

        role = "user"
        content = "User question here"
        other_text = {"case_1": "explicit additional instructions in case_1", "case_2": "explicit additional instructions in case_2"}
        user_dict = {"role": role, "content": content, "definitions": df_definitions, "sections": df_search_sections, "other_text": other_text}

        openai_user_dict = self.chat.create_openai_user_message(user_dict)
        keys_list = list(openai_user_dict.keys())
        assert len(keys_list) == 2
        assert 'role' in keys_list
        assert 'content' in keys_list
        assert openai_user_dict['role'] == role
        expected_content = f'Question: User question here\n\nExtract 1:\nMy definition from WRR\nExtract 2:\nMy definition from Plett\nExtract 3:\nMy Section 1.2 from WRR\nExtract 4:\nMy Section 1.3 from WRR\nExtract 5:\nMy section A.2(A)(i) from Plett\n'
        assert openai_user_dict['content'] == expected_content

    def test__reformat_assistant_answer(self):
        # if there are no sections in the rag, return the raw response 
        df_definitions, df_search_sections = self.create_dummy_definitions_and_search_data()

        result = {"success": True, "path": "ANSWER:", "answer": "some_text_here", "reference": []}

        result["answer"] = "Some random text here."
        result["reference"] = [1, 2, 3, 5]
        formatted_response = self.chat._reformat_assistant_answer(result, df_definitions, df_search_sections)
        raw_response = 'Some random text here.  \nReference:  \nDefinition 1 from Navigating Whale Rock Ridge  \nDefinition A.1 from Navigating Plett  \nSection 1.2 from Navigating Whale Rock Ridge  \nSection A.2(A)(i) from Navigating Plett  \n'
        assert formatted_response == raw_response

        result["answer"] = "Hi"
        result["reference"] = [] # no references
        formatted_response = self.chat._reformat_assistant_answer(result, df_definitions, df_search_sections)
        assert result["answer"] == formatted_response

    def test_create_openai_assistant_message(self):
        df_definitions, df_search_sections = self.create_dummy_definitions_and_search_data()

        role = "assistant"
        content = "ANSWER: Some random text here. Reference: 1, 2, 3, 5"
        other_text = {"case_1": "explicit additional instructions in case_1", "case_2": "explicit additional instructions in case_2"}
        assistant_dict = {"role": role, "content": content, "definitions": df_definitions, "sections": df_search_sections, "other_text": other_text}

        openai_assistant_dict = self.chat.create_openai_assistant_message(assistant_dict)
        keys_list = list(openai_assistant_dict.keys())
        assert len(keys_list) == 2
        assert 'role' in keys_list
        assert 'content' in keys_list
        assert openai_assistant_dict['role'] == role
        expected_content = 'Some random text here.  \nReference:  \nDefinition 1 from Navigating Whale Rock Ridge  \nDefinition A.1 from Navigating Plett  \nSection 1.2 from Navigating Whale Rock Ridge  \nSection A.2(A)(i) from Navigating Plett  \n'
        assert openai_assistant_dict['content'] == expected_content




    def test_append_content(self):
        df_definitions, df_search_sections = self.create_dummy_definitions_and_search_data()
        other_text = {"case_1": "explicit additional instructions in case_1", "case_2": "explicit additional instructions in case_2"}
        
        self.chat.append_content(role ='user', content = 'Question: What documents are required', df_definitions = df_definitions, df_sections = df_search_sections, other_text = other_text)
        assert len(self.chat.messages_intermediate) == 1
        assert self.chat.messages_intermediate[-1]['content'] == 'Question: What documents are required'
        assert self.chat.messages_intermediate[-1]['role'] == 'user'
        # add the same item twice in a row
        self.chat.append_content(role ='user', content = 'Question: What documents are required', df_definitions = df_definitions, df_sections = df_search_sections, other_text = other_text)
        assert len(self.chat.messages_intermediate) == 1


        # Try to add content for a role that does not exist
        self.chat.reset_conversation_history()
        self.chat.append_content('other_role', 'Question: What documents are required')
        assert len(self.chat.messages_intermediate) == 0

        self.chat.reset_conversation_history()
        self.chat.append_content('assistant', 'Answer here')
        assert len(self.chat.messages_intermediate) == 1
        assert self.chat.messages_intermediate[-1]['content'] == 'Answer here'
        assert self.chat.messages_intermediate[-1]['role'] == 'assistant'

    def test_format_messages_for_openai(self):
        self.chat.reset_conversation_history()
        df_definitions, df_search_sections = self.create_dummy_definitions_and_search_data()
        df_definitions = pd.DataFrame()
        other_text = {"case_1": "explicit additional instructions in case_1", "case_2": "explicit additional instructions in case_2"}
        user_dict = {'role': 'user', 'content': 'What documents are required', 'definitions':  df_definitions, 'sections': df_search_sections, 'other_text': other_text}
        self.chat.append_content(role ='user', content = 'Question: What documents are required', df_definitions = df_definitions, df_sections = df_search_sections, other_text = other_text)

        content = "ANSWER: Some random text here. Reference: 1, 2, 3, 5"
        other_text = {"case_1": "explicit additional instructions in case_1", "case_2": "explicit additional instructions in case_2"}
        df_definitions, df_search_sections = self.create_dummy_definitions_and_search_data()
        assistant_dict = {"role": 'assistant', "content": content, "definitions": df_definitions, "sections": df_search_sections, "other_text": other_text}
        self.chat.append_content(role ='assistant', content = content, df_definitions = df_definitions, df_sections = df_search_sections, other_text = other_text)

        messages = self.chat.format_messages_for_openai()
        assert len(messages) == 2

        manual_messages = []
        manual_messages.append(self.chat.create_openai_user_message(user_dict))
        manual_messages.append(self.chat.create_openai_assistant_message(assistant_dict))

        assert messages[-1]['content'] == manual_messages[-1]['content']

    def test_similarity_search(self):
        # Check that random chit-chat to the main dataset does not return any hits from the embeddings
        text = "Hi"
        workflow_triggered, df_definitions, df_search_sections = self.chat.similarity_search(text)
        assert len(df_definitions) == 0
        assert len(df_search_sections) == 0 
        # now move to the testing dataset for fine grained tests
        user_content = "How do I get to South Gate?"
        workflow_triggered, relevant_definitions, relevant_sections = self.chat.similarity_search(user_content)
        assert len(relevant_definitions) == 0
        assert len(relevant_sections) == 3
        assert relevant_sections.iloc[0]["section_reference"] == '1.3' # this is the answer we want
        assert relevant_sections.iloc[1]["section_reference"] == '1.1' # order of these is not important
        assert relevant_sections.iloc[2]["section_reference"] == '1.2' # order of these is not important

    def test__create_system_message(self):
        ref_string = r"[1-9](\.[1-9]){0,2}"  # the text_pattern from SimpleReferenceChecker which is the reference checker for the main document in the "navigating" Corpus (WRR)
        user_type = "a Visitor"
        corpus_description = "the Simplest way to Navigate Plett"

        expected_message = f"You are answering questions about {corpus_description} for {user_type} based only on the reference extracts provided. You have 3 options:\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference:'. Do not include the word Extract, only provide the number(s).\n2) Request additional documentation. If, in the body of the extract(s) provided, there is a reference to another section that is directly relevant and not already provided, respond with the word 'SECTION:' followed by 'Extract extract_number, Reference: section_reference' - for example SECTION: Extract 1, Reference: {ref_string}.\n3) State 'NONE:' and nothing else in all other cases\n"
        assert self.chat._create_system_message(number_of_options=3, review = False) == expected_message

        expected_message = f"You are answering questions about {corpus_description} for {user_type} based only on the reference extracts provided. You have 2 options:\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference:'. Do not include the word Extract, only provide the number(s).\n2) State 'NONE:' and nothing else in all other cases\n"
        assert self.chat._create_system_message(number_of_options=2, review = False) == expected_message

        expected_message = f"Please review your answer. You were asked to assist the user by responding to their question in 1 of 3 ways but your response does not follow the expected format. Please reformat your response so that it follows the requested format.\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference:'. Do not include the word Extract, only provide the number(s).\n2) Request additional documentation. If, in the body of the extract(s) provided, there is a reference to another section that is directly relevant and not already provided, respond with the word 'SECTION:' followed by 'Extract extract_number, Reference: section_reference' - for example SECTION: Extract 1, Reference: {ref_string}.\n3) State 'NONE:' and nothing else in all other cases\n"
        assert self.chat._create_system_message(number_of_options=3, review = True) == expected_message

        expected_message = f"Please review your answer. You were asked to assist the user by responding to their question in 1 of 2 ways but your response does not follow the expected format. Please reformat your response so that it follows the requested format.\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference:'. Do not include the word Extract, only provide the number(s).\n2) State 'NONE:' and nothing else in all other cases\n"
        assert self.chat._create_system_message(number_of_options=2, review = True) == expected_message



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

    def test__check_response(self):
        df_definitions, df_search_sections = self.create_dummy_definitions_and_search_data()

        response = self.chat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value
        check_result =  self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["success"]
        assert check_result["path"] == "ERROR:"
        assert check_result["answer"] == 'This app demonstrates Retrieval Augmented Generation. Behind the scenes, instructions are issued to a Large Language Model (LLM) and then verified. Occasionally, due to the statistical nature of the model, the LLM may not follow instructions correctly. In such cases, I am programmed not to respond but to ask you to clear the conversation history and try asking your question again. This usually resolves the issue. However, if the same error persists in the same spot, it likely indicates a bug rather than a statistical anomaly. Bugs are logged and will be addressed in due course. For now, please clear the conversation history and retry your query.'
        assert check_result["reference"] == []

        # Check if the response does not contain any keywords
        response = "I did not follow any instructions"
        check_result =  self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert not check_result["success"]
        assert check_result["path"] == "NONE:"
        assert check_result["llm_followup_instruction"] == "Your response, did not begin with one of the keywords, 'ANSWER:', 'SECTION:' or 'NONE:'. Please review the question and provide an answer in the required format. Also make sure the referenced extracts are quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference:'. Do not include the word Extract, only provide the number(s).\n"


        # Check if the response contains multiple instances of the keyword "References:"
        # NOTE: this is not a good representative response from the LLM. It is only used here for testing
        response = "ANSWER: There are a few points to consider. \na) In this case you need Reference: 1, 3. \nb) In this case you need Reference: 2, 4. \n\n Which means you need Reference: 1, 2, 3, 4"
        check_result =  self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert not check_result["success"]
        assert check_result["path"] == "ANSWER:"
        assert check_result["llm_followup_instruction"] == f"When answering the question, you used the keyword '{self.chat.reference_key_word}' more than once. It is vitally important that this keyword is only used once in your answer and then only at the end of the answer followed only by an integer, comma separated list of the extracts used. Please reformat your response so that there is only one instance of the keyword '{self.chat.reference_key_word}' and it is at the end of the answer."


        # Check ANSWER:
        path = "ANSWER:"
        text = "Here is an answer with no refs. It should pass because the instruction specifically said provide references *if* you used them"
        references = ""
        response = f"{path} {text} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["path"] == path
        assert check_result["success"]
        assert check_result["answer"] == text
        assert check_result["reference"] == []
        # and now with empty references
        path = "ANSWER:"
        text = "Here is an answer with no refs. It should pass because the instruction specifically said provide references *if* you used them"
        references = "Reference: "
        response = f"{path} {text} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["path"] == path
        assert check_result["success"]
        assert check_result["answer"] == text
        assert check_result["reference"] == []

        # TODO: Check it used the word References: (plural)

        path = "ANSWER:"
        text = "Here is a response but where the references are formatted incorrectly but I can still work with then."
        references = "Reference: Extract 1, Extract 3"
        response = f"{path} {text} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["path"] == path
        assert check_result["success"]
        assert check_result["answer"] == text
        assert check_result["reference"] == [1, 3]

        path = "ANSWER:"
        text = "Here is a response but where the references are formatted correctly."
        references = "Reference: 1,  3, 2, 4"
        response = f"{path} {text} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["path"] == path
        assert check_result["success"]
        assert check_result["answer"] == text
        assert check_result["reference"] == [1, 3, 2, 4]

        path = "ANSWER:"
        text = "Here is a response but where the references are formatted correctly but refer to something invalid."
        references = "Reference: 6"
        response = f"{path} {text} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["path"] == path
        assert not check_result["success"]
        assert check_result["llm_followup_instruction"] == "When answering the question, you have made reference to an extract number that was not provided. Please re-write your references and only refer to the extracts provided by their number"

        path = "ANSWER:"
        text = "Here is a response but where the references are formatted correctly but refer to something invalid."
        references = "Reference: b"
        response = f"{path} {text} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["path"] == path
        assert not check_result["success"]
        assert check_result["llm_followup_instruction"] == "When answering the question, you have made reference to an extract but I am unable to extract the number from your reference. Please re-write your answer using integer extract number(s)"

        # Check NONE:
        response = "NONE: "
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["success"]
        assert check_result["path"] == "NONE:"

        # # Check SECTION: 
        path = "SECTION:"
        references = "Extract 1, Reference 1.1"
        response = f"{path} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["success"]
        assert check_result["path"] == path
        assert check_result["document"] == 'WRR'
        assert check_result["section"] == '1.1'

        path = "SECTION:"
        references = "Extract (3), Reference Article 1.1"
        response = f"{path} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert not check_result["success"]
        assert check_result["path"] == path
        assert check_result["llm_followup_instruction"] == 'When requesting an additional section, you did not use the format r"Extract (\d+), Reference (.+)" or you included additional text. Please re-write your response using this format'

        path = "SECTION:"
        references = "Extract 6, Reference Article 1.1"
        response = f"{path} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert not check_result["success"]
        assert check_result["path"] == path
        assert check_result["llm_followup_instruction"] == 'When requesting an additional section, you have made reference to an extract number that was not provided. Please re-write your answer and use a valid extract number'

        path = "SECTION:"
        references = "Extract 3, Reference 1.1"
        response = f"{path} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["success"]
        assert check_result["path"] == path
        assert check_result["document"] == 'WRR'
        assert check_result["section"] == '1.1'

        path = "SECTION:"
        references = "Extract 5, Reference 1.1" # doc 5 is from Plett but it has a reference to a WRR section 
        response = f"{path} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert check_result["success"]
        assert check_result["path"] == path
        assert check_result["document"] == 'WRR'
        assert check_result["section"] == '1.1'

        path = "SECTION:"
        references = "Extract 3, Reference XXX" 
        response = f"{path} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert not check_result["success"]
        assert check_result["path"] == path
        ref_string = r"[1-9](\.[1-9]){0,2}"
        assert check_result["llm_followup_instruction"] == f"The reference XXX does not appear to be a valid reference for the document. Try using the format {ref_string}"

        path = "SECTION:"
        references = "Extract 5, Reference XXX" # doc 5 is article_30_5 but it may have a GDPR reference
        response = f"{path} {references}"
        check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert not check_result["success"]
        assert check_result["path"] == path
        ref_string = r"[A-Z]\.\d{0,2}\([A-Z]\)\((?:i|ii|iii|iv|v|vi)\)\([a-z]\)\([a-z]{2}\)\(\d+\)" + ", or " + r"[1-9](\.[1-9]){0,2}"
        assert check_result["llm_followup_instruction"] == f"The reference XXX does not appear to be a valid reference for the document. Try using the format {ref_string}"

#         # TODO: I need to test the case where the base document has a non-empty reference checker and that fails along with the GDPR reference checker
#         # e.g.
#         # response = "SECTION: Extract 5, Reference XXX" # doc 5 is article_30_5 but it may have a GDPR reference
#         # check_result =    self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
#         # assert not check_result["success"]
#         # assert check_message == "The reference XXX does not appear to be a valid reference for the document. Try using the format (\d{1,2})(?:\((\d{1,2})\))?(?:\(([a-z])\))?"
#         # but with Extract 5 from a non-GDPR document with an index.


    @patch.object(CorpusChat, '_get_api_response')
    def test_resource_augmented_query(self, mock__get_api_response):
        # NOTE: I am not going to test the openai api call, I'm going to use use mocking from the unittest.mock module to "hardcode" api responses

        self.chat.reset_conversation_history()
        self.chat.system_state = self.chat.State.RAG
        user_content = "How do I get to the gym?"
        workflow_triggered, relevant_definitions, relevant_sections = self.chat.similarity_search(user_content)

        # Check that if the api returns an answer: 
        mock__get_api_response.return_value = "ANSWER: test to see what happens when if the API believes it successfully answered the question with the resources provided"
        result = self.chat.resource_augmented_query(user_question = user_content, 
                                                    df_definitions = relevant_definitions, 
                                                    df_search_sections = relevant_sections)
        assert result["success"]
        assert result["path"] == "ANSWER:"
        assert result["answer"] == "test to see what happens when if the API believes it successfully answered the question with the resources provided"
        assert result["reference"] == []

        # # Check when the api returns NONE: 
        mock__get_api_response.return_value = "NONE:"
        result = self.chat.resource_augmented_query(user_question = user_content, 
                                                            df_definitions = relevant_definitions, 
                                                            df_search_sections = relevant_sections)
        assert result["success"]
        assert result["path"] == self.chat.Prefix.NONE.value
        
        
        # # Check the "section_reference" branch
        mock__get_api_response.return_value = "SECTION: Extract 1, Reference 1.1"
        result = self.chat.resource_augmented_query(user_question = user_content,
                                                    df_definitions = relevant_definitions, 
                                                    df_search_sections = relevant_sections)
        assert result["success"]
        assert result["path"] == self.chat.Prefix.SECTION.value
        assert result["extract"] == 1
        assert result["document"] == 'WRR'
        assert result["section"] == '1.1'

        # # Check the despondent user branch
        mock__get_api_response.side_effect = ["Test to see what happens when if the API cannot listen to instructions", "SECTION: Extract 1, Reference 1.1"]
        result = self.chat.resource_augmented_query(user_question = user_content, 
                                                    df_definitions = relevant_definitions, 
                                                    df_search_sections = relevant_sections)
        assert result["success"]
        assert result["path"] == self.chat.Prefix.SECTION.value
        assert result["extract"] == 1
        assert result["document"] == 'WRR'
        assert result["section"] == '1.1'

        # # Check the stubbornly disobedient branch
        mock__get_api_response.side_effect = ["Test to see what happens when if the API cannot listen to instructions", "But even after marking its own homework it cannot listen to instructions"]

        result = self.chat.resource_augmented_query(user_question = user_content, 
                                                    df_definitions = relevant_definitions, 
                                                    df_search_sections = relevant_sections)
        assert not result["success"]
        assert result["path"] == "NONE:"
        assert result["assistant_response"] == self.chat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value


    # NOTE: I'm going to use use mocking from the unittest.mock module to "hardcode" api responses
    @patch.object(CorpusChat, '_get_api_response')
    def test_user_provides_input(self, mock__get_api_response):
        self.chat.reset_conversation_history()
        # check the response if the system is stuck
        self.chat.system_state = self.chat.State.STUCK
        user_content = "How do I get to the Gym?"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == self.chat.Errors.STUCK.value
        assert self.chat.system_state == self.chat.State.STUCK

        # check the response if the system is in an unknown state
        self.chat.reset_conversation_history()
        self.chat.system_state = "random state not in list"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == self.chat.Errors.UNKNOWN_STATE.value

        # test the workflow if the system answers the question as hoped
        self.chat.reset_conversation_history()
        self.chat.system_state = self.chat.State.RAG
        flag = "ANSWER: "
        input_response = 'Drive to West Gate. Reference: 2'
        output_response = 'Drive to West Gate.  \nReference:  \nSection A.2(A) from Navigating Plett'
        mock__get_api_response.return_value = flag + input_response
        self.chat.user_provides_input(user_content)
        messages = self.chat.format_messages_for_openai()
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].strip() == output_response
        assert self.chat.system_state == self.chat.State.RAG # rag
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"].strip() == 'Question: How do I get to the Gym?\n\nExtract 1:\nThe Gym: The Health and Fitness Center on Piesang Valley Road\nExtract 2:\nA.2 Directions\nA.2(A) To the Gym\nA.2(A)(i) From West Gate (see 1.1)\nTurn left into Longships Drive and right at the T-junction into Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nA.2(A)(ii) From Main Gate (see 1.2)\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nA.2(A)(iii) From South Gate (see 1.3)\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym'
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"].strip() == 'How do I get to the Gym?'
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"].strip() == flag + input_response


        # test the workflow if the system cannot find useful content in the supplied data
        self.chat.system_state = self.chat.State.RAG
        flag = "NONE: "
        response = ""
        mock__get_api_response.return_value = flag 
        self.chat.user_provides_input(user_content)
        messages = self.chat.format_messages_for_openai()
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].strip() == self.chat.Errors.NO_RELEVANT_DATA.value.replace("ERROR: ", "")
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"].strip() == 'Question: How do I get to the Gym?\n\nExtract 1:\nThe Gym: The Health and Fitness Center on Piesang Valley Road\nExtract 2:\nA.2 Directions\nA.2(A) To the Gym\nA.2(A)(i) From West Gate (see 1.1)\nTurn left into Longships Drive and right at the T-junction into Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nA.2(A)(ii) From Main Gate (see 1.2)\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nA.2(A)(iii) From South Gate (see 1.3)\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym'
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"].strip() == self.chat.Errors.NO_RELEVANT_DATA.value
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"].strip() == user_content
        assert self.chat.system_state == self.chat.State.RAG

        # test the workflow if the system needs additional content
        self.chat.system_state = self.chat.State.RAG
        flag = "SECTION: "
        response = "Extract 2, Reference 1.1"
        first_response = flag + response
        # now the response once it has received the additional data
        flag = "ANSWER: "
        input_response = 'Drive to the West Gate. Reference: 2'
        # NOTE this output_response differs to the previous one because we have changed the search_sections while leaving the input_response to refer to extract 2 (which is now different)
        output_response = 'Drive to the West Gate.  \nReference:  \nSection A.2(A) from Navigating Plett'
        second_response = flag + input_response
        mock__get_api_response.side_effect = [first_response, second_response]
        self.chat.user_provides_input(user_content)
        messages = self.chat.format_messages_for_openai()
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].strip() == output_response
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"].strip() == 'Question: How do I get to the Gym?\n\nExtract 1:\nThe Gym: The Health and Fitness Center on Piesang Valley Road\nExtract 2:\nA.2 Directions\nA.2(A) To the Gym\nA.2(A)(i) From West Gate (see 1.1)\nTurn left into Longships Drive and right at the T-junction into Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nA.2(A)(ii) From Main Gate (see 1.2)\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nA.2(A)(iii) From South Gate (see 1.3)\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nExtract 3:\n# 1 Navigating Whale Rock Ridge\n\n## 1.1 To West Gate\n\nTurn right out driveway. At the traffic circle, take the first exit. Proceed to West Gate'
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"].strip() == second_response
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"].strip() == user_content
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
        mock__get_api_response.side_effect = [first_response, second_response]
        self.chat.user_provides_input(user_content)
        messages = self.chat.format_messages_for_openai()
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].strip() == 'Drive to the West Gate.  \nReference:  \nSection A.2(A) from Navigating Plett'
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"].strip() == 'Question: How do I get to the Gym?\n\nExtract 1:\nThe Gym: The Health and Fitness Center on Piesang Valley Road\nExtract 2:\nA.2 Directions\nA.2(A) To the Gym\nA.2(A)(i) From West Gate (see 1.1)\nTurn left into Longships Drive and right at the T-junction into Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nA.2(A)(ii) From Main Gate (see 1.2)\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nA.2(A)(iii) From South Gate (see 1.3)\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym'
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"].strip() == second_response
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"].strip() == user_content
        assert self.chat.system_state == self.chat.State.RAG

        #test what happens if the LLM does not listen to instructions and returns something random
        self.chat.reset_conversation_history()
        self.chat.system_state = self.chat.State.RAG
        response = "None of the supplied documentation was relevant"
        first_response = response
        second_response = response
        mock__get_api_response.side_effect = [first_response, second_response]
        self.chat.user_provides_input(user_content)
        messages = self.chat.format_messages_for_openai()
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].strip() == self.chat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value.replace("ERROR: ", "")
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"].strip() == 'Question: How do I get to the Gym?\n\nExtract 1:\nThe Gym: The Health and Fitness Center on Piesang Valley Road\nExtract 2:\nA.2 Directions\nA.2(A) To the Gym\nA.2(A)(i) From West Gate (see 1.1)\nTurn left into Longships Drive and right at the T-junction into Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nA.2(A)(ii) From Main Gate (see 1.2)\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\nA.2(A)(iii) From South Gate (see 1.3)\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym'
        assert self.chat.system_state == self.chat.State.STUCK


    def test_add_section_to_resource(self):
        df_definitions, df_search_sections = self.create_dummy_definitions_and_search_data()
        
        # check if the section string passes validation but does not refer to something in the document
        result = {"success": True, "path": "SECTION:", "extract": 1, "document": 'WRR', "section": "9.1"}

        df_updated = self.chat.add_section_to_resource(result, df_definitions, df_search_sections)
        assert len(df_updated) == 3
        assert df_updated.iloc[0]['section_reference'] == "1.2"
        assert df_updated.iloc[1]['section_reference'] == '1.3'
        assert df_updated.iloc[2]['section_reference'] == 'A.2(A)(i)'

        # Adding a valid string
        result = {"success": True, "path": "SECTION:", "extract": 5, "document": 'WRR', "section": "1.1"}
        df_updated = self.chat.add_section_to_resource(result, df_definitions, df_search_sections)
        assert len(df_updated) == 4
        assert df_updated.iloc[0]['section_reference'] == "1.2"
        assert df_updated.iloc[1]['section_reference'] == '1.3'
        assert df_updated.iloc[2]['section_reference'] == 'A.2(A)(i)'
        assert df_updated.iloc[3]['section_reference'] == '1.1'

        # check if the section string only comes from the definitions
        result = {"success": True, "path": "SECTION:", "extract": 1, "document": 'WRR', "section": "1.1"}
        df_updated = self.chat.add_section_to_resource(result, df_definitions, df_search_sections)
        assert len(df_updated) == 4
        assert df_updated.iloc[0]['section_reference'] == "1.2"
        assert df_updated.iloc[1]['section_reference'] == '1.3'
        assert df_updated.iloc[2]['section_reference'] == 'A.2(A)(i)'
        assert df_updated.iloc[3]['section_reference'] == '1.1'


