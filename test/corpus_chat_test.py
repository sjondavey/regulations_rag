import os
import pandas as pd
from openai import OpenAI

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

        expected_message = f"You are answering questions about {corpus_description} for {user_type} based only on the reference extracts provided. You have 3 options:\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference: '. Do not include the word Extract, only provide the number(s).\n2) Request additional documentation. If, in the body of the extract(s) provided, there is a reference to another section that is directly relevant and not already provided, respond with the word 'SECTION:' followed by 'Extract extract_number, Reference section_reference' - for example SECTION: Extract 1, Reference {ref_string}.\n3) State 'NONE:' and nothing else in all other cases\n"
        assert self.chat._create_system_message(number_of_options=3, review = False) == expected_message

        expected_message = f"You are answering questions about {corpus_description} for {user_type} based only on the reference extracts provided. You have 2 options:\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference: '. Do not include the word Extract, only provide the number(s).\n2) State 'NONE:' and nothing else in all other cases\n"
        assert self.chat._create_system_message(number_of_options=2, review = False) == expected_message

        expected_message = f"Please review your answer. You were asked to assist the user by responding to their question in 1 of 3 ways but your response does not follow the expected format. Please reformat your response so that it follows the requested format.\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference: '. Do not include the word Extract, only provide the number(s).\n2) Request additional documentation. If, in the body of the extract(s) provided, there is a reference to another section that is directly relevant and not already provided, respond with the word 'SECTION:' followed by 'Extract extract_number, Reference section_reference' - for example SECTION: Extract 1, Reference {ref_string}.\n3) State 'NONE:' and nothing else in all other cases\n"
        assert self.chat._create_system_message(number_of_options=3, review = True) == expected_message

        expected_message = f"Please review your answer. You were asked to assist the user by responding to their question in 1 of 2 ways but your response does not follow the expected format. Please reformat your response so that it follows the requested format.\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference: '. Do not include the word Extract, only provide the number(s).\n2) State 'NONE:' and nothing else in all other cases\n"
        assert self.chat._create_system_message(number_of_options=2, review = True) == expected_message


    def test__add_rag_data_to_question(self):
        dfns = []
        dfns.append(["WRR", "1", "My definition from WRR"])
        dfns.append(["Plett", "A.1", "My definition from Plett"])
        df_definitions = pd.DataFrame(dfns, columns = ["document", "section_reference", "definition"])
        sections = []
        sections.append(["WRR", "1.2", "My Section 1.2 from WRR"])
        sections.append(["WRR", "1.3", "My Section 1.3 from WRR"])
        sections.append(["Plett", "A.2(A)(i)", "My section A.2(A)(i) from Plett"])
        df_search_sections = pd.DataFrame(sections, columns = ["document", "section_reference", "regulation_text"])

        question = "user asks question"
        output_string = self.chat._add_rag_data_to_question(question, df_definitions, df_search_sections)
        expected_text = f'Question: user asks question\n\nExtract 1:\nMy definition from WRR\nExtract 2:\nMy definition from Plett\nExtract 3:\nMy Section 1.2 from WRR\nExtract 4:\nMy Section 1.3 from WRR\nExtract 5:\nMy section A.2(A)(i) from Plett\n'
        assert output_string == expected_text

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
        dfns = []
        dfns.append(["WRR", "1", "My definition from WRR"])
        dfns.append(["Plett", "A.1", "My definition from Plett"])
        df_definitions = pd.DataFrame(dfns, columns = ["document", "section_reference", "definition"])
        sections = []
        sections.append(["WRR", "1.2", "My Section 1.2 from WRR"])
        sections.append(["WRR", "1.3", "My Section 1.3 from WRR"])
        sections.append(["Plett", "A.2(A)(i)", "My section A.2(A)(i) from Plett"])
        df_search_sections = pd.DataFrame(sections, columns = ["document", "section_reference", "regulation_text"])

        # Check if the response does not contain any keywords
        response = "I did not follow any instructions"
        check_result =  self.chat._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
        assert not check_result["success"]
        assert check_result["path"] == "NONE"
        assert check_result["llm_followup_instruction"] == "Your response, did not begin with one of the keywords, 'ANSWER:', 'SECTION:' or 'NONE:'. Please review the question and provide an answer in the required format. Also make sure the referenced extracts are quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference: '. Do not include the word Extract, only provide the number(s).\n"

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


    def test_resource_augmented_query(self):
        # NOTE: I am not going to test the openai api call. I am going to use 'testing' mode with canned answers from the "api call"

        self.chat.reset_conversation_history()
        self.chat.system_state = self.chat.State.RAG
        user_content = "How do I get to the gym?"
        workflow_triggered, relevant_definitions, relevant_sections = self.chat.similarity_search(user_content)

        # Check that if the api returns NONE: 
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("ANSWER: test to see what happens when if the API believes it successfully answered the question with the resources provided")
        result = self.chat.resource_augmented_query(user_question = user_content, 
                                                    df_definitions = relevant_definitions, 
                                                    df_search_sections = relevant_sections,
                                                    testing = testing,
                                                    manual_responses_for_testing = manual_responses_for_testing)
        assert result["success"]
        assert result["path"] == "ANSWER:"
        assert result["answer"] == "test to see what happens when if the API believes it successfully answered the question with the resources provided"
        assert result["reference"] == []

        # Check that if the api returns NONE: 
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("NONE:") 
        result = self.chat.resource_augmented_query(user_question = user_content, 
                                                            df_definitions = relevant_definitions, 
                                                            df_search_sections = relevant_sections,
                                                            testing = testing,
                                                            manual_responses_for_testing = manual_responses_for_testing)
        assert result["success"]
        assert result["path"] == self.chat.Prefix.NONE.value
        
        
        # Check the "section_reference" branch
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("SECTION: Extract 1, Reference 1.1")
        result = self.chat.resource_augmented_query(user_question = user_content,
                                                    df_definitions = relevant_definitions, 
                                                    df_search_sections = relevant_sections,
                                                    testing = testing,
                                                    manual_responses_for_testing = manual_responses_for_testing)
        assert result["success"]
        assert result["path"] == self.chat.Prefix.SECTION.value
        assert result["extract"] == 1
        assert result["document"] == 'WRR'
        assert result["section"] == '1.1'

        # Check the despondent user branch
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("Test to see what happens when if the API cannot listen to instructions")
        manual_responses_for_testing.append("SECTION: Extract 1, Reference 1.1") # but after marking its own homework it behaves

        result = self.chat.resource_augmented_query(user_question = user_content, 
                                                    df_definitions = relevant_definitions, 
                                                    df_search_sections = relevant_sections,
                                                    testing = testing,
                                                    manual_responses_for_testing = manual_responses_for_testing)
        assert result["success"]
        assert result["path"] == self.chat.Prefix.SECTION.value
        assert result["extract"] == 1
        assert result["document"] == 'WRR'
        assert result["section"] == '1.1'

        # Check the stubbornly disobedient branch
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("Test to see what happens when if the API cannot listen to instructions")
        manual_responses_for_testing.append("But even after marking its own homework it cannot listen to instructions")

        result = self.chat.resource_augmented_query(user_question = user_content, 
                                                    df_definitions = relevant_definitions, 
                                                    df_search_sections = relevant_sections,
                                                    testing = testing,
                                                    manual_responses_for_testing = manual_responses_for_testing)
        assert not result["success"]
        assert result["path"] == "NONE"
        assert result["assistant_response"] == self.chat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value


    def test_collect_references(self):
        dfns = []
        dfns.append(["WRR", "1", "My definition from WRR"])
        dfns.append(["Plett", "A.1", "My definition from Plett"])
        df_definitions = pd.DataFrame(dfns, columns = ["document", "section_reference", "definition"])
        sections = []
        sections.append(["WRR", "1.2", "My Section 1.2 from WRR"])
        sections.append(["WRR", "1.3", "My Section 1.3 from WRR"])
        sections.append(["Plett", "A.2(A)(i)", "My section A.2(A)(i) from Plett"])
        df_search_sections = pd.DataFrame(sections, columns = ["document", "section_reference", "regulation_text"])

        df_references = self.chat.collect_references(df_definitions, df_search_sections)
        assert len(df_references) == 5

        dfns = []
        df_definitions = pd.DataFrame(dfns, columns = ["document", "section_reference", "definition"])
        sections = []
        df_search_sections = pd.DataFrame(sections, columns = ["document", "section_reference", "regulation_text"])
        df_references = self.chat.collect_references(df_definitions, df_search_sections)
        assert len(df_references) == 0

        sections = []
        sections = []
        sections.append(["WRR", "1.2", "My Section 1.2 from WRR"])
        df_search_sections = pd.DataFrame(sections, columns = ["document", "section_reference", "regulation_text"])
        df_references = self.chat.collect_references(df_definitions, df_search_sections)
        assert len(df_references) == 1


    def test_reformat_assistant_answer(self):
        # if there are no sections in the rag, return the raw response 
        dfns = []
        dfns.append(["WRR", "1", "My definition from WRR"])
        dfns.append(["Plett", "A.1", "My definition from Plett"])
        df_definitions = pd.DataFrame(dfns, columns = ["document", "section_reference", "definition"])
        sections = []
        sections.append(["WRR", "1.2", "My Section 1.2 from WRR"])
        sections.append(["WRR", "1.3", "My Section 1.3 from WRR"])
        sections.append(["Plett", "A.2(A)(i)", "My section A.2(A)(i) from Plett"])
        df_search_sections = pd.DataFrame(sections, columns = ["document", "section_reference", "regulation_text"])

        result = {"success": True, "path": "ANSWER:", "answer": "some_text_here", "reference": []}

        result["answer"] = "Some random text here."
        result["reference"] = [1, 2, 3, 5]
        formatted_response, used_definitions, used_sections = self.chat.reformat_assistant_answer(result, df_definitions, df_search_sections)
        raw_response = 'Some random text here.  \nReference:  \nDefinition 1 from Navigating Whale Rock Ridge  \nDefinition A.1 from Navigating Plett  \nSection 1.2 from Navigating Whale Rock Ridge  \nSection A.2(A)(i) from Navigating Plett  \n'
        assert formatted_response == raw_response
        assert len(used_definitions) == 2
        assert len(used_sections) == 2

        result["answer"] = "Hi"
        result["reference"] = [] # no references
        formatted_response, used_definitions, used_sections = self.chat.reformat_assistant_answer(result, df_definitions, df_search_sections)
        assert result["answer"] == formatted_response
        assert len(used_definitions) == 2 # even though there are no references, I keep the definitions
        assert len(used_sections) == 0 # no references


    def test_user_provides_input(self):
        # check the response if the system is stuck
        self.chat.system_state = self.chat.State.STUCK
        user_content = "How do I get to the Gym?"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"] == self.chat.Errors.STUCK.value
        assert self.chat.system_state == self.chat.State.STUCK

        # check the response if the system is in an unknown state
        self.chat.system_state = "random state not in list"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"] == self.chat.Errors.UNKNOWN_STATE.value

        # test the workflow if the system answers the question as hoped
        self.chat.system_state = self.chat.State.RAG
        testing = True # don't make call to openai API, use the canned response below
        flag = "ANSWER:"
        input_response = 'Drive to West Gate. Reference: 2'
        output_response = 'Yes, there is a derogation for organizations with fewer than 250 employees with regard to record-keeping under Article 30 of the GDPR. However, this derogation is not absolute and does not apply to processing that is likely to result in a risk to the rights and freedoms of data subjects, processing that is not occasional, or processing that includes special categories of data or personal data relating to criminal convictions and offences. Therefore, small companies may be exempt from the record-keeping obligation if they meet the criteria specified in Article 30(5).  \nReference:  \nSection all from WORKING PARTY 29 POSITION PAPER on the derogations from the obligation to maintain records of processing activities pursuant to Article 30(5) GDPR'
        manual_responses_for_testing = [flag + input_response]
        self.chat.user_provides_input(user_content,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == output_response
        assert self.chat.system_state == self.chat.State.RAG # rag
        assert self.chat.messages[-2]["role"] == "user"
        assert self.chat.messages[-2]["content"].strip() == 'Question: Are there exemptions from GDPR for small companies?\n\nExtract 1:\n# 2 Material scope\n\n&nbsp;&nbsp;&nbsp;&nbsp;1. This Regulation applies to the processing of personal data wholly or partly by automated means and to the processing other than by automated means of personal data which form part of a filing system or are intended to form part of a filing system.\n\n&nbsp;&nbsp;&nbsp;&nbsp;2. This Regulation does not apply to the processing of personal data:\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(a) in the course of an activity which falls outside the scope of Union law;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b) by the Member States when carrying out activities which fall within the scope of Chapter 2 of Title V of the TEU;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(c) by a natural person in the course of a purely personal or household activity;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(d) by competent authorities for the purposes of the prevention, investigation, detection or prosecution of criminal offences or the execution of criminal penalties, including the safeguarding against and the prevention of threats to public security.\n\n&nbsp;&nbsp;&nbsp;&nbsp;3. For the processing of personal data by the Union institutions, bodies, offices and agencies, Regulation (EC) No 45/2001 applies. Regulation (EC) No 45/2001 and other Union legal acts applicable to such processing of personal data shall be adapted to the principles and rules of this Regulation in accordance with Article 98.\n\n&nbsp;&nbsp;&nbsp;&nbsp;4. This Regulation shall be without prejudice to the application of Directive 2000/31/EC, in particular of the liability rules of intermediary service providers in Articles 12 to 15 of that Directive.\n\n\nExtract 2:\n\nThe Working Party 29 has examined the obligation, under Article 30 of the GDPR, for controllers and processors to maintain a record of processing activities. This paper sets out the WP29\'s position on the derogation from this obligation. Recital 13 of the GDPR says:\n\'To take account of the specific situation of micro, small and medium-sized enterprises, this Regulation includes a derogation for organisations with fewer than 250 employees with regard to record-keeping\'.\nArticle 30(5) gives effect to Recital 13. It says that the obligation to keep a record of processing activities does not apply \'to an enterprise or an organisation employing fewer than 250 persons unless the processing it carries out is likely to result in a risk to the rights and freedoms of data subjects, the processing is not occasional, or the processing includes special categories of data as referred to in Article 9(1) or personal data relating to criminal convictions and offences referred to in Article 10.\' Some clarifications on the interpretation of this provision appear necessary, as shown by the high number of requests coming from companies and received in the last few months by national Supervisory Authorities.\nThe derogation provided by Article 30(5) is not absolute. There are three types of processing to which it does not apply. These are:\n·         Processing that is likely to result in a risk to the rights and freedoms of data subjects.\n·         Processing that is not occasional.\n·         Processing that includes special categories of data or personal data relating to criminal convictions and offences.\n\nThe WP29 underlines that the wording of Article 30(5) is clear in providing that the three types of processing to which the derogation does not apply are alternative ("or") and the occurrence of any one of them alone triggers the obligation to maintain the record of processing activities.\nTherefore, although endowed with less than 250 employees, data controllers or processors who find themselves in the position of either carrying out processing likely to result in a risk (not just a high risk) to the rights of the data subjects, or processing personal data on a non-occasional basis, or processing special categories of data under Article 9(1) or data relating to criminal convictions under Article 10 are obliged to maintain the record of processing activities. \nHowever, such organisations need only maintain records of processing activities for the types of processing mentioned by Article 30(5).\nFor example, a small organisation is likely to regularly process data regarding its employees. As a result, such processing cannot be considered "occasional" and must therefore be included in the record of processing activities.1 Other processing activities which are in fact "occasional", however, do not need to be included in the record of processing activities, provided they are unlikely to result in a risk to the right and freedoms of data subjects and do not involve special categories of data or personal data relating to criminal convictions and offences.\nThe WP29 highlights that the record of processing activities is a very useful means to support an analysis of the implications of any processing whether existing or planned. The record facilitates the factual assessment of the risk of the processing activities performed by a controller or processor on individuals\' rights, and the identification and implementation of appropriate security measures to safeguard personal data – both key components of the principle of accountability contained in the GDPR.\nFor many micro, small and medium-sized organisations, maintaining a record of processing activities is unlikely to constitute a particularly heavy burden. However, the WP29 recognises that Article 30 represents a new administrative requirement for controllers and processors, and therefore encourages national Supervisory Authorities to support SMEs by providing tools to facilitate the set up and management of records of processing activities. For example, a Supervisory Authority might make available on its website a simplified model that can be used by SMEs to keep records of processing activities not covered by the derogation in Article 30(5).\n\n\n1 The WP29 considers that a processing activity can only be considered as "occasional" if it is not carried out regularly, and occurs outside the regular course of business or activity of the controller or processor. See WP29 Guidelines on Article 49 of Regulation 2016/679 (WP262).'
        assert self.chat.messages_without_rag[-2]["role"] == "user"
        assert self.chat.messages_without_rag[-2]["content"].strip() == 'Are there exemptions from GDPR for small companies?'
        assert self.chat.messages_without_rag[-1]["role"] == "assistant"
        assert self.chat.messages_without_rag[-1]["content"].strip() == output_response


#         # test the workflow if the system cannot find useful content in the supplied data
#         self.chat.system_state = self.chat.State.RAG
#         user_content = "Who can trade gold?" # there are hits in the KB for this
#         testing = True # don't make call to openai API, use the canned response below
#         flag = "NONE:"
#         response = "None of the supplied documentation was relevant"
#         manual_responses_for_testing = []
#         manual_responses_for_testing.append(flag + response)
#         self.chat.user_provides_input(user_content,
#                                        testing = testing,
#                                        manual_responses_for_testing = manual_responses_for_testing)
#         assert self.chat.messages[-1]["role"] == "assistant"
#         assert self.chat.messages[-1]["content"].strip() == self.chat.Errors.NO_RELEVANT_DATA.value
#         assert self.chat.messages[-2]["role"] == "user"
#         assert self.chat.messages[-2]["content"].strip() == "Question: Who can trade gold?"
#         assert self.chat.messages_without_rag[-2]["role"] == "user"
#         assert self.chat.messages_without_rag[-2]["content"].strip() == user_content
#         assert self.chat.system_state == self.chat.State.RAG

#         # test the workflow if the system needs additional content
#         self.chat.system_state = self.chat.State.RAG
#         user_content = "Are there exemptions from GDPR for small companies?" # there are hits in the KB for this
#         testing = True # don't make call to openai API, use the canned response below        
#         flag = "SECTION:"
#         response = "Extract 2, Reference 30(5)"
#         manual_responses_for_testing = []
#         manual_responses_for_testing.append(flag + " " + response)
#         # now the response once it has received the additional data
#         flag = "ANSWER:"
#         input_response = 'Yes, there is a derogation for organizations with fewer than 250 employees with regard to record-keeping under Article 30 of the GDPR. However, this derogation is not absolute and does not apply to processing that is likely to result in a risk to the rights and freedoms of data subjects, processing that is not occasional, or processing that includes special categories of data or personal data relating to criminal convictions and offences. Therefore, small companies may be exempt from the record-keeping obligation if they meet the criteria specified in Article 30(5). Reference: 2'
#         # NOTE this output_response differs to the previous one because we have changed the search_sections while leaving the input_response to refer to extract 2 (which is now different)
#         output_response = 'Yes, there is a derogation for organizations with fewer than 250 employees with regard to record-keeping under Article 30 of the GDPR. However, this derogation is not absolute and does not apply to processing that is likely to result in a risk to the rights and freedoms of data subjects, processing that is not occasional, or processing that includes special categories of data or personal data relating to criminal convictions and offences. Therefore, small companies may be exempt from the record-keeping obligation if they meet the criteria specified in Article 30(5).  \nReference:  \nSection all from WORKING PARTY 29 POSITION PAPER on the derogations from the obligation to maintain records of processing activities pursuant to Article 30(5) GDPR'
#         manual_responses_for_testing.append(flag + input_response)


#         self.chat.user_provides_input(user_content,
#                                        testing = testing,
#                                        manual_responses_for_testing = manual_responses_for_testing)

#         assert self.chat.messages[-1]["role"] == "assistant"
#         assert self.chat.messages[-1]["content"].strip() == output_response
#         assert self.chat.messages[-2]["role"] == "user"
#         assert self.chat.messages[-2]["content"].strip() == 'Question: Are there exemptions from GDPR for small companies?\n\nExtract 1:\n# 2 Material scope\n\n&nbsp;&nbsp;&nbsp;&nbsp;1. This Regulation applies to the processing of personal data wholly or partly by automated means and to the processing other than by automated means of personal data which form part of a filing system or are intended to form part of a filing system.\n\n&nbsp;&nbsp;&nbsp;&nbsp;2. This Regulation does not apply to the processing of personal data:\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(a) in the course of an activity which falls outside the scope of Union law;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b) by the Member States when carrying out activities which fall within the scope of Chapter 2 of Title V of the TEU;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(c) by a natural person in the course of a purely personal or household activity;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(d) by competent authorities for the purposes of the prevention, investigation, detection or prosecution of criminal offences or the execution of criminal penalties, including the safeguarding against and the prevention of threats to public security.\n\n&nbsp;&nbsp;&nbsp;&nbsp;3. For the processing of personal data by the Union institutions, bodies, offices and agencies, Regulation (EC) No 45/2001 applies. Regulation (EC) No 45/2001 and other Union legal acts applicable to such processing of personal data shall be adapted to the principles and rules of this Regulation in accordance with Article 98.\n\n&nbsp;&nbsp;&nbsp;&nbsp;4. This Regulation shall be without prejudice to the application of Directive 2000/31/EC, in particular of the liability rules of intermediary service providers in Articles 12 to 15 of that Directive.\n\n\nExtract 2:\n\nThe Working Party 29 has examined the obligation, under Article 30 of the GDPR, for controllers and processors to maintain a record of processing activities. This paper sets out the WP29\'s position on the derogation from this obligation. Recital 13 of the GDPR says:\n\'To take account of the specific situation of micro, small and medium-sized enterprises, this Regulation includes a derogation for organisations with fewer than 250 employees with regard to record-keeping\'.\nArticle 30(5) gives effect to Recital 13. It says that the obligation to keep a record of processing activities does not apply \'to an enterprise or an organisation employing fewer than 250 persons unless the processing it carries out is likely to result in a risk to the rights and freedoms of data subjects, the processing is not occasional, or the processing includes special categories of data as referred to in Article 9(1) or personal data relating to criminal convictions and offences referred to in Article 10.\' Some clarifications on the interpretation of this provision appear necessary, as shown by the high number of requests coming from companies and received in the last few months by national Supervisory Authorities.\nThe derogation provided by Article 30(5) is not absolute. There are three types of processing to which it does not apply. These are:\n·         Processing that is likely to result in a risk to the rights and freedoms of data subjects.\n·         Processing that is not occasional.\n·         Processing that includes special categories of data or personal data relating to criminal convictions and offences.\n\nThe WP29 underlines that the wording of Article 30(5) is clear in providing that the three types of processing to which the derogation does not apply are alternative ("or") and the occurrence of any one of them alone triggers the obligation to maintain the record of processing activities.\nTherefore, although endowed with less than 250 employees, data controllers or processors who find themselves in the position of either carrying out processing likely to result in a risk (not just a high risk) to the rights of the data subjects, or processing personal data on a non-occasional basis, or processing special categories of data under Article 9(1) or data relating to criminal convictions under Article 10 are obliged to maintain the record of processing activities. \nHowever, such organisations need only maintain records of processing activities for the types of processing mentioned by Article 30(5).\nFor example, a small organisation is likely to regularly process data regarding its employees. As a result, such processing cannot be considered "occasional" and must therefore be included in the record of processing activities.1 Other processing activities which are in fact "occasional", however, do not need to be included in the record of processing activities, provided they are unlikely to result in a risk to the right and freedoms of data subjects and do not involve special categories of data or personal data relating to criminal convictions and offences.\nThe WP29 highlights that the record of processing activities is a very useful means to support an analysis of the implications of any processing whether existing or planned. The record facilitates the factual assessment of the risk of the processing activities performed by a controller or processor on individuals\' rights, and the identification and implementation of appropriate security measures to safeguard personal data – both key components of the principle of accountability contained in the GDPR.\nFor many micro, small and medium-sized organisations, maintaining a record of processing activities is unlikely to constitute a particularly heavy burden. However, the WP29 recognises that Article 30 represents a new administrative requirement for controllers and processors, and therefore encourages national Supervisory Authorities to support SMEs by providing tools to facilitate the set up and management of records of processing activities. For example, a Supervisory Authority might make available on its website a simplified model that can be used by SMEs to keep records of processing activities not covered by the derogation in Article 30(5).\n\n\n1 The WP29 considers that a processing activity can only be considered as "occasional" if it is not carried out regularly, and occurs outside the regular course of business or activity of the controller or processor. See WP29 Guidelines on Article 49 of Regulation 2016/679 (WP262). \n\nExtract 3:\n# 30 Records of processing activities\n\n&nbsp;&nbsp;&nbsp;&nbsp;5. The obligations referred to in paragraphs 1 and 2 shall not apply to an enterprise or an organisation employing fewer than 250 persons unless the processing it carries out is likely to result in a risk to the rights and freedoms of data subjects, the processing is not occasional, or the processing includes special categories of data as referred to in Article 9(1) or personal data relating to criminal convictions and offences referred to in Article 10.'
#         assert self.chat.messages_without_rag[-2]["role"] == "user"
#         assert self.chat.messages_without_rag[-2]["content"].strip() == user_content
#         assert self.chat.system_state == self.chat.State.RAG



#         # Test what happens if it calls for a section that it already has
#         self.chat.reset_conversation_history()
#         self.chat.system_state = self.chat.State.RAG
#         user_content = "Are there exemptions from GDPR for small companies?" # there are hits in the KB for this
#         testing = True # don't make call to openai API, use the canned response below        
#         flag = "SECTION:"
#         response = "Extract 2, Reference 2"
#         manual_responses_for_testing = []
#         manual_responses_for_testing.append(flag + " " + response)
#         # now the response once it has received the additional data
#         flag = "ANSWER:"
#         input_response = 'Yes, there is a derogation for organizations with fewer than 250 employees with regard to record-keeping under Article 30 of the GDPR. However, this derogation is not absolute and does not apply to processing that is likely to result in a risk to the rights and freedoms of data subjects, processing that is not occasional, or processing that includes special categories of data or personal data relating to criminal convictions and offences. Therefore, small companies may be exempt from the record-keeping obligation if they meet the criteria specified in Article 30(5). Reference: 2'
#         # NOTE this output_response differs to the previous one because we have changed the search_sections while leaving the input_response to refer to extract 2 (which is now different)
#         manual_responses_for_testing.append(flag + input_response)
#         self.chat.user_provides_input(user_content,
#                                        testing = testing,
#                                        manual_responses_for_testing = manual_responses_for_testing)

#         assert self.chat.messages[-1]["role"] == "assistant"
#         assert self.chat.messages[-1]["content"].strip() == 'Yes, there is a derogation for organizations with fewer than 250 employees with regard to record-keeping under Article 30 of the GDPR. However, this derogation is not absolute and does not apply to processing that is likely to result in a risk to the rights and freedoms of data subjects, processing that is not occasional, or processing that includes special categories of data or personal data relating to criminal convictions and offences. Therefore, small companies may be exempt from the record-keeping obligation if they meet the criteria specified in Article 30(5).  \nReference:  \nSection all from WORKING PARTY 29 POSITION PAPER on the derogations from the obligation to maintain records of processing activities pursuant to Article 30(5) GDPR'
#         assert self.chat.messages[-2]["role"] == "user"
#         assert self.chat.messages[-2]["content"].strip() == 'Question: Are there exemptions from GDPR for small companies?\n\nExtract 1:\n# 2 Material scope\n\n&nbsp;&nbsp;&nbsp;&nbsp;1. This Regulation applies to the processing of personal data wholly or partly by automated means and to the processing other than by automated means of personal data which form part of a filing system or are intended to form part of a filing system.\n\n&nbsp;&nbsp;&nbsp;&nbsp;2. This Regulation does not apply to the processing of personal data:\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(a) in the course of an activity which falls outside the scope of Union law;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b) by the Member States when carrying out activities which fall within the scope of Chapter 2 of Title V of the TEU;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(c) by a natural person in the course of a purely personal or household activity;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(d) by competent authorities for the purposes of the prevention, investigation, detection or prosecution of criminal offences or the execution of criminal penalties, including the safeguarding against and the prevention of threats to public security.\n\n&nbsp;&nbsp;&nbsp;&nbsp;3. For the processing of personal data by the Union institutions, bodies, offices and agencies, Regulation (EC) No 45/2001 applies. Regulation (EC) No 45/2001 and other Union legal acts applicable to such processing of personal data shall be adapted to the principles and rules of this Regulation in accordance with Article 98.\n\n&nbsp;&nbsp;&nbsp;&nbsp;4. This Regulation shall be without prejudice to the application of Directive 2000/31/EC, in particular of the liability rules of intermediary service providers in Articles 12 to 15 of that Directive.\n\n\nExtract 2:\n\nThe Working Party 29 has examined the obligation, under Article 30 of the GDPR, for controllers and processors to maintain a record of processing activities. This paper sets out the WP29\'s position on the derogation from this obligation. Recital 13 of the GDPR says:\n\'To take account of the specific situation of micro, small and medium-sized enterprises, this Regulation includes a derogation for organisations with fewer than 250 employees with regard to record-keeping\'.\nArticle 30(5) gives effect to Recital 13. It says that the obligation to keep a record of processing activities does not apply \'to an enterprise or an organisation employing fewer than 250 persons unless the processing it carries out is likely to result in a risk to the rights and freedoms of data subjects, the processing is not occasional, or the processing includes special categories of data as referred to in Article 9(1) or personal data relating to criminal convictions and offences referred to in Article 10.\' Some clarifications on the interpretation of this provision appear necessary, as shown by the high number of requests coming from companies and received in the last few months by national Supervisory Authorities.\nThe derogation provided by Article 30(5) is not absolute. There are three types of processing to which it does not apply. These are:\n·         Processing that is likely to result in a risk to the rights and freedoms of data subjects.\n·         Processing that is not occasional.\n·         Processing that includes special categories of data or personal data relating to criminal convictions and offences.\n\nThe WP29 underlines that the wording of Article 30(5) is clear in providing that the three types of processing to which the derogation does not apply are alternative ("or") and the occurrence of any one of them alone triggers the obligation to maintain the record of processing activities.\nTherefore, although endowed with less than 250 employees, data controllers or processors who find themselves in the position of either carrying out processing likely to result in a risk (not just a high risk) to the rights of the data subjects, or processing personal data on a non-occasional basis, or processing special categories of data under Article 9(1) or data relating to criminal convictions under Article 10 are obliged to maintain the record of processing activities. \nHowever, such organisations need only maintain records of processing activities for the types of processing mentioned by Article 30(5).\nFor example, a small organisation is likely to regularly process data regarding its employees. As a result, such processing cannot be considered "occasional" and must therefore be included in the record of processing activities.1 Other processing activities which are in fact "occasional", however, do not need to be included in the record of processing activities, provided they are unlikely to result in a risk to the right and freedoms of data subjects and do not involve special categories of data or personal data relating to criminal convictions and offences.\nThe WP29 highlights that the record of processing activities is a very useful means to support an analysis of the implications of any processing whether existing or planned. The record facilitates the factual assessment of the risk of the processing activities performed by a controller or processor on individuals\' rights, and the identification and implementation of appropriate security measures to safeguard personal data – both key components of the principle of accountability contained in the GDPR.\nFor many micro, small and medium-sized organisations, maintaining a record of processing activities is unlikely to constitute a particularly heavy burden. However, the WP29 recognises that Article 30 represents a new administrative requirement for controllers and processors, and therefore encourages national Supervisory Authorities to support SMEs by providing tools to facilitate the set up and management of records of processing activities. For example, a Supervisory Authority might make available on its website a simplified model that can be used by SMEs to keep records of processing activities not covered by the derogation in Article 30(5).\n\n\n1 The WP29 considers that a processing activity can only be considered as "occasional" if it is not carried out regularly, and occurs outside the regular course of business or activity of the controller or processor. See WP29 Guidelines on Article 49 of Regulation 2016/679 (WP262).'
#         assert self.chat.messages_without_rag[-2]["role"] == "user"
#         assert self.chat.messages_without_rag[-2]["content"].strip() == user_content
#         assert self.chat.system_state == self.chat.State.RAG


#         #test what happens if the LLM does not listen to instructions and returns something random
#         self.chat.reset_conversation_history()
#         self.chat.system_state = self.chat.State.RAG
#         user_content = "Are there exemptions from GDPR for small companies?" # there are hits in the KB for this
#         testing = True # don't make call to openai API, use the canned response below
#         response = "None of the supplied documentation was relevant"
#         manual_responses_for_testing = []
#         manual_responses_for_testing.append(response)
#         manual_responses_for_testing.append(response) # need to add it twice when checking this branch
#         self.chat.user_provides_input(user_content, 
#                                        testing = testing,
#                                        manual_responses_for_testing = manual_responses_for_testing)
#         assert self.chat.messages[-1]["role"] == "assistant"
#         assert self.chat.messages[-1]["content"].strip() == self.chat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value
#         assert self.chat.messages[-2]["role"] == "user"
#         assert self.chat.messages[-2]["content"].strip() == 'Question: Are there exemptions from GDPR for small companies?\n\nExtract 1:\n# 2 Material scope\n\n&nbsp;&nbsp;&nbsp;&nbsp;1. This Regulation applies to the processing of personal data wholly or partly by automated means and to the processing other than by automated means of personal data which form part of a filing system or are intended to form part of a filing system.\n\n&nbsp;&nbsp;&nbsp;&nbsp;2. This Regulation does not apply to the processing of personal data:\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(a) in the course of an activity which falls outside the scope of Union law;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b) by the Member States when carrying out activities which fall within the scope of Chapter 2 of Title V of the TEU;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(c) by a natural person in the course of a purely personal or household activity;\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(d) by competent authorities for the purposes of the prevention, investigation, detection or prosecution of criminal offences or the execution of criminal penalties, including the safeguarding against and the prevention of threats to public security.\n\n&nbsp;&nbsp;&nbsp;&nbsp;3. For the processing of personal data by the Union institutions, bodies, offices and agencies, Regulation (EC) No 45/2001 applies. Regulation (EC) No 45/2001 and other Union legal acts applicable to such processing of personal data shall be adapted to the principles and rules of this Regulation in accordance with Article 98.\n\n&nbsp;&nbsp;&nbsp;&nbsp;4. This Regulation shall be without prejudice to the application of Directive 2000/31/EC, in particular of the liability rules of intermediary service providers in Articles 12 to 15 of that Directive.\n\n\nExtract 2:\n\nThe Working Party 29 has examined the obligation, under Article 30 of the GDPR, for controllers and processors to maintain a record of processing activities. This paper sets out the WP29\'s position on the derogation from this obligation. Recital 13 of the GDPR says:\n\'To take account of the specific situation of micro, small and medium-sized enterprises, this Regulation includes a derogation for organisations with fewer than 250 employees with regard to record-keeping\'.\nArticle 30(5) gives effect to Recital 13. It says that the obligation to keep a record of processing activities does not apply \'to an enterprise or an organisation employing fewer than 250 persons unless the processing it carries out is likely to result in a risk to the rights and freedoms of data subjects, the processing is not occasional, or the processing includes special categories of data as referred to in Article 9(1) or personal data relating to criminal convictions and offences referred to in Article 10.\' Some clarifications on the interpretation of this provision appear necessary, as shown by the high number of requests coming from companies and received in the last few months by national Supervisory Authorities.\nThe derogation provided by Article 30(5) is not absolute. There are three types of processing to which it does not apply. These are:\n·         Processing that is likely to result in a risk to the rights and freedoms of data subjects.\n·         Processing that is not occasional.\n·         Processing that includes special categories of data or personal data relating to criminal convictions and offences.\n\nThe WP29 underlines that the wording of Article 30(5) is clear in providing that the three types of processing to which the derogation does not apply are alternative ("or") and the occurrence of any one of them alone triggers the obligation to maintain the record of processing activities.\nTherefore, although endowed with less than 250 employees, data controllers or processors who find themselves in the position of either carrying out processing likely to result in a risk (not just a high risk) to the rights of the data subjects, or processing personal data on a non-occasional basis, or processing special categories of data under Article 9(1) or data relating to criminal convictions under Article 10 are obliged to maintain the record of processing activities. \nHowever, such organisations need only maintain records of processing activities for the types of processing mentioned by Article 30(5).\nFor example, a small organisation is likely to regularly process data regarding its employees. As a result, such processing cannot be considered "occasional" and must therefore be included in the record of processing activities.1 Other processing activities which are in fact "occasional", however, do not need to be included in the record of processing activities, provided they are unlikely to result in a risk to the right and freedoms of data subjects and do not involve special categories of data or personal data relating to criminal convictions and offences.\nThe WP29 highlights that the record of processing activities is a very useful means to support an analysis of the implications of any processing whether existing or planned. The record facilitates the factual assessment of the risk of the processing activities performed by a controller or processor on individuals\' rights, and the identification and implementation of appropriate security measures to safeguard personal data – both key components of the principle of accountability contained in the GDPR.\nFor many micro, small and medium-sized organisations, maintaining a record of processing activities is unlikely to constitute a particularly heavy burden. However, the WP29 recognises that Article 30 represents a new administrative requirement for controllers and processors, and therefore encourages national Supervisory Authorities to support SMEs by providing tools to facilitate the set up and management of records of processing activities. For example, a Supervisory Authority might make available on its website a simplified model that can be used by SMEs to keep records of processing activities not covered by the derogation in Article 30(5).\n\n\n1 The WP29 considers that a processing activity can only be considered as "occasional" if it is not carried out regularly, and occurs outside the regular course of business or activity of the controller or processor. See WP29 Guidelines on Article 49 of Regulation 2016/679 (WP262).'
#         assert self.chat.system_state == self.chat.State.STUCK









#     def test_add_section_to_resource(self):
#         dfns = []
#         dfns.append(["GDPR", "4(2)", "My definition from GDPR"])
#         dfns.append(["Article_30_5", "", "My definition from article_30_5"])
#         df_definitions = pd.DataFrame(dfns, columns = ["document", "section_reference", "definition"])
#         sections = []
#         sections.append(["GDPR", "1", "My Section 1 from GDPR"])
#         sections.append(["GDPR", "2", "My Section 2 from GDPR"])
#         sections.append(["Article_30_5", "A.4(i)", "Fake section from Article_30_5"])
#         df_search_sections = pd.DataFrame(sections, columns = ["document", "section_reference", "regulation_text"])

#         # check if the section string passes validation but does not refer to something in the document
#         result = {"success": True, "path": "SECTION:", "extract": 1, "document": 'GDPR', "section": "50(5)(b)"}

#         df_updated = self.chat.add_section_to_resource(result, df_definitions, df_search_sections)
#         assert len(df_updated) == 3
#         assert df_updated.iloc[0]['section_reference'] == "1"
#         assert df_updated.iloc[1]['section_reference'] == '2'
#         assert df_updated.iloc[2]['section_reference'] == 'A.4(i)'

#         # Adding a valid string
#         result = {"success": True, "path": "SECTION:", "extract": 5, "document": 'GDPR', "section": "50"}
#         df_updated = self.chat.add_section_to_resource(result, df_definitions, df_search_sections)
#         assert len(df_updated) == 4
#         assert df_updated.iloc[0]['section_reference'] == "1"
#         assert df_updated.iloc[1]['section_reference'] == '2'
#         assert df_updated.iloc[2]['section_reference'] == "A.4(i)"
#         assert df_updated.iloc[3]['section_reference'] == '50'


#         # check if the section string only comes from the definitions
#         result = {"success": True, "path": "SECTION:", "extract": 1, "document": 'GDPR', "section": "50"}
#         df_updated = self.chat.add_section_to_resource(result, df_definitions, df_search_sections)
#         assert len(df_updated) == 4
#         assert df_updated.iloc[0]['section_reference'] == "1"
#         assert df_updated.iloc[1]['section_reference'] == '2'
#         assert df_updated.iloc[2]['section_reference'] == "A.4(i)"
#         assert df_updated.iloc[3]['section_reference'] == '50'


# #     def test_enrich_user_request_for_documentation(self):
# #         messages_without_rag = [{'role': 'user', 'content': 'Can foreign nationals send money home?'},
# #                                 {'role': 'assistant', 'content': 'Yes, foreign nationals can send money abroad if they meet certain conditions. Foreign nationals temporarily in South Africa are required to declare whether they are in possession of foreign assets upon arrival. If they complete the necessary declarations and undertakings, they may be permitted to conduct their banking on a resident basis, dispose of or invest their foreign assets, conduct non-resident or foreign currency accounts, and transfer funds abroad. However, they must be able to substantiate the source of the funds and the value of the funds should be reasonable in relation to their income generating activities in South Africa. The completed declarations and undertakings must be retained by the Authorised Dealers for a period of five years. There are also exemptions for single remittance transactions up to R5,000 and transactions where a business relationship has been established. (B.5(A)(i)(d), B.5(A)(i)(e))'}]
# #         user_content = 'Is there any documentation required?'
# #         model_to_use = "gpt-3.5-turbo"
# #         response = self.chat.enrich_user_request_for_documentation(user_content, messages_without_rag, model_to_use)
# #         print(response)
# #         assert(response.startswith('What documentation is required as evidence for'))

