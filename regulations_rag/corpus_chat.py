import logging
import pandas as pd
from openai import OpenAI
import re
import copy
import fnmatch
import regex # fuzzy lookup of references in a section of text
from enum import Enum

from regulations_rag.string_tools import match_strings_to_reference_list
                           
from regulations_rag.embeddings import get_ada_embedding, \
                           get_closest_nodes, \
                           num_tokens_from_string,  \
                           num_tokens_from_messages, \
                           EmbeddingParameters

from regulations_rag.rerank import RerankAlgos, rerank

from regulations_rag.corpus import Corpus

logger = logging.getLogger(__name__)

# Create custom log levels for the really detailed logs
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       
###############################################
## DO this in the main file, before importing chat_bot.py
# # Set up basic configuration first
# log_file = '.....'
# logging_level = logging.INFO
# if log_file == '':
#     logging.basicConfig(level=logging_level)
# else: 
#     logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging_level)

# # Then import chat_bot.py so it inherits these settings
###############################################


class ChatParameters:
    def __init__(self, chat_model, temperature, max_tokens):
        self.model = chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.tested_models = ["gpt-4o", "gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]
        untested_models = ["gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo"]
        if self.model not in self.tested_models:
            if self.model not in untested_models:
                raise ValueError("You are attempting to use a model that does not seem to exist")
            else: 
                logger.info("You are attempting to use a model that has not been tested")


class CorpusChat():
    class State(Enum):
        RAG = "rag"
        NO_DATA = "no_relevant_embeddings"
        NEEDS_DATA = "requires_additional_sections"
        STUCK = "stuck"

    '''
    The assistant's response MUST begin with one of these. Much of the effort below will be to ensure this. When the assistant message
    is parsed, this prefix will be stripped out and used to determine what to do with the remaining string - especially how to present it
    in a UI
    '''
    class Prefix(Enum):
        ANSWER = "ANSWER:"
        SECTION = "SECTION:"
        NONE = "NONE:"
        ERROR = "ERROR:"

    class Errors(Enum):
        NO_DATA = "ERROR: This app demonstrates Retrieval Augmented Generation (RAG). It is designed not to respond if it cannot retrieve relevant document sections to reference when answering a question. There are several reasons why valid questions may not yield relevant results, but often, minor rewording can help. A well-formed question is complete (i.e., it does not rely on previous conversation history for context and consists of more than just keywords or short phrases) and includes sufficient detail. Please try rephrasing your question, and I’ll see if I can find relevant sections to reference."
        NO_RELEVANT_DATA = "ERROR: This app is an example of Retrieval-Augmented Generation. Once promising sections of the source documents are identified, they are checked for relevance. In this case, the retrieval (search) step found sections that seemed promising but, upon inspection, were not relevant. In situations like this, I have been programmed not to respond, but rather to ask you to rephrase your question so I can try again. Questions with the highest chance of being answered are complete (i.e., do not rely on conversation history and contain more than just keywords or phrases) and provide sufficient detail. Please try rephrasing your question, and I'll try again."
        STUCK = "ERROR: Unfortunately the system is in an unrecoverable state. Please clear the chat history and retry your query"
        UNKNOWN_STATE = "ERROR: The system is in an unknown state and cannot proceed. Please clear the chat history and retry your query"
        NOT_FOLLOWING_INSTRUCTIONS = "ERROR: This app demonstrates Retrieval Augmented Generation. Behind the scenes, instructions are issued to a Large Language Model (LLM) and then verified. Occasionally, due to the statistical nature of the model, the LLM may not follow instructions correctly. In such cases, I am programmed not to respond but to ask you to clear the conversation history and try asking your question again. This usually resolves the issue. However, if the same error persists in the same spot, it likely indicates a bug rather than a statistical anomaly. Bugs are logged and will be addressed in due course. For now, please clear the conversation history and retry your query."

    def __init__(self, 
                 openai_client, 
                 embedding_parameters,
                 chat_parameters,
                 corpus_index,
                 rerank_algo = RerankAlgos.NONE,   
                 user_name_for_logging = 'test_user'): 

        self.user_name = user_name_for_logging
        self.openai_client = openai_client
        self.embedding_parameters = embedding_parameters
        self.chat_parameters = chat_parameters

        self.index = corpus_index
        self.corpus = self.index.corpus
        self.primary_document = self.corpus.get_primary_document()
        self.has_primary_document = False
        self.has_primary_document = self.primary_document != ""

        self.rerank_algo = rerank_algo
        self.reset_conversation_history()
        self.token_limit_when_truncating_message_queue = 3500

        # Key to CorpusChat is the notion of a reference. The LLM will be asked to answer the question based on a list of definitions
        # and sections. The LLM will be "forced" to create a list of references at the end (AND only the end) of its response or its
        # request for additional information. The text of the LLM response will be checked for (one) instance of this keyword and
        # all the numbered references will be used to filter the initial list of references and to create a nicely formatted 
        # response including the text of the reference(s). The keyword is defined as a variable because it is used in multiple places
        # so none of these are missed if a change is necessary
        self.reference_key_word = "Reference:"

    def reset_conversation_history(self):
        logger.log(ANALYSIS_LEVEL, f"{self.user_name}: Reset Conversation History")        
        self.messages_intermediate = []
        self.system_state = CorpusChat.State.RAG


    '''
    The idea is to use an 'OpenAI like' dictionary of message contexts but with additional fields that contain text that may need to be treaded differently e.g. references.
    The self.messages_intermediate can be used as the source of all information and can be formatted depending on who is using it (like OpenAI or the front end)

    To "future proof" this method, a dictionary other_content is provided. In this base class, it is not used. Override the methods


    in need. One envisaged use is if you want to het your assistant to provide alternative wordings of a user question or an "explore further" or "delve deeper" suggestions
    '''
    def append_content(self, role, content, df_definitions = None, df_sections = None, other_text = {}):
        if self.messages_intermediate and (self.messages_intermediate[-1]["role"] == role and self.messages_intermediate[-1]["content"] == content): # don't duplicate messages in the list
            return

        if role == "system":
            self.messages_intermediate.append({"role": role, "content": content, "other_text": other_text})
        elif role in ["user", "assistant"]:
            if df_definitions is None:
                df_definitions = pd.DataFrame()
            if df_sections is None:
                df_sections = pd.DataFrame()
            self.messages_intermediate.append({"role": role, "content": content, "definitions": df_definitions.copy(), "sections": df_sections.copy(), "other_text": other_text})
        else:
            logger.error(f"Tried to add a message for the role {role} which is not a valid role")

        return

    def format_messages_for_openai(self):
        messages = []
        for row in self.messages_intermediate:
            role = row['role']
            if role == 'system':
                messages.append(self.create_openai_system_message(row))
            elif role == 'user':
                messages.append(self.create_openai_user_message(row))
            elif role == "assistant":
                messages.append(self.create_openai_assistant_message(row))
        return messages

    '''
    Overwrite this message in your class if you need to incorporate anything from the "other_text" dictionary in the system prompt.
    This default implementation ignores the "other_text" dictionary
    '''
    def create_openai_system_message(self, row):
        return {"role": row['role'], "content": row['content']}

    '''
    Overwrite this message in your class if you need to incorporate anything from the "other_text" dictionary in the user prompt.
    This default implementation ignores the "other_text" dictionary but incorporates RAG data from the definitions and sections
    dictionary
    '''
    def create_openai_user_message(self, row):
        question = row['content']
        df_definitions = row['definitions']
        df_search_sections = row['sections']
        content_with_rag = self._add_rag_data_to_question(question, df_definitions, df_search_sections)
        return {"role": row['role'], "content": content_with_rag}

    def _add_rag_data_to_question(self, question, df_definitions, df_search_sections):
        """
        Appends relevant definitions and sections from the Manual to the question to provide context.

        This method formats the question with additional information from the Manual, including definitions and referenced sections, to aid in answering the question more accurately.

        Parameters:
        - question (str): The original question to be answered.
        - df_definitions (DataFrame): A DataFrame containing definitions from the Manual.
        - df_search_sections (DataFrame): A DataFrame containing sections from the Manual relevant to the question.

        Returns:
        - str: The question appended with definitions and sections from the Manual.
        """

        user_content = f'Question: {question}\n\n'
        counter = 1
        if not df_definitions.empty:
            for definition in df_definitions['definition'].to_list():
                # Append the formatted string with the definition and a newline character
                user_content += f"Extract {counter}:\n{definition}\n"
                counter += 1

        if not df_search_sections.empty:
            for section in df_search_sections['regulation_text'].to_list():
                # Append the formatted string with the definition and a newline character
                user_content += f"Extract {counter}:\n{section}\n"
                counter += 1

        return user_content


    def create_openai_assistant_message(self, row):
        llm_reply = row['content']
        df_definitions = row['definitions']
        df_search_sections = row['sections']

        response_dict = self._check_response(llm_reply, df_definitions, df_search_sections)
        content_with_rag = self._reformat_assistant_answer(response_dict, df_definitions, df_search_sections)
        return {"role": row['role'], "content": content_with_rag}

    ''' 
    An intermediate method to extract the LLM and the references from the "ANSWER:" path which can then be format
    for the various output (or input) formats
    '''
    def _extract_assistant_answer_and_references(self, result, df_definitions, df_search_sections):
        references_list = [] # columns = ["document_key", "document_name", "section_reference", "is_definition", "text"]
        
        #if not (result["success"] == True and result["path"] == self.Prefix.ANSWER.value):
        if not (result["success"] == True): # works for ANSWER and ERROR paths
            if 'assistant_response' in result:
                return result['assistant_response']
            return self.Errors.UNKNOWN_STATE.value
        # Extract and clean references from the raw response
        cleaned_references = result["reference"]

        # Early return if no references found. Keep the definitions but empty the search sections
        if not cleaned_references:
            headings = df_search_sections.columns.to_list()
            empty_results = pd.DataFrame([], columns = headings)
            # return result["answer"], df_definitions, empty_results
            df_references_list = pd.DataFrame(references_list, columns = ["document_key", "document_name", "section_reference", "is_definition", "text"])
            return result["answer"], df_references_list

        integer_references = cleaned_references

        used_definitions = []
        used_sections = []
        reference_string = ""
        number_of_definitions = len(df_definitions)
        for reference in integer_references:
            if reference <= number_of_definitions:
                row_number = reference - 1
                document_key = df_definitions.iloc[row_number]["document"]
                document_name = self.corpus.get_document(df_definitions.iloc[row_number]["document"]).name
                section_reference = df_definitions.iloc[row_number]["section_reference"] # can be "" i.e. no reference
                text = df_definitions.iloc[row_number]["definition"]
                references_list.append([document_key, document_name, section_reference, True, text])
            else:
                row_number = reference - number_of_definitions - 1
                document_key = df_search_sections.iloc[row_number]["document"]
                document_name = self.corpus.get_document(document_key).name
                section_reference = df_search_sections.iloc[row_number]["section_reference"]                
                text = self.corpus.get_text(document_name, section_reference, add_markdown_decorators=True, add_headings=True, section_only=False)
                #text = df_search_sections.iloc[row_number]["regulation_text"]
                references_list.append([document_key, document_name, section_reference, False, text])

        df_references_list = pd.DataFrame(references_list, columns = ["document_key", "document_name", "section_reference", "is_definition", "text"])
        return result['answer'], df_references_list

    def _reformat_assistant_answer(self, result, df_definitions, df_search_sections):
        """
        Reformats the "Reference:" section from the LLM's response to ensure consistency with the correctly formatted
        sections from the data used to populate the RAG content. It identifies each reference in the LLM's response,
        replaces it with the closest match from a list of provided references, and removes duplicates.

        Parameters:
        - result = {"success": True, "path": "ANSWER:", "answer": llm_text_without_references, "reference": references_as_integers}
        - df_definitions
        - sections_in_rag (list): A list of correctly formatted reference sections.

        Returns:
        - str: The reformatted response with consistent reference formatting and no duplicate references.
        If however there were problems extracting the references, this will return the original inputs
        - DataFrame: ALL the input df_definitions 
        - DataFrame: a subset of the input df_search_sections - the ones referenced in the LLM answer
        """
        llm_answer, df_references_list = self._extract_assistant_answer_and_references(result, df_definitions, df_search_sections)

        reference_string = ""
        formatted_references = ""
        for index, row in df_references_list.iterrows():
            document_name = row["document_name"]
            section_reference = row["section_reference"]
            if row["is_definition"]:
                if section_reference == "":
                    reference_string += f"The definitions in {document_name}  \n"
                else:
                    reference_string += f"Definition {section_reference} from {document_name}  \n"
            else:
                if section_reference == "":
                    reference_string += f"The document {document_name}  \n"
                else:
                    reference_string += f"Section {section_reference} from {document_name}  \n"
        if len(df_references_list) > 0:
            formatted_references = f"  \n{self.reference_key_word}  \n{reference_string}"
        
        return result['answer'] + formatted_references



    def _create_system_message(self, number_of_options = 3, review = False):
        """
        Generates a system message instructing the system on how to answer questions using specific guidelines.

        The message includes the pattern for referencing sections from the South African Exchange Control Manual and outlines three 
        options for responding to questions: answering directly, requesting additional documentation, or stating none if no other options apply.
        The optional parameter number_of_options can be set to 2 if you do not want the system to ask for more information. This can be  
        used when the system asks for more information but the information is already in the RAG prompt (which happens surprisingly often in some models)

        Parameters:
        - number_of_options (in): Can only be 3 (answer, ask for more info, none) or 2 (answer, none)


        Returns:
            str: A formatted message detailing how to respond to questions based on the manual's guidelines.
        """
        if not review:
            sys_instruction = f"You are answering questions about {self.index.corpus_description} for {self.index.user_type} based only on the reference extracts provided. You have {number_of_options} options:\n"
        else:
            sys_instruction = f"Please review your answer. You were asked to assist the user by responding to their question in 1 of {number_of_options} ways but your response does not follow the expected format. Please reformat your response so that it follows the requested format.\n" 

        if self.has_primary_document:
            sample_reference = self.corpus.get_document(self.primary_document).reference_checker.text_version
        else:
            sample_reference = "[Insert Reference Value Here]"
        
        sys_option_ans  = f"Answer the question. Preface an answer with the tag '{CorpusChat.Prefix.ANSWER.value}'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword '{self.reference_key_word}'. Do not include the word Extract, only provide the number(s).\n"
        sys_option_sec  = f"Request additional documentation. If, in the body of the extract(s) provided, there is a reference to another section that is directly relevant and not already provided, respond with the word '{CorpusChat.Prefix.SECTION.value}' followed by 'Extract extract_number, {self.reference_key_word} section_reference' - for example SECTION: Extract 1, {self.reference_key_word} {sample_reference}.\n"
        sys_option_none = f"State '{CorpusChat.Prefix.NONE.value}' and nothing else in all other cases\n"

        if number_of_options == 2:
            sys_instruction = f"{sys_instruction}1) {sys_option_ans}2) {sys_option_none}"
        elif number_of_options == 3:
            sys_instruction = f"{sys_instruction}1) {sys_option_ans}2) {sys_option_sec}3) {sys_option_none}"
        else:
            logger.log(DEV_LEVEL, f"Forcing the number of options in the system message to be 3. They were {number_of_options}")
            number_of_options == 3
            sys_instruction = f"{sys_instruction}1) {sys_option_ans}2) {sys_option_sec}3) {sys_option_none}"

        return sys_instruction


    def _check_response(self, llm_response_text, df_definitions, df_sections):
        # Return dictionaries:
        #{"success": False, "path": "SECTION:"/"ANSWER:", "llm_followup_instruction": llm_instruction} 
        #{"success": True, "path": "SECTION:", "document": 'GDPR', "section": section_reference}
        #{"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
        #{"success": True, "path": "NONE:"}


        if llm_response_text.startswith(CorpusChat.Prefix.ANSWER.value):
            prefix = CorpusChat.Prefix.ANSWER.value
            answer = llm_response_text[len(prefix):].strip()

            # the keyword 'self.reference_key_word' appears multiple times. I don't want this because I want to filter the DataFrames to only contain the 
            # subset of information that was used and I have seen instances were, if there are multiple instances of the keyword 'self.reference_key_word'  
            # will note make sense after the DataFrames have been filtered.
            count = answer.count(self.reference_key_word)
            if count > 1:
                llm_instruction = f"When answering the question, you used the keyword '{self.reference_key_word}' more than once. It is vitally important that this keyword is only used once in your answer and then only at the end of the answer followed only by an integer, comma separated list of the extracts used. Please reformat your response so that there is only one instance of the keyword '{self.reference_key_word}' and it is at the end of the answer."
                return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction}


            # Extract and clean references from the raw response
            references = answer.split(self.reference_key_word)[-1].split(",") if self.reference_key_word in answer else []
            cleaned_references = [ref.strip() for ref in references if ref.strip()]
            llm_text = answer[:answer.rfind(self.reference_key_word)].strip() if self.reference_key_word in answer else answer

            # Did not supply any references which is in line with the Instruction
            if not cleaned_references:                
                return {"success": True, "path": prefix, "answer": llm_text, "reference": []}

            # Loop through each item in the list
            references_as_integers = []
            for item in cleaned_references:
                try:
                    # Attempt to convert the item to an integer
                    integer_value = int(item)
                    if integer_value < 1 or integer_value > len(df_definitions) + len(df_sections):
                        llm_instruction = "When answering the question, you have made reference to an extract number that was not provided. Please re-write your references and only refer to the extracts provided by their number"
                        return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction}
                    references_as_integers.append(integer_value)
                except ValueError:
                    match = re.search(r'\d', item)
                    if not match:
                        llm_instruction = "When answering the question, you have made reference to an extract but I am unable to extract the number from your reference. Please re-write your answer using integer extract number(s)"
                        return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction}
                    else:
                        integer_value = int(match.group(0))
                        if integer_value < 1 or integer_value > len(df_definitions) + len(df_sections):
                            llm_instruction = "When answering the question, you have made reference to an extract number that was not provided. Please re-write your references and only refer to the extracts provided by their number"
                            return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction}
                        references_as_integers.append(integer_value)
            

            return {"success": True, "path": prefix, "answer": llm_text, "reference": references_as_integers}

        elif llm_response_text.startswith(CorpusChat.Prefix.SECTION.value):
            prefix = CorpusChat.Prefix.SECTION.value
            request = llm_response_text[len(prefix):].strip()

            document_name = ""
            section_reference = ""
            pattern = r"Extract (\d+), Reference (.+)"
            match = re.match(pattern, request)
            if match:
                extract_number = int(match.group(1))
                if extract_number < 1 or extract_number > len(df_definitions) + len(df_sections):
                        llm_instruction = "When requesting an additional section, you have made reference to an extract number that was not provided. Please re-write your answer and use a valid extract number"
                        return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction} 
                if extract_number < len(df_definitions):
                    document_name = df_definitions.iloc[extract_number-1]["document"]
                else:
                    document_name = df_sections.iloc[extract_number-len(df_definitions)-1]["document"]
                doc = self.corpus.get_document(document_name)
                section_reference = match.group(2)

                document_index = doc.reference_checker.text_version
                if doc.reference_checker.is_valid(section_reference):
                    section_reference = doc.reference_checker.extract_valid_reference(section_reference)
                    return {"success": True, "path": prefix, "extract": extract_number, "document": document_name, "section": section_reference}
                elif self.has_primary_document and document_name != self.primary_document: # articles in other documents can refer to the primary document so it makes sense to check if it is a primary document reference as well
                    primary_doc = self.corpus.get_document(self.primary_document)
                    if primary_doc.reference_checker.is_valid(section_reference):
                        section_reference = primary_doc.reference_checker.extract_valid_reference(section_reference)
                        return {"success": True, "path": prefix, "extract": extract_number, "document": self.primary_document, "section": section_reference}

                    if document_index == "":
                        document_index = primary_doc.reference_checker.text_version
                    else:
                        document_index += ", or " + primary_doc.reference_checker.text_version

                llm_instruction = f'The reference {section_reference} does not appear to be a valid reference for the document. Try using the format {document_index}'
                return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction} 
            else:
                llm_instruction = 'When requesting an additional section, you did not use the format r"Extract (\d+), Reference (.+)" or you included additional text. Please re-write your response using this format'
                return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction} 

            return {"success": True, "path": prefix, "extract": extract_number, "document": document_name, "section": section_reference}

        elif llm_response_text.startswith(CorpusChat.Prefix.NONE.value):
            return {"success": True, "path": CorpusChat.Prefix.NONE.value}

        elif llm_response_text.startswith(CorpusChat.Prefix.ERROR.value):
            prefix = CorpusChat.Prefix.ERROR.value
            answer = llm_response_text[len(prefix):].strip()

            references = []
            cleaned_references = []
            llm_text = answer

            # Did not supply any references which is in line with the Instruction
            return {"success": True, "path": prefix, "answer": llm_text, "reference": []}


        llm_instruction = f"Your response, did not begin with one of the keywords, '{CorpusChat.Prefix.ANSWER.value}', '{CorpusChat.Prefix.SECTION.value}' or '{CorpusChat.Prefix.NONE.value}'. Please review the question and provide an answer in the required format. Also make sure the referenced extracts are quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword '{self.reference_key_word}'. Do not include the word Extract, only provide the number(s).\n"
        return {"success": False, "path": "NONE:", "llm_followup_instruction": llm_instruction}
        




    def similarity_search(self, user_content):
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
        question_embedding = get_ada_embedding(self.openai_client, user_content, self.embedding_parameters.model, self.embedding_parameters.dimensions)        
        logger.log(DEV_LEVEL, "#################   Similarity Search       #################")
        if len(self.index.workflow) > 0:
            relevant_workflows = self.index.get_relevant_workflow(user_content = user_content, user_content_embedding = question_embedding, threshold = self.embedding_parameters.threshold)
            if not relevant_workflows.empty > 0:
                # relevant_workflows is sorted in the get_closest_nodes method
                most_relevant_workflow_score = relevant_workflows.iloc[0]['cosine_distance']
                workflow_triggered = relevant_workflows.iloc[0]['workflow']
                logger.info(f"Found a potentially relevant workflow: {workflow_triggered}")
            else:
                logger.log(DEV_LEVEL, "No relevant workflow found")
                most_relevant_workflow_score = 1.0
                workflow_triggered = "none"
        else:
            most_relevant_workflow_score = 1.0
            workflow_triggered = "none"

        relevant_definitions = self.index.get_relevant_definitions(user_content = user_content, user_content_embedding = question_embedding, threshold = self.embedding_parameters.threshold_definitions)
        if not relevant_definitions.empty:

            most_relevant_definition_score = relevant_definitions.iloc[0]['cosine_distance']

            if most_relevant_definition_score < most_relevant_workflow_score: # there is something more relevant than a workflow
                logger.log(DEV_LEVEL, f"Found a definition that was more relevant than the workflow: {workflow_triggered}")
                workflow_triggered = "none"        


        relevant_sections = self.index.get_relevant_sections(user_content = user_content, 
                                                             user_content_embedding = question_embedding, 
                                                             threshold = self.embedding_parameters.threshold, 
                                                             rerank_algo = self.rerank_algo)
        if not relevant_sections.empty:    
        
            if not relevant_sections.empty:    
                most_relevant_section_score = relevant_sections.iloc[0]['cosine_distance']

                if most_relevant_section_score < most_relevant_workflow_score and workflow_triggered != "none": # there is something more relevant than a workflow
                    logger.log(DEV_LEVEL, f"Found a section that was more relevant than the workflow: {workflow_triggered}")
                    workflow_triggered = "none"

        return workflow_triggered, relevant_definitions, relevant_sections

    def _get_api_response(self, messages):
        """
        Fetches a response from the OpenAI API (use unittest.mock module to "hardcode" api responses)

        Parameters:
        - messages (list): The list of messages to send as context to the OpenAI API.

        Returns:
        - str: The response from the OpenAI API.
        """
        model_to_use = self.chat_parameters.model
        total_tokens = num_tokens_from_messages(messages, model_to_use)
        
        if total_tokens > 15000:
            return "The is too much information in the prompt so we are unable to answer this question. Please try again or word the question differently"

        # Adjust model based on token count, similar to your original logic
        if (model_to_use in ["gpt-3.5-turbo", "gpt-4"]) and total_tokens > 3500:
            logger.warning("Switching to the gpt-3.5-turbo-16k model due to long prompt.")                
            model_to_use = "gpt-3.5-turbo-16k"
        
        response = self.openai_client.chat.completions.create(
                        model=model_to_use,
                        temperature=self.chat_parameters.temperature,
                        max_tokens=self.chat_parameters.max_tokens,
                        messages=messages
                    )
        response_text = response.choices[0].message.content
        return response_text

    def _truncate_message_list(self, system_message, message_list, token_limit = 3500):
        """
        Truncates the message list to fit within a specified token limit, ensuring the inclusion of the system message 
        and the most recent messages from the message list. The function guarantees that the returned list always contains 
        the system message and at least the last message from the message list, even if their combined token count exceeds 
        the token limit.

        Parameters:
        - system_message (list): A list containing a single dictionary with the system message.
        - message_list (list): A list of dictionaries representing the user and assistant messages.
        - token_limit (int, optional): The maximum allowed number of tokens. Defaults to 2000.

        Returns:
        - list: A list of messages truncated to meet the token limit, including the system message and the last messages.
        """
        if not message_list:
            return system_message

        # Initialize the token count with the system message and the last message in the list
        token_count = sum(num_tokens_from_string(msg["content"]) for msg in system_message + [message_list[-1]])
        number_of_messages = 1

        # Add messages from the end of the list until the token limit is reached or all messages are included
        while number_of_messages < len(message_list) and token_count < token_limit:
            next_message = message_list[-(number_of_messages + 1)]
            next_message_token_count = num_tokens_from_string(next_message["content"])

            # Check if adding the next message would exceed the token limit
            if token_count + next_message_token_count > token_limit:
                break
            # else keep going
            token_count += next_message_token_count
            number_of_messages += 1

        number_of_messages_excluding_system = max(1, number_of_messages - 1)
        # Compile the truncated list of messages, always including the system message and the most recent messages
        truncated_messages = [system_message[0]] + message_list[-number_of_messages_excluding_system:]

        return truncated_messages


    def resource_augmented_query(self, user_question, df_definitions, df_search_sections, number_of_options = 3):

        # This function's job is to make sure the response follows the rules or it fails. To this end, it will try a second call to the LLM 
        # if the fist call does not follow the rules. The second call will include a description of what was wrong with the fist answer. If,
        # after the second call, the LLM still does not return a response that follows the rules, this will return a False value for success.

        # NOTE: This function does not alter the user question nor the input RAG data so the RAG version of the user question is 
        #        self._add_rag_data_to_question(user_question, df_definitions, df_search_sections)

        # Returns
        # Unsuccessful path
        #   {"success": False, "path": ANSWER" / SECTION: / NONE:, "assistant_response": content} 
        # i.e replace the "llm_followup_instruction" key from the the dictionary created in self._check_response() with the key "assistant_message"

        # successful returns are the same as the dictionary created in self._check_response() 
        #   {"success": True, "path": "SECTION:", "extract", extract_num_as_int "document": document_name, "section": section_reference} NB the document may not be the same as the document in extract_num_as_int
        #   {"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
        #   {"success": True, "path": "NONE:"}

        # TODO: Ensure that the user_provides_input method takes responsibility for setting self.system_state = CorpusChat.State.STUCK
        #       if the result is unsuccessful

        # Here are the dictionary items returned from self._check_response()
        #{"success": False, "path": "SECTION:"/"ANSWER:", "llm_followup_instruction": llm_instruction, "openai_response": unedited_response_from_openai} 
        #{"success": True, "path": "SECTION:", "document": 'GDPR', "section": section_reference, "openai_response": unedited_response_from_openai}
        #{"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers, "openai_response": unedited_response_from_openai}
        #{"success": True, "path": "NONE:", "openai_response": unedited_response_from_openai}

        if self.system_state != CorpusChat.State.RAG:
            logger.error("resource_augmented_query method called but the the system is not in rag state")
            return {"success": False, "path": "NONE", "assistant_response": CorpusChat.State.STUCK.value, "openai_response": ""} 
        

        if len(self.messages_intermediate) > 1 or len(df_definitions) + len(df_search_sections) > 0: # should always be the case as we check this in the control loop
            logger.log(DEV_LEVEL, "#################   RAG Prompts   #################")

            system_content = self._create_system_message(number_of_options, review=False)
            logger.log(DEV_LEVEL, "System Prompt:\n" + system_content)

            # Replace the user question with the RAG version of it
            user_question = self._add_rag_data_to_question(user_question, df_definitions, df_search_sections)
            logger.log(DEV_LEVEL, "User Prompt with RAG:\n" + user_question) # this will be output with ANALYSIS_LEVEL

            chat_messages = self.format_messages_for_openai()
            chat_messages.append({"role": "user", "content": user_question})

            # Create a temporary message list. We will only add the messages to the chat history if we get well formatted answers
            system_message = [{"role": "system", "content": system_content}]
            truncated_chat = self._truncate_message_list(system_message, chat_messages, self.token_limit_when_truncating_message_queue)

            response = self._get_api_response(messages = truncated_chat)

            check_result = self._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
            check_result["openai_response"] = response
            if check_result["success"]:
                return check_result

            # The model did not perform as instructed so we not ask it to check its work
            logger.info(f"Initial chat API response did not follow instructions. New instruction: {check_result['llm_followup_instruction']}")

            # despondent_user_content = f"Please check your answer and make sure you preface your response using only one of the three permissible words, {CorpusChat.Prefix.ANSWER.value}, {CorpusChat.Prefix.SECTION.value} or {CorpusChat.Prefix.NONE.value}"
            # system_content = self._create_system_message(number_of_options, review=True)

            despondent_user_messages = truncated_chat + [
                                        {"role": "assistant", "content": response},
                                        {"role": "user", "content": check_result["llm_followup_instruction"]}]

            response = self._get_api_response(messages = despondent_user_messages)

            check_result = self._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
            check_result["openai_response"] = response
            if check_result["success"]:
                return check_result
            else: 
                msg = f"Even after trying a second time, the LLM was not following instructions. New instruction: {check_result['llm_followup_instruction']}"
                logger.error(msg)
                return {"success": False, "path": check_result["path"], "assistant_response": CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value, "openai_response": response} 


        msg = "A call to resource_augmented_query was made with insufficient information"
        logger.error(msg)
        return {"success": False, "path": "NONE", "assistant_response": CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value, "openai_response": ""} 
        



    # Note: To test the workflow I need some way to control the openai API responses. I have chosen to do this with unittest.mock module to "hardcode" api responses
    def user_provides_input(self, user_content):
        
        if user_content is None:
            logger.error(f"{self.user_name}: user_provides_input() function received an empty input. This should not have happened and is an indication there is a bug in the frontend. The system will be placed into a 'stuck' status")
            return self.place_in_stuck_state()

        if self.system_state == CorpusChat.State.STUCK:
            logger.log(DEV_LEVEL, "Unable to execute corpus_chat.user_provides_input because the system_state == CorpusChat.State.STUCK")
            self.append_content("user", user_content)
            self.append_content("assistant", CorpusChat.Errors.STUCK.value)
            return 

        elif self.system_state == CorpusChat.State.RAG:            
            logger.log(ANALYSIS_LEVEL, f"{self.user_name} question: {user_content}")        
            workflow_triggered, df_definitions, df_search_sections = self.similarity_search(user_content) # df_search_sections MUST not have "document"
            
            if workflow_triggered != "none":
                logger.log(DEV_LEVEL, f"Workflow triggered: {workflow_triggered}")
                workflow_triggered, df_definitions, df_search_sections = self.execute_workflow(workflow_triggered, user_content)

            if (len(df_definitions) + len(df_search_sections) == 0): # unable to find any relevant text in the database
                if len(self.messages_intermediate) < 2:
                    logger.log(DEV_LEVEL, "Executing path for no retrieval and no conversation history")
                    return self.execute_path_no_retrieval_no_conversation_history(user_content)
                    
                else:
                    logger.log(DEV_LEVEL, "Executing path for no retrieval but with conversation history")
                    return self.execute_path_no_retrieval_with_conversation_history(user_content)
                    

            else: # Retrieval step returns data
                result = self.resource_augmented_query(user_question = user_content, df_definitions = df_definitions, df_search_sections = df_search_sections, number_of_options=3)

                return self.select_path_and_execute(user_content, df_definitions, df_search_sections, result)
        else:
            logger.error("The system is in an unknown state")
            self.append_content("assistant", CorpusChat.Errors.UNKNOWN_STATE.value)
            return

    def select_path_and_execute(self, user_content, df_definitions, df_search_sections, result):
        if not result["success"]:
            logger.error("corpus_chat.resource_augmented_query did not return result[\"success\"] == True.")
            return self.execute_path_for_unsuccessful_rag(user_content, df_definitions, df_search_sections)
            
        if result["path"] == CorpusChat.Prefix.ANSWER.value:
            #   result = {"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
            logger.log(DEV_LEVEL, "corpus_chat.resource_augmented_query answered the question using the Retrieved text")
            return self.execute_path_for_successful_rag(user_content, df_definitions, df_search_sections, result)

        elif result["path"] == CorpusChat.Prefix.NONE.value:
            #   result {"success": True, "path": "NONE:"}
            logger.info("corpus_chat.resource_augmented_query was not not able to find relevant information in the retrieved text")
            return self.execute_path_for_no_relevant_information_in_retrieved_text(user_content, df_definitions, df_search_sections)

        elif result["path"] == CorpusChat.Prefix.SECTION.value:
            #   result = {"success": True, "path": "SECTION:", "extract", extract_num_as_int "document": document_name, "section": section_reference} NB the document may not be the same as the document in extract_num_as_int
            logger.log(DEV_LEVEL, f"System requested for more info: Extract {result['extract']} requested section {result['section']}")
            return self.execute_path_for_additional_sections_requested(user_content, df_definitions, df_search_sections, result)

        else:
            logger.error("Note: RAG returned an unexpected response")
            self.append_content("assistant", CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value)
            self.system_state = CorpusChat.State.STUCK # We are at a dead end.
            return


    def add_section_to_resource(self, result, df_definitions, df_search_sections):
        '''
        Returns a DataFrame that should be be used in RAG that is a replacement for df_search_sections.

        If there is an issue, this will return df_search_sections otherwise it will create a DataFrame with the the entry from 
        result["document"], result["section"] and df_search_sections.iloc[result["extract"] - len(df_definitions) - 1]

        '''
        # result = {"success": True, "path": "SECTION:", "extract": extract_num_as_int, "document": document_name, "section": section_reference} NB the document may not be the same as the document in extract_num_as_int
        # Step 1) confirm it is requesting something that passes validation
        doc = self.corpus.get_document(result["document"])
        modified_section_to_add = doc.reference_checker.extract_valid_reference(result["section"])
        
        if modified_section_to_add is None:
            logger.info(f"Tried to add {section_to_add} the Valid_Index object could not extract a valid reference from this")
            self.system_state = CorpusChat.State.STUCK 
            return df_search_sections
        
        text_to_add = ""
        try: # passes index verification but there is an error retrieving the section
            text_to_add = doc.get_text(modified_section_to_add)

        except Exception as e:
            logger.log(DEV_LEVEL, f"Tried to add {modified_section_to_add} but a call to get this regulation resulted in an exception {e}")
            self.system_state = CorpusChat.State.STUCK 
            return df_search_sections

        if text_to_add == "":
            logger.log(DEV_LEVEL, f"The section {modified_section_to_add} does not have any text. Is it a valid reference")
            return df_search_sections
          
        # prepare the new data
        new_row = [result["document"], result["section"], text_to_add]
        new_sections = pd.DataFrame([new_row], columns = ["document", "section_reference", "regulation_text"])


        if result["extract"] > len(df_definitions): # Delete the other sections, keep the referring section and the new data
            row_to_keep = result["extract"] - len(df_definitions) - 1
        # sections_to_keep = df_search_sections.iloc[[row_to_keep]]
        sections_to_keep = df_search_sections # keep everything - the context window is long enough
        new_sections = pd.concat([sections_to_keep, new_sections], ignore_index = True)

        return new_sections

    def place_in_stuck_state(self):
        logger.error("The system is now in 'stuck' mode")
        self.append_content("assistant", CorpusChat.Errors.UNKNOWN_STATE.value)
        self.system_state = CorpusChat.State.STUCK
        return

    """ 
    Override this method if you have created a table of workflows for your chat bot

    returns workflow_triggered, df_definitions, df_search_sections
    """
    def execute_workflow(self, workflow_triggered, user_content):
        logger.log(DEV_LEVEL, "Executing default corpus_chat.execute_workflow() i.e. not executing any workflow, only re-running the similarity search and returning the results")
        workflow_triggered, df_definitions, df_search_sections = self.similarity_search(user_content)
        return workflow_triggered, df_definitions, df_search_sections


    """ 
    Override this method if you want to execute something specific when the user question does not result in any hits in the database
    and there is no conversation history that may otherwise allow the system to infer some context and phrase a better question.

    The default behaviour is not to go to the LLM but simply return the hardcoded CorpusChat.Errors.NO_DATA.value error message
    """ 
    def execute_path_no_retrieval_no_conversation_history(self, user_content):
        logger.log(DEV_LEVEL, "Executing default corpus_chat.execute_path_no_retrieval_no_conversation_history() i.e. bypassing the LLM and forcing the assistant to respond with CorpusChat.Errors.NO_DATA.value")
        self.system_state = CorpusChat.State.RAG         
        self.append_content("user", user_content)       
        self.append_content("assistant", CorpusChat.Errors.NO_DATA.value)
        return

    """ 
    Override this method if you want to execute something specific when the user question does not result in any hits in the database
    BUT there IS conversation history that can be used to phrase a better question

    The default behaviour is not to go to the LLM but simply return the hardcoded CorpusChat.Errors.NO_DATA.value error message
    """ 
    def execute_path_no_retrieval_with_conversation_history(self, user_content):
        logger.log(DEV_LEVEL, "Executing default corpus_chat.execute_path_no_retrieval_with_conversation_history() i.e. bypassing the LLM and forcing the assistant to respond with CorpusChat.Errors.NO_DATA.value")
        self.system_state = CorpusChat.State.RAG         
        self.append_content("user", user_content)       
        self.append_content("assistant", CorpusChat.Errors.NO_DATA.value)
        return

    """ 
    Override this method if you want to execute something specific when the retrieval augmented generation step returns result["success"] != True

    The default behaviour here is to respond with CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value AND put the system in STUCK state
    """ 
    def execute_path_for_unsuccessful_rag(self, user_content, df_definitions, df_search_sections):
        logger.log(DEV_LEVEL, "Executing default corpus_chat.execute_path_for_unsuccessful_rag() i.e. bypassing the LLM and forcing the assistant to respond with CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value and updating system_state to CorpusChat.State.STUCK")
        self.append_content("user", user_content, df_definitions, df_search_sections)
        self.append_content("assistant", CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value)
        self.system_state = CorpusChat.State.STUCK # We are at a dead end.
        return


    def execute_path_for_successful_rag(self, user_content, df_definitions, df_search_sections, result):
        # result = {"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
        logger.log(DEV_LEVEL, "Executing default corpus_chat.execute_path_for_successful_rag() i.e. enriching the user question with the Retrieved text, reformatting the references")

        self.append_content("user", user_content, df_definitions, df_search_sections)
        # reformatted_response = self._reformat_assistant_answer(result, df_definitions = df_definitions, df_search_sections = df_search_sections)
        # collect references
        # HERE: The message must be result["answer"] + "Reference: " + the references
        self.append_content("assistant", result["openai_response"], df_definitions, df_search_sections)
        self.system_state = CorpusChat.State.RAG 
        return 

    def execute_path_for_no_relevant_information_in_retrieved_text(self, user_content, df_definitions, df_search_sections):
        # result {"success": True, "path": "NONE:"}
        logger.log(DEV_LEVEL, "Executing default corpus_chat.execute_path_for_no_relevant_information_in_retrieved_text() i.e. forcing the assistant response to CorpusChat.Errors.NO_RELEVANT_DATA.value")

        self.append_content("user", user_content, df_definitions, df_search_sections)
        self.append_content("assistant", CorpusChat.Errors.NO_RELEVANT_DATA.value, df_definitions, df_search_sections)
        self.system_state = CorpusChat.State.RAG
        return


    def execute_path_for_additional_sections_requested(self, user_content, df_definitions, df_search_sections, result):
        logger.log(DEV_LEVEL, "Executing default corpus_chat.execute_path_for_additional_sections_requested()")

        # Asking for an invalid section or a section that is already in the RAG
        force = False
        document = self.corpus.get_document(result['document'])
        if document.reference_checker.is_reference_or_parents_in_list(result['section'], df_search_sections["section_reference"].tolist()):
            logger.log(DEV_LEVEL, f"But {result['section']} is already in the RAG data, so now forcing the system to answer or opt out")
            force = True

        if force:
            result = self.resource_augmented_query(user_question = user_content, df_definitions = df_definitions, df_search_sections = df_search_sections, number_of_options=2)

        else:
            df_search_sections = self.add_section_to_resource(result, df_definitions, df_search_sections)
            if self.system_state == CorpusChat.State.STUCK: # failed to add the sections
                # TODO: Do you want to ask the user for help?
                logger.log(DEV_LEVEL, "Note: Request to add resources failed")
                self.append_content("assistant", CorpusChat.Errors.STUCK.value)
                return

            # ... and try again with new resources
            result = self.resource_augmented_query(user_question = user_content, df_definitions = df_definitions, df_search_sections = df_search_sections, number_of_options=3)
        
        if result["path"] == CorpusChat.Prefix.ANSWER.value:
            logger.log(DEV_LEVEL, "Note: Question answered with the additional information")
            return self.execute_path_for_successful_rag(user_content, df_definitions, df_search_sections, result)
        
        else: 
            logger.log(DEV_LEVEL, "Note: Even with the additional information, they system was unable to answer the question. Placing the system in 'stuck' mode")
            logger.log(DEV_LEVEL, f"The response from the query with additional resources was: \n{result}")
            msg = "A call for additional sections did not result in sufficient information to answer the question. The system is now stuck. Please clear the chat history and retry your query"
            self.append_content("assistant", msg)
            self.system_state = CorpusChat.State.STUCK
            return
