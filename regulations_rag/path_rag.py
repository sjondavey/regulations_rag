import pandas as pd
import re
from enum import Enum
import logging
from regulations_rag.corpus_index import CorpusIndex

from regulations_rag.corpus_chat_tools import ChatParameters, get_caveat_for_no_rag_response

from regulations_rag.data_classes import AnswerWithRAGResponse, AnswerWithoutRAGResponse, \
    NoAnswerClassification, NoAnswerResponse, ErrorClassification, ErrorResponse

logger = logging.getLogger(__name__)
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       

# *If* everything works, a search step (in a different path and hence not in this file) will provide DataFrames with 
# definitions and sections that may be to the users question. Some of these will be used to answer the question amd the 
# dictionary produced will look like this:
# {
#   "role": "assistant", 
#   "content": "The answer, formatted for OpenAI history, including the text from the references", 
#   "assistant_response": AssistantResponse object
# }
#
# The steps to get here are:
#
# 1) resource_augmented_query(...): This takes number_of_options as input and returns a a message:
#   [{"role": "assistant", "content": text_llm_response}]
# where text_llm_response = prefix + answer + reference_key_word_RAG + references
#
# 2) check_response_RAG(...) checks this message using the reference definitions and sections dataframes. 
#   If the LLM followed instructions, the AssistantMessageClassification.ANSWER_WITH_RAG
#   message is created and returned. If a genuine error is detected, the AssistantMessageClassification.ERROR message is 
#   created and returned. 
#   If the LLM asks for additional resouces or does not follow instructions, internal (to this RAG Path and hence only in this file) 
#   messages are created with information on how to proceed. 
#
# 3) process_llm_response(...): Takes the internal messages and tries to use them to get an answer. At the end of this function,
#   the final response (ANSWER or ERROR) is created and returned.


class PathRAG:

    # Key to RAG is the notion of a reference. The LLM will be asked to answer the question based on a list of definitions
    # and sections. The LLM will be "forced" to create a list of references at the end (AND only the end) of its response or its
    # request for additional information. The text of the LLM response will be checked for (one) instance of this keyword and
    # all the numbered references will be used to filter the initial list of references and to create a nicely formatted 
    # response including the text of the reference(s). The keyword is defined as a variable because it is used in multiple places
    # so none of these are missed if a change is necessary
    def __init__(self, corpus_index: CorpusIndex, chat_parameters: ChatParameters):
        self.reference_key_word_RAG = "Reference:"
        self.corpus_index = corpus_index
        self.corpus = corpus_index.corpus
        self.chat_parameters = chat_parameters
        self.execution_path = [] # used to track the execution path for testing and analysis

    def _track_path(self, step):
        self.execution_path.append(step)    

    def remove_rag_data_from_message_history(self, message_history: list):
        stripped_message_history = []
        for message in message_history:
            stripped_message = {"role": message["role"], "content": message["content"]}
            stripped_message_history.append(stripped_message)
        return stripped_message_history


    def perform_RAG_path(self, message_history: list, current_user_message: dict):
        """
        Performs the RAG (Retrieval-Augmented Generation) path based on the provided chat data for RAG.

        This function orchestrates the entire process of RAG, from initializing the RAG path to processing the LLM response.


        Returns:
        dict: A final response dictionary representing the result of the RAG process. It includes:
        - 'role': Always 'assistant'.
        - 'content': The content of the response, which could be an answer, an error message,
        or a request for more information.
        - 'assistant_response': A dictionary containing metadata about the response, such as:
        - 'AssistantMessageClassification': An enum value indicating the type of response.
        - 'AnswerClassification' / 'NoAnswerClassification' / 'ErrorClassification': An enum from the relvant enum class.
        - 'answer': text that may be relevant.
        - 'reference': A DataFrame with all references used (if applicable).
        """
        self._track_path("PathRAG.perform_RAG_path")
        if "reference_material" not in current_user_message:
            current_user_message["reference_material"] = {
                "definitions": pd.DataFrame(),
                "sections": pd.DataFrame()
            }
        if "definitions" not in current_user_message["reference_material"]:
            current_user_message["reference_material"]["definitions"] = pd.DataFrame()
        if "sections" not in current_user_message["reference_material"]:
            current_user_message["reference_material"]["sections"] = pd.DataFrame()


        rag_response = self.resource_augmented_query(message_history = message_history, 
                                                     current_user_message = current_user_message, 
                                                     number_of_options = 3)
        if "assistant_response" in rag_response:
            return rag_response

        llm_checked_response = self.check_response_RAG(
                                                llm_message_response = rag_response, 
                                                df_definitions = current_user_message["reference_material"]["definitions"], 
                                                df_sections = current_user_message["reference_material"]["sections"])
        if "assistant_response" in llm_checked_response:
            return llm_checked_response

        final_response = self.process_llm_response(llm_checked_response = llm_checked_response, message_history=message_history, current_user_message=current_user_message)
        return final_response

    '''
    The LLM (or system logic in the function before the call to the LLM) will preface a text answer with one of 
    these words. These will then be used to determine the how to turn the text answer into a final response.

    After it is used for classification, the prefix is stripped out of the assistant's message, and the remaining text will be used as the 
    assistant's content.
    '''
    class LLMPrefix(Enum):
        ANSWER = "ANSWER:"
        SECTION = "SECTION:"
        NONE = "NONE:"

    class RAGPath(Enum):
        SECTION = "System requires additional documents to answer the question"
        FOLLOWUP = "The system did not follow instructions. A followup instruction has been provided"

    def format_user_question(self, question, df_definitions, df_search_sections):
        """
        Adds references with identifying extract numbers to the user question. The system message will instruct the LLM to
        use these extract numbers to create a list (subset) of references that were *used* to answer the question.
        In other methods, the *used* references will be reformatted into one DataFrame and saved in the assistant's message.

        Parameters:
        - question (str): The original question to be answered.
        - df_definitions (DataFrame): A DataFrame containing definitions from the corpus.
        - df_search_sections (DataFrame): A DataFrame containing sections from the corpus relevant to the question.

        Returns:
        - str: The input question followed by the relevant text from *all* definitions and sections. Each definition 
        or section is prefixed with the string "Extract N:" (where N starts at 1). (Note: These extact numbers will differ 
        from the extract numbers used to create the user question which numbers *all* the definitions and sections.)
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


    def create_system_message_RAG(self, number_of_options=3, review=False):
        """
        Generates a system message instructing the AI on how to answer questions using specific guidelines.

        Parameters:
        - corpus_index: The corpus_index object used to get the corpus description, user type, and main document.
        - number_of_options (int): Can only be 2 or 3. 
            3: (answer, ask for more info, none)
            2: (answer, none) - forces the system to answer the question with only the RAG data at hand.
        - review (bool): If True, the system will be instructed to review its answer. This is used when the LLM 
            did not preface its answer with one of the keywords in LLMPrefix.

        Returns:
            str: A formatted message detailing how to respond to questions based on the corpus guidelines.
        """
        if not review:
            sys_instruction = f"You are answering questions about {self.corpus_index.corpus_description} for {self.corpus_index.user_type} based only on the reference extracts provided. You have {number_of_options} options:\n"
        else:
            sys_instruction = f"Please review your answer. You were asked to assist the user by responding to their question in 1 of {number_of_options} ways but your response does not follow the expected format. Please reformat your response so that it follows the requested format.\n" 


        if self.corpus_index.corpus.get_primary_document() != "":
            primary_document = self.corpus_index.corpus.get_document(self.corpus_index.corpus.get_primary_document())
            sample_reference = primary_document.reference_checker.text_version
        else:
            sample_reference = "[Insert Reference Value Here]"

        sys_option_ans  = f"Answer the question. Preface an answer with the tag '{self.LLMPrefix.ANSWER.value}'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword '{self.reference_key_word_RAG}'. Do not include the word Extract, only provide the number(s).\n"
        sys_option_sec  = f"Request additional documentation. If, in the body of the extract(s) provided, there is a reference to another section that is directly relevant and not already provided, respond with the word '{self.LLMPrefix.SECTION.value}' followed by 'Extract extract_number, {self.reference_key_word_RAG} section_reference' - for example SECTION: Extract 1, {self.reference_key_word_RAG} {sample_reference}.\n"
        sys_option_none = f"State '{self.LLMPrefix.NONE.value}' and nothing else in all other cases\n"

        if number_of_options == 2:
            sys_instruction = f"{sys_instruction}1) {sys_option_ans}2) {sys_option_none}"
        elif number_of_options == 3:
            sys_instruction = f"{sys_instruction}1) {sys_option_ans}2) {sys_option_sec}3) {sys_option_none}"
        else:
            logger.log(DEV_LEVEL, f"Forcing the number of options in the system message to be 3. They were {number_of_options}")
            number_of_options == 3
            sys_instruction = f"{sys_instruction}1) {sys_option_ans}2) {sys_option_sec}3) {sys_option_none}"

        return sys_instruction


    def check_response_RAG(self, llm_message_response, df_definitions, df_sections):
        """
        Checks the response returned by resource_augmented_query(...) to determine how to proceed.

        The input llm_message_response is a dictionary with a "content" key. The value of the "content" key 
        is the string returned from OpenAI that needs to be checked.

        This function returns a dictionary. If no further action is required or can be performed, the dictionary will be in its final
        form and should be passed back to CorpusChat. You will know this is the case when the dictionary contains the keys "role", "content",
        AND "assistant_response".
        
        If, however, the LLM response is deemed to need more work (more sections, followup instruction), the dictionary 
        will contain the key "RAGPath" with the details necessary to follow up.

        Returns:    

        Dictionaries when more work is required:
        1) Needs to check its work
            {
                "RAGPath": RAGPath.FOLLOWUP.value, 
                "content": llm_message_response["content"], # we need this in later steps
                "llm_followup_instruction": llm_instruction
            }

        2) Requires additional information
            {
                "RAGPath": RAGPath.SECTION.value, 
                "extract": extract_number, # Which extract (from the formatted user question) referenced the additional section
                "document": document_name, # the document that contains the additional section
                "section": section_reference # the reference to the additional section
            }

        Dictionaries in final form that can be passed to CorpusChat:

        1) Success
            {
                "role": "assistant", 
                "content": content formatted for OpenAI history, 
                "assistant_response": AnswerWithRAGResponse / AnswerWithoutRAGResponse object
            }
                
        2) LLM is not able to answer the question with the provided information
            {
                "role": "assistant", 
                "content": "The system was not able to answer the question using the provided documents", 
                "assistant_response": NoAnswerResponse object
            }

        NOTE: There are no errors here. Errors are caught and handled elsewhere. Here we can either provide answer or no answer;
            or produce an internal message to follow up and produce an answer or no answer.
        """
        self._track_path("PathRAG.check_response_RAG")
        
        full_llm_text_response = llm_message_response["content"] #includes prefix and references IF all went well

        if full_llm_text_response.startswith(self.LLMPrefix.ANSWER.value):
            prefix = self.LLMPrefix.ANSWER.value
            llm_text_without_prefix = full_llm_text_response[len(prefix):].strip()

            # I don't want the keyword 'reference_key_word_RAG' to appear more than once in the answer. I want to filter the DataFrames to only 
            # contain the subset of information that was used and I have seen instances were, if there are multiple instances 
            # f the keyword 'reference_key_word_RAG'  will note make sense after the DataFrames have been filtered.
            count = llm_text_without_prefix.count(self.reference_key_word_RAG)
            if count > 1:
                llm_instruction = f"When answering the question, you used the keyword '{self.reference_key_word_RAG}' more than once. It is vitally important that this keyword is only used once in your answer and then only at the end of the answer followed only by an integer, comma separated list of the extracts used. Please reformat your response so that there is only one instance of the keyword '{self.reference_key_word_RAG}' and it is at the end of the answer."
                return {
                    "RAGPath": self.RAGPath.FOLLOWUP.value, 
                    "content": full_llm_text_response,
                    "llm_followup_instruction": llm_instruction
                }

            # Extract and clean references from the raw response
            references = llm_text_without_prefix.split(self.reference_key_word_RAG)[-1].split(",") if self.reference_key_word_RAG in llm_text_without_prefix else []
            cleaned_references = [ref.strip() for ref in references if ref.strip()]
            llm_text = llm_text_without_prefix[:llm_text_without_prefix.rfind(self.reference_key_word_RAG)].strip() if self.reference_key_word_RAG in llm_text_without_prefix else llm_text_without_prefix

            # Did not supply any references which is in line with the Instruction so a final version of the answer is returned
            if not cleaned_references:                
                assistant_response = AnswerWithoutRAGResponse(
                    answer=llm_text,
                    caveat= get_caveat_for_no_rag_response() ## TODO: This may not work BUT I may also want a different caveat for different types of questions
                )
                return {
                    "role": "assistant", 
                    "content": assistant_response.create_openai_content(), 
                    "assistant_response": assistant_response
                }


            # Loop through each item in the list
            references_as_integers = []
            for item in cleaned_references:
                try:
                    # Attempt to convert the item to an integer
                    integer_value = int(item)
                    if integer_value < 1 or integer_value > len(df_definitions) + len(df_sections):
                        llm_instruction = "When answering the question, you made reference to an extract number that was not provided. Please re-write your answer and only refer to the extracts provided by their number"
                        return {
                            "RAGPath": self.RAGPath.FOLLOWUP.value, 
                            "content": full_llm_text_response,
                            "llm_followup_instruction": llm_instruction
                        }
                    
                    references_as_integers.append(integer_value)
                
                except ValueError:
                    match = re.search(r'\d+', item)
                    if not match:
                        llm_instruction = "When answering the question, you have made reference to an extract but I am unable to extract the number from your reference. Please re-write your answer using integer extract number(s)"
                        return {
                            "RAGPath": self.RAGPath.FOLLOWUP.value, 
                            "content": full_llm_text_response,
                            "llm_followup_instruction": llm_instruction
                        }
                    else:
                        integer_value = int(match.group(0))
                        if integer_value < 1 or integer_value > len(df_definitions) + len(df_sections):
                            llm_instruction = "When answering the question, you have made reference to an extract number that was not provided. Please re-write your answer and only refer to the extracts provided by their number"
                            return {
                                "RAGPath": self.RAGPath.FOLLOWUP.value, 
                                "content": full_llm_text_response,
                                "llm_followup_instruction": llm_instruction
                            }                        
                        
                        references_as_integers.append(integer_value)
            
            if len(references_as_integers) == 0:
                assistant_response = AnswerWithoutRAGResponse(
                    answer = llm_text,
                    caveat = get_caveat_for_no_rag_response() ## TODO: This may not work BUT I may also want a different caveat for different types of questions
                )
                return {
                    "role": "assistant", 
                    "content": assistant_response.create_openai_content(), 
                    "assistant_response": assistant_response
                }
                

            # Everything worked as expected so now format the final response
            df_used_references = self.extract_used_references(
                                                        integer_list_of_used_references = references_as_integers, 
                                                        df_definitions = df_definitions, 
                                                        df_search_sections = df_sections)
            assistant_response = AnswerWithRAGResponse(
                answer = llm_text,
                references = df_used_references
            )
            assist_response_formatted_for_chat = assistant_response.create_openai_content()
            return {
                "role": "assistant", 
                "content": assist_response_formatted_for_chat, 
                "assistant_response": assistant_response
            }

        elif full_llm_text_response.startswith(self.LLMPrefix.SECTION.value):
            prefix = self.LLMPrefix.SECTION.value
            request = full_llm_text_response[len(prefix):].strip()

            document_name = ""
            section_reference = ""
            pattern = r"extract\s*:?\s*(\d+).*reference\s*:?\s*(.+)"
            match = re.match(pattern, request, re.IGNORECASE)
            if match:
                extract_number = int(match.group(1))
                if extract_number < 1 or extract_number > len(df_definitions) + len(df_sections):
                        llm_instruction = "When requesting an additional section, you have made reference to an extract number that was not provided. Please re-write your answer and use a valid extract number"
                        return {
                            "RAGPath": self.RAGPath.FOLLOWUP.value, 
                            "content": full_llm_text_response,
                            "llm_followup_instruction": llm_instruction
                        } 
                if extract_number < len(df_definitions):
                    document_name = df_definitions.iloc[extract_number-1]["document"]
                else:
                    document_name = df_sections.iloc[extract_number-len(df_definitions)-1]["document"]
                doc = self.corpus.get_document(document_name)
                section_reference = match.group(2)

                document_index = doc.reference_checker.text_version
                if doc.reference_checker.is_valid(section_reference):
                    section_reference = doc.reference_checker.extract_valid_reference(section_reference)
                    return {
                        "RAGPath": self.RAGPath.SECTION.value, 
                        "extract": extract_number, 
                        "document": document_name, 
                        "section": section_reference
                    }
                elif self.corpus.get_primary_document() != "" and document_name != self.corpus.get_primary_document(): # articles in other documents can refer to the primary document so it makes sense to check if it is a primary document reference as well
                    primary_doc = self.corpus.get_document(self.corpus.get_primary_document())
                    if primary_doc.reference_checker.is_valid(section_reference):
                        section_reference = primary_doc.reference_checker.extract_valid_reference(section_reference)
                        return {
                            "RAGPath": self.RAGPath.SECTION.value, 
                            "extract": extract_number, 
                            "document": self.corpus.get_primary_document(), 
                            "section": section_reference
                        }

                    if document_index == "":
                        document_index = primary_doc.reference_checker.text_version
                    else:
                        document_index += ", or " + primary_doc.reference_checker.text_version

                llm_instruction = f'The reference {section_reference} does not appear to be a valid reference for the document. Try using the format {document_index}'
                return {
                    "RAGPath": self.RAGPath.FOLLOWUP.value, 
                    "content": full_llm_text_response,
                    "llm_followup_instruction": llm_instruction
                } 
            else:
                llm_instruction = r'When requesting an additional section, you did not use the format "Extract (\d+), Reference (.+)" or you included additional text. Please re-write your response using this format'
                return {
                    "RAGPath": self.RAGPath.FOLLOWUP.value, 
                    "content": full_llm_text_response,
                    "llm_followup_instruction": llm_instruction
                } 

            return {
                "RAGPath": self.RAGPath.SECTION.value, 
                "extract": extract_number, 
                "document": document_name, 
                "section": section_reference
            }

        elif full_llm_text_response.startswith(self.LLMPrefix.NONE.value):
            return {
                "role": "assistant", 
                "content": "The system was not able to answer the question using the provided documents", 
                "assistant_response": NoAnswerResponse(
                    classification = NoAnswerClassification.NO_RELEVANT_DATA,
                )
            }
        
        llm_instruction = f"Your response, did not begin with one of the keywords, '{self.LLMPrefix.ANSWER.value}', '{self.LLMPrefix.SECTION.value}' or '{self.LLMPrefix.NONE.value}'. Please review the question and provide an answer in the required format. Also make sure the referenced extracts are quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword '{self.reference_key_word_RAG}'. Do not include the word Extract, only provide the number(s).\n"
        return {
            "RAGPath": self.RAGPath.FOLLOWUP.value, 
            "content": full_llm_text_response,
            "llm_followup_instruction": llm_instruction
        }


    def resource_augmented_query(self, message_history, current_user_message,  number_of_options = 3):
        '''
        Generates a message with "role" and "content" keys from one pass of the LLM, which needs to be checked
        to decide how to proceed.

        Any errors are reported as final, 'external' messages that can be returned to CorpusChat. These
        messages are identified by the presence of the "assistant_response" key. When calling this function
        always check for the presence of "assistant_response" and create a path for these messages as they
        are final and do not need further processing.

        Parameters:
        - number_of_options (int): The number of response options available to the LLM (default is 3).

        Returns:
        dict: A message dictionary with the following structure:
            {
                "role": "assistant",
                "content": str  # The LLM's response
            }

        If the "assistant_response" key is also present in the returned dictionary, the message
        is assumed to be the final message formatted for CorpusChat and should be returned immediately.

        Note:
        This function does not implement multiple attempts or error handling for non-compliant responses.
        Such logic, is implemented in the check_response_RAG and process_llm_response functions.
        '''
        
        self._track_path("PathRAG.resource_augmented_query")

        df_definitions = current_user_message["reference_material"]["definitions"]
        df_search_sections = current_user_message["reference_material"]["sections"]
        if  len(df_definitions) + len(df_search_sections) == 0:
            logger.log(DEV_LEVEL, "resource_augmented_query called with no RAG data.")
            # A final, 'external' message - identified by the presence of the "assistant_response" key.
            assistant_response = NoAnswerResponse(
                classification = NoAnswerClassification.NO_DATA,
            )
            return {
                "role": "assistant", 
                "content": assistant_response.create_openai_content(), 
                "assistant_response": assistant_response
            }

        system_content = self.create_system_message_RAG(number_of_options, review=False)
        logger.log(DEV_LEVEL, "resource_augmented_query called with System Prompt:\n" + system_content)

        # Replace the user question with the RAG version of it
        user_question = self.format_user_question(current_user_message['content'], df_definitions, df_search_sections)
        logger.log(DEV_LEVEL, "resource_augmented_query called with User Prompt:\n" + user_question) 

        local_messages = message_history.copy()
        local_messages.append({"role": "user", "content": user_question})

        # Create a temporary message list. We will only add the messages to the chat history if we get well formatted answers
        system_message = [{"role": "system", "content": system_content}]
        text_llm_response = self.chat_parameters.get_api_response(system_message, local_messages)
        # An intermediate response which needs to be checked - identified by the absence of the "assistant_response" key.
        return {
            "role": "assistant", 
            "content": text_llm_response
        } 


    def process_llm_response(self, llm_checked_response, message_history, current_user_message):
        """
        Process 'internal' RAG messages e.g. paths that need additional sections or 
        require follow up calls to the LLM, to produce an 'external' message (dictionary) that 
        can be returned to CorpusChat.

        Parameters:
        - llm_checked_response (dict): The result (an 'internal' message) of checking the LLM's 
        text response, which describes the work to be done to get to a final, 'external' message
        for CorpusChat (e.g., requesting more sections or following up).

        Returns:
        dict: A final, 'external' message representing the final RAG response. It includes:
        - 'role': Always 'assistant'.
        - 'content': The content of the response, which could be an answer, an error message,
        or a request for more information.
        - 'assistant_response': An object of type AnswerWithRAGResponse, AnswerWithoutRAGResponse or NoAnswerResponse

        The function handles three main scenarios:
        1. RAGPath.SECTION: When additional sections are requested.
        2. RAGPath.FOLLOWUP: When a follow-up query to the LLM is needed.
        3. Error handling: When unexpected scenarios occur.
        """
        self._track_path("PathRAG.process_llm_response")
        check_result = llm_checked_response
        df_definitions = current_user_message["reference_material"]["definitions"]
        df_search_sections = current_user_message["reference_material"]["sections"]
        if check_result["RAGPath"] == self.RAGPath.SECTION.value:
            success, df_updated = self.add_section_to_resource(result = check_result, df_definitions=df_definitions, df_search_sections=df_search_sections)
            if success:
                updated_user_message = current_user_message.copy()
                updated_user_message["reference_material"]["sections"] = df_updated
                updated_response = self.resource_augmented_query(number_of_options = 3, message_history=message_history, current_user_message=updated_user_message)
                result = self.check_response_RAG(llm_message_response=updated_response, df_definitions=df_definitions, df_sections=df_search_sections)
                if "assistant_response" in result:
                    return result
                else: # Even after adding a new section to the RAG data, the LLM still did not produce a valid response.
                    assistant_response = ErrorResponse(
                        classification = ErrorClassification.NOT_FOLLOWING_INSTRUCTIONS,
                    )
                    return {
                        "role": "assistant", 
                        "content": updated_response['content'],
                        "assistant_response": assistant_response
                    }
            else: # the LLM requested a section and it could not be added to the RAG data
                assistant_response = ErrorResponse(
                    classification = ErrorClassification.CALL_FOR_MORE_DOCUMENTS_FAILED,
                )
                return {
                    "role": "assistant", 
                    "content": f"The section requested was: {check_result['extract']} from {check_result['document']} with reference {check_result['section']}",
                    "assistant_response": assistant_response
                }
            
        if check_result["RAGPath"] == self.RAGPath.FOLLOWUP.value:
            updated_messages = self.remove_rag_data_from_message_history(message_history)
            updated_messages.append(current_user_message)
            updated_messages.append({"role": "assistant", "content": llm_checked_response["content"]})
            updated_messages.append({"role": "user", "content": check_result["llm_followup_instruction"]})
            system_message = []
            #truncated_chat = truncate_message_list(system_message, updated_messages, chat_data.chat_parameters.token_limit_when_truncating_message_queue)
            response = self.chat_parameters.get_api_response(system_message, updated_messages)
            llm_response_message = {"role": "assistant", "content": response}
            result = self.check_response_RAG(llm_message_response=llm_response_message, df_definitions=df_definitions, df_sections=df_search_sections)
            if "assistant_response" in result:
                return result
            else: # Even after the second attempt, the LLM still did not produce a valid response OR it required ANOTHER followup call. The invalid response was: {response}
                # TODO: Think about the case were the the LLM now asks for a section. Should this fail?
                assistant_response = ErrorResponse(
                    classification = ErrorClassification.NOT_FOLLOWING_INSTRUCTIONS,
                )
                return {
                    "role": "assistant", 
                    "content": response,
                    "assistant_response": assistant_response
                }

        assistant_response = ErrorResponse(
            classification = ErrorClassification.ERROR,
        )
        return {
            "role": "assistant", 
            "content": "Unable to process the llm response in the RAG path. The RAGPath was: " + check_result["RAGPath"],
            "assistant_response": assistant_response
        }


    def add_section_to_resource(self, result, df_definitions, df_search_sections):
        '''
        Adds a new section to the RAG data and returns a boolean indicating success and an updated DataFrame.

        Parameters:
        - corpus: The corpus object containing document information.
        - result: A dictionary with the following keys:
            {"RAGPath": RAGPath.SECTION.value, "extract": extract_number, "document": document_name, "section": section_reference}
        - df_definitions: DataFrame containing definitions (unused in this function).
        - df_search_sections: DataFrame containing the current search sections.

        Returns:
        - tuple: (success: bool, updated_df: DataFrame)
            - success: True if the section was successfully added, False otherwise.
            - updated_df: An updated DataFrame to be used for RAG, replacing df_search_sections.

        If there is an issue adding the new section, this function will return (False, df_search_sections).
        Otherwise, it will create a new DataFrame with the entry from result["document"] and result["section"] 
        appended to the existing df_search_sections.
        '''
        self._track_path("PathRAG.add_section_to_resource")

        # Confirm that the result dictionary is of the correct form
        if not all(key in result for key in ["RAGPath", "extract", "document", "section"]):
            logger.error(f"add_section_to_resource called with a result dictionary that does not have the required keys: {result}")
            return False, df_search_sections
        if result["RAGPath"] != self.RAGPath.SECTION.value:
            logger.error(f"add_section_to_resource called with a result dictionary that does not have RAGPath.SECTION.value: {result}")
            return False, df_search_sections

        # Step 1) confirm it is requesting something that passes validation
        doc = self.corpus.get_document(result["document"])
        modified_section_to_add = doc.reference_checker.extract_valid_reference(result["section"])
        
        if modified_section_to_add is None:
            logger.log(DEV_LEVEL, f"add_section_to_resource tried to add {result['section']} but the Valid_Index object could not extract a valid reference from this")
            return False, df_search_sections
        
        text_to_add = ""
        try: # passes index verification but there is an error retrieving the section
            text_to_add = doc.get_text(modified_section_to_add)
        except Exception as e:
            logger.log(DEV_LEVEL, f"add_section_to_resource tried to add {modified_section_to_add} but a call to get this regulation resulted in an exception {e}")
            return False, df_search_sections

        if text_to_add == "":
            logger.log(DEV_LEVEL, f"add_section_to_resource tried to add {modified_section_to_add} but the section does not have any text. Is it a valid reference")
            return False, df_search_sections
            
        # prepare the new data
        new_row = [result["document"], result["section"], text_to_add]
        new_sections = pd.DataFrame([new_row], columns = ["document", "section_reference", "regulation_text"])

        sections_to_keep = df_search_sections # keep everything - the context window is long enough
        new_sections = pd.concat([sections_to_keep, new_sections], ignore_index = True)

        return True, new_sections


    def extract_used_references(self, integer_list_of_used_references, df_definitions, df_search_sections):
        ''' 
        An intermediate method that returns a single DataFrame containing only the *used* references (definitions and sections).

        Parameters:
        - corpus: The corpus object containing document name and exact text.
        - integer_list_of_used_references: A list of integers representing the used reference indices.
        - df_definitions: DataFrame containing definitions.
        - df_search_sections: DataFrame containing search sections.

        Returns:
        - DataFrame with columns: ["document_key", "document_name", "section_reference", "is_definition", "text"]
        containing only the used references.
        '''
        self._track_path("PathRAG.extract_used_references")

        columns = ["document_key", "document_name", "section_reference", "is_definition", "text"]

        references_list = []
        cleaned_references = integer_list_of_used_references
        if not cleaned_references:
            return pd.DataFrame([], columns = columns)

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
                text = self.corpus.get_text(document_key, section_reference, add_markdown_decorators=True, add_headings=True, section_only=False)
                references_list.append([document_key, document_name, section_reference, False, text])

        df_used_references = pd.DataFrame(references_list, columns = columns)
        return df_used_references





