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
from regulations_rag.data_classes import AnswerWithRAGResponse, AnswerWithoutRAGResponse, \
    NoAnswerClassification, NoAnswerResponse, ErrorClassification, ErrorResponse

from regulations_rag.corpus_chat_tools import ChatParameters, get_caveat_for_no_rag_response

from regulations_rag.path_search import ChatDataForSearch, similarity_search
from regulations_rag.path_no_rag_data import ChatDataForNoRAGData, query_no_rag_data
from regulations_rag.path_rag import ChatDataForRAG, perform_RAG_path

logger = logging.getLogger(__name__)
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


class CorpusChat():
    '''
    Works by creating a series of messages between the user and the assistant. These are stored in the list self.messages_intermediate
    These messages use an openai format with an additional dictionary for content used in the RAG process:
    [
      {"role": "user", "content": raw_user_prompt_no_references, "reference_material": {"definitions": df_definitions, "sections": df_search_sections}}, 
      {"role": "assistant", "content": assistant_response_with_full_text_references, "assistant_response": A class from data_classes.py}
      {"role": "user", "content": raw_user_prompt_no_references, "reference_material": {"definitions": df_definitions, "sections": df_search_sections}}, 
      {"role": "assistant", "content": assistant_response_with_full_text_references, "assistant_response": A class from data_classes.py}
    ]
    The assistant_response class stores components of the answer in case they need to be formatted differently to 'content' or error messages
    that detail why the RAG process failed or was not attempted.


    A system message:
      {"role": "system", "content": system_prompt}, 
    is generated for each path but is never stored in the messages_intermediate list. When it is used, it is added to a copy of the messages_intermediate list 
    so that, at any one time, there is only one system message in a list that will get sent to OpenAI.
    '''

    class State(Enum):
        RAG = "rag"
        NO_DATA = "no relevant documents"
        NEEDS_DATA = "requires additional documents"
        STUCK = "stuck"


    def __init__(self, 
                 embedding_parameters,
                 chat_parameters,
                 corpus_index,
                 rerank_algo = RerankAlgos.NONE,   
                 user_name_for_logging = 'test_user'): 

        self.user_name = user_name_for_logging
        self.openai_client = chat_parameters.openai_client
        self.embedding_parameters = embedding_parameters
        self.chat_parameters = chat_parameters

        self.index = corpus_index
        self.corpus = self.index.corpus
        self.primary_document = self.corpus.get_primary_document()
        self.has_primary_document = self.primary_document != ""

        self.rerank_algo = rerank_algo
        self.reset_conversation_history()

        # this is used when creating assistant responses which may need to include information about the app layout
        self.assume_streamlit_ui = False
        # only answer if there is supporting information. If False, a general call will be placed to the LLM. The response, 
        # if it chooses to respond, will be caveated
        self.strict_rag = True

    def reset_conversation_history(self):
        logger.log(DEV_LEVEL, f"{self.user_name}: Reset Conversation History")        
        self.messages_intermediate = []
        self.system_state = CorpusChat.State.RAG

    '''
    The idea is to use an 'OpenAI like' dictionary of message content along with a dictionary of additional content that can be used for special formatting in different contexts, for example references, followup questions, etc
    '''
    def append_content(self, message):

        # don't duplicate messages in the list
        if self.messages_intermediate and (self.messages_intermediate[-1]["role"] == message["role"] and self.messages_intermediate[-1]["content"] == message["content"]): 
            logger.log(DEV_LEVEL, f"CorpusChat.append_content: Not adding duplicate message. Role: {message.get('role', 'Unknown')}, Content: {message.get('content', 'No content')[:50]}...")
            return

        # Making paths in case I need them later
        if message["role"] == "system":
            self.messages_intermediate.append(message)

        elif message["role"] == "user":
            self.messages_intermediate.append(message)

        elif message["role"] == "assistant":
            self.messages_intermediate.append(message)

        else:
            logger.error(f"CorpusChat.append_content: Tried to add a message for the role {message['role']} which is not a valid role")

        return


    def place_in_stuck_state(self, error_classification = ErrorClassification.ERROR):
        # once the system is in a stuck state, it will remain in that state until it is reset by a call to reset_conversation_history()
        logger.error(f"CorpusChat.place_in_stuck_state() called with an optional message: {error_classification.value}")

        assistant_response = ErrorResponse(error_classification)

        assistant_message = {
            "role": "assistant", 
            "content": assistant_response.create_openai_content(),
            "assistant_response": assistant_response
        }
        self.append_content(assistant_message)
        self.system_state = CorpusChat.State.STUCK
        return

    def user_provides_input(self, user_content):
        
        if user_content is None:
            message = "user_provides_input() function received an empty input. This should not have happened and is an indication there is a bug in the frontend. The system will be placed into a 'stuck' status"
            logger.error(message)
            return self.place_in_stuck_state(ErrorClassification.STUCK)

        if self.system_state == CorpusChat.State.STUCK:
            return self.place_in_stuck_state(ErrorClassification.STUCK)
             

        elif self.system_state == CorpusChat.State.RAG:            
            logger.log(DEV_LEVEL, f"CorpusChat.user_provides_input() processing {self.user_name}'s question: {user_content}")      
            search_data = ChatDataForSearch(self.index, self.chat_parameters, self.embedding_parameters, self.rerank_algo)
            workflow_triggered, df_definitions, df_search_sections = similarity_search(search_data, user_content) 
            
            if workflow_triggered != "none":
                logger.log(DEV_LEVEL, f"Workflow triggered: {workflow_triggered}")
                return self.execute_path_workflow(workflow_triggered, user_content)

            if (len(df_definitions) + len(df_search_sections) == 0): # unable to find any relevant text in the database
                if len(self.messages_intermediate) < 2:
                    return self.execute_path_no_retrieval_no_conversation_history(user_content)
                else:
                    return self.execute_path_no_retrieval_with_conversation_history(user_content)
                    
            else: # Retrieval step returns data
                chat_data = ChatDataForRAG(corpus_index = self.index, 
                                             chat_parameters = self.chat_parameters, 
                                             messages = self.messages_intermediate, 
                                             user_message = {"role": "user", "content": user_content, "reference_material": {"definitions": df_definitions, "sections": df_search_sections}})
                result = perform_RAG_path(chat_data)

                if isinstance(result["assistant_response"], NoAnswerResponse) and self.strict_rag == False:
                    if result["assistant_response"].classification == NoAnswerClassification.QUESTION_NOT_RELEVANT: # if we no it is not relevant, don't retest it
                        self.append_content({"role": "user", "content": user_content, "reference_material": {"definitions": df_definitions, "sections": df_search_sections}})
                        self.append_content(result)                
                        return
                    else:
                        return self.execute_path_answer_question_with_no_data(user_content)


                else:
                    self.append_content({"role": "user", "content": user_content, "reference_material": {"definitions": df_definitions, "sections": df_search_sections}})
                    self.append_content(result)                
                    return
        else:
            message = "CorpusChat.user_provides_input() did not process user input because of an unknown system_state"
            logger.error(message)
            self.place_in_stuck_state(ErrorClassification.STUCK)
            return


    """ 
    Override this method if you have created a table of workflows for your chat bot
    """
    def execute_path_workflow(self, workflow_triggered, user_content):
        if self.system_state != CorpusChat.State.RAG:
            return
        user_message = {"role": "user", "content": user_content}
        self.append_content(user_message)
        self.place_in_stuck_state(ErrorClassification.WORKFLOW_NOT_IMPLEMENTED)
        return

    """ 
    Override this method if you want to execute something specific when the user question does not result in any hits in the database
    and there is no conversation history that may otherwise allow the system to infer some context and phrase a better question.

    The default behaviour is not to go to the LLM but simply return the hardcoded CorpusChat.Errors.NO_DATA.value error message
    """ 
    def execute_path_no_retrieval_no_conversation_history(self, user_content):
        if self.system_state != CorpusChat.State.RAG:
            return
        if self.strict_rag:
            logger.log(DEV_LEVEL, "Executing default CorpusChat.execute_path_no_retrieval_no_conversation_history() i.e. bypassing the LLM and forcing the assistant to respond with CorpusChat.Errors.NO_DATA.value")
            self.system_state = CorpusChat.State.RAG         
            self.append_content({"role": "user", "content": user_content})       
            assistant_response = NoAnswerResponse(NoAnswerClassification.NO_DATA)
            assistant_message = {
                "role": "assistant", 
                "content": assistant_response.create_openai_content(),
                "assistant_response": assistant_response
                }
            self.append_content(assistant_message)
            return
        else:
            logger.log(DEV_LEVEL, "Executing CorpusChat.execute_path_no_retrieval_no_conversation_history() for permissive question answering i.e. asking the LLM to answer the question without supporting documents")
            chat_data = ChatDataForNoRAGData(corpus_index = self.index, 
                                             chat_parameters = self.chat_parameters, 
                                             messages = self.messages_intermediate, 
                                             user_message = {"role": "user", "content": user_content})
            response = query_no_rag_data(chat_data)
            self.append_content({"role": "user", "content": user_content})       
            self.append_content(response)
            return


    def execute_path_answer_question_with_no_data(self, user_content):
        if self.system_state != CorpusChat.State.RAG:
            return

        if self.strict_rag == False:           
            chat_data = ChatDataForNoRAGData(corpus_index = self.index, 
                                                chat_parameters = self.chat_parameters, 
                                                messages = self.messages_intermediate, 
                                                user_message = {"role": "user", "content": user_content})
            result = query_no_rag_data(chat_data)
            self.append_content({"role": "user", "content": user_content})       
            self.append_content(result)
            return
        else:
            self.append_content({"role": "user", "content": user_content})       
            assistant_response = NoAnswerResponse(NoAnswerClassification.NO_DATA)
            result = {"role": "assistant", "content": NoAnswerClassification.NO_DATA.value, "assistant_response": assistant_response}
            self.append_content(result)
            return

    """ 
    Override this method if you want to execute something specific when the user question does not result in any hits in the database
    BUT there IS conversation history that can be used to phrase a better question
    """ 
    def execute_path_no_retrieval_with_conversation_history(self, user_content):
        if self.system_state != CorpusChat.State.RAG:
            return

        logger.log(DEV_LEVEL, "Executing default corpus_chat.execute_path_no_retrieval_with_conversation_history()")
        return self.execute_path_no_retrieval_no_conversation_history(user_content)


    def execute_path_for_no_relevant_information_in_retrieved_text(self, user_content, df_definitions, df_search_sections):
        if self.system_state != CorpusChat.State.RAG:
            return

        # result {"success": True, "path": "NONE:"}
        logger.log(DEV_LEVEL, "Executing default corpus_chat.execute_path_for_no_relevant_information_in_retrieved_text() i.e. forcing the assistant response to CorpusChat.Errors.NO_RELEVANT_DATA.value")

        user_message = {"role": "user", "content": user_content, "reference_material": {"definitions": df_definitions, "sections": df_search_sections}} 
        self.append_content(user_message)
        assistant_response = ErrorResponse(ErrorClassification.NO_RELEVANT_DATA)
        assistant_message = {"role": "assistant", "content": assistant_response.create_openai_content(), "assistant_response": assistant_response}
        self.append_content(assistant_message)
        self.system_state = CorpusChat.State.RAG
        return


