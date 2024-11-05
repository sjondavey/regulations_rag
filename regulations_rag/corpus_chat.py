import logging
import re
from enum import Enum
                          
from regulations_rag.rerank import RerankAlgos

from regulations_rag.data_classes import AnswerWithRAGResponse, AnswerWithoutRAGResponse, \
    NoAnswerClassification, NoAnswerResponse, ErrorClassification, ErrorResponse

from regulations_rag.corpus_chat_tools import ChatParameters

from regulations_rag.path_search import PathSearch
from regulations_rag.path_no_rag_data import PathNoRAGData
from regulations_rag.path_rag import PathRAG

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

        # True = only answer if there is supporting information. 
        # False = check if the LLM can answer the question without supporting information. Responses are caveated
        self.strict_rag = True        

        self.progress_callback = None # used in the streamlit ui to display progress
        self.execution_path = [] # used to track the execution path for testing and analysis

        self.path_search = self._create_path_search()
        self.path_no_rag_data = self._create_path_no_rag_data()
        self.path_rag = self._create_path_rag()

    def _track_path(self, step):
        self.execution_path.append(step)

    def _reset_execution_path(self):
        self.execution_path = []
        self.path_search.execution_path = []
        self.path_no_rag_data.execution_path = []
        self.path_rag.execution_path = []

    def _create_path_search(self):
        """Factory method to create PathSearch instance. Can be overridden by child classes that need a different PathSearch object."""
        return PathSearch(
            corpus_index=self.index,
            chat_parameters=self.chat_parameters,
            embedding_parameters=self.embedding_parameters,
            rerank_algo=self.rerank_algo
        )

    def _create_path_no_rag_data(self):
        """Factory method to create PathNoRAGData instance. Can be overridden by child classes that need a different PathNoRAGData object."""
        return PathNoRAGData(corpus_index=self.index, chat_parameters=self.chat_parameters)

    def _create_path_rag(self):
        """Factory method to create PathRAG instance. Can be overridden by child classes that need a different PathRAG object."""
        return PathRAG(corpus_index=self.index, chat_parameters=self.chat_parameters)

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def reset_conversation_history(self):
        logger.log(DEV_LEVEL, f"{self.user_name}: Reset Conversation History")        
        self.messages_intermediate = []
        self.system_state = CorpusChat.State.RAG

    def append_content(self, message):
        '''  
        Placeholder for future functionality
        '''
        # don't duplicate messages in the list
        if self.messages_intermediate and (self.messages_intermediate[-1]["role"] == message["role"] and self.messages_intermediate[-1]["content"] == message["content"]): 
            logger.log(DEV_LEVEL, f"CorpusChat.append_content: Not adding duplicate message. Role: {message.get('role', 'Unknown')}, Content: {message.get('content', 'No content')[:50]}...")
            return

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
        self._track_path("CorpusChat.place_in_stuck_state")
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
        self._track_path("CorpusChat.user_provides_input")       
        
        if user_content is None:
            message = "user_provides_input() function received an empty input. This should not have happened and is an indication there is a bug in the frontend. The system will be placed into a 'stuck' status"
            logger.error(message)
            return self.place_in_stuck_state(ErrorClassification.STUCK)

        if self.system_state == CorpusChat.State.STUCK:
            return self.place_in_stuck_state(ErrorClassification.STUCK)
             

        elif self.system_state == CorpusChat.State.RAG:     
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: {user_content}")      
            
            if self.progress_callback:
                self.progress_callback("Searching for relevant information...")

            workflow_triggered, df_definitions, df_search_sections = self.path_search.similarity_search(user_content) 
            for step in self.path_search.execution_path:
                self._track_path(step)
            
            if workflow_triggered != "none":
                logger.log(ANALYSIS_LEVEL, f"{self.user_name} triggering workflow: {workflow_triggered} ...")
                return self.execute_path_workflow(workflow_triggered, user_content)

            return self.run_base_rag_path(user_content, df_definitions, df_search_sections)
        else:
            message = "CorpusChat.user_provides_input() did not process user input because of an unknown system_state"
            logger.error(message)
            self.place_in_stuck_state(ErrorClassification.STUCK)
            return

    def run_base_rag_path(self, user_content, df_definitions, df_search_sections):
        self._track_path("CorpusChat.run_base_rag_path")
        if (len(df_definitions) + len(df_search_sections) == 0): # unable to find any relevant text in the database
            if len(self.messages_intermediate) < 2:
                logger.log(ANALYSIS_LEVEL, f"{self.user_name}: run_base_rag_path() calling execute_path_no_retrieval_no_conversation_history() ...")
                return self.execute_path_no_retrieval_no_conversation_history(user_content)
            else:
                logger.log(ANALYSIS_LEVEL, f"{self.user_name}: run_base_rag_path() calling execute_path_no_retrieval_with_conversation_history() ...")
                return self.execute_path_no_retrieval_with_conversation_history(user_content)
                    
        else: # Retrieval step returns data            

            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: performing RAG ...")
            if self.progress_callback:
                self.progress_callback("Checking references and generating response...")            
            result = self.path_rag.perform_RAG_path(message_history=self.messages_intermediate, current_user_message={"role": "user", "content": user_content, "reference_material": {"definitions": df_definitions, "sections": df_search_sections}})
            for step in self.path_rag.execution_path:
                self._track_path(step)

            if isinstance(result["assistant_response"], NoAnswerResponse) and self.strict_rag == False:                
                if result["assistant_response"].classification == NoAnswerClassification.QUESTION_NOT_RELEVANT: # if we know it is not relevant, don't retest it
                    self.append_content({"role": "user", "content": user_content, "reference_material": {"definitions": df_definitions, "sections": df_search_sections}})
                    self.append_content(result)   
                    logger.log(ANALYSIS_LEVEL, f"{self.user_name}: RAG returned a NoAnswerClassification.QUESTION_NOT_RELEVANT response.")             
                    return
                else:
                    logger.log(ANALYSIS_LEVEL, f"{self.user_name}: RAG returned a NoAnswerResponse. Trying without the retrieved data ...")
                    if self.progress_callback:
                        self.progress_callback("References not useful. Trying to answer without them...")
                    return self.execute_path_answer_question_with_no_data(user_content)

            else:
                self.append_content({"role": "user", "content": user_content, "reference_material": {"definitions": df_definitions, "sections": df_search_sections}})
                self.append_content(result)   
                logger.log(ANALYSIS_LEVEL, f"{self.user_name}: RAG returned the response.")             
                return


    def execute_path_workflow(self, workflow_triggered, user_content):
        '''  
        Placeholder for future functionality
        '''
        if self.system_state != CorpusChat.State.RAG:
            return
        self._track_path("CorpusChat.execute_path_workflow")
        user_message = {"role": "user", "content": user_content}
        self.append_content(user_message)
        self.place_in_stuck_state(ErrorClassification.WORKFLOW_NOT_IMPLEMENTED)
        logger.error(f"{self.user_name}: CorpusChat.execute_path_workflow() was called but no workflow was implemented for {workflow_triggered}.")
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
            self._track_path("CorpusChat.execute_path_no_retrieval_no_conversation_history. Strict RAG")
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
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: execute_path_no_retrieval_no_conversation_history with strict_rag = True. Returning a NoAnswerResponse")
            return
        else:
            self._track_path("CorpusChat.execute_path_no_retrieval_no_conversation_history. Permissive RAG")
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: execute_path_no_retrieval_no_conversation_history with strict_rag = False. Trying to answer without supporting documents ...")
            if self.progress_callback:
                self.progress_callback("No supporting documents found. Trying to answer without them ...")

            user_message = {"role": "user", "content": user_content}
            response = self.path_no_rag_data.query_no_rag_data(message_history=self.messages_intermediate, current_user_message=user_message)
            for step in self.path_no_rag_data.execution_path:
                self._track_path(step)
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: query_no_rag_data() returned the response.")
            self.append_content(user_message)       
            self.append_content(response)
            return


    def execute_path_answer_question_with_no_data(self, user_content):
        if self.system_state != CorpusChat.State.RAG:
            return
        if self.strict_rag == False:           
            self._track_path("CorpusChat.execute_path_answer_question_with_no_data. Permissive RAG")
            if self.progress_callback:
                self.progress_callback("Trying to answer without supporting documents...")
            user_message = {"role": "user", "content": user_content}
            result = self.path_no_rag_data.query_no_rag_data(message_history=self.messages_intermediate, current_user_message=user_message)
            
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: query_no_rag_data() returned the response.")
            self.append_content({"role": "user", "content": user_content})       
            self.append_content(result)
            return
        else:
            self._track_path("CorpusChat.execute_path_answer_question_with_no_data. Strict RAG")
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: execute_path_answer_question_with_no_data with strict_rag = True. Returning a NoAnswerResponse")
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
        self._track_path("CorpusChat.execute_path_no_retrieval_with_conversation_history")
        logger.log(ANALYSIS_LEVEL, f"{self.user_name}: corpus_chat.execute_path_no_retrieval_with_conversation_history() defaulting to execute_path_no_retrieval_no_conversation_history()")
        return self.execute_path_no_retrieval_no_conversation_history(user_content)


    def execute_path_for_no_relevant_information_in_retrieved_text(self, user_content, df_definitions, df_search_sections):
        if self.system_state != CorpusChat.State.RAG:
            return
        if self.strict_rag:
            self._track_path("CorpusChat.execute_path_for_no_relevant_information_in_retrieved_text. Strict RAG")
            # result {"success": True, "path": "NONE:"}
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: Executing default corpus_chat.execute_path_for_no_relevant_information_in_retrieved_text() i.e. forcing the assistant response to CorpusChat.Errors.NO_RELEVANT_DATA.value")
            user_message = {"role": "user", "content": user_content, "reference_material": {"definitions": df_definitions, "sections": df_search_sections}} 
            self.append_content(user_message)
            assistant_response = ErrorResponse(ErrorClassification.NO_RELEVANT_DATA)
            assistant_message = {"role": "assistant", "content": assistant_response.create_openai_content(), "assistant_response": assistant_response}
            self.append_content(assistant_message)
            self.system_state = CorpusChat.State.RAG
            return
        else:
            self._track_path("CorpusChat.execute_path_for_no_relevant_information_in_retrieved_text. Permissive RAG")
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: execute_path_for_no_relevant_information_in_retrieved_text with strict_rag = False. Trying without the retrieved data ...")
            return self.execute_path_answer_question_with_no_data(user_content)


