import logging
import re
from regulations_rag.data_classes import AnswerWithRAGResponse, AnswerWithoutRAGResponse, \
    NoAnswerClassification, NoAnswerResponse, ErrorClassification, ErrorResponse

from regulations_rag.corpus_chat_tools import ChatParameters, get_caveat_for_no_rag_response


logger = logging.getLogger(__name__)
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       




class ChatDataForNoRAGData:
    def __init__(self, corpus_index, chat_parameters, messages, user_message):
        self.corpus_index = corpus_index
        self.corpus = corpus_index.corpus
        self.chat_parameters = chat_parameters
        self.messages = messages.copy() # don't make any changes to the original message list
        self.user_question = user_message["content"]



def query_no_rag_data(chat_data_for_no_rag_data):

    # Returns
    # Unsuccessful path
    #   {
    #       "role": "assistant", 
    #       "content": tap_out_phrase,
    #       "assistant_response": NoAnswerResponse(
    #           classification=NoAnswerClassification.QUESTION_NOT_RELEVANT,
    #       )
    #   } 

    # successful path 
    # {
    #     "role": "assistant", 
    #     "content": llm output with caveate formatted for openai,
    #     "assistant_response": AnswerWithoutRAGResponse(
    #         answer=response,
    #         caveat=caveate,
    #     )
    # } 

    logger.log(DEV_LEVEL, "CorpusChat.query_without_resources() called")
    
    cd = chat_data_for_no_rag_data

    relevant, reason = is_user_content_relevant(chat_data_for_no_rag_data=cd)            
    if relevant:           
        tap_out_phrase = "No Answer"
        system_content = create_system_content_no_rag_data(corpus_description=cd.corpus_index.corpus_description, user_type=cd.corpus_index.user_type, tap_out_phrase=tap_out_phrase)
        local_messages = cd.messages.copy()
        local_messages.append({"role": "user", "content": cd.user_question})

        # Create a temporary message list. We will only add the messages to the chat history if we get well formatted answers
        system_message = [{"role": "system", "content": system_content}]
        response = cd.chat_parameters.get_api_response(system_message, local_messages)

        if response.lower().strip() == tap_out_phrase.lower().strip():
            logger.log(DEV_LEVEL, "CorpusChat.query_without_resources() did not want to answer the user's question")
            # Because we already should have checekd the the question is relevant, we only get here if the system is unable to answer the question with the data in its heistory
            return {
                "role": "assistant", 
                "content": tap_out_phrase,
                "assistant_response": NoAnswerResponse(
                    classification=NoAnswerClassification.UNABLE_TO_ANSWER,
                )
            } 
        caveate = get_caveat_for_no_rag_response()
        caveated_response = caveate + "\n\n" + response
        return {
            "role": "assistant", 
            "content": caveated_response,
            "assistant_response": AnswerWithoutRAGResponse(
                answer=response,
                caveat=caveate,
            )
        }
    else:
        #self.append_content({"role": "user", "content": user_content})       
        # Trim any leading punctuation or whitespace from 'reason'
        reason = reason.lstrip('.,;:!? \t\n\r')
        assistant_response = NoAnswerResponse(NoAnswerClassification.QUESTION_NOT_RELEVANT, reason)
        result = {"role": "assistant", "content": reason, "assistant_response": assistant_response}
        #self.append_content(result)
        return result



# override this message in your implementing class if you need to refine the details about what constitutes a relevant topic
def create_system_content_no_rag_data(corpus_description, user_type, tap_out_phrase):
    '''
    Before calling this function, make sure that the question is relevant to the topic.
    '''
    logger.log(DEV_LEVEL, "create_system_content_no_rag_data() called with default system content for RAG without supporting data")

    #TODO I need another option here or I need to run the relevant / not relvant branch and then check if it can answer after that

    system_message = f"You are answering questions about {corpus_description} for {user_type}. Based on an initial search of the relevant document database, no reference documents could be found to assist in answering the users question. Please review the user question. If you are able to answer the question, please do so. If you are not able to answer the question, respond with the words {tap_out_phrase} without punctuation or any other text."

    return system_message




# override this if the default message is not performing well in the implementing class
def create_system_message_is_user_content_relevant(corpus_description):
    
    return f"You are assisting a user answer technical questions about the {corpus_description}. \nYour task is to determine if their question is about this subject matter or not. It is possible the user may be engaging in pleasantries, small talk, may just be testing the bounds of the system or may be asking about how to circumvent to topic. For now please respond with one of only two responses: Relevant if the question, with the conversation history is about subject matter or how to comply with the regulations; or Not Relevant if the topic of the question is anything else. If the question is Not Relevant, please provide a short explaination why this is the case after the word Not Relevant."


def is_user_content_relevant(chat_data_for_no_rag_data):
    logger.log(DEV_LEVEL, "Executing is_user_content_relevant() i.e. Checking to see if should engage with the user or not")
    cd = chat_data_for_no_rag_data
    system_content = create_system_message_is_user_content_relevant(cd.corpus_index.corpus_description)
    # Create a complete list of messages excluding the system message

    local_messages = cd.messages.copy()
    local_messages.append({"role": "user", "content": cd.user_question})

    system_message=[{'role': 'system', 'content': system_content}]
    initial_response = cd.chat_parameters.get_api_response(system_message=system_message, message_list=local_messages)
    if initial_response.lower().strip() == 'relevant':
        logger.log(DEV_LEVEL, "CorpusChat.is_user_content_relevant() determined that the content was relevant")
        return True, ""
    else: # instead of placing the system in "stuck" mode, just continue as if the question was not relevant
        logger.log(DEV_LEVEL, f"CorpusChat.is_user_content_relevant() determined that the content \'{initial_response}\' was not relevant")
        # remove any instance of the phrase "Not relevant" from the response
        stripped_response = re.sub(r'(?i)not\s+relevant', '', initial_response).strip()
        return False, stripped_response

