import logging
from openai import OpenAI
from enum import Enum
from dataclasses import dataclass
import pandas as pd

from regulations_rag.corpus_index import CorpusIndex
from regulations_rag.embeddings import num_tokens_from_string, num_tokens_from_messages

logger = logging.getLogger(__name__)
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       


# in case you need to overwrite this in an implementing class
def get_caveat_for_no_rag_response():
    caveated_response = "NOTE: The following answer is provided without references and should therefore be treated with caution."
    return caveated_response



class ChatParameters:
    def __init__(self, chat_model, api_key, temperature, max_tokens, token_limit_when_truncating_message_queue):
        self.model = chat_model
        self.temperature = temperature
        self.token_limit_when_truncating_message_queue = token_limit_when_truncating_message_queue
        self.max_tokens = max_tokens # maximum number of returned tokens
        self.openai_client = OpenAI(api_key=api_key)

        self.tested_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]
        untested_models = ["gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo"]
        if self.model not in self.tested_models:
            if self.model not in untested_models:
                raise ValueError("You are attempting to use a model that does not seem to exist")
            else: 
                logger.log(DEV_LEVEL, "You are attempting to use a model that has not been tested")


    def truncate_message_list(self, system_message, message_list):
        """
        Truncates the message list to fit within a specified token limit, ensuring the inclusion of the system message 
        and the most recent messages from the message list. The function guarantees that the returned list always contains 
        the system message and at least the last message from the message list, even if their combined token count exceeds 
        the token limit.

        Parameters:
        - system_message (list): A list containing a single dictionary with the system message.
        - message_list (list): A list of dictionaries representing the user and assistant messages.

        Returns:
        - list: A list of messages truncated to meet the token limit, including the system message and the last messages.
        """
        if not message_list:
            return system_message

        # Initialize the token count with the system message and the last message in the list
        try:
            token_count = sum(num_tokens_from_string(msg["content"]) for msg in system_message + [message_list[-1]])
        
        # KeyError: If "content" key is missing
        # ValueError: If the token counting fails due to invalid input
        # TypeError: If the message is not in the expected format
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error calculating number of tokens for system message: {system_message} and last message: {message_list[-1]}. Error: {e}")
            token_count = 500
        number_of_messages = 1

        # Add messages from the end of the list until the token limit is reached or all messages are included
        while number_of_messages < len(message_list) + 1 and token_count < self.token_limit_when_truncating_message_queue:
            next_message = message_list[-number_of_messages]
            try:
                next_message_token_count = num_tokens_from_string(next_message["content"])
            except:
                logger.error(f"Error calculating number of tokens for message: {next_message}")
                next_message_token_count = 500

            # Check if adding the next message would exceed the token limit
            if token_count + next_message_token_count > self.token_limit_when_truncating_message_queue:
                break
            # else keep going
            token_count += next_message_token_count
            number_of_messages += 1

        number_of_messages_excluding_system = max(1, number_of_messages - 1)
        # Compile the truncated list of messages, always including the system message and the most recent messages
        if system_message == []:
            truncated_messages = message_list[-number_of_messages_excluding_system:]
        else:
            truncated_messages = [system_message[0]] + message_list[-number_of_messages_excluding_system:]

        # strip out anything that is not part of the openai dict
        stripped_messages = []
        for msg in truncated_messages:
            stripped_messages.append({"role": msg["role"], "content": msg["content"]})
        return stripped_messages



    def get_api_response(self, system_message, message_list):
        """
        Fetches a response from the OpenAI API (use unittest.mock module to "hardcode" api responses)

        Parameters:
        - messages (list): The list of messages to send as context to the OpenAI API.

        Returns:
        - str: The response from the OpenAI API.
        """
        # I seem to make the mistake of passing a single system message dict instead of a list of system messages
        if isinstance(system_message, dict):
            system_message = [system_message]
        truncated_messages = self.truncate_message_list(system_message, message_list)

        model_to_use = self.model
        try:
            total_tokens = num_tokens_from_messages(truncated_messages, model_to_use)
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error calculating number of tokens for messages: {truncated_messages} and model: {model_to_use}")
            total_tokens = 500
        
        if total_tokens > 15000:
            return "The is too much information in the prompt so we are unable to answer this question. Please try again or word the question differently"

        # Adjust model based on token count, similar to your original logic
        if (model_to_use in ["gpt-3.5-turbo", "gpt-4"]) and total_tokens > 3500:
            logger.warning("Switching to the gpt-3.5-turbo-16k model due to long prompt.")                
            model_to_use = "gpt-3.5-turbo-16k"
        
        response = self.openai_client.chat.completions.create(
                        model=model_to_use,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=truncated_messages
                    )
        response_text = response.choices[0].message.content
        return response_text


