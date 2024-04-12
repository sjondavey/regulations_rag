import logging
import pandas as pd
from openai import OpenAI
import fnmatch
import regex # fuzzy lookup of references in a section of text
import copy
from enum import Enum



# import src.data
# importlib.reload(src.data)
from regulations_rag.regulation_index import RegulationIndex, EmbeddingParameters

from regulations_rag.string_tools import match_strings_to_reference_list

from regulations_rag.regulation_reader import RegulationReader
                           
from regulations_rag.embeddings import get_ada_embedding, \
                           get_closest_nodes, \
                           num_tokens_from_string,  \
                           num_tokens_from_messages

from regulations_rag.rerank import RerankAlgos, rerank


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

        self.tested_models = ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]
        untested_models = ["gpt-4-1106-preview", "gpt-4-0125-preview"]
        if self.model not in self.tested_models:
            if self.model not in untested_models:
                raise ValueError("You are attempting to use a model that does not seem to exist")
            else: 
                logger.info("You are attempting to use a model that has not been tested")


class RegulationChat():
    class State(Enum):
        RAG = "rag"
        NO_DATA = "no_relevant_embeddings"
        NEEDS_DATA = "requires_additional_sections"
        STUCK = "stuck"

    class Prefix(Enum):
        ANSWER = "ANSWER:"
        SECTION = "SECTION:"
        NONE = "NONE:"
        FAIL = "FAIL:"

    class Errors(Enum):
        NO_DATA = "I was unable to find any relevant documentation to assist in answering the question. Can you try rephrasing the question?"
        NO_RELEVANT_DATA = "The documentation I have been provided does not help me answer the question. Please rephrase it and let's try again?"
        STUCK = "Unfortunately the system is in an unrecoverable state. Please clear the chat history and retry your query"
        UNKNOWN_STATE = "The system is in an unknown state and cannot proceed. Please clear the chat history and retry your query"
        NOT_FOLLOWING_INSTRUCTIONS = "The call to the LLM resulted in a response that did not fit parameters, even after retrying it. Please clear the chat history and retry your query."

    def __init__(self, 
                 openai_client, 
                 embedding_parameters,
                 chat_parameters,
                 regulation_reader,
                 regulation_index,
                 rerank_algo = RerankAlgos.NONE,   
                 user_name_for_logging = 'test_user'): 

        self.user_name = user_name_for_logging
        self.openai_client = openai_client
        self.embedding_parameters = embedding_parameters
        self.chat_parameters = chat_parameters
        self.reader = regulation_reader
        self.index = regulation_index
        self.rerank_algo = rerank_algo
        self.reset_conversation_history()

    def reset_conversation_history(self):
        self.messages = []
        self.messages_without_rag = []
        self.system_state = RegulationChat.State.RAG


    def append_content(self, role, content):
        if role not in ["user", "assistant", "system"]:
            logger.error(f"Tried to add a message for the role {role} which is not a valid role")
            return

        content_without_rag = content

        if role == "user":            
            if content.startswith("Question:"):
                content_without_rag = self._extract_question_from_rag_data(content)

        # nothing special for assistant or system messages

        if self.messages and (self.messages[-1]["role"] == role and self.messages[-1]["content"] == content): # don't duplicate messages in the list
            return

        self.messages.append({"role": role, "content": content})
        logger.log(ANALYSIS_LEVEL, f"{role} to {self.user_name}: {content}")        
        self.messages_without_rag.append({"role": role, "content": content_without_rag})        


    def chat_completion(self, user_content, testing = False, manual_responses_for_testing = []):
        self.user_provides_input(user_content, testing, manual_responses_for_testing)                        
        return self.messages[-1]["content"]

    # Note: To test the workflow I need some way to control the openai API responses. I have chosen to do this with the two parameters
    #       testing: a flag. If false the function will run calling the openai api for responses. If false the function will 
    #                        select the response from the list of responses manual_responses_for_testing
    #       manual_responses_for_testing: A list of text. If testing == True, these values will be used as if they were the 
    #                                     the response from the API. This function can make multiple calls to the API so the i-th
    #                                     row in the list corresponds to the i-th call of the API
    #  
    def user_provides_input(self, user_content, testing = False, manual_responses_for_testing = []):
        
        if user_content is None:
            logger.error(f"{self.user_name}: user_provides_input() function received an empty input. This should not have happened and is an indication there is a bug in the frontend. The system will be placed into a 'stuck' status")
            self.append_content("assistant", RegulationChat.Errors.UNKNOWN_STATE.value)
            self.system_state = RegulationChat.State.STUCK
            return

        workflow_triggered, df_definitions, df_search_sections = self.similarity_search(user_content)
        if workflow_triggered == "documentation":
            user_content = self.enrich_user_request_for_documentation(user_content, self.messages_without_rag)
            workflow_triggered, df_definitions, df_search_sections = self.similarity_search(user_content)

        self.append_content("user", user_content)

        if self.system_state == RegulationChat.State.STUCK:            
            self.append_content("assistant", RegulationChat.Errors.STUCK.value)
            return 

        elif self.system_state == RegulationChat.State.RAG:            
            if len(self.messages) < 2 and (len(df_definitions) + len(df_search_sections) == 0):
                logger.log(DEV_LEVEL, "Note: Unable to find any definitions or text related to this query")
                self.system_state = RegulationChat.State.RAG                
                self.append_content("assistant", RegulationChat.Errors.NO_DATA.value)
                return
            else:
                flag, response = self.resource_augmented_query(df_definitions = df_definitions, df_search_sections = df_search_sections, number_of_options=3,
                                                               testing = testing, manual_responses_for_testing = manual_responses_for_testing)
                if flag == RegulationChat.Prefix.ANSWER:
                    sections_in_rag = df_search_sections["section_reference"].tolist()
                    actual_references_used_in_rag, reformatted_response = self.reformat_assistant_answer(response.strip(), sections_in_rag)
                    # Change the user message so that it only contains the actual_references_used_in_rag references
                    relevant_sections = pd.DataFrame(actual_references_used_in_rag, columns = ["section_reference"])
                    # relevant_sections["regulation_text"] = relevant_sections["section_reference"].apply(self.get_regulation_detail)
                    self.messages[-1]["content"] = self._add_rag_data_to_question(user_content, df_definitions, relevant_sections)

                    # once the user message is updated, we can update the response
                    self.append_content("assistant", reformatted_response)
                    self.system_state = RegulationChat.State.RAG 
                    return 
                elif flag == RegulationChat.Prefix.SECTION:
                    logger.info(f"System requested for more info:\n{response}")
                    # Asking for an invalid section or a section that is already in the RAG
                    force = False
                    if not self.reader.reference_checker.is_valid(response):
                        logger.info(f"But \n{response} is not a valid reference, so now forcing the system to answer or opt out")
                        force = True
                    if self.reader.reference_checker.is_reference_or_parents_in_list(response, df_search_sections["section_reference"].tolist()):
                        logger.info(f"But \n{response} is already in the RAG data, so now forcing the system to answer or opt out")
                        force = True

                    if force:
                        flag, response = self.resource_augmented_query(df_definitions, df_search_sections, number_of_options=2,
                                                                       testing = testing, manual_responses_for_testing = manual_responses_for_testing[1:])

                    else:
                        df_search_sections = self.add_section_to_resource(response, df_search_sections)
                        if self.system_state == RegulationChat.State.STUCK: # failed to add the sections
                            # TODO: Do you want to ask the user for help?
                            logger.info("Note: Request to add resources failed")
                            self.append_content("assistant", RegulationChat.Errors.STUCK.value)
                            return

                        # reset user_content i.e. remove the previous RAG data
                        self.messages[-1]["content"] = user_content
                        # ... and try again with new resources
                        flag, response = self.resource_augmented_query(df_definitions = df_definitions, df_search_sections = df_search_sections, number_of_options=3,
                                                                    testing = testing, manual_responses_for_testing = manual_responses_for_testing[1:])
                    
                    if flag == RegulationChat.Prefix.ANSWER:
                        logger.info("Note: Question answered with the additional information")
                        self.append_content("assistant", response.strip())
                        self.system_state = RegulationChat.State.RAG
                        return
                    
                    else: 
                        logger.info("Note: Even with the additional information, they system was unable to answer the question. Placing the system in 'stuck' mode")
                        logger.info(f"The response from the query with additional resources was: \n{response}")
                        msg = "A call for additional sections did not result in sufficient information to answer the question. The system is now stuck. Please clear the chat history and retry your query"
                        self.append_content("assistant", msg)
                        self.system_state = RegulationChat.State.STUCK
                        return

                elif flag == RegulationChat.Prefix.NONE:
                    logger.info("Note: The LLM was not able to find anything relevant in the supplied sections")
                    self.append_content("assistant", RegulationChat.Errors.NO_RELEVANT_DATA.value)
                    self.system_state = RegulationChat.State.RAG
                    return

                else:
                    logger.error("Note: RAG returned an unexpected response")
                    self.append_content("assistant", RegulationChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value)
                    self.system_state = RegulationChat.State.STUCK # We are at a dead end.
                    return
        else:
            logger.error("The system is in an unknown state")
            self.append_content("assistant", RegulationChat.Errors.UNKNOWN_STATE.value)
            return

    def add_section_to_resource(self, section_to_add, df_search_sections):
        # Step 1) confirm it is requesting something that passes validation
        modified_section_to_add = self.reader.reference_checker.extract_valid_reference(section_to_add)
        
        if modified_section_to_add is None:
            logger.info(f"Tried to add {section_to_add} the Valid_Index object could not extract a valid reference from this")
            self.system_state = RegulationChat.State.STUCK 
            return df_search_sections
        
        try: # passes index verification but there is an error retrieving the section
            self.get_regulation_detail(modified_section_to_add)

        except Exception as e:
            logger.info(f"Tried to add {modified_section_to_add} but a call to get this regulation resulted in an exception {e}")
            self.system_state = RegulationChat.State.STUCK 
            return df_search_sections
          
        referring_sections = self._find_reference_that_calls_for(modified_section_to_add, df_search_sections)
        
        if len(referring_sections) > 0: # Delete the other sections, keep the referring section and the new data
            referring_sections.append(modified_section_to_add)
            # now create the new RAG df_search_sections
            manual_data = []
            for i in range(len(referring_sections)):
                section = referring_sections[i]
                count = 1
                raw_text = self.get_regulation_detail(section)
                token_count = num_tokens_from_string(raw_text)
                manual_data.append([section, 1.0, count, raw_text, token_count])            
            df_manual_data = pd.DataFrame(manual_data, columns = ["section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
            return df_manual_data
            
        else: # Just add the new data and hope the total context is not too long
            section = modified_section_to_add
            count = 1
            raw_text = self.get_regulation_detail(section)
            token_count = num_tokens_from_string(raw_text)
            df_to_add = pd.DataFrame([[section, 1.0, count, raw_text, token_count]], columns = ["section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
            df_search_sections = pd.concat([df_search_sections, df_to_add]).reset_index(drop=True)
            #df_search_sections.loc[len(df_search_sections.index)] = [section, 1.0, count, raw_text, token_count]
            return df_search_sections


    def _find_reference_that_calls_for(self, valid_section_index, df_search_sections):
        """
        Identifies references within the provided DataFrame that call for a specific valid section index.

        This function searches through a DataFrame of search sections for references that match a given
        valid section index, using fuzzy matching on the raw text of each section.

        Parameters:
        - valid_section_index (DocumentIndex): The index of the valid section being searched for.
        - df_search_sections (DataFrame): A DataFrame containing the search sections, each with a 'reference' and 'regulation_text'.

        Returns:
        - list: A list of references that match the given valid section index. If no matches are found, returns an empty list.
        """
        referring_section = []
        for _, row in df_search_sections.iterrows():
            match = self._find_fuzzy_reference(row["regulation_text"], valid_section_index)
            if match:
                referring_section.append(row["section_reference"])

        if not referring_section:
            logger.error(f"The LLM asked for an additional valid reference {valid_section_index} but we could not determine which section referred to it")

        return referring_section       

    # TODO: Think about replacing this with just the function DocumentIndex.extract_valid_reference
    # I think that if we delete the first line of raw_text then we should be able to run DocumentIndex.extract_valid_reference
    # on the remaining text to see if the valid_section_index was in the raw_section. It will get rid of some code and cater
    # also for cases with the reference in the raw_section is not correctly formatted rather than allowing for some random 
    # number of mismatches as I do here
    def _find_fuzzy_reference(self, raw_section, valid_section_index):
        # Enabling fuzzy matching with 2 insertions/deletions/substitutions to cater for stray spaces that may creep into the text
        pattern = r'(%s){e<=2}' % regex.escape(valid_section_index)
        match = regex.search(pattern, raw_section)
        if match:
            return match.group()
        else:
            return None

    def _create_system_message(self, number_of_options = 3):
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
        #short_pattern = r"[A-Z]\.\d{0,2}\([A-Z]\)\(\b(?:i|ii|iii|iv|v|vi)\b\)\([a-z]\)\([a-z]{2}\)\(\d+\)"
        short_pattern = self.index.reference_checker.text_version
        if number_of_options == 2:
            return f"You are answering questions for {self.index.user_type} based only on the sections from the {self.index.regulation_name} that are provided. Please use the manual's index pattern when referring to sections: {short_pattern}. You have two options:\n\
1) Answer the question. Preface an answer with the tag '{RegulationChat.Prefix.ANSWER.value}'. End the answer with 'Reference: ' and a comma separated list of the section you used to answer the question if you used any.\n\
2) State '{RegulationChat.Prefix.NONE.value}' and nothing else if you cannot answer the question with the resources provided\n\n\""

        return f"You are answering questions for {self.index.user_type} based only on the sections from the {self.index.regulation_name} that are provided. Please use the manual's index pattern when referring to sections: {short_pattern}. You have three options:\n\
1) Answer the question. Preface an answer with the tag '{RegulationChat.Prefix.ANSWER.value}'. End the answer with 'Reference: ' and a comma separated list of the section you used to answer the question if you used any.\n\
2) Request additional documentation. If, in the body of the sections provided, there is a reference to another section of the Manual that is directly relevant and not already provided, respond with the word '{RegulationChat.Prefix.SECTION.value}' followed by the full section reference.\n\
3) State '{RegulationChat.Prefix.NONE.value}' and nothing else in all other cases\n\n\""


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
        if not df_definitions.empty:
            definitions_content = "Definitions from the Manual\n" + "\n".join(df_definitions['definition']) + "\n"
            user_content += definitions_content


        if not df_search_sections.empty:
            df_search_sections["regulation_text"] = df_search_sections["section_reference"].apply(
                lambda x: self.reader.get_regulation_detail(x)
            )        
            #df_search_sections["regulation_text"] = self.reader.get_regulation_detail(df_search_sections["section_reference"])        

            sections_content = "Sections from the Manual\n" + "\n".join(df_search_sections['regulation_text']) + "\n"
            user_content += sections_content

        return user_content


    def _extract_question_from_rag_data(self, decorated_question):
        return decorated_question.split("\n")[0][len("Question: "):]


    def _get_api_response(self, messages, testing=False, manual_responses_for_testing=[], response_index=0):
        """
        Fetches a response from the OpenAI API or uses a canned response based on the testing flag.

        Parameters:
        - messages (list): The list of messages to send as context to the OpenAI API.
        - testing (bool): Flag to determine whether to use the OpenAI API or canned responses.
        - manual_responses_for_testing (list of str): A list of canned responses to use if testing is True.
        - response_index (int): The index of the response to use from manual_responses_for_testing.

        Returns:
        - str: The response from the OpenAI API or the selected canned response.

        NOTE: In some tests I want a manual response to the first call so that I can test the rest of the code. To do that
              I set testing = True but only pass one message in manual_responses_for_testing 
        """
        if testing and response_index < len(manual_responses_for_testing):
            # NOTE: In some tests I force the first response but want to test the API call on the second attempt 
            #       so I set testing = True but only pass one message in manual_responses_for_testing 
            response_text = manual_responses_for_testing[response_index]
        else:
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
        return self._check_response_prefix(response_text)

    def _check_response_prefix(self, response):
        """
        Checks if the response starts with any of the expected prefixes in RegulationChat.Prefix and trims the prefix from the response.

        Parameters:
        - response (str): The response text to check for prefixes.
        - prefixes (list): A list of expected prefixes (RegulationChat.Prefix enum values).

        Returns:
        - tuple: (matched prefix or None, trimmed response or original response if no prefix matched, success flag)
        """
        for prefix in RegulationChat.Prefix:
            if response.startswith(prefix.value):
                return prefix, response[len(prefix.value):].strip(), True  # Trim prefix and return
        return None, response, False  # No prefix matched
            

    def resource_augmented_query(self, df_definitions, df_search_sections, number_of_options = 3,
                                 testing = False, manual_responses_for_testing = []):
        """
        Executes a Resource-Augmented Query (RAQ) using user input, definitions, and search sections. This function
        modifies the last user message to include Resource-Augmented Generation (RAG) data, then queries the OpenAI API
        or uses predefined responses based on the testing mode.

        Parameters:
        - df_definitions (DataFrame): DataFrame containing relevant definitions to be included in the RAG.
        - df_search_sections (DataFrame): DataFrame containing relevant sections to be included in the RAG.
        - testing (bool): If True, uses manual_responses_for_testing instead of calling the OpenAI API. 
                        If False, calls the OpenAI API for responses.
        - manual_responses_for_testing (list of str): Predefined responses to use in testing mode. The i-th item
                                                    is used as the i-th API call response.

        Returns:
        - tuple: A prefix indicating the type of response obtained (from RegulationChat.Prefix) and the trimmed
                response text or an error message if no acceptable answer is provided by the LLM.

        Ensures that the last entry in self.messages is from the user, and modifies it to include RAG data before
        querying for a response. If the initial response does not follow the expected format, a follow-up attempt
        is made to correct the response.
        """

        if len(self.messages) == 0 or self.messages[-1]["role"] != "user": 
            logger.error("resource_augmented_query method called but the last message on the stack was not from the user")
            self.system_state = RegulationChat.State.STUCK
            return RegulationChat.State.STUCK, RegulationChat.State.STUCK
        if self.system_state != RegulationChat.State.RAG:
            logger.error("resource_augmented_query method called but the the system is not in rag state")
            self.system_state = RegulationChat.State.STUCK # stuck
            return RegulationChat.State.STUCK, RegulationChat.State.STUCK
        

        if len(self.messages) > 1 or len(df_definitions) + len(df_search_sections) > 0: # should always be the case as we check this in the control loop
            logger.log(DEV_LEVEL, "#################   RAG Prompts   #################")

            system_content = self._create_system_message(number_of_options)
            logger.log(DEV_LEVEL, "System Prompt:\n" + system_content)

            # Replace the user question with the RAG version of it
            user_question = self.messages[-1]["content"]
            if user_question.startswith("Question:"):
                user_question = self._extract_question_from_rag_data(user_question)
            self.messages[-1]["content"] = self._add_rag_data_to_question(user_question, df_definitions, df_search_sections)
            logger.log(DEV_LEVEL, "User Prompt with RAG:\n" + self.messages[-1]["content"])


            # Create a temporary message list. We will only add the messages to the chat history if we get well formatted answers
            system_message = [{"role": "system", "content": system_content}]
            truncated_chat = self._truncate_message_list(system_message, self.messages, 2000)

            prefix, trimmed_response, success = self._get_api_response(messages = truncated_chat, testing=testing, manual_responses_for_testing=manual_responses_for_testing, response_index = 0)
            if success:
                return prefix, trimmed_response

            # The model did not perform as instructed so we not ask it to check its work
            logger.info("Initial chat API response did not result in a response with the correct format. It returned")
            logger.info(f"Prefix: {prefix}\nResponse: {trimmed_response}")            
            logger.info("We will not ask the model to check its original run and comply with the instructions")

            despondent_user_content = f"Please check your answer and make sure you preface your response using only one of the three permissible words, {RegulationChat.Prefix.ANSWER.value}, {RegulationChat.Prefix.SECTION.value} or {RegulationChat.Prefix.NONE.value}"
            despondent_user_messages = truncated_chat + [
                                        {"role": "assistant", "content": trimmed_response},
                                        {"role": "user", "content": despondent_user_content}]
                                        
            prefix, trimmed_response, success = self._get_api_response(messages = despondent_user_messages, testing=testing, manual_responses_for_testing=manual_responses_for_testing, response_index = 1)
            
            if success:
                return prefix, trimmed_response

        return RegulationChat.Prefix.FAIL, "The LLM was not able to return an acceptable answer. "
        

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
            relevant_workflows = self.index.get_relevant_workflow(user_content_embedding = question_embedding, threshold = self.embedding_parameters.threshold)
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

        relevant_definitions = self.index.get_relevant_definitions(user_content_embedding = question_embedding, threshold = self.embedding_parameters.threshold)
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
        


    def get_regulation_detail(self, node_str):
        valid_reference = self.reader.reference_checker.extract_valid_reference(node_str)
        if not valid_reference:
            return "The reference did not conform to this documents standard"
        else:
            if valid_reference != node_str:
                logger.info(f"The string {node_string} is not a valid index. Defaulting to {valid_reference}")
            return self.reader.get_regulation_detail(valid_reference)

    def _truncate_message_list(self, system_message, message_list, token_limit=2000):
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


    def reformat_assistant_answer(self, raw_response, sections_in_rag):
        """
        Reformats the "Reference" section from the LLM's response to ensure consistency with the correctly formatted
        sections from the data used to populate the RAG content. It identifies each reference in the LLM's response,
        replaces it with the closest match from a list of provided references, and removes duplicates.

        Parameters:
        - raw_response (str): The raw response from the LLM, potentially containing references.
        - sections_in_rag (list): A list of correctly formatted reference sections.

        Returns:
        - list: A list of the references from sections_in_rag that were referenced in the answer
        - str: The reformatted response with consistent reference formatting and no duplicate references.
        """
        # Early return if no references found. This can happen if the answer is in the definitions for example
        if not sections_in_rag:
            return raw_response

        # Extract and clean references from the raw response
        references = raw_response.split("Reference:")[1].split(",") if "Reference:" in raw_response else []
        cleaned_references = [ref.strip() for ref in references if ref.strip()]

        # Early return if no references found
        if not cleaned_references:
            return [], raw_response.split("Reference:")[0].strip()

        unique_references = match_strings_to_reference_list(cleaned_references, sections_in_rag)

        # Get headings for matched references
        reference_headings = [self.reader.get_regulation_heading(ref) for ref in unique_references]

        # Reconstruct the answer with reformatted references
        answer_base = raw_response.split("Reference:")[0].strip()
        formatted_references = "  \nReference:" + "".join(f"  \n{ref}: {heading}" for ref, heading in zip(unique_references, reference_headings))

        return unique_references, answer_base + formatted_references


    def enrich_user_request_for_documentation(self, user_content, messages_without_rag, model_to_use="gpt-3.5-turbo"):
        """
        Enhances a user's request for documentation based on the conversation history. It constructs a standalone request
        for documentation, utilizing the most recent conversation history to formulate a question that specifies what documentation
        is required.

        Parameters:
        - user_content (str): The latest user content to be used for generating documentation requests.
        - messages_without_rag (list): A list of message dictionaries that exclude RAG content, to be used as conversation history.
        - model_to_use (str, optional): Specifies the AI model to use for generating the documentation request. Defaults to "gpt-3.5-turbo".

        Returns:
        - str: The enhanced documentation request generated by the model.
        """
        logger.info("Enriching user request for documentation based on conversation history.")

        # Preparing the initial system message to guide the model in request generation
        system_content = "You are assisting a user to construct a stand alone request for documentation from a conversation. \
At the end of the conversation they have asked a question about the documentation they require. Your job is to review the conversation history and to respond with this question \
'What documentation is required as evidence for ...' where you need to replace the ellipses with a short description of the most recent conversation history. Try to keep the question short and general."
        
        # Create a complete list of messages excluding the system message
        messages_copy = copy.deepcopy(messages_without_rag)
        messages_copy.append({'role': 'user', 'content': user_content})
        
        # Truncate messages list to meet a specific token limit and ensure there is space for the system message
        system_message={'role': 'system', 'content': system_content}
        truncated_messages = self._truncate_message_list([system_message], messages_copy, token_limit=3500)
        # NOTE, the truncated_messages will now contain the system message

        # Generate the enhanced documentation request using the specified AI model
        response = self.openai_client.chat.completions.create(
            model=model_to_use,
            temperature=1.0,
            max_tokens=200,
            messages=truncated_messages
        )

        # Extract the initial response and log information for review
        initial_response = response.choices[0].message.content
        logger.info(f"{self.user_name} original question: {user_content}")
        logger.info(f"System enhanced question: {initial_response}")

        # Check if the response starts as expected; log a warning if not
        if not initial_response.startswith('What documentation is required as evidence for'):
            logger.warning("The function did not enrich the user request for documentation as expected, which may create problems.")

        return initial_response