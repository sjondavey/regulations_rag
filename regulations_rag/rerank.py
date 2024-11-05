from openai import APIConnectionError
import pandas as pd
import copy
import logging
from enum import Enum
from collections import Counter

from regulations_rag.string_tools import match_strings_to_reference_list


# Create a logger for this module
logger = logging.getLogger(__name__)
DEV_LEVEL = 15
logging.addLevelName(DEV_LEVEL, 'DEV')       


mandatory_columns = ["section_reference", "text", "source", "cosine_distance"] # input and output
''' 
Functions that take an input DataFrame 'relevant_sections' with columns "mandatory_columns" and 
extracts a subset of these that are "most relevant" to the user question. Some of these functions may add columns to
the input DataFrame to get their job done but you should not rely on this. You MUST assume the returned DataFrame 
only contains the mandatory_columns
'''

class RerankAlgos(Enum):
    NONE = ("none", {"initial_section_number_cap": 15, "final_token_cap": 3500, "user_question": None})
    MOST_COMMON = ("most_common", {"initial_section_number_cap": 15, "final_token_cap": 3500, "user_question": None})
    LLM = ("llm", {"initial_section_number_cap": 15, "final_token_cap": 3500, "openai_client": None, "model_to_use": None, "user_question": None, "user_type": "", "corpus_description": ""})

    def __init__(self, algo, params):
        self.algo = algo
        self.params = params



def check_rerank_columns(dataframe):
    dataframe_columns = dataframe.columns.to_list()
    for column in mandatory_columns:
        if column not in dataframe_columns:
            return False
    return True

def rerank(relevant_sections, rerank_algo):
    ''' 
    Parameters:
    -----------
    relevant_sections : DataFrame
        Must contain the 'mandatory_columns'
    rerank_algo : RerankAlgos
        An enum that will define how the dataframe in re-ranked

    Returns:
        DataFrame that contains the "mandatory_columns". NOTE, some of the reranking algorithms may have additional columns 
        but you should not rely on this. You MUST assume the returned DataFrame only contains the "mandatory_columns"
    '''

    if relevant_sections.empty:
        return pd.DataFrame([], columns = mandatory_columns)

    if not check_rerank_columns(relevant_sections):
        raise AttributeError("The dataframe to rerank does not have the correct columns")
    
    if rerank_algo.algo == "none":
        logger.log(DEV_LEVEL, f"No re-ranking of the relevant sections")
        # relevant_sections = relevant_sections
    elif rerank_algo.algo == 'most_common':
        logger.log(DEV_LEVEL, f"Re-ranking using most_common")
        relevant_sections = rerank_most_common(relevant_sections)
    elif rerank_algo.algo == "llm":
        logger.log(DEV_LEVEL, f"Re-ranking using LLM")
        relevant_sections = rerank_llm(relevant_sections, 
                                       openai_client = rerank_algo.params["openai_client"], 
                                       model_to_use=rerank_algo.params["model_to_use"], 
                                       user_question = rerank_algo.params["user_question"],
                                       user_type = rerank_algo.params["user_type"],
                                       corpus_description = rerank_algo.params["corpus_description"])
    else:        
        raise NotImplementedError()
    
    return relevant_sections
    

def rerank_most_common(relevant_sections):
    """
    Refines and selects the most relevant sections to be sent to the LLM based on the provided dataframe of sections,
    their cosine distances, and occurrences. It prioritizes the top result, the most common section (mode), and other
    frequently found sections that are not the top result or mode. It also ensures diversity in the selection by including
    sections based on their frequency and cosine distance.

    Parameters:
    - relevant_sections (DataFrame): A dataframe containing sections with their cosine distances.

    Returns:
    - DataFrame: A dataframe with the selected references, their minimum cosine distances, and count.
    """
    
    search_sections = []

    # Top result
    top_result = relevant_sections.iloc[0].copy()
    count = (relevant_sections['section_reference'] == top_result["section_reference"]).sum()
    top_result['count'] = count # add the field count
    search_sections.append(top_result)
    logger.log(DEV_LEVEL, f'Top result: {top_result["section_reference"]} with a cosine distance of {top_result["cosine_distance"]:.4f}')

    # Mode
    mode_value_list = relevant_sections['section_reference'].mode()
    if len(mode_value_list) == 1:
        mode_value = mode_value_list[0]
        if mode_value != top_result["section_reference"]:
            mode_sections = relevant_sections[relevant_sections['section_reference'] == mode_value]
            mode_result = mode_sections.iloc[0].copy()
            mode_result['count'] = len(mode_sections)
            search_sections.append(mode_result)
            logger.log(DEV_LEVEL, f"Most common section: {mode_result['section_reference']} with a minimum cosine distance of {mode_result['cosine_distance']:.4f}")

    elif not mode_value_list.empty:
        # If there are multiple modes, treat as if no mode by setting mode_value_list to be empty
        # This block assumes that mode_value_list should be considered empty if the mode's occurrence is not unique
        mode_value_list = pd.Series(dtype='object') 
        logger.log(DEV_LEVEL, "Multiple modes found, treated as no unique mode.")

    else:
        logger.log(DEV_LEVEL, "No mode")

    # Frequent references excluding top result and mode
    count_dict = Counter(relevant_sections['section_reference'])
    for section, freq in count_dict.items():
        if freq > 1 and section not in [top_result["section_reference"], mode_value_list[0] if not mode_value_list.empty else None]:
            sub_frame = relevant_sections[relevant_sections['section_reference'] == section]
            repeat_find = sub_frame.iloc[0].copy()
            repeat_find['count'] = len(sub_frame)
            search_sections.append(repeat_find)
            logger.log(DEV_LEVEL, f"Reference: {repeat_find['section_reference']}, Count: {repeat_find['count']}, Min Cosine-Distance: {repeat_find['cosine_distance']:.4f}")
    
    # Additional checks for diversity if only one section is selected initially
    if len(search_sections) == 1 and len(relevant_sections) > 1:
        logger.log(DEV_LEVEL, 'Only the top result added but more were found. Adding the next most likely answer(s).')
        remaining_sections = relevant_sections[~relevant_sections['section_reference'].isin([top_result["section_reference"]])]
        added_count = 0
        for index, row in remaining_sections.iterrows():
            if added_count >= 2:  # Break after adding two additional results
                break
            next_most_likely = row.copy()
            next_most_likely['count']= 1

            logger.log(DEV_LEVEL, f"Reference: {next_most_likely['section_reference']}, Count: {next_most_likely['count']}, Min Cosine-Distance: {next_most_likely['cosine_distance']:.4f}")

            search_sections.append(next_most_likely)
            added_count += 1

    # Note the order of the search_section is preserved        
    return pd.DataFrame(search_sections)



def rerank_llm(relevant_sections, openai_client, model_to_use, user_question, user_type, corpus_description):

    list_of_sections_and_text = []
    counter = 1
    for index, row in relevant_sections.iterrows():
        list_of_sections_and_text.append(f"Index {counter}: {row['text']}")
        counter += 1
    string_of_sections_and_text = "\n".join(list_of_sections_and_text)

    
    system_content = f"You are answering {user_type} answer questions on {corpus_description}. You will be given the users question followed \
by a list of index items. An index item is a description of what is contained in a document. It is either a summary of the document or a question that is \
answered in the document. Your job is to use the index items to determine which documents are likely to contain an answer to the users question. List the \
number of the index items in a pipe delimited list. Do not respond with any other text. Just the pipe delimited list of integer index numbers."

    
    user_content = f"### Question: {user_question}\n### Index items: \n{string_of_sections_and_text}"

    messages = []
    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    try:
        response = openai_client.chat.completions.create(
                                    model=model_to_use,
                                    temperature=0,
                                    max_tokens=500,
                                    messages=messages
                                )
        response_text = response.choices[0].message.content
    except APIConnectionError as e:
        print("API connection error")


    relevant_section_references = [item.strip() for item in response_text.split('|')]    
    unique_items = []
    references_as_integers = []
    found_at_least_one_reference = False
    for item in relevant_section_references:
        try:
            # Attempt to convert the item to an integer
            integer_value = int(item)
            if integer_value < 1 or integer_value > len(relevant_sections):
                logger.log(DEV_LEVEL, f"{item} should be an integer between 1 and {len(relevant_sections)} but it is not")
            else:
                search_section_row_to_add = integer_value - 1
                doc_section_par = f"{relevant_sections.iloc[search_section_row_to_add]['document']}_{relevant_sections.iloc[search_section_row_to_add]['section_reference']}"
                if len(unique_items) == 0 or doc_section_par not in unique_items:
                    unique_items.append(doc_section_par)
                    references_as_integers.append(search_section_row_to_add)
                    found_at_least_one_reference = True
        except ValueError:
            logger.log(DEV_LEVEL, f"{item} should be an integer between 1 and {len(relevant_sections)} but it is not")

    # remove duplicates
    document_section = set()

    logger.log(DEV_LEVEL, "--   results requested by LLM filter")
    for section in references_as_integers:
        logger.log(DEV_LEVEL, f'{relevant_sections.iloc[section]["section_reference"]}')


    # Create a DataFrame from the list of best match rows
    subset_df = relevant_sections.iloc[references_as_integers]

    return subset_df
