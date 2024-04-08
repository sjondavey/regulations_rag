import pandas as pd
import logging
from enum import Enum
from collections import Counter

from regulations_rag.reg_tools import get_regulation_detail
from regulations_rag.string_tools import match_strings_to_reference_list


# Create a logger for this module
logger = logging.getLogger(__name__)
DEV_LEVEL = 15
logging.addLevelName(DEV_LEVEL, 'DEV')       

''' 
Functions that take an input DataFrame 'relevant_sections' with columns ["section_reference", "text", "source", "cosine_distance"] and 
extracts a subset of these that are 'most relevant' to the user question
'''

class RerankAlgos(Enum):
    NONE = ("none", {"initial_section_number_cap": 15, "final_token_cap": 3500, "user_question": None})
    MOST_COMMON = ("most_common", {"initial_section_number_cap": 15, "final_token_cap": 3500, "user_question": None})
    LLM = ("llm", {"initial_section_number_cap": 15, "final_token_cap": 3500, "openai_client": None, "model_to_use": None, "user_question": None})

    def __init__(self, algo, params):
        self.algo = algo
        self.params = params


mandatory_columns = ["section_reference", "text", "source", "cosine_distance"]

def check_columns(relevant_sections):
    dataframe_columns = relevant_sections.columns.to_list()
    for column in mandatory_columns:
        if column not in dataframe_columns:
            return False
    return True

def rerank(relevant_sections, rerank_algo):
    if not check_columns(relevant_sections):
        raise AttributeError("The dataframe to rerank does not have the correct columns")
    if rerank_algo == RerankAlgos.NONE:
         logger.log(DEV_LEVEL, f"No re-ranking of the relevant sections")
         return relevant_sections
    elif rerank_algo == RerankAlgos.MOST_COMMON:
        return rerank_most_common(relevant_sections, initial_cap_sections=rerank_algo.params["initial_section_number_cap"], final_cap_tokens=rerank_algo.params["final_token_cap"])
    elif rerank_algo == RerankAlgos.LLM:
        return rerank_llm(relevant_sections, openai_client = rerank_algo.params["openai_client"], model_to_use=rerank_algo.params["model_to_use"], user_question = rerank_algo.params["user_question"])
    else:        
        raise NotImplementedError()


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
    
    if relevant_sections.empty:
        columns = mandatory_columns
        columns.append("count")
        return pd.DataFrame([], columns = mandatory_columns)

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



def rerank_llm(relevant_sections, openai_client, model_to_use, user_question):
    if relevant_sections.empty:
        return pd.DataFrame([], columns = ["section_reference", "text", "source", "cosine_distance", "count"])

    list_of_sections_and_text = []
    for index, row in relevant_sections.iterrows():
        list_of_sections_and_text.append(f"{row['section_reference']}: {row['text']}")
    string_of_sections_and_text = "\n".join(list_of_sections_and_text)

    short_pattern = r"[A-Z]\.\d{0,2}\([A-Z]\)\(\b(?:i|ii|iii|iv|v|vi)\b\)\([a-z]\)\([a-z]{2}\)\(\d+\)"
    system_content = "You are assisting and Autorised Dealer to answer questions from the Currency and Exchange Control Manual for Authorised Dealers (CEMAD). \
You will be given the users question followed by a list of sections from CEMAD along with the section's heading, summary or a questions which is answered in section. \
Your job is to choose which sections are likely to contain an answer and output that list as a pipe delimited list. Do not respond with any other text. Just the pipe \
delimited list paying attention to ensure the format of the section reference is {short_pattern} with no following text."
    user_content = f"### Question: {user_question}### CEMAD Sections with heading, summary or a question that is answered in the section: \n{string_of_sections_and_text}"

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

    checked_list_of_strings = match_strings_to_reference_list(list_of_strings = relevant_section_references, reference_list_of_strings = relevant_sections['section_reference'].to_list())

    logger.log(DEV_LEVEL, "--   results requested by LLM filter")
    for section in checked_list_of_strings:
        logger.log(DEV_LEVEL, f'{section}')

    # Initialize a list to hold the rows for the subset
    subset_rows = []

    # Iterate over the list of strings
    for string in checked_list_of_strings:
        # Filter the DataFrame for rows matching the current string
        matches = relevant_sections[relevant_sections['section_reference'].eq(string)]
        
        # If there are matches, find the one with the minimum 'cosine_distance'
        if not matches.empty:
            best_match = matches.loc[matches['cosine_distance'].idxmin()]
            # Add the best match row to the list
            subset_rows.append(best_match)


    # Create a DataFrame from the list of best match rows
    subset_df = pd.DataFrame(subset_rows).reset_index(drop=True)

    if len(checked_list_of_strings) != len(subset_df):
        logger.log(DEV_LEVEL, f'Not all requested sections could be matched. The ones that could are:')
        for index, row in subset_df.iterrows():
            logger.log(DEV_LEVEL,f'{row["section_reference"]}')


    return subset_df
