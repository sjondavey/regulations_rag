from thefuzz import process


def match_strings_to_reference_list(list_of_strings, reference_list_of_strings):
    """
    Matches each string in `list_of_strings` to the closest string in `reference_list_of_strings`
    based on a similarity score, and returns a list of unique matched strings that meet or exceed
    a score cutoff of 80. It removes duplicate entries both in the input reference list and in
    the final list of matched strings.
    
    Parameters:
    - list_of_strings (list of str): A list of strings to match against the reference list.
    - reference_list_of_strings (list of str): A list of reference strings to be matched against.
    
    Returns:
    - list of str: A list of unique matched strings from the reference list that have a similarity
      score of at least 80 with any of the strings in `list_of_strings`. The list does not include
      any duplicates.
    
    Note:
    This function utilizes the `process.extractOne` method from the `thefuzz` library to find the best
    match for each string in `list_of_strings` against the entries in `reference_list_of_strings`.
    Only matches with a score of 80 or higher are included in the returned list. The `reference_list_of_strings`
    is first converted to a set to remove any duplicate entries, ensuring each reference string is unique.
    Similarly, before returning, the list of matched strings is also converted to a set to remove duplicates,
    ensuring that each matched string is unique.
    """
    # Remove duplicates while preserving order - I probably don't need to worry about the order here but I do in the next de-dup
    de_duped_reference_list_of_strings = []
    for item in reference_list_of_strings:
        if item not in de_duped_reference_list_of_strings:
            de_duped_reference_list_of_strings.append(item)

    checked_list_of_strings = []
    for item in list_of_strings:
        match = process.extractOne(item, reference_list_of_strings, score_cutoff=80)
        if match:
            checked_list_of_strings.append(match[0])
    
    # Remove duplicates while preserving order
    de_duped_checked_list_of_strings = []
    for item in checked_list_of_strings:
        if item not in de_duped_checked_list_of_strings:
            de_duped_checked_list_of_strings.append(item)

    return de_duped_checked_list_of_strings
