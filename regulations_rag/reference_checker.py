import re

class ReferenceChecker:
    """
    A helper class for managing and validating legal document section references in a tree structure format.
    
    Legal documents often utilize a hierarchical numbering system for sections, which this class aims to support.
    The indexing might start with a capital letter, followed by an indented Roman numeral, then a double-indented lowercase letter in brackets, etc.
    This class facilitates the management of such indices, including the ability to specify exceptions that do not follow the standard indexing pattern,
    for example "Overview" or "Definitions".
    
    Attributes:
        index_patterns (list): A list of regex patterns that define valid index formats.
        text_version (str): A text description of the index that will be used in the System message to describe to the LLM what format to use when referring to sections of the document.
        exclusion_list (list): A list of index patterns to be excluded from validation, representing exceptions in the document structure.
        
    Parameters:
        regex_list_of_indices (list): A list of regex patterns specifying the valid formats for indices.
        text_version (str, optional): A text description of the index format. Defaults to an empty string.
        exclusion_list (list, optional): A list of index formats that should be excluded from validation. Defaults to an empty list.
    """
    
    def __init__(self, regex_list_of_indices, text_version="", exclusion_list=[]):
        self.index_patterns = regex_list_of_indices
        self.exclusion_list = exclusion_list
        if text_version:
            self.text_version = text_version
        else:
            combined_pattern = "".join(f"({pattern.lstrip('^')})" for pattern in regex_list_of_indices)
            self.text_version = "r'" + combined_pattern + "'"

    def is_valid(self, reference):
        """
        Validates if a reference is valid based on predefined patterns and exclusions.

        Parameters:
            reference (str): The reference string to validate.

        Returns:
            bool: True if the reference is valid or in the exclusion list, False otherwise.
        """
        if reference in self.exclusion_list:
            return True

        reference_copy = reference
        pattern_matched = False
        for pattern in self.index_patterns:
            if reference_copy:
                match = re.match(pattern, reference_copy)
                if match:
                    reference_copy = reference_copy[match.end():]
                    pattern_matched = True
                else:
                    if pattern_matched:
                        # If a pattern was matched previously but the current pattern doesn't match,
                        # the reference is invalid.
                        return False
        if reference_copy:  # i.e. there is more text than there are regex patterns
            return False
        return pattern_matched

    def extract_valid_reference(self, input_string):
        """
        Attempts to extract a valid reference from a string that may contain additional text or characters.

        The method iteratively searches for patterns defined in index_patterns within the input_string,
        constructing a 'partial_ref' by appending matched patterns. It removes matched patterns and any text
        to the left, continuing with the next pattern until all patterns are attempted or no further matches
        can be found.

        For example, given the testing reference pattern, this method will provide the following output
        # print(extract_valid_reference('B.18 Gold (B)(i)(b)'))  # Output: 'B.18(B)(i)(b)'
        # print(extract_valid_reference('B.18 Gold (B)(a)(b)'))  # Output: 'B.18(B) because after (B) we need a Roman Numeral
        # print(extract_valid_reference('A.1'))  # Output: 'A.1'

        Parameters:
            input_string (str): The string from which to extract a valid reference.

        Returns:
            str or None: The extracted reference if successful, or None if a valid reference cannot be
                        fully constructed based on the index_patterns or if the remaining text suggests
                        an incomplete reference.
        """
        if input_string.strip() in self.exclusion_list:
            return input_string.strip()

        partial_ref = ""
        remaining_str = input_string

        for pattern in self.index_patterns:
            # the caret "^" is used in the index pattern because we only want the index at the start of the section but this causes potential issues here so it is removed 
            if pattern[0] == "^":
                pattern = pattern[1:]
            match = re.search(pattern, remaining_str)
            if match:
                partial_ref += match.group()
                remaining_str = remaining_str[match.end():]
            else:
                if remaining_str and "(" in remaining_str:
                    return partial_ref  # this will deal with some cases but may result in undesired behaviour for invalid strings of the form ('B.18 Gold (B)(a)(b)'))

        return partial_ref if partial_ref else None

    def split_reference(self, reference):
        """
        Splits a given reference into its constituent parts based on the index_patterns.

        This method attempts to match each part of the reference with the regex patterns provided in
        index_patterns. It collects matched components into a list. If a part of the reference does not match
        any pattern or if there is any leftover string after attempting all patterns, it raises a ValueError.

        Parameters:
            reference (str): The reference string to split into components.

        Returns:
            list: A list of components that were successfully matched against the index_patterns.

        Raises:
            ValueError: If the reference does not fully match the provided patterns or if there's unmatched
                        text remaining.
        """
        components = []
        if reference == "":
            return components
        if reference in self.exclusion_list:
            components.append(reference)
            return components

        # Initialize variables
        reference_copy = reference
        pattern_matched = False

        for pattern in self.index_patterns:
            if reference_copy:
                match = re.match(pattern, reference_copy)
                if match:
                    components.append(match.group(0))
                    reference_copy = reference_copy[match.end():]
                    pattern_matched = True
                else:
                    if pattern_matched:
                        raise ValueError(f'The input index {reference} did not comply with the schema')

        # If there's anything left in the reference after all patterns have been attempted, it's invalid.
        if reference_copy:
            raise ValueError(f'The input index {reference} did not comply with the schema')

        return components

    def get_parent_reference(self, input_string):
        """
        Determines the parent reference of a given reference string.

        This method uses split_reference to decompose the input_string into its components based on the
        index_patterns. It then reconstructs the parent reference by joining all but the last component.
        If the input_string is empty or cannot be decomposed into valid components, it raises a ValueError.

        Parameters:
            input_string (str): The reference string for which the parent reference is sought.

        Returns:
            str: The parent reference of the input string.

        Raises:
            ValueError: If the input_string is empty or if valid components cannot be extracted from it.
        """
        if input_string == "":
            return ""
            # raise ValueError(f"Unable to get parent string for empty input")

        split_reference = self.split_reference(input_string)

        parent_reference = ''

        if not split_reference:
            raise ValueError(f"Unable to extract valid indexes from the string {input_string}")

        for i in range(0, len(split_reference) - 1):
            parent_reference += split_reference[i]

        return parent_reference

    def get_current_and_parent_references(self, reference):
        """
        Retrieves a list of the given reference and all its parent references.

        Parameters:
            reference (str): The reference string for which to retrieve the list of references.

        Returns:
            list: A list containing the given reference and all its parent references.
        """
        parents = [reference]
        while reference:
            reference = self.get_parent_reference(reference)
            if reference:
                parents.append(reference)
        return parents

    def is_reference_or_parents_in_list(self, reference, list_of_references):
        """
        Checks if a given reference or any of its parent references are present in a list of references.

        This method first checks if the provided reference is directly in the list of references.
        If not, it retrieves all parent references of the given reference and checks if any of them
        are in the list.

        Parameters:
            reference (str): The reference string to check.
            list_of_references (list): A list of reference strings to check against.

        Returns:
            bool: True if the reference or any of its parent references are in the list, False otherwise.
        """
        if reference in list_of_references:
            return True
        parents = self.get_current_and_parent_references(reference)
        return any(parent in list_of_references for parent in parents)


    def _extract_reference_from_string(self, s):
        """
        Attempts to extract an index that matches predefined patterns from the start of a string.

        This method iterates through the index_patterns to find a match at the beginning of the string.
        If a match is found, it returns the matched index and the rest of the string minus the index and any
        subsequent space. If no match is found and the string matches an item in the exclusion_list, it returns
        the item and an empty string. Otherwise, it returns an empty string for the index and the original string.

        Note: This method does not verify the correctness of the indentation level of the index within the context
        of the document. It simply extracts what appears to be an index based on the provided patterns because at this 
        stage, with only one line of data it is impossible to know, for example if '(i)' is a roman numeral or the 
        single lowercase letter. At this stage we just strip matching strings that look like index values from the 
        rest of the string. We only check if (i) is a indented correctly later when we have the full reference.

        Parameters:
            s (str): The string from which to extract the index.

        Returns:
            tuple: A tuple containing the extracted index (or an empty string if no index is found) and the
                remaining string after removing the index and any immediate following space.
        """
        for pattern in self.index_patterns:
            match = re.match(pattern, s)
            if match:
                # If a match is found, return the matched index and the remaining string
                return match.group(0), s[match.end()+1:] # there is always a space after the index

        for exclusion_item in self.exclusion_list:
            if s.strip() == exclusion_item:
                return exclusion_item, ''

        # If no match is found, return an empty string for the index and the original string
        return '', s


class EmptyReferenceChecker(ReferenceChecker):
    def __init__(self):
        exclusion_list = []
        index_patterns = [r""]    
        text_pattern = ""

        super().__init__(regex_list_of_indices = index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)
        self.text_version = text_pattern

    def is_valid(self, reference):
        if reference == None or reference == "":
            return True
        if reference == "all":
            return True
        return False

    def extract_valid_reference(self, input_string):
        return ""
    
    def split_reference(self, reference):
        return ""

    def get_parent_reference(self, input_string):
        return ""

    def get_current_and_parent_references(self, reference):
        return [""]

    def is_reference_or_parents_in_list(self, reference, list_of_references):
        super().is_reference_or_parents_in_list(reference, list_of_references) 

    def _extract_reference_from_string(self, s):
        return ""


class MultiReferenceChecker(ReferenceChecker):
    def __init__(self, list_of_reference_checkers):
        self.list_of_reference_checkers = list_of_reference_checkers

    def is_valid(self, reference):
        for ref_checker in self.list_of_reference_checkers:
            if ref_checker.is_valid(reference):
                return True
        return False

    def extract_valid_reference(self, reference):
        raise NotImplementedError()

    def split_reference(self, reference):
        for ref_checker in self.list_of_reference_checkers:
            if ref_checker.is_valid(reference):
                return ref_checker.split_reference(reference)
        return ""

    def get_parent_reference(self, reference):
        for ref_checker in self.list_of_reference_checkers:
            if ref_checker.is_valid(reference):
                return ref_checker.get_parent_reference(reference)
        return ""
    
    def get_current_and_parent_references(self, reference):
        raise NotImplementedError()

    def is_reference_or_parents_in_list(self, reference, list_of_references):
        raise NotImplementedError()

    def _extract_reference_from_string(self, s):
        raise NotImplementedError()


