from regulations_rag.reference_checker import ReferenceChecker


class TESTReferenceChecker(ReferenceChecker):
    def __init__(self):
        cemad_exclusion_list = ['Legal context', 'Introduction']
        cemad_index_patterns = [
            r'^[A-Z]\.\d{0,2}',             # Matches capital letter followed by a period and up to two digits.
            r'^\([A-Z]\)',                  # Matches single capital letters within parentheses.
            r'^\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx|xxi|xxii|xxiii|xxiv|xxv|xxvi|xxvii)\)', # Matches Roman numerals within parentheses.
            r'^\([a-z]\)',                  # Matches single lowercase letters within parentheses.
            r'^\([a-z]{2}\)',               # Matches two lowercase letters within parentheses.
            r'^\((?:[1-9]|[1-9][0-9])\)',   # Matches numbers within parentheses, excluding leading zeros.
        ]    
        text_pattern = r"[A-Z]\.\d{0,2}\([A-Z]\)\((?:i|ii|iii|iv|v|vi)\)\([a-z]\)\([a-z]{2}\)\(\d+\)"

        super().__init__(regex_list_of_indices = cemad_index_patterns, text_version = text_pattern, exclusion_list=cemad_exclusion_list)

class SimpleReferenceChecker(ReferenceChecker):
    def __init__(self):
        exclusion_list = []
        simple_index_patterns = [
            r'^[1-9]',          # Matches a single digit from 1 to 9
            r'^\.[1-9]',        # Matches a dot followed by a single digit from 1 to 9
            r'^\.[1-9]',        # Matches a dot followed by a single digit from 1 to 9 again for the third level
        ]
        text_pattern = r"[1-9](\.[1-9]){0,2}"
        super().__init__(regex_list_of_indices=simple_index_patterns, text_version=text_pattern)



