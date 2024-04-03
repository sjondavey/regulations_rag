import pytest
import pandas as pd

from src.regulations_rag.section_reference_checker import SectionReferenceChecker

from src.regulations_rag.reg_tools import  get_regulation_detail, \
                                           get_regulation_heading


exclusion_list = ['Legal context', 'Introduction']
index_patterns = [
    r'^[A-Z]\.\d{0,2}',             # Matches capital letter followed by a period and up to two digits.
    r'^\([A-Z]\)',                  # Matches single capital letters within parentheses.
    r'^\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx|xxi|xxii|xxiii|xxiv|xxv|xxvi|xxvii)\)', # Matches Roman numerals within parentheses.
    r'^\([a-z]\)',                  # Matches single lowercase letters within parentheses.
    r'^\([a-z]{2}\)',               # Matches two lowercase letters within parentheses.
    r'^\((?:[1-9]|[1-9][0-9])\)',   # Matches numbers within parentheses, excluding leading zeros.
]    
text_pattern = r"[A-Z]\.\d{0,2}\([A-Z]\)\((?:i|ii|iii|iv|v|vi)\)\([a-z]\)\([a-z]{2}\)\(\d+\)"

section_reference_checker = SectionReferenceChecker(regex_list_of_indices=index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)


def test_get_regulation_detail():
    df = pd.read_csv("./test/inputs/manual.csv", sep="|", encoding="utf-8")
    
    response = get_regulation_detail('A.3(E)(viii)(a)(bb)', df, section_reference_checker)
    expected_response = 'A.3 Duties and responsibilities of Authorised Dealers\n    (E) Transactions with Common Monetary Area residents\n        (viii) As an exception to (vi) above, Authorised Dealers may:\n            (a) sell foreign currency to:\n                (bb) CMA residents in South Africa, to cover unforeseen incidental costs whilst in transit, subject to viewing a passenger ticket confirming a destination outside the CMA;'
    assert response == expected_response


def test_get_regulation_heading():
    df = pd.read_csv("./test/inputs/manual.csv", sep="|", encoding="utf-8")

    response = get_regulation_heading('A.3(E)(viii)(a)(bb)', df, section_reference_checker)
    expected_response = 'A.3 Duties and responsibilities of Authorised Dealers. (E) Transactions with Common Monetary Area residents.'
    assert response == expected_response

