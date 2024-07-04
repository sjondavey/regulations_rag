import pytest
import pandas as pd

from regulations_rag.reference_checker import ReferenceChecker, EmptyReferenceChecker, MultiReferenceChecker
from test.reference_checker_samples import TESTReferenceChecker, SimpleReferenceChecker

class TestReferenceChecker:

    reference_checker = TESTReferenceChecker()

    def test_is_valid(self):
        blank_reference = ""
        assert not self.reference_checker.is_valid(blank_reference)

        long_reference = 'G.1(C)(xviii)(c)(dd)(9)'
        assert self.reference_checker.is_valid(long_reference)
        very_long_reference = 'G.1(C)(xviii)(c)(dd)(9)(10)'
        assert not self.reference_checker.is_valid(very_long_reference)

        short_reference = 'G.1(C)'        
        assert self.reference_checker.is_valid(short_reference)

        reference_on_exclusion_list = 'Legal context'
        assert self.reference_checker.is_valid(reference_on_exclusion_list)

        invalid_reference = 'G.1(C)(xviii)(c)(c)(9)'
        assert not self.reference_checker.is_valid(invalid_reference)
        invalid_reference = 'G.1(C)(xviii)(c)(DD)(9)'
        assert not self.reference_checker.is_valid(invalid_reference)
        invalid_reference = 'G.1(C)(xviii)(c)(9)(dd)'
        assert not self.reference_checker.is_valid(invalid_reference)
        invalid_reference = 'G.1(xviii)'
        assert not self.reference_checker.is_valid(invalid_reference)

    def test_extract_valid_reference(self):
        assert self.reference_checker.extract_valid_reference('B.18 Gold (B)(i)(b)') == 'B.18(B)(i)(b)'
        assert self.reference_checker.extract_valid_reference('   B.18 Gold (B)(i)(b)') == 'B.18(B)(i)(b)'
        #assert self.reference_checker.extract_valid_reference('B.18 Gold (B)(i)(ii)')  is None
        assert self.reference_checker.extract_valid_reference('B.18 Gold (B)(i)(ii)')  == 'B.18(B)(i)'
        assert self.reference_checker.extract_valid_reference('A.1') == 'A.1'
        assert self.reference_checker.extract_valid_reference('B.18 Gold (B)(i)(b) hello') == 'B.18(B)(i)(b)' # text at the end
        #assert self.reference_checker.extract_valid_reference('B.18 Gold (B)(i)(b) (hello)') == None  # the text at the end contains an "("
        assert self.reference_checker.extract_valid_reference('B.18 Gold (B)(i)(b) (hello)') == 'B.18(B)(i)(b)'

    def test_split_reference(self):
        long_reference = 'G.1(C)(xviii)(c)(dd)(9)'
        components = self.reference_checker.split_reference(long_reference)
        assert len(components) == 6
        assert components[0] == 'G.1'
        assert components[1] == '(C)'
        assert components[2] == '(xviii)'
        assert components[3] == '(c)'
        assert components[4] == '(dd)'
        assert components[5] == '(9)'

        short_reference = 'G.1(C)'        
        components = self.reference_checker.split_reference(short_reference)
        assert len(components) == 2
        assert components[0] == 'G.1'
        assert components[1] == '(C)'


        invalid_reference = 'G.1(C)(xviii)(c)(DD)(9)'
        with pytest.raises(ValueError):
            components = self.reference_checker.split_reference(invalid_reference)

        invalid_reference = 'G.1(C)(xviii)(c)(d)(9)'
        with pytest.raises(ValueError):
            components = self.reference_checker.split_reference(invalid_reference)

        reference_on_exclusion_list = 'Legal context'
        components = self.reference_checker.split_reference(reference_on_exclusion_list)
        assert components[0] == reference_on_exclusion_list


    def test_get_parent_reference(self):
        reference = 'G.1(C)(xviii)(c)(dd)(9)'
        assert self.reference_checker.get_parent_reference(reference) == 'G.1(C)(xviii)(c)(dd)'
        components = self.reference_checker.get_parent_reference("")
        assert components == ""
        # Test for a single component reference
        reference = 'G.1'
        assert self.reference_checker.get_parent_reference(reference) == ''

    def test_get_current_and_parent_references(self):
        reference = 'G.1(C)(xviii)(c)(dd)(9)'
        current_and_parent = ['G.1(C)(xviii)(c)(dd)(9)', 'G.1(C)(xviii)(c)(dd)', 'G.1(C)(xviii)(c)', 'G.1(C)(xviii)', 'G.1(C)', 'G.1']
        assert self.reference_checker.get_current_and_parent_references(reference) == current_and_parent
        # Test for a single component reference
        reference = 'G.1'
        current_and_parent = ['G.1']
        assert self.reference_checker.get_current_and_parent_references(reference) == current_and_parent

    def test_is_reference_or_parents_in_list(self):
        reference = 'G.1(C)(xviii)(c)(dd)(9)'
        list_of_references = ['A.1', 'B.1', 'C.1']
        assert not self.reference_checker.is_reference_or_parents_in_list(reference, list_of_references)
        list_of_references = ['A.1', 'B.1', 'G.1']
        assert self.reference_checker.is_reference_or_parents_in_list(reference, list_of_references)
        # Test when the reference itself is in the list
        reference = 'G.1(C)(xviii)(c)(dd)(9)'
        list_of_references = ['A.1', 'B.1', 'G.1(C)(xviii)(c)(dd)(9)']
        assert self.reference_checker.is_reference_or_parents_in_list(reference, list_of_references)

        # Test when a parent of the reference is in the list
        reference = 'G.1(C)(xviii)(c)(dd)(9)'
        list_of_references = ['A.1', 'B.1', 'G.1(C)(xviii)(c)']
        assert self.reference_checker.is_reference_or_parents_in_list(reference, list_of_references)

    def test___extract_reference_from_string(self):
        string_with_no_reference = 'Africa means any country forming part of the African Union.'
        index, string = self.reference_checker._extract_reference_from_string(string_with_no_reference)
        assert index == ""
        assert string == string_with_no_reference

        # tests for each of the numbering patters used in excon_index_patterns
        string_with_reference = 'A.1 Definitions'
        index, string = self.reference_checker._extract_reference_from_string(string_with_reference)
        assert index == "A.1"
        assert string == 'Definitions'

        string_with_reference = '(A) Authorised Dealers'
        index, string = self.reference_checker._extract_reference_from_string(string_with_reference)
        assert index == "(A)"
        assert string == 'Authorised Dealers'

        string_with_reference = '(xxiii) Authorised Dealers must reset their application numbering systems to zero at the beginning of each calendar year.'
        index, string = self.reference_checker._extract_reference_from_string(string_with_reference)
        assert index == "(xxiii)"
        assert string == 'Authorised Dealers must reset their application numbering systems to zero at the beginning of each calendar year.'

        string_with_reference = '(a) a list of application numbers generated but not submitted to the Financial Surveillance Department;'
        index, string = self.reference_checker._extract_reference_from_string(string_with_reference)
        assert index == "(a)"
        assert string == 'a list of application numbers generated but not submitted to the Financial Surveillance Department;'

        string_with_reference = '(dd) CMA residents who travel overland to and from other CMA countries through a SADC country up to an amount not exceeding R25 000 per calendar year. This allocation does not form part of the permissible travel allowance for residents; and'
        index, string = self.reference_checker._extract_reference_from_string(string_with_reference)
        assert index == "(dd)"
        assert string == 'CMA residents who travel overland to and from other CMA countries through a SADC country up to an amount not exceeding R25 000 per calendar year. This allocation does not form part of the permissible travel allowance for residents; and'

        string_with_reference = '(1) the full names and identity number of the applicant;'
        index, string = self.reference_checker._extract_reference_from_string(string_with_reference)
        assert index == "(1)"
        assert string == 'the full names and identity number of the applicant;'

        heading_on_exclusion_list = 'Legal context'
        index, string = self.reference_checker._extract_reference_from_string(heading_on_exclusion_list)
        assert index == heading_on_exclusion_list
        assert string == ""

        # Test for string with no reference and additional spaces
        # string_with_no_reference = '    Africa means any country forming part of the African Union.'
        # index, string = self.reference_checker._extract_reference_from_string(string_with_no_reference)
        # assert index == ""
        # assert string == string_with_no_reference.strip()

        # Test for string with reference and additional text at the end
        string_with_reference = 'A.1 Definitions and more text'
        index, string = self.reference_checker._extract_reference_from_string(string_with_reference)
        assert index == "A.1"
        assert string == 'Definitions and more text'

        # Test for string with reference and additional spaces
        # string_with_reference = '    A.1    Definitions'
        # index, string = self.reference_checker._extract_reference_from_string(string_with_reference)
        # assert index == "A.1"
        # assert string == 'Definitions'

        def test_another_reference_tester(self):
            class ManualReferenceChecker(ReferenceChecker):
                def __init__(self):
                    exclusion_list = [] # none of this will be indexed
                    # NOTE: I do not include 'analysis' in the reference checker because I will never have a user search for sections with this as the reference
                    index_patterns = [
                        r'\bApplication\b',
                        r'\.\s(Part|Annex)\s\d+', # "". Part" or ".Annex"
            #            r'(\d+)',
                        r'\.\d+',
                    ]
                    text_pattern = r'Application. (Part/Annex\s\d+)?(\.(\d+))?'


                    super().__init__(regex_list_of_indices = index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)



class TestEmptyReferenceChecker():
    no_reference = EmptyReferenceChecker()

    def test_construction(self):
        assert True

    def test_is_valid(self):
        assert self.no_reference.is_valid("")
        assert not self.no_reference.is_valid("A")
        assert self.no_reference.is_valid(None)

        data = ["", "article_30_5"]
        df = pd.DataFrame([data], columns=["section_reference", "document"])
        empty_df_entry = df.iloc[0]["section_reference"]
        assert self.no_reference.is_valid(empty_df_entry)

        data = [None, "article_30_5"]
        df = pd.DataFrame([data], columns=["section_reference", "document"])
        empty_df_entry = df.iloc[0]["section_reference"]
        assert self.no_reference.is_valid(empty_df_entry)




class MainSection(ReferenceChecker):
    def __init__(self):
        exclusion_list = [] 
        index_patterns = [
            r'^\d+',   
            r'^\.\d+', 
            r'^\.\d+', 
            r'^\.\d+', 
        ]    
        text_pattern = r'(\d+(\.\d+)?(\.\d+)?(\.\d+)?)'

        super().__init__(regex_list_of_indices = index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)

class AltSection(ReferenceChecker):
    def __init__(self):
        exclusion_list = [] #
        index_patterns = [
            r'\bApplication\b',
            r'\.\s(Part|Annex)\s\d+', # "". Part" or ".Annex"
            r'\.\d+',
        ]
        text_pattern = r'Application. (Part/Annex\s\d+)?(\.(\d+))?'

        super().__init__(regex_list_of_indices = index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)

class TestMultiReferenceChecker():

    main = MainSection()
    alt = AltSection()
    doc_ref_checker = MultiReferenceChecker([main, alt])

    def test_is_valid(self):
        assert self.doc_ref_checker.is_valid("3.4.2.1")
        assert self.doc_ref_checker.is_valid("Application. Annex 2.1")
        assert not self.doc_ref_checker.is_valid("Application. Annex")

    def test_get_parent_reference(self):
        assert self.doc_ref_checker.get_parent_reference("3.4.2.1") == "3.4.2"
        assert self.doc_ref_checker.get_parent_reference("Application. Part 2") == "Application"


class TestSimpleReferenceChecker():
    rc = SimpleReferenceChecker()

    def test_is_valid(self):
        assert self.rc.is_valid("1")
        assert self.rc.is_valid("1.1")
        assert self.rc.is_valid("1.1.1")