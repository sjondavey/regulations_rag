import pytest
#import pandas as pd

from regulations_rag.reference_checker import TESTReferenceChecker

from regulations_rag.regulation_index import RegulationIndex
from regulations_rag.regulation_index_tree import TableOfContentTree, StandardTableOfContentTree, split_tree
from regulations_rag.regulation_reader import load_csv_data
from regulations_rag.regulation_reader import TESTReader

# from regulations_rag.file_tools import process_lines, add_full_reference


class TestTree:
    index_checker = TESTReferenceChecker()

    def test_add_to_toc(self):
        tree = TableOfContentTree("Excon", self.index_checker)
        invalid_reference = 'G.1(C)(xviii)(c)(DD)(9)'
        with pytest.raises(ValueError):
            tree.add_to_toc(invalid_reference, heading_text='')

        #Check all nodes get added
        valid_index = 'G.1(C)(xviii)(c)(dd)(9)'
        tree.add_to_toc(valid_index, heading_text='Some really deep heading here')
        number_of_nodes = sum(1 for _ in tree.root.descendants) # excludes the root node
        assert number_of_nodes == 6

        #check that if a duplicate is added, it does not increase the node count
        sub_index = 'G.1(C)(xviii)'
        tree.add_to_toc(valid_index, heading_text='Some less deep heading here')
        number_of_nodes = sum(1 for _ in tree.root.descendants) # excludes the root node
        assert number_of_nodes == 6



    def test_get_node(self):
        tree = TableOfContentTree("Excon", self.index_checker)
        invalid_reference = 'G.1(C)(xviii)(c)(DD)(9)'
        with pytest.raises(ValueError):
            tree.get_node(invalid_reference)
        invalid_reference = ''
        with pytest.raises(ValueError):
            tree.get_node(invalid_reference)
        
        assert tree.get_node("Excon") == tree.root
        assert tree.get_node("Excon").full_node_name == ""
        assert tree.get_node("Excon").heading_text == ""

        excon_description = "Exchange control manual hierarchy"
        tree.add_to_toc("Excon", heading_text=excon_description)
        assert tree.get_node("Excon").heading_text == excon_description

        valid_index = 'G.1(C)(xviii)(c)(dd)(9)'
        tree.add_to_toc(valid_index, heading_text='Some really deep heading here')
        assert tree.get_node(valid_index).heading_text == 'Some really deep heading here'
        sub_index = 'G.1(C)(xviii)'
        assert tree.get_node(sub_index).heading_text == ''
        tree.add_to_toc(sub_index, heading_text='Some less deep heading here')
        assert tree.get_node(sub_index).heading_text == 'Some less deep heading here'


class TestStandardTableOfContentTree:
    index_checker = TESTReferenceChecker()

    def test_construction(self):
        df = load_csv_data("./test/inputs/tree_test_data.csv")
        tree = StandardTableOfContentTree("Excon", self.index_checker, df)
        assert True


def test_split_tree():
    reference_checker = TESTReferenceChecker()
    df = load_csv_data("./test/inputs/tree_test_data.csv")
    toc = StandardTableOfContentTree("Excon", reference_checker, df)
    reader = TESTReader(reference_checker, df)

    #node_list=[]
    token_limit_per_chunk = 125
    chunked_df = split_tree(toc.root, reader, toc, token_limit_per_chunk)
    assert len(chunked_df) == 7
    assert chunked_df.iloc[0]['section_reference'] == 'A.3(A)(i)'
    assert chunked_df.iloc[1]['section_reference'] == 'A.3(A)(ii)'
    assert chunked_df.iloc[2]['section_reference'] == 'A.3(B)(i)'
    assert chunked_df.iloc[6]['section_reference'] == 'A.3(E)(viii)(a)(cc)'

