import pytest
import pandas as pd
from unittest.mock import Mock
from regulations_rag.reference_checker import ReferenceChecker
from regulations_rag.regulation_table_of_content import StandardTableOfContent, TableOfContentEntry, split_tree
from test.reference_checker_samples import SimpleReferenceChecker

       

@pytest.fixture
def sample_dataframe():
    data = {
        'text': ['Text 1', 'Text 2', 'Text 3'],
        'heading': [True, True, True],
        'section_reference': ['1', '1.1', '1.1.1']
    }
    return pd.DataFrame(data)

@pytest.fixture
def reference_checker():
    return SimpleReferenceChecker()

@pytest.fixture
def standard_toc(sample_dataframe, reference_checker):
    return StandardTableOfContent(root_node_name="root", reference_checker=reference_checker, regulation_df=sample_dataframe)

def test_initialization(standard_toc):
    assert standard_toc.root.name == "root"
    assert len(standard_toc.root.children) == 1  # only the first valid section should be added
    assert standard_toc.root.children[0].name == "1"

def test_add_to_toc(standard_toc):
    standard_toc.add_to_toc('1.2', 'New heading')
    node = standard_toc.get_node('1.2')
    assert node.heading_text == 'New heading'

def test_add_to_toc_invalid(standard_toc):
    with pytest.raises(ValueError):
        standard_toc.add_to_toc('invalid', 'Invalid heading')

def test_get_node(standard_toc):
    node = standard_toc.get_node('1')
    assert node.name == '1'
    assert node.heading_text == 'Text 1'

def test_get_node_invalid(standard_toc):
    with pytest.raises(ValueError):
        standard_toc.get_node('invalid')

def test_print_tree(standard_toc, capsys):
    standard_toc.print_tree()
    captured = capsys.readouterr()
    assert "root" in captured.out
    assert "1" in captured.out

def test_list_node_children(standard_toc):
    node = standard_toc.get_node('1')
    result = standard_toc._list_node_children(node)
    assert '1 Text 2' in result

def test_remove_footnotes(standard_toc):
    text = "This is a text[^1]\n[^1]: This is a footnote"
    cleaned_text = standard_toc.remove_footnotes(text)
    assert "[^1]" not in cleaned_text

def test_check_columns(standard_toc):
    assert standard_toc.check_columns()

def test_split_tree(standard_toc):
    document = Mock()
    document.get_text.side_effect = lambda x: "sample text"
    result_df = split_tree(standard_toc.root, document, standard_toc, 100)
    assert len(result_df) > 0
    assert 'section_reference' in result_df.columns
    assert 'text' in result_df.columns
    assert 'token_count' in result_df.columns
