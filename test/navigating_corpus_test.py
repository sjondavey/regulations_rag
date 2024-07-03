import pytest
from regulations_rag.document import Document
from .navigating_corpus import NavigatingCorpus


@pytest.fixture
def navigating_corpus():
    return NavigatingCorpus()

def test_get_primary_document(navigating_corpus):
    assert navigating_corpus.get_primary_document() == "WRR"


def test_get_text(navigating_corpus):
    text = navigating_corpus.get_text("WRR", "1", add_markdown_decorators=False, section_only = True)
    expected_text = '1 Navigating Whale Rock Ridge\nWhale Rock Ridge is a large complex. Here are directions to help you[^1].\n\n  \n[^1]: Directions from 11 Turnstone'
    assert text == expected_text

def test_get_document(navigating_corpus):
    doc = navigating_corpus.get_document("Plett")
    expected_text = "From South Gate, turn left into Whale Rock Drive. Continue straight to Robberg Nature Reserve"
    assert doc.get_text("A.2(B)(iii)", add_markdown_decorators = False, add_headings = False, section_only = True)