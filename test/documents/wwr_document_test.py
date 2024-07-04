import pytest
import pandas as pd
from regulations_rag.document import Document
from test.reference_checker_samples import SimpleReferenceChecker
from regulations_rag.regulation_table_of_content import StandardTableOfContent
from .wrr_document import WRR

@pytest.fixture
def wrr_document():
    return WRR()

def test_get_text(wrr_document):
    text = wrr_document.get_text("all")
    expected_text = ""
    assert text == expected_text
    
    text = wrr_document.get_text("1.2")
    expected_text = '# 1 Navigating Whale Rock Ridge\n\n## 1.2 To Main Gate\n\nTurn left out driveway. Road turns left. At the first stop street, turn right. Proceed to Gate'
    assert text == expected_text

    text = wrr_document.get_text("1.2", add_markdown_decorators=False)
    expected_text = '1 Navigating Whale Rock Ridge\n1.2 To Main Gate\nTurn left out driveway. Road turns left. At the first stop street, turn right. Proceed to Gate'
    assert text == expected_text

    text = wrr_document.get_text("1.2", add_markdown_decorators=False, add_headings=False)
    expected_text = '1.2 To Main Gate\nTurn left out driveway. Road turns left. At the first stop street, turn right. Proceed to Gate'
    assert text == expected_text
    
    text = wrr_document.get_text("1", add_markdown_decorators=False, section_only=True)
    expected_text = '1 Navigating Whale Rock Ridge\nWhale Rock Ridge is a large complex. Here are directions to help you[^1].\n\n  \n[^1]: Directions from 11 Turnstone'
    assert text == expected_text

def test_get_heading(wrr_document):
    heading = wrr_document.get_heading("1")
    expected_heading = "1 Navigating Whale Rock Ridge"
    assert heading == expected_heading

def test_get_toc(wrr_document):
    toc = wrr_document.get_toc()
    assert toc.root.name == "Navigating Whale Rock Ridge"

def test_extract_footnotes(wrr_document):
    text = "Here are directions to help you[^1].\n[^1]: Directions from 11 Turnstone"
    footnote_pattern = r'^\[\^\d+\]\:'
    footnotes, remaining_text = wrr_document._extract_footnotes(text, footnote_pattern)
    assert footnotes == ["[^1]: Directions from 11 Turnstone"]
    assert remaining_text == "Here are directions to help you[^1]."

def test_format_line(wrr_document):
    row = {"section_reference": "1", "heading": True}
    text_extract = "Navigating Whale Rock Ridge"
    formatted_line = wrr_document._format_line(row, text_extract, True)
    expected_line = "# 1 Navigating Whale Rock Ridge\n\n"
    assert formatted_line == expected_line

def test_format_text_and_footnotes(wrr_document):
    text = "Whale Rock Ridge is a large complex."
    footnotes = ["[^1]: Directions from 11 Turnstone"]
    formatted_text = wrr_document._format_text_and_footnotes(text, footnotes)
    expected_text = "Whale Rock Ridge is a large complex.\n\n  \n[^1]: Directions from 11 Turnstone"
    assert formatted_text == expected_text

def test_get_text_and_footnotes(wrr_document):
    text, footnotes = wrr_document.get_text_and_footnotes("1.1")
    expected_text = '# 1 Navigating Whale Rock Ridge\n\n## 1.1 To West Gate\n\nTurn right out driveway. At the traffic circle, take the first exit. Proceed to West Gate\n\n'
    expected_footnotes = []
    assert text == expected_text
    assert footnotes == expected_footnotes

def test_get_text_and_footnotes_with_footnotes(wrr_document):
    text, footnotes = wrr_document.get_text_and_footnotes("1", add_markdown_decorators=False, section_only=True)
    expected_text = '1 Navigating Whale Rock Ridge\nWhale Rock Ridge is a large complex. Here are directions to help you[^1].\n'
    expected_footnotes = ['[^1]: Directions from 11 Turnstone']
    assert text == expected_text
    assert footnotes == expected_footnotes
