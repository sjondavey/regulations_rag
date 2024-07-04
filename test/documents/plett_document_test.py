import pytest
from .plett_document import Plett

@pytest.fixture
def plett_document():
    return Plett()

def test_get_text(plett_document):
    #something that is not a valid reference
    text = plett_document.get_text("all")
    expected_text = ""
    assert text == expected_text
    
    text = plett_document.get_text("A.2(A)")
    expected_text = '# A.2 Directions\n\n## A.2(A) To the Gym\n\n### A.2(A)(i) From West Gate (see 1.1)\n\nTurn left into Longships Drive and right at the T-junction into Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\n\n### A.2(A)(ii) From Main Gate (see 1.2)\n\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym\n\n### A.2(A)(iii) From South Gate (see 1.3)\n\nTurn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym'
    assert text == expected_text

#     text = plett_document.get_text("1.2", add_markdown_decorators=False)
#     expected_text = '1 Navigating Whale Rock Ridge\n1.2 To Main Gate\nTurn left out driveway. Road turns left. At the first stop street, turn right. Proceed to Gate'
#     assert text == expected_text

#     text = plett_document.get_text("1.2", add_markdown_decorators=False, add_headings = False)
#     expected_text = '1.2 To Main Gate\nTurn left out driveway. Road turns left. At the first stop street, turn right. Proceed to Gate'
#     assert text == expected_text
    
#     text = plett_document.get_text("1", add_markdown_decorators=False, section_only = True)
#     expected_text = '1 Navigating Whale Rock Ridge\nWhale Rock Ridge is a large complex. Here are directions to help you[^1].\n\n  \n[^1]: Directions from 11 Turnstone'
#     assert text == expected_text
    

# def test_get_heading(plett_document):
#     heading = plett_document.get_heading("1")
#     expected_heading = "1 Navigating Whale Rock Ridge"
#     assert heading == expected_heading

# def test_get_toc(plett_document):
#     toc = plett_document.get_toc()
#     assert toc.root.name == "Navigating Whale Rock Ridge"

# def test_extract_footnotes(plett_document):
#     text = "Here are directions to help you[^1].\n[^1]: Directions from 11 Turnstone"
#     footnote_pattern = r'^\[\^\d+\]\:'
#     footnotes, remaining_text = plett_document._extract_footnotes(text, footnote_pattern)
#     assert footnotes == ["[^1]: Directions from 11 Turnstone"]
#     assert remaining_text == "Here are directions to help you[^1]."

# def test_format_line(plett_document):
#     row = {"section_reference": "1", "heading": True}
#     text_extract = "Navigating Whale Rock Ridge"
#     formatted_line = plett_document._format_line(row, text_extract, True)
#     expected_line = "# 1 Navigating Whale Rock Ridge\n\n"
#     assert formatted_line == expected_line

# def test_format_text_and_footnotes(plett_document):
#     text = "Whale Rock Ridge is a large complex."
#     footnotes = ["[^1]: Directions from 11 Turnstone"]
#     formatted_text = plett_document._format_text_and_footnotes(text, footnotes)
#     expected_text = "Whale Rock Ridge is a large complex.\n\n  \n[^1]: Directions from 11 Turnstone"
#     assert formatted_text == expected_text