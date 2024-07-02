import re
import pandas as pd
from regulations_rag.document import Document
from test.reference_checker_samples import SimpleReferenceChecker
from regulations_rag.regulation_table_of_content import StandardTableOfContent

# Defining test data
columns = ["section_reference", "heading", "text"]
document_as_list = [
    ["1", True, "Navigating Whale Rock Ridge"],
    ["1", False, "Whale Rock Ridge is a large complex. Here are directions to help you[^1].\n[^1]: Directions from 11 Turnstone"],
    ["1.1", True, "To West Gate"],
    ["1.1", False, "Turn right out driveway. At the traffic circle, take the first exit. Proceed to West Gate"],
    ["1.2", True, "To Main Gate"],
    ["1.2", False, "Turn left out driveway. Road turns left. At the first stop street, turn right. Proceed to Gate"],
    ["1.3", True, "To South Gate"],
    ["1.2", False, "Turn left out driveway. Road turns left. At the first stop street, turn left. Follow road to Gate"]
]
document_as_df = pd.DataFrame(document_as_list, columns=columns)

class WRR(Document):
    def __init__(self):

        reference_checker =  SimpleReferenceChecker()
        self.document_as_df = document_as_df

        document_name = "Navigating Whale Rock Ridge"
        super().__init__(document_name, reference_checker=reference_checker)

    def get_text(self, section_reference, add_markdown_decorators = True, add_headings = True, section_only = False):               
        text, footnotes = super().get_text_and_footnotes(section_reference, add_markdown_decorators, add_headings, section_only)
        return super()._format_text_and_footnotes(text, footnotes)

    def get_heading(self, section_reference, add_markdown_decorators = False):
        return super().get_heading(section_reference, add_markdown_decorators)

    def get_toc(self):
        return StandardTableOfContent(root_node_name = self.name, reference_checker = self.reference_checker, regulation_df = self.document_as_df)

    def _extract_footnotes(self, text, footnote_pattern):
        return super()._extract_footnotes(text, footnote_pattern)

    def _format_line(self, row, text_extract, add_markdown_decorators):
        return super()._format_line(row, text_extract, add_markdown_decorators)

    def get_text_and_footnotes(self, section_reference, add_markdown_decorators = True, add_headings = True, section_only = False):
        return super().get_text_and_footnotes(section_reference)

    def _format_text_and_footnotes(self, text, footnotes):
        return super()._format_text_and_footnotes(text, footnotes)

