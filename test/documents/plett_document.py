import re
import pandas as pd
from regulations_rag.document import Document
from test.reference_checker_samples import TESTReferenceChecker
from regulations_rag.regulation_table_of_content import StandardTableOfContent

# Defining test data
columns = ["section_reference", "heading", "text"]
document_as_list = [
    ["A.", True, "Navigating Plettenberg Bay"],
    ["A.", False, "Plett is a small town. Here are directions to help you[^1].\n[^1]: Directions from Whale Rock Ridge"],
    ["A.1", True, "Definitions"],
    ["A.1(A)", False, "The Gym: The Health and Fitness Center on Piesang Valley Road"],
    ["A.1(B)", False, "The Robberg Nature Reserve: The Cape Nature park at the end of the Robberg Peninsula"],
    ["A.2", True, "Directions"],
    ["A.2(A)", True, "To the Gym"],
    ["A.2(A)(i)", True, "From West Gate (see 1.1)"],
    ["A.2(A)(i)", False, "Turn left into Longships Drive and right at the T-junction into Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym"],
    ["A.2(A)(ii)", True, "From Main Gate (see 1.2)"],
    ["A.2(A)(ii)", False, "Turn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym"],
    ["A.2(A)(iii)", True, "From South Gate (see 1.3)"],
    ["A.2(A)(iii)", False, "Turn right Whale Rock Drive. At the T-junction turn right into Robberg Road. Turn left into Green Point Avenue and arrive at the gym"],
    ["A.2(B)", True, "To Robberg Nature Reserve"],
    ["A.2(B)(i)", True, "From West Gate (see 1.1)"],
    ["A.2(B)(i)", False, "Turn left into Longships Drive and left at the T-junction into Whale Rock Drive. Continue straight to Robberg Nature Reserve"],
    ["A.2(B)(ii)", True, "From Main Gate (see 1.2)"],
    ["A.2(B)(ii)", False, "Turn left into Whale Rock Drive. Continue straight to Robberg Nature Reserve"],
    ["A.2(B)(iii)", True, "From South Gate (see 1.3)"],
    ["A.2(B)(iii)", False, "Turn left into Whale Rock Drive. Continue straight to Robberg Nature Reserve"],
]
document_as_df = pd.DataFrame(document_as_list, columns=columns)

class Plett(Document):
    def __init__(self):
        reference_checker = TESTReferenceChecker()
        self.document_as_df = document_as_df
        document_name = "Navigating Plett"
        super().__init__(document_name, reference_checker=reference_checker)

    def get_text(self, section_reference, add_markdown_decorators=True, add_headings=True, section_only=False):
        """
        Get the text for a given section reference.
        
        Args:
            section_reference (str): The section reference to get the text for.
            add_markdown_decorators (bool): Whether to add markdown decorators.
            add_headings (bool): Whether to add headings.
            section_only (bool): Whether to include only the section text.
        
        Returns:
            str: The formatted text and footnotes.
        """
        text, footnotes = super().get_text_and_footnotes(section_reference, add_markdown_decorators, add_headings, section_only)
        return super()._format_text_and_footnotes(text, footnotes)

    def get_heading(self, section_reference, add_markdown_decorators=False):
        """
        Get the heading for a given section reference.
        
        Args:
            section_reference (str): The section reference to get the heading for.
            add_markdown_decorators (bool): Whether to add markdown decorators.
        
        Returns:
            str: The heading text.
        """
        return super().get_heading(section_reference, add_markdown_decorators)

    def get_toc(self):
        """
        Get the table of contents for the document.
        
        Returns:
            StandardTableOfContent: The table of contents.
        """
        return StandardTableOfContent(root_node_name=self.name, reference_checker=self.reference_checker, regulation_df=self.document_as_df)

