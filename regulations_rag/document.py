import os
import re
import pandas as pd
from abc import ABC, abstractmethod
from regulations_rag.reference_checker import ReferenceChecker
from regulations_rag.regulation_table_of_content import StandardTableOfContent



class Document(ABC):
    def __init__(self, document_name, reference_checker):
        self.name = document_name 
        self.reference_checker = reference_checker

    @abstractmethod    
    def get_text(self, section_reference, add_markdown_decorators = True, add_headings = True, section_only = False):
        '''
        NOTE: When used with the Table of Content to break up a document into chunks, the call to get_text("") should return the entire text of the document
        '''
        pass

    @abstractmethod
    def get_toc(self):
        pass

    def _extract_footnotes(self, text, footnote_pattern):
        ''' 
        This assumes that the DataFrame that stores the document is formatted such that the footnotes are saved with the paragraph of text that
        refers to them. This method extracts the footnote(s) from a section of text so the text and footnotes for a section can be re-assembled 
        separately
        ''' 
        footnotes = []
        remaining_text = []

        if footnote_pattern == "":
            return footnotes, text

        lines = text.split('\n')
        for line in lines:
            if re.match(footnote_pattern, line):
                footnotes.append(line)
            else:
                remaining_text.append(line)

        text = '\n'.join(remaining_text)
        return footnotes, text


    def _format_line(self, row, text_extract, add_markdown_decorators):
        ''' 
        This method 
            - adds the section_reference to any line labelled as a heading
            - adds the markdown # to headings if add_markdown_decorators == True
            - ends the line with 
                - "\n" if add_markdown_decorators = False
                - "\n\n" if add_markdown_decorators = True
        ''' 
        if row["heading"]:
            line = ""
            if add_markdown_decorators:
                depth = self.reference_checker.split_reference(row["section_reference"])
                line = "#" * len(depth)  + " " +  row["section_reference"] + " " + text_extract + "\n\n"
            else:
                line = row["section_reference"] + " " + text_extract + "\n"

            return line
            
        else:    
            if add_markdown_decorators and not text_extract.startswith("|"): # if "|" it is a table
                return text_extract + "\n\n"
            else:
                return text_extract + "\n"


    def get_text_and_footnotes(self, section_reference, add_markdown_decorators = True, add_headings = True, section_only = False):
        ''' 
        A commonly used pattern for traditional documents where we select only the text "below" and the headings and stop when we reach the next heading
        - even if that is a sub - heading. We also do not select any text from "above" the section reference. 
        Contrast this with a pattern used in legal documents where we would also select the text, all sub-sections and any text "above" the section reference
        back to the root node
        '''
        if section_reference != "" and not self.reference_checker.is_valid(section_reference):
            return "" 
        else:
            footnote_pattern = r'^\[\^\d+\]\:'

            text = ""
            all_footnotes = []
            if section_reference == "":
                subset = self.document_as_df
            else:
                subset = self.document_as_df[self.document_as_df["section_reference"] == section_reference] 

            if len(subset) == 0:
                return "", []
            for index, row in subset.iterrows():
                footnotes, text_extract = self._extract_footnotes(row["text"], footnote_pattern)
                text_extract = text_extract.strip()
                all_footnotes = all_footnotes + footnotes
                if text.strip().endswith("|") and not text_extract.strip().startswith("|"): # we are done with the table
                    text += "\n"    
                text += self._format_line(row, text_extract, add_markdown_decorators)

            if add_headings:
                build_up = ""
                buildup_footnotes = []
                parent = self.reference_checker.get_parent_reference(section_reference)
                while parent != "":
                    subset = self.document_as_df[self.document_as_df["section_reference"] == parent]
                    for index, row in subset[::-1].iterrows(): # backwards
                        if row["heading"]:
                            footnotes, text_extract = self._extract_footnotes(row["text"], footnote_pattern)
                            text_extract = text_extract.strip()
                            buildup_footnotes = buildup_footnotes + footnotes
                            build_up = self._format_line(row, text_extract, add_markdown_decorators) + build_up

                    parent = self.reference_checker.get_parent_reference(parent)

                text = build_up + text
                all_footnotes = buildup_footnotes + all_footnotes

            if section_reference != "" and not section_only:
                children = ""
                children_footnotes = []
                toc = self.get_toc()
                children_nodes = toc.get_node(section_reference).children
                for child_node in children_nodes:
                    child_section_reference = child_node.full_node_name # could be empty ?
                    if child_section_reference != "":
                        child_text, child_footnotes = self.get_text_and_footnotes(child_section_reference, add_markdown_decorators, add_headings = False, section_only = section_only)
                        text = text + child_text
                        all_footnotes = all_footnotes + child_footnotes

        return text, all_footnotes

    def _format_text_and_footnotes(self, text, footnotes):
        text = text.strip() + "\n\n"
        for footnote in footnotes:
            text = text + "  \n" + footnote.strip() 

        return text.strip()


    def get_heading(self, section_reference, add_markdown_decorators = False):
        '''
        Some heading have footnotes :-(
        No markdown formatting is added to the heading text
        '''
        footnote_pattern = r'^\[\^\d+\]\:'
        if not self.reference_checker.is_valid(section_reference):
            return "" 
        else:
            text = ""
            all_footnotes = []
            subset = self.document_as_df[self.document_as_df["section_reference"] == (section_reference)]
            if len(subset) > 0:
                for index, row in subset.iterrows():
                    footnotes, text_extract = self._extract_footnotes(row["text"], footnote_pattern)
                    text_extract = text_extract.strip()
                    if row['heading']:
                        formatted_text = self._format_line(row, text_extract, add_markdown_decorators)
                        if formatted_text:
                            all_footnotes = all_footnotes + footnotes
                            text += formatted_text

                parent = self.reference_checker.get_parent_reference(section_reference)
                build_up = ""
                while parent != "":
                    subset = self.document_as_df[self.document_as_df["section_reference"] == parent]
                    for index, row in subset[::-1].iterrows(): # backwards
                        if row['heading']:
                            footnotes, text_extract = self._extract_footnotes(row["text"], footnote_pattern)
                            text_extract = text_extract.strip()
                            formatted_text = self._format_line(row, text_extract, add_markdown_decorators)
                            if formatted_text:
                                all_footnotes = all_footnotes + footnotes
                                build_up = formatted_text + build_up

                    parent = self.reference_checker.get_parent_reference(parent)
                text = build_up + text
                # for footnote in all_footnotes: # Don't display the actual footnotes to the headings. The Markers like [^3] will still be in them
                #     text = text + "\n" + footnote

            return text.strip("\n")
