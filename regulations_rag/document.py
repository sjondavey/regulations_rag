import os
import re
import pandas as pd
from abc import ABC, abstractmethod
from regulations_rag.reference_checker import ReferenceChecker


class Document(ABC):
    def __init__(self, document_name, reference_checker):
        self.name = document_name
        self.reference_checker = reference_checker

    @abstractmethod
    def get_text(self, section_reference, add_markdown_decorators=True, add_headings=True, section_only=False):
        """
        Abstract method to get the text of the document.
        When used with the Table of Content to break up a document into chunks, 
        the call to get_text("") should return the entire text of the document.
        """
        pass

    @abstractmethod
    def get_toc(self):
        """Abstract method to get the table of contents of the document."""
        pass

    def _extract_footnotes(self, text, footnote_pattern):
        """
        Extract footnotes from the text based on the provided footnote pattern.

        Args:
            text (str): The text from which footnotes need to be extracted.
            footnote_pattern (str): The regex pattern to identify footnotes.

        Returns:
            tuple: A tuple containing a list of footnotes and the remaining text.
        """
        footnotes, remaining_text = [], []

        if not footnote_pattern:
            return footnotes, text

        for line in text.split('\n'):
            if re.match(footnote_pattern, line):
                footnotes.append(line)
            else:
                remaining_text.append(line)

        return footnotes, '\n'.join(remaining_text)


    def _format_line(self, row, text_extract, add_markdown_decorators):
        """
        Format a line of text, adding markdown decorators and section references if necessary.

        Args:
            row (pd.Series): A row from the document DataFrame.
            text_extract (str): The text to format.
            add_markdown_decorators (bool): Whether to add markdown decorators.

        Returns:
            str: The formatted text.
        """
        if row["heading"]:
            depth = len(self.reference_checker.split_reference(row["section_reference"]))
            line = f"{'#' * depth} {row['section_reference']} {text_extract}\n\n" if add_markdown_decorators else f"{row['section_reference']} {text_extract}\n"
        else:
            line = f"{text_extract}\n\n" if add_markdown_decorators and not text_extract.startswith("|") else f"{text_extract}\n"
        return line


    def get_text_and_footnotes(self, section_reference, add_markdown_decorators=True, add_headings=True, section_only=False):
        """
        Get the text and footnotes for a given section reference.

        Args:
            section_reference (str): The section reference to get the text and footnotes for.
            add_markdown_decorators (bool): Whether to add markdown decorators.
            add_headings (bool): Whether to add headings.
            section_only (bool): Whether to include only the section text.

        Returns:
            tuple: A tuple containing the text and a list of footnotes.
        """
        if section_reference and not self.reference_checker.is_valid(section_reference):
            return "", []

        footnote_pattern = r'^\[\^\d+\]\:'
        text, all_footnotes = "", []

        subset = self.document_as_df if not section_reference else self.document_as_df[self.document_as_df["section_reference"] == section_reference]

        if subset.empty:
            return "", []

        for _, row in subset.iterrows():
            footnotes, text_extract = self._extract_footnotes(row["text"], footnote_pattern)
            all_footnotes.extend(footnotes)
            text += self._format_line(row, text_extract.strip(), add_markdown_decorators)
            if text.strip().endswith("|") and not text_extract.strip().startswith("|"):
                text += "\n"

        if add_headings:
            build_up, buildup_footnotes = "", []
            parent = self.reference_checker.get_parent_reference(section_reference)
            while parent:
                subset = self.document_as_df[self.document_as_df["section_reference"] == parent]
                for _, row in subset.iloc[::-1].iterrows():
                    if row["heading"]:
                        footnotes, text_extract = self._extract_footnotes(row["text"], footnote_pattern)
                        buildup_footnotes.extend(footnotes)
                        build_up = self._format_line(row, text_extract.strip(), add_markdown_decorators) + build_up
                parent = self.reference_checker.get_parent_reference(parent)

            text = build_up + text
            all_footnotes = buildup_footnotes + all_footnotes

        if section_reference and not section_only:
            toc = self.get_toc()
            children_nodes = toc.get_node(section_reference).children
            for child_node in children_nodes:
                child_section_reference = child_node.full_node_name
                if child_section_reference:
                    child_text, child_footnotes = self.get_text_and_footnotes(
                        child_section_reference, add_markdown_decorators, add_headings=False, section_only=section_only
                    )
                    text += child_text
                    all_footnotes.extend(child_footnotes)

        return text, all_footnotes

    def _format_text_and_footnotes(self, text, footnotes):
        """
        Format text and footnotes into a single string.

        Args:
            text (str): The main text.
            footnotes (list): A list of footnotes.

        Returns:
            str: The formatted text with footnotes.
        """
        formatted_text = text.strip() + "\n\n"
        for footnote in footnotes:
            formatted_text += "  \n" + footnote.strip()
        return formatted_text.strip()


    def get_heading(self, section_reference, add_markdown_decorators=False):
        """
        Get the heading text for a given section reference.

        Args:
            section_reference (str): The section reference to get the heading for.
            add_markdown_decorators (bool): Whether to add markdown decorators.

        Returns:
            str: The heading text.
        """
        footnote_pattern = r'^\[\^\d+\]\:'
        if not self.reference_checker.is_valid(section_reference):
            return ""

        text, all_footnotes = "", []
        subset = self.document_as_df[self.document_as_df["section_reference"] == section_reference]

        if not subset.empty:
            for _, row in subset.iterrows():
                if row["heading"]:
                    footnotes, text_extract = self._extract_footnotes(row["text"], footnote_pattern)
                    formatted_text = self._format_line(row, text_extract.strip(), add_markdown_decorators)
                    if formatted_text:
                        all_footnotes.extend(footnotes)
                        text += formatted_text

            parent = self.reference_checker.get_parent_reference(section_reference)
            build_up = ""
            while parent:
                subset = self.document_as_df[self.document_as_df["section_reference"] == parent]
                for _, row in subset.iloc[::-1].iterrows():
                    if row["heading"]:
                        footnotes, text_extract = self._extract_footnotes(row["text"], footnote_pattern)
                        formatted_text = self._format_line(row, text_extract.strip(), add_markdown_decorators)
                        if formatted_text:
                            all_footnotes.extend(footnotes)
                            build_up = formatted_text + build_up
                parent = self.reference_checker.get_parent_reference(parent)

            text = build_up + text

        return text.strip("\n")