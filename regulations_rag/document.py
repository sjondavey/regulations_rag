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

    # @abstractmethod    
    # def get_text(self, section_reference, add_markdown_decorators = True, footnote_pattern = r'^\[\^\d+\]\:'):
    #     '''
    #     NOTE: When used with the Table of Content to break up a document into chunks, the call to get_text("") should return the entire text of the document
    #     '''
    #     pass

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



class TESTDocument(Document):
    def __init__(self, reference_checker, regulation_df):
        super().__init__(document_name = "CEMAD", reference_checker=reference_checker)
        if regulation_df.isna().any().any():
            raise AttributeError("The input DataFrame contains NaNs which will cause issues")
        self.regulation_df = regulation_df
        if not self.check_columns():
            raise AttributeError("The input DataFrame does not have the correct columns")

    def check_columns(self):
        ''' 
            'indent'            : number of indents before the line starts - to help interpret it (i) is the letter or the Roman numeral (for example)
            'reference'         : the part of the section_reference at the start of the line. Can be blank
            'text'              : the text on the line excluding the 'reference' and any special text (identifying headings, page number etc)
            'heading'           : boolean identifying the text as as (sub-) section heading
            'section_reference' : the full reference. Starting at the root node and ending with the value in 'reference'
        '''
        expected_columns = ['indent', 'reference', 'text', 'heading', 'section_reference'] # this is a minimum - it could contain more

        actual_columns = self.regulation_df.columns.to_list()
        for column in expected_columns:
            if column not in actual_columns:
                print(f"{column} not in the DataFrame version of the manual")
                return False
        return True

    # Note: This method will not work correctly if empty values in the dataframe are NaN as is the case when loading
    #       a dataframe form a file without the 'na_filter=False' option. You should ensure that the dataframe does 
    #       not have any NaN value for the text fields. Try running self.regulation_df.isna().any().any() as a test before you get here
    def get_heading(self, section_reference):
        if not self.reference_checker.is_valid(section_reference):
            return "Not a valid reference"
        remaining_reference = section_reference
        heading = ""
        
        while len(remaining_reference) > 0:
            tmp_df = self.regulation_df.loc[(self.regulation_df['section_reference'] == remaining_reference) & (self.regulation_df['heading'] == True)]
            if len(tmp_df) > 1:
                return f"There was more than one heading for the section reference {section_reference}"
            elif len(tmp_df) == 1:
                if heading == "":
                    heading = tmp_df.iloc[0]["reference"] + " " + tmp_df.iloc[0]["text"] + "."
                else:
                    heading = tmp_df.iloc[0]["reference"] + " " + tmp_df.iloc[0]["text"] + ". " + heading
            remaining_reference = self.reference_checker.get_parent_reference(remaining_reference)

        return heading


    # Note: This method will not work correctly if empty values in the dataframe are NaN as is the case when loading
    #       a dataframe form a file without the 'na_filter=False' option. You should ensure that the dataframe does 
    #       not have any NaN value for the text fields. Try running self.regulation_df.isna().any().any() as a test before you get here
    def get_text(self, section_reference):
        # if not self.reference_checker.is_valid(section_reference):
        #     return "The reference did not conform to this documents standard"
        text = ''
        terminal_text_df = self.regulation_df[self.regulation_df['section_reference'].str.startswith(section_reference)]
        if len(terminal_text_df) == 0:
            return f"No section could be found with the reference {section_reference}"
        terminal_text_index = terminal_text_df.index[0]
        terminal_text_indent = 0 # terminal_text_df.iloc[0]['indent']
        for index, row in terminal_text_df.iterrows():
            number_of_spaces = (row['indent'] - terminal_text_indent) * 4
            #set the string "line" to start with the number of spaces
            line = " " * number_of_spaces
            if pd.isna(row['reference']) or row['reference'] == '':
                line = line + row['text']
            else:
                if pd.isna(row['text']):
                    line = line + row['reference']
                else:     
                    line = line + row['reference'] + " " + row['text']
            if text != "":
                text = text + "\n"
            text = text + line

        if section_reference != '': #i.e. there is a parent
            parent_reference = self.reference_checker.get_parent_reference(section_reference)
            all_conditions = ""
            all_qualifiers = ""
            while parent_reference != "":
                parent_text_df = self.regulation_df[self.regulation_df['section_reference'] == parent_reference]
                conditions = ""
                qualifiers = ""
                for index, row in parent_text_df.iterrows():
                    if index < terminal_text_index:
                        number_of_spaces = (row['indent'] - terminal_text_indent) * 4
                        if conditions != "":
                            conditions = conditions + "\n"
                        conditions = conditions + " " * number_of_spaces
                        if (row['reference'] == ''):
                            conditions = conditions + row['text']
                        else:
                            conditions = conditions + row['reference'] + " " +  row['text']
                    else:
                        number_of_spaces = (row['indent'] - terminal_text_indent) * 4
                        if (qualifiers != ""):
                            qualifiers = qualifiers + "\n"
                        qualifiers = qualifiers + " " * number_of_spaces
                        if (row['reference'] == ''):
                            qualifiers = qualifiers + row['text']
                        else:
                            qualifiers = qualifiers + row['reference'] + " " + row['text']

                if conditions != "":
                    all_conditions = conditions + "\n" + all_conditions
                if qualifiers != "":
                    all_qualifiers = all_qualifiers + "\n" + qualifiers
                parent_reference = self.reference_checker.get_parent_reference(parent_reference)

            if all_conditions != "":
                text = all_conditions +  text
            if all_qualifiers != "":
                text = text + all_qualifiers

        return text

    def get_toc(self):
        # create the dataframe using chapters and articles
        gdpr_data_for_tree = []
        chapter_number = ""
        article_number = 0

        for index, row in self.document_as_df.iterrows():
            if row["chapter_number"] != chapter_number:
                chapter_number = row["chapter_number"]
                section_number = ""
                gdpr_data_for_tree.append([row["chapter_number"], True, row["chapter_heading"]])

            if row["article_number"] != article_number:
                article_number = row["article_number"]
                gdpr_data_for_tree.append([f'{chapter_number}.{row["article_number"]}', True, row["article_heading"]])

        gdpr_df_for_tree = pd.DataFrame(gdpr_data_for_tree, columns = ["section_reference", "heading", "text"])

        return StandardTableOfContent(root_node_name = self.name, index_checker = self.GDPRToCReferenceChecker(), regulation_df = gdpr_df_for_tree)

    # Reference checker for TOC only
    class GDPRToCReferenceChecker(ReferenceChecker):
        def __init__(self):
            exclusion_list = []

            gdpr_index_patterns = [
                r'^\b(I|II|III|IV|V|VI|VII|VIII|IX|X|XI)\b',
                r'^\.\d{1,2}',   # Matches numbers, excluding leading zeros. - Article Number
            ]
            
            # ^Article : Matches the beginning of the string, followed by "Article ". - mandatory
            # (\d{1,2}): Captures a one or two digit number immediately following "Article ". - mandatory
            # (?:\((\d{1,2})\))?: An optional non-capturing group that contains a capturing group for a one or two digit number enclosed in parentheses. The entire group is made optional by ?, so it matches 0 or 1 times.
            # (?:\(([a-z])\))?: Another optional non-capturing group that contains a capturing group for a single lowercase letter enclosed in parentheses. This part is also optional.
            text_pattern = r'((I|II|III|IV|V|VI|VII|VIII|IX|X|XI))(?:\((\d{1,2})\))?'

            super().__init__(regex_list_of_indices = gdpr_index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)


