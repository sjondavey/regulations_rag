from abc import ABC, abstractmethod
from regulations_rag.reference_checker import ReferenceChecker

import pandas as pd 

class RegulationReader(ABC):
    def __init__(self, reference_checker):
        self.reference_checker = reference_checker

    @abstractmethod
    def get_regulation_heading(section_reference):
        pass

    @abstractmethod
    def get_regulation_detail(section_reference):
        pass



class TESTReader(RegulationReader):
    def __init__(self, reference_checker, regulation_df):
        super().__init__(reference_checker=reference_checker)
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
    def get_regulation_heading(self, section_reference):
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
    def get_regulation_detail(self, section_reference):
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

