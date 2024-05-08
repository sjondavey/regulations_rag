import pandas as pd
import re
from gdpr_rag.document import Document
from regulations_rag.regulation_reader import  load_regulation_data_from_files

from gdpr_rag.gdpr_reference_checker import GDPRReferenceChecker

class GDPR(Document):
    def __init__(self, path_to_manual_as_csv_file = "./inputs/documents/gdpr.csv"):
        reference_checker = GDPRReferenceChecker()


        self.document_as_df = load_csv_data(path_to_file = path_to_manual_as_csv_file)

        document_name = "General Data Protection Regulation"
        super().__init__(document_name, reference_checker=reference_checker)

    def check_columns(self):
        expected_columns = ["chapter_number", "chapter_heading", "section_number", "section_heading", "article_number", "article_heading", "major_reference", "minor_reference", "content", "section_reference"] # this is a minimum - it could contain more

        actual_columns = self.document_as_df.columns.to_list()
        for column in expected_columns:
            if column not in actual_columns:
                print(f"{column} not in the DataFrame version of the manual")
                return False
        return True


    def get_text(self, section_reference):

        if section_reference == "":
            subframe = self.document_as_df
        else:
            # if not self.reference_checker.is_valid(section_reference):
            #     raise AttributeError(f"{section_reference} is not a valid index")

            pattern = r'^\d{1,2}$' # there is a special case here were if I ask for "2" for example, the "startswith()" query will also include 21, 22 etc
            if re.match(pattern, section_reference):
                tmp1 = self.document_as_df[self.document_as_df["section_reference"].str.startswith(section_reference + "(")] # use startswith to get children as well
                tmp2 = self.document_as_df[self.document_as_df["section_reference"] == section_reference] 
                subframe = pd.concat([tmp1, tmp2]).sort_index()
            else:
                subframe = self.document_as_df[self.document_as_df["section_reference"].str.startswith(section_reference)] # use startswith to get children as well
            parent_reference = self.reference_checker.get_parent_reference(section_reference)
            while parent_reference:
                parent_df = self.document_as_df[self.document_as_df["section_reference"] == (parent_reference)] # use equality to get only the lines for the parent
                subframe = pd.concat([subframe, parent_df]).sort_index()
                parent_reference = self.reference_checker.get_parent_reference(parent_reference)

            if len(subframe) == 0:
                return ""

        formatted_regulation = ""
        # if include_section_and_chapter:
        #     formatted_regulation = f"Chapter {subframe.iloc[0]['chapter_number']} {subframe.iloc[0]['chapter_heading']}\n"  
        #     if subframe.iloc[0]['section_number']:
        #         formatted_regulation = formatted_regulation + f"Section {subframe.iloc[0]['section_number']} {subframe.iloc[0]['section_heading']}\n"  

        #formatted_regulation = formatted_regulation + f"Article {subframe.iloc[0]['article_number']} {subframe.iloc[0]['article_heading']}\n"  
        formatted_regulation = formatted_regulation + f"{subframe.iloc[0]['article_number']} {subframe.iloc[0]['article_heading']}\n"  
        for index, row in subframe.iterrows():
            line = row["content"] + "\n"
            if row["minor_reference"]:
                line = 2 * 4 * " " + f"({row['minor_reference']}) " + line
            elif row["major_reference"]:
                line = 1 * 4 * " " + f"{row['major_reference']}. " + line
            else:
                line = line

            formatted_regulation = formatted_regulation + line

        return formatted_regulation


    # Note: This method will not work correctly if empty values in the dataframe are NaN as is the case when loading
    #       a dataframe form a file without the 'na_filter=False' option. You should ensure that the dataframe does 
    #       not have any NaN value for the text fields. Try running self.document_as_df.isna().any().any() as a test before you get here
    def get_heading(self, section_reference):
        if not self.reference_checker.is_valid(section_reference):
            raise AttributeError(f"{section_reference} is not a valid index")

        pattern = r'^\d{1,2}$' # there is a special case here were if I ask for "2" for example, the "startswith()" query will also include 21, 22 etc
        if re.match(pattern, section_reference):
            tmp1 = self.document_as_df[self.document_as_df["section_reference"].str.startswith(section_reference + "(")] # use startswith to get children as well
            tmp2 = self.document_as_df[self.document_as_df["section_reference"] == section_reference] 
            subframe = pd.concat([tmp1, tmp2]).sort_index()
        else:
            subframe = self.document_as_df[self.document_as_df["section_reference"].str.startswith(section_reference)] # use startswith to get children as well
        parent_reference = self.reference_checker.get_parent_reference(section_reference)
        while parent_reference:
            parent_df = self.document_as_df[self.document_as_df["section_reference"] == (parent_reference)] # use equality to get only the lines for the parent
            subframe = pd.concat([subframe, parent_df]).sort_index()
            parent_reference = self.reference_checker.get_parent_reference(parent_reference)

        if len(subframe) == 0:
            return ""

        formatted_regulation = ""
        formatted_regulation = f"Chapter {subframe.iloc[0]['chapter_number']} {subframe.iloc[0]['chapter_heading']}."  
        if subframe.iloc[0]['section_number']:
            formatted_regulation = formatted_regulation + f" Section {subframe.iloc[0]['section_number']} {subframe.iloc[0]['section_heading']}."  

        formatted_regulation = formatted_regulation + f" Article {subframe.iloc[0]['article_number']} {subframe.iloc[0]['article_heading']}."  

        return formatted_regulation


