import logging
import pandas as pd

from regulations_rag.section_reference_checker import SectionReferenceChecker
from regulations_rag.data_in_dataframes import DataInDataFrames, load_csv_data, append_csv_data, load_parquet_data, append_parquet_data, load_data_from_files


"""
Creates and returns a SectionReferenceChecker instance used for testing.
"""
exclusion_list = ['Legal context', 'Introduction']
index_patterns = [
    r'^[A-Z]\.\d{0,2}',             # Matches capital letter followed by a period and up to two digits.
    r'^\([A-Z]\)',                  # Matches single capital letters within parentheses.
    r'^\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx|xxi|xxii|xxiii|xxiv|xxv|xxvi|xxvii)\)', # Matches Roman numerals within parentheses.
    r'^\([a-z]\)',                  # Matches single lowercase letters within parentheses.
    r'^\([a-z]{2}\)',               # Matches two lowercase letters within parentheses.
    r'^\((?:[1-9]|[1-9][0-9])\)',   # Matches numbers within parentheses, excluding leading zeros.
]    
text_pattern = r"[A-Z]\.\d{0,2}\([A-Z]\)\((?:i|ii|iii|iv|v|vi)\)\([a-z]\)\([a-z]{2}\)\(\d+\)"

section_reference_checker = SectionReferenceChecker(regex_list_of_indices=index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)



def check_definitions(path_to_definitions_as_parquet_file):
    df_definitions_all = pd.read_parquet(path_to_definitions_as_parquet_file, engine='pyarrow')
    assert len(df_definitions_all.columns) == 3

    expected_columns = ['definition', 'source', 'embedding']
    for column_heading in df_definitions_all.columns:
        assert column_heading in expected_columns

def check_index(path_to_index_as_parquet_file):
    df_text_all = pd.read_parquet(path_to_index_as_parquet_file, engine='pyarrow')
    assert len(df_text_all.columns) == 4
    expected_columns = ["section_reference", "text", "source", "embedding"]
    for column_heading in df_text_all.columns:
        assert column_heading in expected_columns


def test_data():
    # Make sure that when you load the manual, there are no NaN values
    manual = load_csv_data("./test/inputs/manual.csv")
    assert not manual.isna().any().any()

    path_to_definitions_as_parquet_file = "./test/inputs/definitions.parquet"
    check_definitions(path_to_definitions_as_parquet_file)

    path_to_index_as_parquet_file = "./test/inputs/index.parquet"
    check_index(path_to_index_as_parquet_file)


def test_load_csv_data():
    df_document = load_csv_data("./test/inputs/manual.csv")
    assert not df_document.isna().any().any()

def test_append_csv_data():
    df_document = load_csv_data("./test/inputs/manual.csv")
    l = len(df_document)
    df_document = append_csv_data("", df_document)
    assert len(df_document) == l
    assert not df_document.isna().any().any()
    df_document = append_csv_data("./test/inputs/manual_plus.csv", df_document)
    assert len(df_document) > l
    assert not df_document.isna().any().any()

def test_load_parquet_data():
    df_text_all = load_parquet_data("./test/inputs/definitions.parquet")
    assert not df_text_all.isna().any().any()

def test_append_parquet_data():
    df_text_all = load_parquet_data("./test/inputs/definitions.parquet")
    l = len(df_text_all)
    df_text_all = append_parquet_data("", df_text_all)
    assert len(df_text_all) == l
    df_text_all = append_parquet_data("./test/inputs/definitions_plus.parquet", df_text_all)
    assert len(df_text_all) > l
    assert not df_text_all.isna().any().any()

def test_load_data():
    user_type = "Authorised Dealer (AD)" 
    regulation_name = "\'Currency and Exchange Manual for Authorised Dealers\' (Manual or CEMAD)"

    path_to_manual_as_csv_file = "./test/inputs/manual.csv"
    path_to_definitions_as_parquet_file = "./test/inputs/definitions.parquet"
    path_to_index_as_parquet_file = "./test/inputs/index.parquet"
    path_to_additional_manual_as_csv_file = ""
    path_to_additional_definitions_as_parquet_file = ""
    path_to_additional_index_as_parquet_file = ""
    path_to_workflow_as_parquet = "./test/inputs/workflow.parquet"

    df_regulations, df_definitions, df_index, df_workflow = load_data_from_files(
                                path_to_manual_as_csv_file, path_to_additional_manual_as_csv_file, 
                                path_to_definitions_as_parquet_file, path_to_additional_definitions_as_parquet_file,
                                path_to_index_as_parquet_file, path_to_additional_index_as_parquet_file,
                                path_to_workflow_as_parquet)
    l_r = len(df_regulations)
    l_d = len(df_definitions)
    l_i = len(df_index)

    path_to_additional_manual_as_csv_file = ""
    path_to_additional_definitions_as_parquet_file = "./test/inputs/definitions_plus.parquet"
    path_to_additional_index_as_parquet_file = ""
    df_regulations, df_definitions, df_index, df_workflow = load_data_from_files(
                                path_to_manual_as_csv_file, path_to_additional_manual_as_csv_file, 
                                                path_to_definitions_as_parquet_file, path_to_additional_definitions_as_parquet_file,
                                                path_to_index_as_parquet_file, path_to_additional_index_as_parquet_file,
                                                path_to_workflow_as_parquet)
    assert len(df_regulations) == l_r
    assert len(df_definitions) > l_d
    assert len(df_index) == l_i


def test_filter_relevant_sections():
    # I am not using data that is loaded here, just testing a method but I need an object
    user_type = "Authorised Dealer (AD)" 
    regulation_name = "\'Currency and Exchange Manual for Authorised Dealers\' (Manual or CEMAD)"

    path_to_manual_as_csv_file = "./test/inputs/manual.csv"
    path_to_definitions_as_parquet_file = "./test/inputs/definitions.parquet"
    path_to_index_as_parquet_file = "./test/inputs/index.parquet"
    path_to_additional_manual_as_csv_file = ""
    path_to_additional_definitions_as_parquet_file = ""
    path_to_additional_index_as_parquet_file = ""
    path_to_workflow_as_parquet = "./test/inputs/workflow.parquet"

    df_regulations, df_definitions, df_index, df_workflow = load_data_from_files(
                                path_to_manual_as_csv_file, path_to_additional_manual_as_csv_file, 
                                path_to_definitions_as_parquet_file, path_to_additional_definitions_as_parquet_file,
                                path_to_index_as_parquet_file, path_to_additional_index_as_parquet_file,
                                path_to_workflow_as_parquet)

    data = DataInDataFrames(user_type = user_type, 
                            regulation_name = regulation_name, 
                            section_reference_checker = section_reference_checker, 
                            df_regulations = df_regulations, 
                            df_definitions = df_definitions, 
                            df_index = df_index, 
                            df_workflow = df_workflow)

    #data = src.data.load_data_from_folders(chat_for_ad = True, base_directory = ".", embeddings_directory = "") 

    output_columns = ["section_reference", "cosine_distance", "source", "text", "count"]
    test_data = []
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 0
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 1
    assert df_filtered_test_data.iloc[0]["count"] == 1
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)

    # Check that duplicate top values are filtered and the cosine distance is the minimum
    test_data.append(['A.1(A)(i)(a)', 0.2, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 1
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[0]["cosine_distance"] == 0.1
    # Check that the top and mode results are returned and the mode value is the lowest distance
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.3, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 2
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.2 # Note the order of the search_section is preserved so the mode should be second
    assert df_filtered_test_data.iloc[1]["count"] == 2

    # Check if there are no duplicate indexes that we still return multiple sections
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 2
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[0]["section_reference"] == 'A.1(A)(i)(a)'
    assert df_filtered_test_data.iloc[0]["cosine_distance"] == 0.1
    assert df_filtered_test_data.iloc[1]["section_reference"] == 'A.1(A)(i)(b)'
    assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.2
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.3, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 3
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[0]["section_reference"] == 'A.1(A)(i)(a)'
    assert df_filtered_test_data.iloc[0]["cosine_distance"] == 0.1
    assert df_filtered_test_data.iloc[1]["section_reference"] == 'A.1(A)(i)(b)'
    assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.2
    assert df_filtered_test_data.iloc[2]["section_reference"] == 'A.1(A)(i)(c)'
    assert df_filtered_test_data.iloc[2]["cosine_distance"] == 0.3
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.3, "question", "some text here"])
    test_data.append(['A.1(A)(i)(d)', 0.3, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 3
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[0]["section_reference"] == 'A.1(A)(i)(a)'
    assert df_filtered_test_data.iloc[0]["cosine_distance"] == 0.1
    assert df_filtered_test_data.iloc[1]["section_reference"] == 'A.1(A)(i)(b)'
    assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.2
    assert df_filtered_test_data.iloc[2]["section_reference"] == 'A.1(A)(i)(c)'
    assert df_filtered_test_data.iloc[2]["cosine_distance"] == 0.3


    # My logic should exclude the second most likely search item
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.3, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.4, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 2
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.3
    # mode plus replete
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.3, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.4, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.5, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.6, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 3
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[1]["section_reference"] == "A.1(A)(i)(c)"
    assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.3
    assert df_filtered_test_data.iloc[1]["count"] == 3
    assert df_filtered_test_data.iloc[2]["section_reference"] == "A.1(A)(i)(b)"
    assert df_filtered_test_data.iloc[2]["cosine_distance"] == 0.2
    assert df_filtered_test_data.iloc[2]["count"] == 2

    # mode plus two repletes
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.3, "question", "some text here"])
    test_data.append(['A.1(A)(i)(d)', 0.35, "question", "some text here"])
    test_data.append(['A.1(A)(i)(e)', 0.375, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.4, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.5, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.6, "question", "some text here"])
    test_data.append(['A.1(A)(i)(d)', 0.7, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 4
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    # Note the order of the search_section is preserved so the mode should be second
    assert df_filtered_test_data.iloc[3]["section_reference"] == "A.1(A)(i)(d)"
    assert df_filtered_test_data.iloc[3]["cosine_distance"] == 0.35
    # no unique mode
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.3, "question", "some text here"])
    test_data.append(['A.1(A)(i)(c)', 0.5, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.6, "question", "some text here"])
    test_data.append(['A.1(A)(i)(d)', 0.7, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = data.filter_relevant_sections(df_test_data)
    assert len(df_filtered_test_data) == 3
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)

# def test_cap_rag_section_token_length():
#     data = src.data.load_data_from_folders(chat_for_ad = True, base_directory = ".", embeddings_directory = "") 
#     test_data = []
#     test_data.append(['B.2(C)(ii)', 0.1, "question", "some text here"])
#     test_data.append(['B.2(C)(i)(a)', 0.2, "question", "some text here"])
#     test_data.append(['B.2(B)(i)', 0.3, "question", "some text here"])
#     test_data.append(['E.(D)', 0.5, "question", "some text here"])
#     test_data.append(['B.2(F)', 0.6, "question", "some text here"])
#     test_data.append(['G.(C)', 0.7, "question", "some text here"])

#     df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
#     df_capped_test_data = data.cap_rag_section_token_length(df_test_data, 10000)
#     assert len(df_capped_test_data) == 5 # hard cap in function

#     df_capped_test_data = data.cap_rag_section_token_length(df_test_data, 1000) # first entry has more that 1000
#     assert len(df_capped_test_data) == 1 

#     df_capped_test_data = data.cap_rag_section_token_length(df_test_data, 2000)
#     assert len(df_capped_test_data) == 2
    
