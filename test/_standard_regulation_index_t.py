import os
import logging
import pandas as pd

from regulations_rag.reference_checker import TESTReferenceChecker
from regulations_rag.regulation_reader import TESTReader, load_regulation_data_from_files, load_csv_data, append_csv_data
from regulations_rag.standard_regulation_index import StandardRegulationIndex, load_parquet_data, append_parquet_data, load_index_data_from_files



reference_checker = TESTReferenceChecker()


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

    key = key = os.getenv('excon_encryption_key')
    df_index = load_parquet_data("./test/inputs/index.parquet", key)
    assert df_index.iloc[0]['text'] == 'Export of gold jewellery by manufacturing jewellers' # check it is decrypted
    df_index_plus = append_parquet_data("./test/inputs/index.parquet", df_index, key)
    assert df_index_plus.iloc[-1]['text'] == 'Applications for the importation of gold' # check it is decrypted


def test_load_data():

    path_to_manual_as_csv_file = "./test/inputs/manual.csv"
    path_to_definitions_as_parquet_file = "./test/inputs/definitions.parquet"
    path_to_index_as_parquet_file = "./test/inputs/index.parquet"
    path_to_additional_manual_as_csv_file = ""
    path_to_additional_definitions_as_parquet_file = ""
    path_to_additional_index_as_parquet_file = ""
    path_to_workflow_as_parquet = "./test/inputs/workflow.parquet"

    df_definitions, df_index, df_workflow = load_index_data_from_files(
                                path_to_definitions_as_parquet_file, path_to_additional_definitions_as_parquet_file,
                                path_to_index_as_parquet_file, path_to_additional_index_as_parquet_file,
                                path_to_workflow_as_parquet)

    l_d = len(df_definitions)
    l_i = len(df_index)

    path_to_additional_manual_as_csv_file = ""
    path_to_additional_definitions_as_parquet_file = "./test/inputs/definitions_plus.parquet"
    path_to_additional_index_as_parquet_file = ""
    df_definitions, df_index, df_workflow = load_index_data_from_files(
                                                path_to_definitions_as_parquet_file, path_to_additional_definitions_as_parquet_file,
                                                path_to_index_as_parquet_file, path_to_additional_index_as_parquet_file,
                                                path_to_workflow_as_parquet)
    assert len(df_definitions) > l_d
    assert len(df_index) == l_i



def test_cap_rag_section_token_length():
    
    path_to_manual_as_csv_file = "./test/inputs/manual.csv"
    path_to_definitions_as_parquet_file = "./test/inputs/definitions.parquet"
    path_to_index_as_parquet_file = "./test/inputs/index.parquet"
    path_to_additional_manual_as_csv_file = ""
    path_to_additional_definitions_as_parquet_file = ""
    path_to_additional_index_as_parquet_file = ""
    path_to_workflow_as_parquet = "./test/inputs/workflow.parquet"

    df_regulations = load_regulation_data_from_files(path_to_manual_as_csv_file, path_to_additional_manual_as_csv_file)

    df_definitions, df_index, df_workflow = load_index_data_from_files(
                                path_to_definitions_as_parquet_file, path_to_additional_definitions_as_parquet_file,
                                path_to_index_as_parquet_file, path_to_additional_index_as_parquet_file,
                                path_to_workflow_as_parquet)

    user_type = "an Authorised Dealer (AD)" 
    # regulation_name = "\'Currency and Exchange Manual for Authorised Dealers\' (Manual or CEMAD)"
    regulation_name = "South African Exchange Control Manual"


    reader = TESTReader(reference_checker = reference_checker, regulation_df = df_regulations)
    data = StandardRegulationIndex(user_type = user_type,
                            regulation_name = regulation_name,
                            regulation_reader = reader,
                            df_definitions = df_definitions, 
                            df_index = df_index, 
                            df_workflow = df_workflow)

    test_data = []
    test_data.append(['A.3(A)', 0.1, "question", "some text here"])
    test_data.append(['A.3(E)', 0.2, "question", "some text here"])
    test_data.append(['Legal Context', 0.3, "question", "some text here"])
    test_data.append(['C.(E)', 0.5, "question", "some text here"])
    test_data.append(['G.(A)', 0.6, "question", "some text here"])
    test_data.append(['C.(F)', 0.7, "question", "some text here"])

    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_capped_test_data = data.cap_rag_section_token_length(df_test_data, 10000)
    assert len(df_capped_test_data) == 5 # hard cap in function

    df_capped_test_data = data.cap_rag_section_token_length(df_test_data, 100) # first entry has more that 100
    assert len(df_capped_test_data) == 1 

    df_capped_test_data = data.cap_rag_section_token_length(df_test_data, 300)
    assert len(df_capped_test_data) == 3
    
