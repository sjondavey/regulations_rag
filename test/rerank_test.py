import pandas as pd
import os
from cryptography.fernet import Fernet


from regulations_rag.rerank import check_columns, top_n_items, get_top_mode_and_repeat_sections
from regulations_rag.data_in_dataframes import load_parquet_data

def test_check_columns():
    df = pd.DataFrame([], columns = ["section_reference", "text", "source", "cosine_distance"])
    assert check_columns(df)

    df = pd.DataFrame([], columns = ["section_reference", "text", "source"])
    assert not check_columns(df)

def test_top_n_items():
    items_list = []
    items_list.append(["A.1", "Hi", "heading", 0.001])
    items_list.append(["A.2", "There", "heading", 0.002])
    items_list.append(["A.3", "How", "question", 0.005])
    items_list.append(["A.1", "Are", "summary", 0.003])
    items_list.append(["A.1", "YOU", "heading", 0.01])
    items_list.append(["A.1", "doing", "heading", 0.003])

    df = pd.DataFrame(items_list, columns = ["section_reference", "text", "source", "cosine_distance"])
    df_filtered = top_n_items(relevant_sections = df, n = 3) # interesting because there are two with the 3rd ranking
    assert len(df_filtered) == 3

def test_get_top_mode_and_repeat_sections():
    # test_file = "./test/inputs/index.parquet"
    # key = os.getenv('excon_encryption_key')

    # df = load_parquet_data(test_file, key)
    # raise NotImplemented()

    output_columns = ["section_reference", "text", "source", "cosine_distance", "count"] # the rerak_most_common adds a "count" column
    test_data = []
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "text", "source", "cosine_distance"])        
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
    assert len(df_filtered_test_data) == 0
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
    assert len(df_filtered_test_data) == 1
    assert df_filtered_test_data.iloc[0]["count"] == 1
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)

    # Check that duplicate top values are filtered and the cosine distance is the minimum
    test_data.append(['A.1(A)(i)(a)', 0.2, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
    assert len(df_filtered_test_data) == 1
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[0]["cosine_distance"] == 0.1
    # Check that the top and mode results are returned and the mode value is the lowest distance
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.3, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
    assert len(df_filtered_test_data) == 2
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.2 # Note the order of the search_section is preserved so the mode should be second
    assert df_filtered_test_data.iloc[1]["count"] == 2

    # Check if there are no duplicate indexes that we still return multiple sections
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
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
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
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
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
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
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
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
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
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
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
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
    df_filtered_test_data = get_top_mode_and_repeat_sections(df_test_data)
    assert len(df_filtered_test_data) == 3
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
