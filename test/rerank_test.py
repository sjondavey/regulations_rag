import pandas as pd
import os
from cryptography.fernet import Fernet
from openai import OpenAI


from regulations_rag.rerank import check_rerank_columns, rerank_most_common, rerank_llm

def test_check_rerank_columns():
    df = pd.DataFrame([], columns = ["section_reference", "text", "source", "cosine_distance"])
    assert check_rerank_columns(df)

    df = pd.DataFrame([], columns = ["section_reference", "text", "source"])
    assert not check_rerank_columns(df)


def test_rerank_most_common():

    output_columns = ["section_reference", "text", "source", "cosine_distance", "count"] # the rerak_most_common adds a "count" column
    test_data = []
    # a empty dataframe will not get to rerank_most_common()
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    assert check_rerank_columns(dataframe = df_test_data)
    df_filtered_test_data = rerank_most_common(df_test_data)
    assert check_rerank_columns(dataframe = df_filtered_test_data)
    assert len(df_filtered_test_data) == 1
    assert df_filtered_test_data.iloc[0]["count"] == 1
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)

    # Check that duplicate top values are filtered and the cosine distance is the minimum
    test_data.append(['A.1(A)(i)(a)', 0.2, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = rerank_most_common(df_test_data)
    assert len(df_filtered_test_data) == 1
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[0]["cosine_distance"] == 0.1
    # Check that the top and mode results are returned and the mode value is the lowest distance
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.3, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = rerank_most_common(df_test_data)
    assert len(df_filtered_test_data) == 2
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)
    assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.2 # Note the order of the search_section is preserved so the mode should be second
    assert df_filtered_test_data.iloc[1]["count"] == 2

    # Check if there are no duplicate indexes that we still return multiple sections
    test_data = []
    test_data.append(['A.1(A)(i)(a)', 0.1, "question", "some text here"])
    test_data.append(['A.1(A)(i)(b)', 0.2, "question", "some text here"])
    df_test_data = pd.DataFrame(test_data, columns = ["section_reference", "cosine_distance", "source", "text"])        
    df_filtered_test_data = rerank_most_common(df_test_data)
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
    df_filtered_test_data = rerank_most_common(df_test_data)
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
    df_filtered_test_data = rerank_most_common(df_test_data)
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
    df_filtered_test_data = rerank_most_common(df_test_data)
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
    df_filtered_test_data = rerank_most_common(df_test_data)
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
    df_filtered_test_data = rerank_most_common(df_test_data)
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
    df_filtered_test_data = rerank_most_common(df_test_data)
    assert len(df_filtered_test_data) == 3
    assert set(df_filtered_test_data.columns.to_list()) == set(output_columns)

def test_rerank_llm():
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
    model_to_use = "gpt-3.5-turbo"
    model_to_use = "gpt-4-1106-preview"

    test_data = []
    test_data.append(['cemad', 'A.1(A)(i)(a)', 0.1, "question", "Dentists can buy gold"])
    test_data.append(['cemad', 'A.1(A)(i)(b)', 0.2, "question", "Trusts can transfer money offshore"])
    test_data.append(['cemad', 'A.1(A)(i)(c)', 0.3, "question", "Producers can sell gold"])
    test_data.append(['cemad', 'A.1(A)(i)(c)', 0.5, "question", "How much money can an individual take offshore?"])
    test_data.append(['cemad', 'A.1(A)(i)(b)', 0.6, "question", "How much cash can I take on holiday?"])
    test_data.append(['cemad', 'A.1(A)(i)(d)', 0.7, "question", "Why is it so cold"])
    relevant_sections = pd.DataFrame(test_data, columns = ["document", "section_reference", "cosine_distance", "source", "text"])        
    
    user_question = "Who can trade gold?"
    user_type = "an Authorised Dealer"
    corpus_description = "the Currency and Exchange Control Manual for Authorised Dealers"
    reranked_sections = rerank_llm(relevant_sections = relevant_sections, openai_client = openai_client, model_to_use = model_to_use, user_question = user_question, user_type = user_type, corpus_description = corpus_description)
    assert len(reranked_sections) == 2 # should return "Dentists can buy gold" and "Producers can sell gold"
