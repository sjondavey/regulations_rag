from regulations_rag.string_tools import match_strings_to_reference_list

def test_match_strings_to_reference_list():
    # simple test of the main usecase
    reference_list = ['B.2(C)(ii)', 'B.2(C)(i)(a)', 'B.2(B)(i)', 'B.2(C)(iii)', 'B.2(C)(ii)']
    llm_selected_list_no_problems =  ['B.2(C)(ii)', 'B.2(C)(i)(a)']
    results_list = match_strings_to_reference_list(llm_selected_list_no_problems, reference_list)
    assert set(results_list) == set(llm_selected_list_no_problems)

    llm_selected_list_format_problems =  ['B2(C)(ii)', '(C)(i)(a)'] #miss "." then miss B.2 but should still match
    results_list = match_strings_to_reference_list(llm_selected_list_format_problems, reference_list)
    assert set(results_list) == set(llm_selected_list_no_problems)

    llm_selected_list_format_problems =  ['B.2(C)(ii)', 'hello'] #Should not return a match for hello
    results_list = match_strings_to_reference_list(llm_selected_list_format_problems, reference_list)
    assert len(results_list) == 1
    assert results_list[0] == llm_selected_list_no_problems[0]
