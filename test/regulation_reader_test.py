import pytest
import pandas as pd

from regulations_rag.reference_checker import TESTReferenceChecker

from regulations_rag.regulation_reader import  TESTReader, load_csv_data, load_regulation_data_from_files


reference_checker = TESTReferenceChecker()

#df = pd.read_csv("./test/inputs/manual.csv", sep="|", encoding="utf-8", na_filter="")
#df = load_csv_data("./test/inputs/manual.csv")
df = load_regulation_data_from_files("./test/inputs/manual.csv", "")
test_reader = TESTReader(reference_checker = reference_checker, regulation_df = df)

def test_get_regulation_detail():


    response = test_reader.get_regulation_detail('A.3(E)(viii)(a)(bb)')
    expected_response = 'A.3 Duties and responsibilities of Authorised Dealers\n    (E) Transactions with Common Monetary Area residents\n        (viii) As an exception to (vi) above, Authorised Dealers may:\n            (a) sell foreign currency to:\n                (bb) CMA residents in South Africa, to cover unforeseen incidental costs whilst in transit, subject to viewing a passenger ticket confirming a destination outside the CMA;'
    assert response == expected_response


def test_get_regulation_heading():
    response = test_reader.get_regulation_heading('A.3(E)(viii)(a)(bb)')
    expected_response = 'A.3 Duties and responsibilities of Authorised Dealers. (E) Transactions with Common Monetary Area residents.'
    assert response == expected_response

