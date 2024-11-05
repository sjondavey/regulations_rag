import os
from regulations_rag.corpus_index import DataFrameCorpusIndex
from test.documents.gdpr import GDPR
from test.documents.consent import Consent

class GDPRCorpus(Corpus):
    def __init__(self, folder):
        # document_path = "e:/code/chat/regulations_rag/test/documents"
        # document_files = ["gdpr.py", "consent.py"]


        document_dictionary = {}
        document_dictionary["gdpr"] = GDPR()
        document_dictionary["consent"] = Consent()
        super().__init__(document_dictionary)

    def get_primary_document(self):
        return "GDPR"


class GDPRCorpusIndex(DataFrameCorpusIndex):
    def __init__(self, key):
        #key = os.getenv('encryption_key_gdpr')


        corpus = GDPRCorpus("e:/code/chat/gdpr_rag/gdpr_rag/documents/")
        index_folder = "e:/code/chat/gdpr_rag/inputs/index/"
        index_df = pd.DataFrame()
        for filename in os.listdir(index_folder):
            if filename.endswith(".parquet"):  
                filepath = os.path.join(index_folder, filename)
                df = load_parquet_data(filepath, key)
                index_df = pd.concat([index_df, df], ignore_index = True)

        user_type = "a Controller"
        corpus_description = "the General Data Protection Regulation (GDPR)"


        super.__init__(user_type, corpus_description, corpus, definitions, index, workflow)

class TestGDPRCorpusIndex():
    def test_construction(self):
        key = os.getenv('encryption_key_gdpr')
        gdpr_corpus = GDPRCorpusIndex(key)
        assert True