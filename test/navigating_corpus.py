from regulations_rag.corpus import Corpus
from test.documents.wrr_document import WRR
from test.documents.plett_document import Plett



class NavigatingCorpus(Corpus):
    def __init__(self):
        document_dictionary = {}
        document_dictionary["WRR"] = WRR()
        document_dictionary["Plett"] = Plett()
        super().__init__(document_dictionary)

    def get_primary_document(self):
        return "WRR"
