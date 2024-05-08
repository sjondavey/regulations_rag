import os
import pandas as pd
from abc import ABC, abstractmethod
from regulations_rag.reference_checker import ReferenceChecker


class Document(ABC):
    def __init__(self, document_name, reference_checker):
        self.name = document_name 
        self.reference_checker = reference_checker

    @abstractmethod
    def get_heading(self, section_reference):
        pass

    @abstractmethod    
    def get_text(self, section_reference):
        '''
        NOTE: When used with the Table of Content to break up a document into chunks, the call to get_text("") should return the entire text of the document
        '''
        pass


