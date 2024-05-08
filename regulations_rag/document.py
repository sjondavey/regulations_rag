import logging

import os
import pandas as pd
from abc import ABC, abstractmethod
from regulations_rag.reference_checker import ReferenceChecker


class Document(ABC):
    def __init__(self, document_name, reference_checker):
        if document_as_df.isna().any().any():
            raise AttributeError("The input DataFrame contains NaNs which will cause issues")

        self.document_as_df = document_as_df

        if not self.check_columns():
            raise AttributeError("The input DataFrame does not have the correct columns")

        self.name = document_name 
        self.reference_checker = reference_checker

    @abstractmethod
    def check_columns(self):
        pass

    @abstractmethod
    def get_heading(self, section_reference):
        pass

    @abstractmethod
    def get_text(self, section_reference):
        pass


