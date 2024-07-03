import pytest
from .navigating_index import NavigatingIndex

import pandas as pd
import os
from openai import OpenAI
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
from regulations_rag.embeddings import get_ada_embedding
from regulations_rag.rerank import RerankAlgos


@pytest.fixture
def navigating_index():
    return NavigatingIndex()

def test_get_relevant_definitions(navigating_index):
    user_content = "What is the Gym?"
    model = "text-embedding-3-large"
    dimensions = 1024
    user_content_embedding = get_ada_embedding(openai_client = openai_client, text = user_content, model = model, dimensions = dimensions)
    threshold = 0.38
    dfns = navigating_index.get_relevant_definitions(user_content, user_content_embedding, threshold)
    assert len(dfns) == 1 


def test_get_relevant_sections(navigating_index):
    user_content = "How do I get to South Gate?"
    model = "text-embedding-3-large"
    dimensions = 1024
    user_content_embedding = get_ada_embedding(openai_client = openai_client, text = user_content, model = model, dimensions = dimensions)
    threshold = 0.38
    sections = navigating_index.get_relevant_sections(user_content, user_content_embedding, threshold, rerank_algo = RerankAlgos.NONE)
    assert len(sections) == 3
    assert sections.iloc[0]["section_reference"] == "1.3"
    assert sections.iloc[0]["document"] == "WRR"


#     def cap_rag_section_token_length(self, relevant_sections, capped_number_of_tokens):
#     def get_relevant_workflow(self, user_content_embedding, threshold):
