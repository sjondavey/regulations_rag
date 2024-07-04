import pytest
import os
import pandas as pd
from openai import OpenAI
from .navigating_index import NavigatingIndex
from regulations_rag.embeddings import get_ada_embedding
from regulations_rag.rerank import RerankAlgos

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@pytest.fixture
def navigating_index():
    return NavigatingIndex()

def test_get_relevant_definitions(navigating_index):
    user_content = "What is the Gym?"
    model = "text-embedding-3-large"
    dimensions = 1024
    user_content_embedding = get_ada_embedding(openai_client=openai_client, text=user_content, model=model, dimensions=dimensions)
    threshold = 0.38
    dfns = navigating_index.get_relevant_definitions(user_content, user_content_embedding, threshold)
    assert len(dfns) == 1 
    assert dfns.iloc[0]["section_reference"] == "A.1(A)"
    assert dfns.iloc[0]["document"] == "Plett"

def test_get_relevant_sections(navigating_index):
    user_content = "How do I get to South Gate?"
    model = "text-embedding-3-large"
    dimensions = 1024
    user_content_embedding = get_ada_embedding(openai_client=openai_client, text=user_content, model=model, dimensions=dimensions)
    threshold = 0.38
    sections = navigating_index.get_relevant_sections(user_content, user_content_embedding, threshold, rerank_algo=RerankAlgos.NONE)
    assert len(sections) == 3
    assert sections.iloc[0]["section_reference"] == "1.3"
    assert sections.iloc[0]["document"] == "WRR"

def test_get_relevant_workflow(navigating_index):
    user_content = "Show me the map"
    model = "text-embedding-3-large"
    dimensions = 1024
    user_content_embedding = get_ada_embedding(openai_client=openai_client, text=user_content, model=model, dimensions=dimensions)
    threshold = 0.38
    workflow = navigating_index.get_relevant_workflow(user_content, user_content_embedding, threshold)
    assert len(workflow) == 1
    assert workflow.iloc[0]["workflow"] == "map"
    assert workflow.iloc[0]["text"] == "Can you show this on a map?"

def test_cap_rag_section_token_length(navigating_index):
    user_content = "How do I get to South Gate?"
    model = "text-embedding-3-large"
    dimensions = 1024
    user_content_embedding = get_ada_embedding(openai_client=openai_client, text=user_content, model=model, dimensions=dimensions)
    threshold = 0.38
    sections = navigating_index.get_relevant_sections(user_content, user_content_embedding, threshold, rerank_algo=RerankAlgos.NONE)
    capped_sections = navigating_index.cap_rag_section_token_length(sections, capped_number_of_tokens=100)
    assert len(capped_sections) <= 5  # Assuming cap limits the number of sections
    assert "token_count" in capped_sections.columns
    assert capped_sections["token_count"].sum() <= 100

