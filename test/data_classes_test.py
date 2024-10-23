import pandas as pd
import pytest
from regulations_rag.data_classes import AnswerWithRAGResponse, AnswerWithoutRAGResponse, \
    NoAnswerClassification, NoAnswerResponse, ErrorClassification, ErrorResponse


def test_answer_with_rag_response():
    references = pd.DataFrame({
        "document_key": ["1", "2", "3"],
        "document_name": ["Doc1", "Doc2", "Doc3"],
        "section_reference": ["1.1", "2.2", "3.3"],
        "is_definition": [True, False, True],
        "text": ["The text of definition no 1", "The text of section 2", "The text of definition no 3"]
    })
    # Example usage:
    response_with_rag = AnswerWithRAGResponse(
        answer="This is the answer using RAG.",
        references=references
    )
    assert response_with_rag.create_openai_content() == 'This is the answer using RAG. \n\nReference: \n\nDefinition 1.1 from Doc1: \n\nThe text of definition no 1  \n\nSection 2.2 from Doc2: \n\nThe text of section 2  \n\nDefinition 3.3 from Doc3: \n\nThe text of definition no 3  \n\n'

    # Test that AnswerWithRAGResponse raises ValueError when DataFrame doesn't have expected columns
    with pytest.raises(ValueError) as excinfo:
        response_with_rag = AnswerWithRAGResponse(
            answer="This is the answer using RAG.",
            references=pd.DataFrame()
        )
    assert str(excinfo.value) == "References DataFrame must have columns: ['document_key', 'document_name', 'section_reference', 'is_definition', 'text']"

    # Test that AnswerWithRAGResponse raises ValueError when answer is empty
    with pytest.raises(ValueError) as excinfo:
        response_with_rag = AnswerWithRAGResponse(
            answer="",
            references=references
        )
    assert str(excinfo.value) == "Answer cannot be empty"

def test_answer_without_rag_response():
    response_without_rag = AnswerWithoutRAGResponse(
        answer="This is the answer without RAG.",
        caveat="This is the caveat"
    )
    assert response_without_rag.create_openai_content() == 'This is the caveat \n\nThis is the answer without RAG.'

    # Test that AnswerWithoutRAGResponse raises ValueError when caveat is empty
    with pytest.raises(ValueError) as excinfo:
        response_without_rag = AnswerWithoutRAGResponse(
            answer="This is the answer without RAG.",
            caveat=""
        )
