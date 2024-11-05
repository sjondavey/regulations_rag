from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
from abc import ABC, abstractmethod

# # class AssistantMessageClassification(Enum):
# #     ANSWER = "The system answered the user question"
# #     NO_ANSWER = "The system did not answer the user question"
# #     ERROR = "An error occoured"

@dataclass
class AssistantResponse(ABC):
    @abstractmethod
    def create_openai_content(self) -> str:
        pass
    def get_text_for_streamlit(self) -> str:
        pass


@dataclass
class AnswerWithRAGResponse(AssistantResponse):
    answer: str
    references: pd.DataFrame
    """
    answer (str): The value returned by the LLM, stripped of any prefix or references
    references (pd.DataFrame): A DataFrame containing reference information.
    
    Expected columns:
    - document_key (str): Unique identifier for the document
    - document_name (str): Name of the document
    - section_reference (str): Reference to the specific section
    - is_definition (bool): Whether this reference is a definition
    - text (str): The relevant text from the reference
    """

    # __post_init__ runs after the object is initialized. 
    # This method checks if all the expected columns are present in the DataFrame.
    def __post_init__(self):
        if self.answer == "":
            raise ValueError("Answer cannot be empty")

        """Validate the DataFrame structure."""
        expected_columns = ['document_key', 'document_name', 'section_reference', 'is_definition', 'text']
        if not all(col in self.references.columns for col in expected_columns):
            raise ValueError(f"References DataFrame must have columns: {expected_columns}")

    # create the full text with a reference description (document name, section reference) and the full text of the reference
    def create_openai_content(self) -> str:
        reference_string = ""
        formatted_references = ""
        for index, row in self.references.iterrows():
            document_name = row["document_name"]
            section_reference = row["section_reference"]
            if row["is_definition"]: # note the Markdown formatting
                if section_reference == "":
                    reference_string += f"The definitions in {document_name}: \n\n{row['text']}  \n\n"
                else:
                    reference_string += f"Definition {section_reference} from {document_name}: \n\n{row['text']}  \n\n"
            else:
                if section_reference == "":
                    reference_string += f"The document {document_name}: \n\n{row['text']}  \n\n"
                else:
                    reference_string += f"Section {section_reference} from {document_name}: \n\n{row['text']}  \n\n"
        if len(self.references) > 0:
            formatted_references = f" \n\nReference: \n\n{reference_string}"
        
        open_ai_text = self.answer + formatted_references
        return open_ai_text

    def get_text_for_streamlit(self) -> str:
        return self.answer

@dataclass
class AnswerWithoutRAGResponse(AssistantResponse):
    """
    answer (str): The value returned by the LLM, stripped of any prefix or references
    caveat (str): This will be added before the answer so the user knows the answer did not use RAG as designed
    """
    answer: str
    caveat: str

    def __post_init__(self):
        if self.answer == "":
            raise ValueError("Answer cannot be empty")
        if self.caveat == "":
            raise ValueError("Caveat cannot be empty")

    def create_openai_content(self) -> str:
        return f"{self.caveat} \n\n{self.answer}"

    def get_text_for_streamlit(self) -> str:
        return f"{self.caveat} \n\n{self.answer}"

@dataclass
class AlternativeQuestionResponse(AssistantResponse):
    alternatives: list

    def create_openai_content(self) -> str:
        return f"The system has suggested the following alternative questions: \n\n" + "\n".join(self.alternatives)

    def get_text_for_streamlit(self) -> str:
        if len(self.alternatives) == 1:
            return f"The model was unable to retrieve any reference documentation to answer your question. Below is a rephrased version of your question that matches some reference material. \n\nNOTE: While this alternative suggestion has been matched with reference material in the database, the reference material has *not* yet been verified as relevant. \n\n"
        else:
            return f"The model was unable to retrieve any reference documentation to answer your question. Below are a few suggestions on how to rephrase your question to match available reference material. \n\nNOTE: While these alternative suggestions have been matched with reference material in the database, the reference material has *not* yet been verified as relevant. \n\n"

class NoAnswerClassification(Enum):
    NO_DATA = "The model was asked to perform strict RAG without any data being provided"
    NO_RELEVANT_DATA = "The model was asked to perform strict RAG but no data provided was not deemed relevant"
    QUESTION_NOT_RELEVANT = "The model determined that the question was not relevant to the corpus"
    UNABLE_TO_ANSWER = "The model was unable to answer the question"

@dataclass
class NoAnswerResponse(AssistantResponse):
    classification: NoAnswerClassification
    additional_text: str = ""

    def __post_init__(self):
        if not isinstance(self.classification, NoAnswerClassification):
            raise ValueError("Classification must be an instance of NoAnswerClassification")

    def create_openai_content(self) -> str:
        return self.classification.value

    def get_text_for_streamlit(self) -> str:
        base_text = "This app demonstrates Retrieval Augmented Generation. Behind the scenes, the model is instructed to use the reference material to answer your question. The model is given an option not to respond for a variety of reasons. In this case it did not respond because "
    
        if self.classification == NoAnswerClassification.QUESTION_NOT_RELEVANT:
            if self.additional_text == "":
                text = base_text + "it did not beleive the question was relevant to the subject matter."
            else:
                text = base_text + "it did not beleive the question was relevant to the subject matter. The reason given is: " + self.additional_text
        elif self.classification == NoAnswerClassification.NO_RELEVANT_DATA:
            text = base_text + "the reference material it initially found, proved not to be of assistance when answering your question."
        elif self.classification == NoAnswerClassification.NO_DATA:
            text = base_text + "it did not find any relevant material in the reference material and RAG enforcement is set to Strict"
        elif self.classification == NoAnswerClassification.UNABLE_TO_ANSWER:
            text = base_text + "it was not comfortable that it had the knowledge to answer the question."
        else:
            text = self.classification.value
        
        text = text + " If you did ask a relevant question, you can try a rephrasing it and asking again. If you are reading this and  beleive your question is relevant and should be answered by this service, please press the 'thumbs down' button which will flag the conversation history for investigation. This will help improve the model over time."
        return text


class ErrorClassification(Enum):
    ERROR = "Unfortunately the system is in an unrecoverable state. Please clear the chat history and retry your query"
    NOT_FOLLOWING_INSTRUCTIONS = "This app demonstrates Retrieval Augmented Generation. Behind the scenes, instructions are issued to a Large Language Model (LLM) and then verified. Occasionally, due to the statistical nature of the model, the LLM may not follow instructions correctly. In such cases, I am programmed not to respond but to ask you to clear the conversation history and try asking your question again. This usually resolves the issue. However, if the same error persists in the same spot, it likely indicates a bug rather than a statistical anomaly. Bugs are logged and will be addressed in due course. For now, please clear the conversation history and retry your query."
    CALL_FOR_MORE_DOCUMENTS_FAILED = "This app demonstrates Retrieval Augmented Generation. While accessing the source documents, the system requested additional material. There was an error in retrieving this additional material."
    STUCK = "Unfortunately the system is in an unrecoverable state. Please clear the chat history and retry your query"
    WORKFLOW_NOT_IMPLEMENTED = "A workflow was triggered was but there is no implementation of the execute_path_workflow() function for this workflow"

@dataclass
class ErrorResponse(AssistantResponse):
    classification: ErrorClassification

    def __post_init__(self):
        if not isinstance(self.classification, ErrorClassification):
            raise ValueError("Classification must be an instance of ErrorClassification")

    def create_openai_content(self) -> str:
        return self.classification.value

    def get_text_for_streamlit(self) -> str:
        base_text = "This app demonstrates Retrieval Augmented Generation. Behind the scenes, instructions are issued to a Large Language Model (LLM) and then verified. Occasionally, due to the statistical nature of the model, things may go wring. In this case, the model has reported that something went wrong because "
        
        if self.classification == ErrorClassification.NOT_FOLLOWING_INSTRUCTIONS:
            text = base_text + "the LLM did not follow instructions correctly."
        elif self.classification == ErrorClassification.CALL_FOR_MORE_DOCUMENTS_FAILED:
            text = base_text + "the model asked for additioinal references to answer your question but it was unable to retrieve these."
        elif self.classification == ErrorClassification.WORKFLOW_NOT_IMPLEMENTED:
            text = base_text + "a workflow was triggered by the question but no matching implementation could be found."
        elif self.classification == ErrorClassification.STUCK:
            text = base_text + "system logic failed. Please rest assured that the developers will be punished for their lazyness and incompetance (except for the masochists)!"
        elif self.classification == ErrorClassification.ERROR:
            text = base_text + "something really unexpected went wrong. Please rest assured that the developers will be punished for their lazyness and incompetance (except for the masochists)!"
        else:
            text = self.classification.value
        text = text + " Sometimes the error is due to a statistical anomaly and clearing the conversation history and retrying will resolve the issue. However, if the same error persists, please press the 'thumbs down' button which will flag the conversation history for investigation. This will help improve the model over time."
        return text


