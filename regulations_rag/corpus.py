import os
import ast

class Corpus:
    """
    A class representing a collection of documents.
    
    Attributes:
    all_documents (dict): A dictionary where keys are the class names and values are instances of these classes.
                          NOTE: In the "document" field of the index database is a text value that will be matched
                                against the key in this dictionary.
    """

    def __init__(self, document_dictionary):
        self.all_documents = document_dictionary

    def get_document(self, document_key):
        return self.all_documents.get(document_key)

    def get_heading(self, document_key, section_reference):
        doc = self.get_document(document_key)
        return doc.get_heading(section_reference) if doc else None

    def get_primary_document(self):
        return ""

    def get_text(self, document_key, section_reference, add_markdown_decorators=True, add_headings=True, section_only=False):
        doc = self.get_document(document_key)
        if doc:
            return doc.get_text(section_reference, add_markdown_decorators, add_headings, section_only)
        return None

def create_document_dictionary_from_folder(folder_name, namespace_dict=None):
    """
    Create a dictionary of document instances from Python classes defined in the files within a given folder.

    Args:
    folder_name (str): The name of the folder where the Python files are located.
    namespace_dict (dict): Optional dictionary to look up classes.

    Returns:
    dict: A dictionary where keys are the class names and values are instances of these classes.
    """
    class_names_dict = find_class_names_in_files(folder_name)
    all_documents = {}
    for class_name, file_class_name in class_names_dict.items():
        doc_class = get_document_class_by_name(file_class_name, namespace_dict)
        if doc_class:
            all_documents[file_class_name] = doc_class()
    return all_documents

def find_class_names_in_files(directory):
    """
    Extracts the first class name from each Python file in a directory.

    Args:
    directory (str): The directory to search for Python files.

    Returns:
    dict: A dictionary where keys are the filenames without extensions and values are the class names found in the files.
    """
    class_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                file_content = file.read()
            tree = ast.parse(file_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    file_name_without_extension = os.path.splitext(filename)[0]
                    class_dict[file_name_without_extension] = class_name
                    break
    return class_dict

def get_document_class_by_name(class_name, namespace_dict=None):
    """
    Retrieve a class object from the global scope by its name.

    Args:
    class_name (str): The name of the class to retrieve.
    namespace_dict (dict): Optional dictionary to look up classes.

    Returns:
    type: The class object if found, None otherwise.
    """
    if namespace_dict is None:
        namespace_dict = globals()
    return namespace_dict.get(class_name)
