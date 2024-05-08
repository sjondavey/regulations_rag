import os
import ast

class Corpus():
    """  
    document_dictionary: A dictionary where keys are the class names and values are instances of these classes.
                         NOTE: In the "document" filed of the index database is a text value that will be matched
                               against the key in this dictionary
    """
    def __init__(self, document_dictionary):
        self.all_documents = document_dictionary

    def get_document(self, document_name):
        return self.all_documents.get(document_name)

    def get_heading(self, document_name, section_reference):
        doc = self.get_document(document_name)
        return doc.get_heading(section_reference)

    def get_text(self, document_name, section_reference):
        doc = self.get_document(document_name)
        return doc.get_text(section_reference)


def create_corpus_from_folder_of_documents(folder_name):
    """
    Create a dictionary of document instances from Python classes defined in the files within a given folder.

    Args:
    folder_name (str): The name of the folder where the Python files are located.

    Returns:
    dict: A dictionary where keys are the class names and values are instances of these classes.

    This function reads Python files in the specified folder, extracts class definitions, and creates an instance
    of each class. It logs the process, noting whether instances are added successfully or if a class was not found.
    """
    class_names_dict = find_class_names_in_files(folder_name)
    all_documents = {}
    for class_name in class_names_dict:

        doc_class = get_document_class_by_name(class_names_dict[class_name])
        if doc_class:
            document_instance = doc_class()
            all_documents[class_names_dict[class_name]] = document_instance
            logger.log(DEV_LEVEL, f"Added instance of {class_name} to all_documents.")
        else:
            logger.log(DEV_LEVEL, f"Class {class_name} not found.")
    return all_documents


def find_class_names_in_files(directory):
    """
    Extracts the first class name from each Python file in a directory.

    Args:
    directory (str): The directory to search for Python files.

    Returns:
    dict: A dictionary where keys are the filenames without extensions and values are the class names found in the files.

    This function searches through all Python files in the specified directory, parsing each file to find class definitions.
    It assumes that there is one class defined per file and stops searching after finding the first class in each file.
    """
    class_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".py"):  # Check for Python files
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                file_content = file.read()
            tree = ast.parse(file_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    file_name_without_extension = os.path.splitext(filename)[0]
                    class_dict[file_name_without_extension] = class_name
                    break  # Assuming one class per file, break after finding the first class
    return class_dict



def get_document_class_by_name(class_name):
    """
    Retrieve a class object from the global scope by its name.

    Args:
    class_name (str): The name of the class to retrieve.

    Returns:
    class: The class object if found, None otherwise.

    This function attempts to find a class by its name in the global scope. If the class exists, it returns the class object;
    otherwise, it returns None.
    """
    return globals().get(class_name)



