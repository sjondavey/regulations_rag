import logging
from anytree import Node, RenderTree, find, LevelOrderIter, AsciiStyle
import re
import pandas as pd
from regulations_rag.embeddings import num_tokens_from_string

logger = logging.getLogger(__name__)

class TableOfContentEntry(Node):
    def __init__(self, name, full_node_name, parent=None, heading_text=''):
        super().__init__(name, parent=parent)
        self.heading_text = heading_text
        self.full_node_name = full_node_name

    def consolidate_from_leaves(self, consolidate_headings):
        if not self.children:
            return self.heading_text
        
        children_headings = [child.consolidate_from_leaves(consolidate_headings) for child in self.children]
        self.heading_text = consolidate_headings(children_headings)
        return self.heading_text

class TableOfContent:
    def __init__(self, root_id, reference_checker):
        self.root = TableOfContentEntry(root_id, "", parent=None, heading_text='')
        self.reference_checker = reference_checker

    def add_to_toc(self, section_reference, heading_text=''):
        if section_reference == self.root.name:
            self.root.heading_text = heading_text
            return

        if not self.reference_checker.is_valid(section_reference):
            raise ValueError(f'{section_reference} is not a valid section_reference')

        node_names = self.reference_checker.split_reference(section_reference)
        current_parent = self.root
        full_node_name = ''

        for i, node_name in enumerate(node_names):
            full_node_name += node_name
            found_node = next((child for child in current_parent.children if child.name == node_name), None)

            if found_node is None:
                heading = heading_text if i == len(node_names) - 1 else ''
                current_parent = TableOfContentEntry(node_name, full_node_name, parent=current_parent, heading_text=heading)
            else:
                current_parent = found_node

            if i == len(node_names) - 1 and not current_parent.heading_text:
                current_parent.heading_text = heading_text

    def get_node(self, section_reference):
        if section_reference == self.root.name:
            return self.root
        if not self.reference_checker.is_valid(section_reference):
            raise ValueError(f'{section_reference} is not a valid section_reference')
        
        current_node = self.root
        node_names = self.reference_checker.split_reference(section_reference)
        for node_name in node_names:
            current_node = next((node for node in current_node.children if node.name == node_name), None)
            if current_node is None:
                raise ValueError(f"Node with path {section_reference} does not exist in the tree")
        return current_node

    def print_tree(self):
        for pre, _, node in RenderTree(self.root, style=AsciiStyle()):
            print(f"{pre}{node.name} [{node.heading_text}]")

    def _list_node_children(self, section_reference, indent=0):
        string = ""
        children_with_text = [child for child in section_reference.children if child.heading_text != '']

        if children_with_text:
            for child in section_reference.children:
                exclusion_list = self.reference_checker.exclusion_list if child.parent == self.root else []
                text = '' if child.name in exclusion_list else f' {child.heading_text}'
                string += ' ' * indent + f'{child.name}{text}\n'
                string += self._list_node_children(child, indent + 4)
        return string


class StandardTableOfContent(TableOfContent):

    def __init__(self, root_node_name, reference_checker, regulation_df):
        """
        Constructs a regulation tree from a DataFrame containing regulation entries.
        
        This function builds a tree structure representing the hierarchical relationship of regulations
        starting from a root section_reference. Each regulation or sub-regulation is added as a section_reference in the tree based
        on its 'section_reference'. The tree can be used to navigate through the regulations efficiently.
        
        Parameters:
        - root_node_name (str): The name of the root section_reference of the tree.
        - regulation_df (pd.DataFrame): DataFrame containing the regulations. Expected to have
        columns 'heading', 'text', and 'section_reference'.
        - reference_checker (object): A ReferenceChecker object.
        
        Raises:
        - ValueError: If any 'full_reference' in the DataFrame is not valid according to `reference_checker`.
        """
        super().__init__(root_node_name, reference_checker=reference_checker)
        self.regulation_df = regulation_df
        if not self.check_columns():
            message = "The input DataFrame did not have the correct headings to build the StandardTableOfContent. Required columns are 'text', 'heading', and 'section_reference'"
            logger.error(message)
            raise AttributeError(message)

        for i, row in regulation_df.iterrows():
            try:
                heading_text = row['text'] if row['heading'] else ''
                heading_text = self.remove_footnotes(heading_text).strip()

                if not reference_checker.is_valid(row['section_reference']):
                    raise ValueError(f"{row['section_reference']} is not a valid reference. See row {i}")

                super().add_to_toc(row['section_reference'], heading_text=heading_text)

            except Exception as e:
                logger.error(f"An error occurred at row {i}:")
                logger.error(regulation_df.iloc[i])
                logger.error(f"Error message: {e}")
                break

    def remove_footnotes(self, text):
        footnote_pattern = r'\[\^\d+\]\:'
        lines = text.split('\n')
        remaining_text = [line for line in lines if not re.match(footnote_pattern, line)]
        text = '\n'.join(remaining_text)
        return re.sub(r'\[\^\d+\]', '', text)

    def check_columns(self):
        """
        Validates the presence of required columns in the DataFrame.
        
        Required columns:
        - 'text': the text on the line excluding the 'reference' and any special text (identifying headings, page number, etc.)
        - 'heading': boolean identifying the text as a (sub-)section heading
        - 'section_reference': the full reference. Starting at the root section_reference and ending with the value in 'reference'
        """
        expected_columns = ['text', 'heading', 'section_reference']
        actual_columns = self.regulation_df.columns.to_list()

        missing_columns = [col for col in expected_columns if col not in actual_columns]
        if missing_columns:
            logger.error(f"Missing columns in DataFrame: {', '.join(missing_columns)}")
            return False
        return True

def _split_recursive(node, document, table_of_content, token_limit, node_list=[]):
#def _split_recursive(node, regulation_reader, table_of_content, token_limit, node_list=[]):
    """    
    Recursively splits nodes based on token limits and collects valid nodes in a list.
    You shouldn't need to call this method. Rather use the "split_tree()" method
    
    Parameters:
    - node (Node): The current node being processed.
    - dataframe (pd.DataFrame): DataFrame containing the regulation details.
    - token_limit (int): The maximum allowed token count per section.
    - reference_checker (callable): Function to check if an index is valid.
    - node_list (list, optional): List to collect nodes meeting the token criteria.
    
    Returns:
    - list: A list of nodes that meet the token criteria.
    """
    if node_list is None:
        node_list = []

    subsection_text = document.get_text(node.full_node_name)
    #subsection_text = regulation_reader.get_regulation_detail(node.full_node_name)
    token_count = num_tokens_from_string(subsection_text)

    if token_count > token_limit:
        if not node.children:
            raise Exception(f'Node {node.full_node_name} has no children but has a token count of {token_count} so it cannot be split into nodes that contain fewer tokens that {token_limit}')
        for child in node.children:
            _split_recursive(child, document, table_of_content, token_limit, node_list)
            #_split_recursive(child, regulation_reader, table_of_content, token_limit, node_list)
    else:
        node_list.append(node)

    return node_list


def split_tree(node, document, table_of_content, token_limit):
#def split_tree(node, regulation_reader, table_of_content, token_limit):
    """
    Splits a tree starting from a given node into sections that don't exceed a token limit.
    
    Initially this is used to set up the base DataFrame using node == root and later it can be used if we want 
    to change the word_limit for a specific piece of regulation to change chunking where it makes sense.

    Parameters:
    - node (Node): The starting node to split the tree.
    - dataframe (pd.DataFrame): DataFrame containing regulation details.
    - token_limit (int): The maximum allowed token count per section.
    - reference_checker (callable): Function to check if an index is valid.
    
    Returns:
    - pd.DataFrame: A DataFrame with columns ['section_reference', 'text', 'token_count'] for each valid section_reference.
    """
    #node_list = _split_recursive(node, regulation_reader, table_of_content, token_limit, node_list=[])
    node_list = _split_recursive(node, document, table_of_content, token_limit, node_list=[])
    section_token_count = [[node.full_node_name, 
                            document.get_text(node.full_node_name),
                            num_tokens_from_string(document.get_text(node.full_node_name))] 
                           for node in node_list]
    # section_token_count = [[node.full_node_name, 
    #                         regulation_reader.get_regulation_detail(node.full_node_name),
    #                         num_tokens_from_string(regulation_reader.get_regulation_detail(node.full_node_name))] 
    #                        for node in node_list]


    return pd.DataFrame(section_token_count, columns=['section_reference', 'text', 'token_count'])


