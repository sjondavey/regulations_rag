import logging
from anytree import Node, RenderTree, find, LevelOrderIter, AsciiStyle
import re
import pandas as pd
from regulations_rag.regulation_index import RegulationIndex
from regulations_rag.embeddings import num_tokens_from_string
        
logger = logging.getLogger(__name__)

class TableOfContentEntry(Node):
    def __init__(self, name, full_node_name, parent=None, heading_text=''):
        super().__init__(name, parent=parent)
        self.heading_text = heading_text
        self.full_node_name = full_node_name

    # Recursive function to consolidate headings from leaves to root
    def consolidate_from_leaves(self, consolidate_headings):
        """
        Recursively consolidates heading texts from leaf nodes up to the root node.
        
        This method is used to aggregate or summarize information from the bottom of the
        tree (leaf nodes) upwards, allowing for the compilation of heading texts at higher
        levels in the hierarchy based on a user-defined consolidation function.
        
        Parameters:
        - consolidate_headings (callable): A function that takes a list of heading texts from
          child nodes and consolidates them into a single heading text.
        
        Returns:
        - str: The consolidated heading text for this node after processing all child nodes.
        """
        # base case: if the node is a leaf node (no children)
        if not self.children:
            return self.heading_text
        
        # Recursive case: if the node has children
        children_headings = [child.consolidate_from_leaves(consolidate_headings) for child in self.children]
        self.heading_text = consolidate_headings(children_headings)

        return self.heading_text

class TableOfContent:
    def __init__(self, root_id, index_checker):
        self.root = TableOfContentEntry(root_id, "", parent=None, heading_text='')
        self.index_checker = index_checker

    def add_to_toc(self, section_reference, heading_text=''):
        """
        Adds a new entry to the Table of Content or updates an existing entry's heading text based on a hierarchical
        section_reference identifier string. This also adds any missing parents (without headings) of the section_reference 
        all the way back to the root

        This method parses the `section_reference` using the `index_checker` to navigate through the tree
        and find the correct position for the new node or to update an existing section_reference.

        Parameters:
        - section_reference (str): The hierarchical identifier of the node to add or update.
        - heading_text (str, optional): The heading text for the section_reference. Defaults to an empty string.

        Raises:
        - ValueError: If `section_reference` is not a valid reference according to `index_checker`.
        """
        if section_reference == self.root.name:
            self.root.heading_text = heading_text
            return

        elif not self.index_checker.is_valid(section_reference):
            raise ValueError(f'{section_reference} is not a valid section_reference reference')

        node_names = self.index_checker.split_reference(section_reference)

        current_parent = self.root
        full_node_name = ''
        previous_full_node_name = ''  # variable to hold previous full section_reference 

        for i, node_name in enumerate(node_names):
            previous_full_node_name = full_node_name  # update previous section_reference before adding current section_reference 
            full_node_name = full_node_name + node_name
            found_node = None

            for child in current_parent.children:
                if child.name == node_name:
                    found_node = child
                    break

            # If the section_reference isn't found, create it
            if found_node is None:
                if i == len(node_names) - 1:  # if this is the last section_reference
                    current_parent = TableOfContentEntry(node_name, previous_full_node_name + node_name, parent=current_parent, heading_text=heading_text)
                else:
                    current_parent = TableOfContentEntry(node_name, previous_full_node_name + node_name, parent=current_parent, heading_text='')
            else:
                current_parent = found_node
            # If this is the last section_reference and it does not have a heading text, assign it
            if i == len(node_names) - 1 and not current_parent.heading_text:
                current_parent.heading_text = heading_text

    def get_node(self, section_reference):
        if section_reference == self.root.name:
            return self.root
        if not self.index_checker.is_valid(section_reference):
            raise ValueError(f'{section_reference} is not a valid section_reference reference')
        # Start search from the root
        current_node = self.root
        node_names = self.index_checker.split_reference(section_reference)
        for node_name in node_names:
            # Look for the section_reference among the children of the current section_reference
            found_node = next((node for node in current_node.children if node.name == node_name), None)
            # If not found, raise a ValueError
            if found_node is None:
                raise ValueError(f"Node with path {section_reference} does not exist in the tree")
            # If found, continue searching from this section_reference
            current_node = found_node
        # Return the section_reference we've found
        return current_node

    def print_tree(self):
        for pre, _, node in RenderTree(self.root, style=AsciiStyle()):
            print(f"{pre}{node.name} [{node.heading_text}]")

    # I use this function when extracting the headings from the manual for indexing. There are no tests for it yet!!
    # TODO: Add tests for this
    def _list_node_children(self, section_reference, indent = 0):
        string = ""
        # For each section_reference, check if at least one child has a non-empty heading text
        children_with_text = [child for child in section_reference.children if child.heading_text != '']

        if children_with_text:
            # If any child has non-empty heading text, print all that section_reference's children with their heading text
            for child in section_reference.children:
                if child.parent == self.root:
                    if child.name in self.index_checker.exclusion_list:
                        string = string + (' ' * indent + f'{child.name}\n')    
                    else:
                        string = string + (' ' * indent + f'{child.name} {child.heading_text}\n')
                else:
                    string = string + (' ' * indent + f'{child.name} {child.heading_text}\n')
                string = string + self._list_node_children(child, indent + 4)
        return string


class StandardTableOfContent(TableOfContent):

    def __init__(self, root_node_name, index_checker, regulation_df):
        """
        Constructs a regulation tree from a DataFrame containing regulation entries.
        
        This function builds a tree structure representing the hierarchical relationship of regulations
        starting from a root section_reference. Each regulation or sub-regulation is added as a section_reference in the tree based
        on its 'section_reference'. The tree can be used to navigate through the regulations efficiently.
        
        Parameters:
        - root_node_name (str): The name of the root section_reference of the tree.
        - regulation_df (pd.DataFrame): DataFrame containing the regulations. Expected to have
        columns 'heading', 'text', and 'section_reference'.
        - index_checker (object): A ReferenceChecker object.
        
        
        Raises:
        - ValueError: If any 'full_reference' in the DataFrame is not valid according to `index_checker`.
        """
        super().__init__(root_node_name, index_checker=index_checker)
        self.regulation_df = regulation_df
        if not self.check_columns():
            message = f"The input DataFrame did not have the correct headings to build the StandardTableOfContent. Required columns are text, heading and section_reference"
            logger.error(message)
            raise AttributeError(message)

        for i, row in regulation_df.iterrows() :
            try:
                heading_text = row['text'] if row['heading'] else ''

                if not index_checker.is_valid(row['section_reference']):
                    raise ValueError(row['section_reference'] + ' is not a valid reference. See row ' + str(i))

                super().add_to_toc(row['section_reference'], heading_text=heading_text)

            except Exception as e:
                logger.error(f"An error occurred at row {i}:")
                logger.error(regulation_df.iloc[i])
                logger.error(f"Error message: {e}")
                break

    def check_columns(self):
        ''' 
            'text'              : the text on the line excluding the 'reference' and any special text (identifying headings, page number etc)
            'heading'           : boolean identifying the text as as (sub-) section heading
            'section_reference' : the full reference. Starting at the root section_reference and ending with the value in 'reference'
        '''
        expected_columns = ['text', 'heading', 'section_reference'] # this is a minimum - it could contain more

        actual_columns = self.regulation_df.columns.to_list()
        for column in expected_columns:
            if column not in actual_columns:
                print(f"{column} not in the DataFrame version of the manual")
                return False
        return True



def _split_recursive(node, regulation_reader, table_of_content, token_limit, node_list=[]):
    """    
    Recursively splits nodes based on token limits and collects valid nodes in a list.
    You shouldn't need to call this method. Rather use the "split_tree()" method
    
    Parameters:
    - node (Node): The current node being processed.
    - dataframe (pd.DataFrame): DataFrame containing the regulation details.
    - token_limit (int): The maximum allowed token count per section.
    - index_checker (callable): Function to check if an index is valid.
    - node_list (list, optional): List to collect nodes meeting the token criteria.
    
    Returns:
    - list: A list of nodes that meet the token criteria.
    """
    if node_list is None:
        node_list = []

    subsection_text = regulation_reader.get_regulation_detail(node.full_node_name)
    token_count = num_tokens_from_string(subsection_text)

    if token_count > token_limit:
        if not node.children:
            raise Exception(f'Node {node.full_node_name} has no children but has a token count of {token_count} so it cannot be split into nodes that contain fewer tokens that {token_limit}')
        for child in node.children:
            _split_recursive(child, regulation_reader, table_of_content, token_limit, node_list)
    else:
        node_list.append(node)

    return node_list


def split_tree(node, regulation_reader, table_of_content, token_limit):
    """
    Splits a tree starting from a given node into sections that don't exceed a token limit.
    
    Initially this is used to set up the base DataFrame using node == root and later it can be used if we want 
    to change the word_limit for a specific piece of regulation to change chunking where it makes sense.

    Parameters:
    - node (Node): The starting node to split the tree.
    - dataframe (pd.DataFrame): DataFrame containing regulation details.
    - token_limit (int): The maximum allowed token count per section.
    - index_checker (callable): Function to check if an index is valid.
    
    Returns:
    - pd.DataFrame: A DataFrame with columns ['section_reference', 'text', 'token_count'] for each valid section_reference.
    """
    node_list = _split_recursive(node, regulation_reader, table_of_content, token_limit, node_list=[])
    section_token_count = [[node.full_node_name, 
                            regulation_reader.get_regulation_detail(node.full_node_name),
                            num_tokens_from_string(regulation_reader.get_regulation_detail(node.full_node_name))] 
                           for node in node_list]


    return pd.DataFrame(section_token_count, columns=['section_reference', 'text', 'token_count'])


