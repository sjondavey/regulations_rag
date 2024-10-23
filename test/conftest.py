import pandas as pd
import pytest

'''  
conftest.py is indeed a special filename for pytest:
    pytest recognizes this filename automatically.
    It's used to share fixtures across multiple test files.
    You don't need to import it explicitly in your test files.
'''

@pytest.fixture
def dummy_definitions():
    data = [
        ["WRR", "1", "My definition from WRR"],
        ["Plett", "A.1", "My definition from Plett"]
    ]
    return pd.DataFrame(data, columns=["document", "section_reference", "definition"])

@pytest.fixture
def dummy_search_sections():
    data = [
        ["WRR", "1.2", "My Section 1.2 from WRR"],
        ["WRR", "1.3", "My Section 1.3 from WRR"],
        ["Plett", "A.2(A)(i)", "My section A.2(A)(i) from Plett"]
    ]
    return pd.DataFrame(data, columns=["document", "section_reference", "regulation_text"])

@pytest.fixture
def dummy_workflow():
    data = [
        ["map", "Can I see this on a map?"],
    ]
    return pd.DataFrame(data, columns=["workflow_name", "trigger"])
