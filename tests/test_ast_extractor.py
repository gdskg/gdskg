import pytest
from analysis.ast_extractor import AstExtractor

def test_ast_extractor():
    file_content = "def hello_world():\n    return 'Hello, World!'\n"
    
    # Extract AST in-memory
    nodes = AstExtractor.get_ast_nodes(file_content, "python")
    
    assert len(nodes) > 0, "AST_NODEs should have been created in-memory"
    
    types = [n["attributes"]["ast_type"] for n in nodes]
    assert "function_definition" in types, "Should contain function_definition"
    assert "return_statement" in types, "Should contain return_statement"

    # Test filtering by query
    filtered_nodes = AstExtractor.get_ast_nodes(file_content, "python", query="hello_world function")
    assert len(filtered_nodes) > 0, "Query filtering should return matching nodes"
    assert "similarity" in filtered_nodes[0]
