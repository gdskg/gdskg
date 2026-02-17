from typing import List, Dict, Tuple, Set
from tree_sitter import Node as TSNode
from analysis.tree_sitter_utils import TreeSitterUtils
import re

class SymbolExtractor:
    def __init__(self):
        pass

    def extract_symbols(self, file_content: str, language: str) -> Dict[str, str]:
        """
        Pass 1: Parse the file and build a Symbol Table (identifier -> canonical_name).
        For now, we'll extract function and class definitions.
        Returns a dict: {identifier: canonical_name}
        """
        parser = TreeSitterUtils.get_parser(language)
        if not parser:
            return {}

        tree = parser.parse(bytes(file_content, "utf8"))
        root_node = tree.root_node
        
        symbols = {}
        
        # Simple query for functions/classes (approximate, varies by language)
        # This is a naive implementation. A robust one needs per-language queries.
        # We will use a generic traversal for now to find "function_definition", "class_definition"
        # and extract their names.
        
        def traverse(node: TSNode, scope: str = ""):
            node_type = node.type
            new_scope = scope
            
            name = None
            if node_type in ["function_definition", "class_definition", "method_definition"]:
                # Find the 'name' child
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = file_content[name_node.start_byte:name_node.end_byte]
                    canonical = f"{scope}.{name}" if scope else name
                    symbols[name] = canonical
                    new_scope = canonical
            
            elif node_type == "identifier" and scope:
                # If we are inside a scope, maybe we want to track it?
                # For GDSKG, we mostly care about DEFINTIONS.
                pass

            for child in node.children:
                traverse(child, new_scope)

        traverse(root_node)
        return symbols

    def filter_diff_symbols(self, file_content: str, language: str, affected_lines: Set[int]) -> List[str]:
        """
        Pass 2: Identify symbols occurring in the affected lines.
        affected_lines: Set of 1-based line numbers.
        Returns a list of Canonical Names.
        """
        parser = TreeSitterUtils.get_parser(language)
        if not parser:
            return []

        tree = parser.parse(bytes(file_content, "utf8"))
        root_node = tree.root_node
        
        matched_symbols = set()

        def traverse(node: TSNode, current_symbol: str = None):
            # Check if this node intersects with affected lines
            # node.start_point is (row, col) 0-indexed.
            start_line = node.start_point.row + 1
            end_line = node.end_point.row + 1
            
            # Intersection logic: if the node overlaps with any affected line
            # This is tricky: we want the *symbol* that is MODIFIED.
            # If a line inside a function is changed, does that count as "Modified Symbol"?
            # Spec says: "Only if the symbol is inside the diff + or - lines."
            # This implies if I change the *body* of a function, the function symbol is modified?
            # Or only if I rename it? "Symbol is inside the diff".
            # Usually strict GDSKG interpretation: The function *definition* creates the symbol.
            # If I stick to specific "identifiers", I should look for identifier nodes.
            
            # Let's try to map "identifiers" in the diff to their definitions.
            
            if node.type == "identifier":
                 if start_line in affected_lines:
                     name = file_content[node.start_byte:node.end_byte]
                     # We assume we have a way to resolve this name to a canonical one.
                     # Without full scope resolution (Pass 1), this is hard.
                     # But for this prototype, we'll return the raw identifier.
                     matched_symbols.add(name)
            
            for child in node.children:
                traverse(child)

        traverse(root_node)
        return list(matched_symbols)
