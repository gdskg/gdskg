from typing import List, Dict, Tuple, Set, Optional
from tree_sitter import Node as TSNode
from analysis.tree_sitter_utils import TreeSitterUtils
import re
import bisect

class SymbolExtractor:
    """
    Extractor for analyzing source code symbols using tree-sitter.
    
    Identifies definitions, imports, and usages of functions, classes, and variables
    across different programming languages to build a symbolic representation 
    of the codebase.
    """

    def __init__(self):
        """
        Initialize the SymbolExtractor with supported definition types.
        """

        self.definition_types = {
            "function_definition", 
            "class_definition", 
            "method_definition",
            "lexical_declaration",
            "variable_declaration",
            "function_declaration"
        }
        self.function_types = {
            "function_definition",
            "method_definition",
            "function_declaration"
        }

    def _extract_name(self, node: TSNode, content: str) -> Optional[str]:
        """
        Internal helper to extract the identifier name from a tree-sitter node.

        Args:
            node (TSNode): The tree-sitter node to analyze.
            content (str): The source code content.

        Returns:
            Optional[str]: The extracted name, or None if not found.
        """

        # 1. Look for child 'name'
        name_node = node.child_by_field_name("name")
        if name_node:
            return content[name_node.start_byte:name_node.end_byte]
            
        # 2. For variable/lexical declarations (TS/JS), the name is in a declarator
        if node.type in ["lexical_declaration", "variable_declaration"]:
            # usually children are variable_declarator
            for child in node.children:
                if child.type == "variable_declarator":
                    return self._extract_name(child, content)
        
        # 3. For variable_declarator, name is 'name' field
        if node.type == "variable_declarator":
             name_node = node.child_by_field_name("name")
             if name_node:
                 return content[name_node.start_byte:name_node.end_byte]
                 
        # 4. For C/C++ function_definition, name is nested in declarator
        if node.type == "function_definition":
             declarator_node = node.child_by_field_name("declarator")
             if declarator_node:
                 if declarator_node.type == "function_declarator":
                     inner_dec = declarator_node.child_by_field_name("declarator")
                     if inner_dec:
                         return content[inner_dec.start_byte:inner_dec.end_byte]
                 else:
                     return content[declarator_node.start_byte:declarator_node.end_byte]
        
        # 5. For React arrow components exporting directly e.g. export const Button
        if node.type == "export_statement":
             declaration_node = node.child_by_field_name("declaration")
             if declaration_node:
                 return self._extract_name(declaration_node, content)
                 
        return None

    def _extract_imports(self, node: TSNode, content: str, language: str) -> Dict[str, str]:
        """
        Extract import statements and resolve aliases for a given language.

        Args:
            node (TSNode): The tree-sitter node representing an import.
            content (str): The source code content.
            language (str): The programming language of the file.

        Returns:
            Dict[str, str]: A mapping of local aliases to canonical symbol names.
        """

        imports = {}
        
        # Python Imports
        if language == "python":
            if node.type == "import_statement":
                # import os, sys
                for child in node.children:
                    if child.type == "dotted_name":
                        name = content[child.start_byte:child.end_byte]
                        imports[name] = name
                    elif child.type == "aliased_import":
                        # import numpy as np
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if name_node and alias_node:
                            name = content[name_node.start_byte:name_node.end_byte]
                            alias = content[alias_node.start_byte:alias_node.end_byte]
                            imports[alias] = name

            elif node.type == "import_from_statement":
                # from os import path
                module_node = node.child_by_field_name("module_name")
                module_name = content[module_node.start_byte:module_node.end_byte] if module_node else ""
                
                for child in node.children:
                    if child.type == "aliased_import":
                         # from x import y as z
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if name_node and alias_node:
                            name = content[name_node.start_byte:name_node.end_byte]
                            alias = content[alias_node.start_byte:alias_node.end_byte]
                            canonical = f"{module_name}.{name}" if module_name else name
                            imports[alias] = canonical
                    elif child.type == "dotted_name" and child != module_node:
                         # from . import y (relative) or similar structure
                         name = content[child.start_byte:child.end_byte]
                         canonical = f"{module_name}.{name}" if module_name else name
                         imports[name] = canonical
                    elif child.type == "identifier" and child != module_node:
                        # from x import y
                        # In some parsers/versions, simple names might be identifiers
                        name = content[child.start_byte:child.end_byte]
                        canonical = f"{module_name}.{name}" if module_name else name
                        imports[name] = canonical

        # TypeScript/JS Imports
        elif language in ["typescript", "tsx", "javascript"]:
            if node.type == "import_statement":
                source = ""
                for child in node.children:
                    if child.type == "string":
                        source = content[child.start_byte:child.end_byte].strip("'\"")
                
                for child in node.children:
                    if child.type == "import_clause":
                        for subchild in child.children:
                            if subchild.type == "named_imports":
                                for specifier in subchild.children:
                                    if specifier.type == "import_specifier":
                                        name_node = specifier.child_by_field_name("name")
                                        alias_node = specifier.child_by_field_name("alias")
                                        
                                        if name_node and alias_node:
                                            name = content[name_node.start_byte:name_node.end_byte]
                                            alias = content[alias_node.start_byte:alias_node.end_byte]
                                            imports[alias] = f"{source}.{name}"
                                        elif name_node: # Sometimes name is just a child identifier
                                            name = content[name_node.start_byte:name_node.end_byte]
                                            imports[name] = f"{source}.{name}"
                                        else:
                                            # Fallback: iterate children for identifier
                                            for deep_child in specifier.children:
                                                if deep_child.type == "identifier":
                                                    name = content[deep_child.start_byte:deep_child.end_byte]
                                                    imports[name] = f"{source}.{name}"
                            elif subchild.type == "identifier":
                                # import Default from 'y'
                                name = content[subchild.start_byte:subchild.end_byte]
                                imports[name] = f"{source}.default"

        return imports

    def build_symbol_table(self, file_content: str, language: str, tree=None) -> Dict[str, str]:
        """
        Build a comprehensive mapping of identifiers to their canonical names.
        
        This includes both locally defined symbols (functions, classes) and 
        imported symbols from external modules.

        Args:
            file_content (str): The content of the file to analyze.
            language (str): The programming language.
            tree: Optional tree-sitter Tree.

        Returns:
            Dict[str, str]: A dictionary mapping identifiers to canonical names.
        """

        if tree is None:
            parser = TreeSitterUtils.get_parser(language)
            if not parser:
                return {}
            tree = parser.parse(bytes(file_content, "utf8"))
            
        root_node = tree.root_node
        
        symbol_table = {}
        
        def traverse(node: TSNode, scope: str = ""):
            # 1. Definitions
            if node.type in self.definition_types:
                name = self._extract_name(node, file_content)
                if name:
                    canonical = f"{scope}.{name}" if scope else name
                    symbol_table[name] = canonical
                    
                    # Recurse into definition with new scope
                    for child in node.children:
                        traverse(child, canonical)
                    return # Skip default recursion since we handled it

            # 2. Imports
            if node.type in ["import_statement", "import_from_statement"]:
                 imports = self._extract_imports(node, file_content, language)
                 symbol_table.update(imports)
            
            # Recurse
            for child in node.children:
                traverse(child, scope)

        traverse(root_node)
        return symbol_table

    def extract_symbols(self, file_content: str, language: str, tree=None) -> Dict[str, str]:
        """
        Extract only the definitions (classes/functions) from the source code.

        Args:
            file_content (str): The source code content.
            language (str): The programming language.
            tree: Optional tree-sitter Tree.

        Returns:
            Dict[str, str]: A mapping of definition names to their canonical scoped names.
        """

        # Backward compatibility / specific definition extraction if needed
        # For now, just return definitions from build_symbol_table?
        # No, extract_symbols meant "What checks do we create SYMBOL nodes for?"
        # We probably only want to create nodes for DEFINITIONS, not IMPORTS.
        # So we should keep a version that only returns definitions.
        
        if tree is None:
            parser = TreeSitterUtils.get_parser(language)
            if not parser:
                return {}
            tree = parser.parse(bytes(file_content, "utf8"))
            
        root_node = tree.root_node
        
        definitions = {}
        def traverse(node: TSNode, scope: str = ""):
             if node.type in self.definition_types:
                name = self._extract_name(node, file_content)
                if name:
                    canonical = f"{scope}.{name}" if scope else name
                    definitions[name] = canonical
                    for child in node.children:
                        traverse(child, canonical)
                    return
             for child in node.children:
                traverse(child, scope)
        traverse(root_node)
        return definitions

    def filter_diff_symbols(self, file_content: str, language: str, affected_lines: Set[int], symbol_table: Dict[str, str] = None, tree=None) -> List[str]:
        """
        Identify symbols that were directly impacted by changes in specific lines.

        Args:
            file_content (str): The source code content.
            language (str): The programming language.
            affected_lines (Set[int]): A set of line numbers changed in the diff.
            symbol_table (Dict[str, str], optional): A pre-built symbol table for resolution.
            tree: Optional tree-sitter Tree.

        Returns:
            List[str]: A list of canonical symbol names modified or used in the diff.
        """

        if symbol_table is None:
            symbol_table = {}
            
        if tree is None:
            parser = TreeSitterUtils.get_parser(language)
            if not parser:
                return []
            tree = parser.parse(bytes(file_content, "utf8"))

        root_node = tree.root_node
        
        matched_symbols = set()
        if not affected_lines:
            return list(matched_symbols)
        sorted_affected = sorted(list(affected_lines))

        def traverse(node: TSNode, current_scope: str = ""):
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            if end_line < sorted_affected[0] or start_line > sorted_affected[-1]:
                return  # Entire subtree is outside the affected range
                
            idx = bisect.bisect_left(sorted_affected, start_line)
            intersects = idx < len(sorted_affected) and sorted_affected[idx] <= end_line
            
            # Update scope if defining
            new_scope = current_scope
            if node.type in self.definition_types:
                name = self._extract_name(node, file_content)
                if name:
                     new_scope = f"{current_scope}.{name}" if current_scope else name
            
            if intersects:
                # Check for Identifiers (Usages)
                if node.type == "identifier":
                    name = file_content[node.start_byte:node.end_byte]
                    if name in symbol_table:
                        matched_symbols.add(symbol_table[name])
                
                # Check for Definitions (Explicit Modification)
                if node.type in self.definition_types:
                    name = self._extract_name(node, file_content)
                    if name:
                         canonical = f"{current_scope}.{name}" if current_scope else name
                         matched_symbols.add(canonical)

                for child in node.children:
                    traverse(child, new_scope)

        traverse(root_node)
        return list(matched_symbols)

    def extract_modified_functions(self, file_content: str, language: str, affected_lines: Set[int], tree=None) -> Dict[str, str]:
        """
        Identify functions and methods that were directly modified in the given lines.

        Args:
            file_content (str): The source code content.
            language (str): The programming language.
            affected_lines (Set[int]): Lines changed in the current commit.
            tree: Optional tree-sitter Tree.

        Returns:
            Dict[str, str]: A mapping of canonical function names to their current source text.
        """
        if tree is None:
            parser = TreeSitterUtils.get_parser(language)
            if not parser:
                return {}
            tree = parser.parse(bytes(file_content, "utf8"))

        root_node = tree.root_node
        
        modified_functions = {}
        if not affected_lines:
            return modified_functions
            
        import bisect
        sorted_affected = sorted(list(affected_lines))
        
        def traverse(node: TSNode, current_scope: str = ""):
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            if end_line < sorted_affected[0] or start_line > sorted_affected[-1]:
                return
                
            idx = bisect.bisect_left(sorted_affected, start_line)
            intersects = idx < len(sorted_affected) and sorted_affected[idx] <= end_line
            
            new_scope = current_scope
            if node.type in self.definition_types:
                name = self._extract_name(node, file_content)
                if name:
                    new_scope = f"{current_scope}.{name}" if current_scope else name
                    
                    if intersects and node.type in self.function_types:
                        content = file_content[node.start_byte:node.end_byte]
                        modified_functions[new_scope] = content
                        
            if intersects:
                for child in node.children:
                    traverse(child, new_scope)

        traverse(root_node)
        return modified_functions

    def extract_comments(self, file_content: str, language: str, affected_lines: Set[int], tree=None) -> List[Tuple[str, str, int]]:
        """
        Extract docstrings and inline comments that intersect with affected lines.

        Args:
            file_content (str): The source code content.
            language (str): The programming language.
            affected_lines (Set[int]): Lines changed in the current commit.
            tree: Optional tree-sitter Tree.

        Returns:
            List[Tuple[str, str, int]]: A list of (content, type, line_number) tuples.
        """
        if tree is None:
            parser = TreeSitterUtils.get_parser(language)
            if not parser:
                return []
            tree = parser.parse(bytes(file_content, "utf8"))

        root_node = tree.root_node
        comments = []
        if not affected_lines:
            return comments
            
        import bisect
        sorted_affected = sorted(list(affected_lines))

        def traverse(node: TSNode):
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            if end_line < sorted_affected[0] or start_line > sorted_affected[-1]:
                return
                
            idx = bisect.bisect_left(sorted_affected, start_line)
            intersects = idx < len(sorted_affected) and sorted_affected[idx] <= end_line
            
            is_comment = False
            comment_type = "inline"

            # Check for standard comments
            if node.type in ["comment", "line_comment", "block_comment"]:
                is_comment = True

            # Check for Python docstrings (expression_statement -> string)
            elif language == "python" and node.type == "expression_statement":
                # A docstring in Python is often an expression statement containing only a string
                if len(node.children) == 1 and node.children[0].type == "string":
                    is_comment = True
                    comment_type = "docstring"

            # Check for JS/TS docstrings (often inside specific comment formats, but handled via 'comment' usually)
            
            if is_comment and intersects:
                content = file_content[node.start_byte:node.end_byte].strip()
                # Clean python strings if docstring
                if comment_type == "docstring":
                   content = content.strip('\'" \n\t')
                
                if content:
                    comments.append((content, comment_type, start_line))

            if intersects:
                for child in node.children:
                    traverse(child)

        traverse(root_node)
        return comments

    def extract_all_symbols(self, file_content: str, language: str, tree=None) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Combined single-traversal extraction of both definitions and full symbol table.

        Args:
            file_content (str): The source code content.
            language (str): The programming language.
            tree: Optional tree-sitter Tree.

        Returns:
             Tuple[Dict[str, str], Dict[str, str]]: A tuple containing (definitions_mapping, full_symbol_table).
        """
        if tree is None:
            parser = TreeSitterUtils.get_parser(language)
            if not parser:
                return {}, {}
            tree = parser.parse(bytes(file_content, "utf8"))

        root_node = tree.root_node
        definitions = {}
        symbol_table = {}

        def traverse(node: TSNode, scope: str = ""):
            if node.type in self.definition_types:
                name = self._extract_name(node, file_content)
                if name:
                    canonical = f"{scope}.{name}" if scope else name
                    definitions[name] = canonical
                    symbol_table[name] = canonical
                    for child in node.children:
                        traverse(child, canonical)
                    return

            if node.type in ["import_statement", "import_from_statement"]:
                imports = self._extract_imports(node, file_content, language)
                symbol_table.update(imports)

            for child in node.children:
                traverse(child, scope)

        traverse(root_node)
        return definitions, symbol_table

    def analyze_diff(self, file_content: str, language: str, affected_lines: Set[int],
                     symbol_table: Dict[str, str] = None, tree=None) -> Tuple[List[str], Dict[str, str], List[Tuple[str, str, int]]]:
        """
        Combined single-traversal analysis of affected lines to identify modified symbols, functions, and comments.

        Args:
            file_content (str): The source code content.
            language (str): The programming language.
            affected_lines (Set[int]): A set of changed line numbers.
            symbol_table (Dict[str, str], optional): A mapping of local aliases to canonical names.
            tree: Optional tree-sitter Tree.

        Returns:
            Tuple[List[str], Dict[str, str], List[Tuple[str, str, int]]]: (modified_symbols, modified_functions, comments).
        """
        empty_result = ([], {}, [])
        if not affected_lines:
            return empty_result

        if symbol_table is None:
            symbol_table = {}

        if tree is None:
            parser = TreeSitterUtils.get_parser(language)
            if not parser:
                return empty_result
            tree = parser.parse(bytes(file_content, "utf8"))

        root_node = tree.root_node
        matched_symbols = set()
        modified_functions = {}
        comments = []
        sorted_affected = sorted(list(affected_lines))
        comment_node_types = {"comment", "line_comment", "block_comment"}

        def traverse(node: TSNode, current_scope: str = ""):
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if end_line < sorted_affected[0] or start_line > sorted_affected[-1]:
                return

            idx = bisect.bisect_left(sorted_affected, start_line)
            intersects = idx < len(sorted_affected) and sorted_affected[idx] <= end_line

            new_scope = current_scope
            if node.type in self.definition_types:
                name = self._extract_name(node, file_content)
                if name:
                    new_scope = f"{current_scope}.{name}" if current_scope else name

            if intersects:
                # filter_diff_symbols logic
                if node.type == "identifier":
                    name = file_content[node.start_byte:node.end_byte]
                    if name in symbol_table:
                        matched_symbols.add(symbol_table[name])

                if node.type in self.definition_types:
                    name = self._extract_name(node, file_content)
                    if name:
                        canonical = f"{current_scope}.{name}" if current_scope else name
                        matched_symbols.add(canonical)
                        # extract_modified_functions logic
                        if node.type in self.function_types:
                            content = file_content[node.start_byte:node.end_byte]
                            modified_functions[canonical] = content

                # extract_comments logic
                is_comment = False
                comment_type = "inline"
                if node.type in comment_node_types:
                    is_comment = True
                elif language == "python" and node.type == "expression_statement":
                    if len(node.children) == 1 and node.children[0].type == "string":
                        is_comment = True
                        comment_type = "docstring"

                if is_comment:
                    content = file_content[node.start_byte:node.end_byte].strip()
                    if comment_type == "docstring":
                        content = content.strip("'\" \n\t")
                    if content:
                        comments.append((content, comment_type, start_line))

                for child in node.children:
                    traverse(child, new_scope)

        traverse(root_node)
        return list(matched_symbols), modified_functions, comments
