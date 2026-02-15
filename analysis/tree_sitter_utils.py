from tree_sitter import Language, Parser
import tree_sitter_languages

class TreeSitterUtils:
    """
    Utility class for interacting with tree-sitter parsers and languages.
    
    Manages parser instantiation and language mapping for various file extensions.
    """

    _languages = {}

    @staticmethod
    def get_parser(lang_name: str) -> Parser:
        """
        Retrieve a tree-sitter parser for the specified language.

        Args:
            lang_name (str): The name of the language (e.g., 'python', 'javascript').

        Returns:
            Parser: A tree-sitter Parser instance, or None if the language is not supported.
        """

        if lang_name not in TreeSitterUtils._languages:
            try:
                language = tree_sitter_languages.get_language(lang_name)
                TreeSitterUtils._languages[lang_name] = language
            except Exception as e:
                print(f"Warning: Could not load language '{lang_name}': {e}")

                return None

        parser = Parser()
        parser.set_language(TreeSitterUtils._languages[lang_name])
        return parser

    @staticmethod
    def map_extension_to_language(extension: str) -> str:
        """
        Map a file extension to its corresponding tree-sitter language name.

        Args:
            extension (str): The file extension (e.g., '.py', '.js').

        Returns:
            str: The tree-sitter language name, or None if the extension is unknown.
        """

        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "c_sharp",
        }
        return ext_map.get(extension.lower())
