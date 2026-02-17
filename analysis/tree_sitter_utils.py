from tree_sitter import Language, Parser
import tree_sitter_languages

class TreeSitterUtils:
    _languages = {}

    @staticmethod
    def get_parser(lang_name: str) -> Parser:
        """
        Get a Tree-sitter parser for the specified language.
        """
        if lang_name not in TreeSitterUtils._languages:
            try:
                language = tree_sitter_languages.get_language(lang_name)
                TreeSitterUtils._languages[lang_name] = language
            except Exception as e:
                # Fallback or error handling for unsupported languages
                print(f"Warning: Could not load language '{lang_name}': {e}")
                return None

        parser = Parser()
        parser.set_language(TreeSitterUtils._languages[lang_name])
        return parser

    @staticmethod
    def map_extension_to_language(extension: str) -> str:
        """
        Map file extension to tree-sitter language name.
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
