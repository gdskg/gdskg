import re
from typing import List, Tuple

class SecretScanner:
    """
    Scanner for identifying potential environment variables and secrets in source code.
    
    Uses regular expressions to detect common patterns of environment variable 
    usage across multiple programming languages.
    """

    # Common patterns for env vars / secrets
    PATTERNS = [
        r"process\.env\.([A-Z_][A-Z0-9_]*)",  # JS/TS
        r"os\.environ\.get\(['\"]([A-Z_][A-Z0-9_]*)['\"]", # Python
        r"os\.getenv\(['\"]([A-Z_][A-Z0-9_]*)['\"]",      # Python
        r"ENV\[['\"]([A-Z_][A-Z0-9_]*)['\"]\]",            # Ruby/Crystal
        r"\$([A-Z_][A-Z0-9_]*)",                           # Shell
        r"([A-Z_][A-Z0-9_]*_KEY)",                         # General Heuristic
        r"([A-Z_][A-Z0-9_]*_SECRET)",                      # General Heuristic
        r"([A-Z_][A-Z0-9_]*_TOKEN)",                       # General Heuristic
    ]
    
    COMPILED = [re.compile(p) for p in PATTERNS]

    def scan(self, content: str) -> List[str]:
        """
        Scan the provided content for potential secrets and environment variables.

        Args:
            content (str): The text content of the file to scan.

        Returns:
            List[str]: A list of unique environment variable or secret names found.
        """

        found = set()
        for pattern in self.COMPILED:
            matches = pattern.findall(content)
            for match in matches:
                # Some groups might be distinct, strict regexes usually return the group
                if isinstance(match, tuple):
                    match = match[0] 
                if match:
                    found.add(match)
        return list(found)
