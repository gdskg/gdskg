import re
from typing import List, Tuple

class SecretScanner:
    """
    Scanner for identifying potential environment variables and secrets in source code.
    
    Uses regular expressions to detect common patterns of environment variable 
    usage across multiple programming languages.
    """

    # Common patterns for env vars / secrets
    COMBINED_PATTERN = re.compile(
        r"process\.env\.([A-Z_][A-Z0-9_]*)|"
        r"os\.environ\.get\(['\"]([A-Z_][A-Z0-9_]*)['\"]|"
        r"os\.getenv\(['\"]([A-Z_][A-Z0-9_]*)['\"]|"
        r"ENV\[['\"]([A-Z_][A-Z0-9_]*)['\"]\]|"
        r"\$([A-Z_][A-Z0-9_]*)|"
        r"([A-Z_][A-Z0-9_]*_KEY)|"
        r"([A-Z_][A-Z0-9_]*_SECRET)|"
        r"([A-Z_][A-Z0-9_]*_TOKEN)"
    )

    def scan(self, content: str) -> List[str]:
        """
        Scan the provided content for potential secrets and environment variables.

        Args:
            content (str): The text content of the file to scan.

        Returns:
            List[str]: A list of unique environment variable or secret names found.
        """

        found = set()
        for match in self.COMBINED_PATTERN.finditer(content):
            for group in match.groups():
                if group:
                    found.add(group)
        return list(found)
