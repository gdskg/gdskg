import re
from typing import List, Tuple

class SecretScanner:
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
        Scan content for secrets/env vars.
        Returns a list of found variable names.
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
