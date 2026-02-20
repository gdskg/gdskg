import typer
import git
import shutil
from pathlib import Path
import time
from rich.console import Console

app = typer.Typer()
console = Console()

def setup_repo(name: str):
    # Create in /tmp or local tmp
    base_dir = Path("/tmp/gdskg_test_repos")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    repo_dir = base_dir / name
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir()
    
    repo = git.Repo.init(repo_dir)
    console.print(f"[green]Initialized repo at {repo_dir}[/green]")
    return repo, repo_dir

@app.command()
def python_repo():
    """Create a Python-focused test repository."""
    repo, repo_path = setup_repo("python_test_repo")
    
    # 1. Initial Commit
    (repo_path / "main.py").write_text("def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()\n")
    (repo_path / "utils.py").write_text("def helper():\n    return 42\n")
    repo.index.add(["main.py", "utils.py"])
    repo.index.commit("Initial commit: Basic structure")
    time.sleep(1)

    # 2. Add Feature
    (repo_path / "core").mkdir()
    (repo_path / "core" / "logic.py").write_text("class Processor:\n    def process(self, data):\n        return data * 2\n")
    repo.index.add(["core/logic.py"])
    repo.index.commit("feat: Add core logic processor")
    time.sleep(1)

    # 3. Refactor (Modify existing)
    (repo_path / "utils.py").write_text("def helper():\n    # Returns the answer to everything\n    return 42\n\ndef new_helper():\n    return 0\n")
    repo.index.add(["utils.py"])
    repo.index.commit("refactor: Update utils with comments and new function")
    time.sleep(1)

    # 4. Introduce Secret (Infrastructure)
    (repo_path / "config.py").write_text("import os\n\nDB_PASSWORD = os.getenv('DB_PASS')\nAWS_SECRET = 'AKIA_FAKE_SECRET'\n")
    repo.index.add(["config.py"])
    repo.index.commit("chore: Add configuration (DO NOT COMMIT SECRETS!)")
    
    console.print(f"[bold]Created Python repo at: {repo_path}[/bold]")

@app.command()
def typescript_repo():
    """Create a TypeScript-focused test repository."""
    repo, repo_path = setup_repo("typescript_test_repo")
    
    # 1. Init
    (repo_path / "package.json").write_text('{"name": "ts-app", "version": "1.0.0"}')
    (repo_path / "src").mkdir()
    (repo_path / "src" / "index.ts").write_text("console.log('Starting app...');\n")
    repo.index.add(["package.json", "src/index.ts"])
    repo.index.commit("chore: Initial project setup")
    time.sleep(1)

    # 2. Add Component
    (repo_path / "src" / "Component.tsx").write_text("import React from 'react';\n\nexport const Button = () => <button>Click me</button>;\n")
    repo.index.add(["src/Component.tsx"])
    repo.index.commit("feat: Add Button component")
    time.sleep(1)

    # 3. Modify Component
    (repo_path / "src" / "Component.tsx").write_text("import React from 'react';\n\nexport const Button = ({ label }) => <button>{label}</button>;\n")
    repo.index.add(["src/Component.tsx"])
    repo.index.commit("fix: Make Button label dynamic")
    time.sleep(1)
    
    console.print(f"[bold]Created TypeScript repo at: {repo_path}[/bold]")

@app.command()
def react_repo():
    """Create a React-focused test repository."""
    repo, repo_path = setup_repo("react_test_repo")
    
    # 1. Init
    (repo_path / "package.json").write_text('{"name": "react-app", "version": "1.0.0"}')
    (repo_path / "src").mkdir()
    (repo_path / "src" / "App.jsx").write_text("import React from 'react';\nfunction App() { return <div>App</div>; }\nexport default App;\n")
    repo.index.add(["package.json", "src/App.jsx"])
    repo.index.commit("chore: Initial project setup")
    time.sleep(1)

    # 2. Modify component
    (repo_path / "src" / "App.jsx").write_text("import React from 'react';\nfunction App() { return <div><h1>App</h1></div>; }\nexport default App;\n")
    repo.index.add(["src/App.jsx"])
    repo.index.commit("feat: Make App component fancier")
    time.sleep(1)
    
    console.print(f"[bold]Created React repo at: {repo_path}[/bold]")

@app.command()
def c_repo():
    """Create a C-focused test repository."""
    repo, repo_path = setup_repo("c_test_repo")
    
    # 1. Init
    (repo_path / "src").mkdir()
    (repo_path / "src" / "main.c").write_text("#include <stdio.h>\n\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}\n")
    repo.index.add(["src/main.c"])
    repo.index.commit("chore: Initial project setup")
    time.sleep(1)

    # 2. Add header and function
    (repo_path / "src" / "utils.h").write_text("#ifndef UTILS_H\n#define UTILS_H\n\nint add(int a, int b);\n\n#endif\n")
    (repo_path / "src" / "utils.c").write_text("#include \"utils.h\"\n\nint add(int a, int b) {\n    return a + b;\n}\n")
    (repo_path / "src" / "main.c").write_text("#include <stdio.h>\n#include \"utils.h\"\n\nint main() {\n    printf(\"%d\\n\", add(1, 2));\n    return 0;\n}\n")
    repo.index.add(["src/utils.h", "src/utils.c", "src/main.c"])
    repo.index.commit("feat: Extract add and call from main")
    time.sleep(1)
    
    console.print(f"[bold]Created C repo at: {repo_path}[/bold]")

@app.command()
def misc_repo():
    """Create a repository with miscellaneous file types for testing."""
    repo, repo_path = setup_repo("misc_test_repo")
    
    # 1. Initial Commit
    (repo_path / "index.html").write_text("<!DOCTYPE html>\n<html><body><h1>Hello</h1></body></html>\n")
    (repo_path / "style.css").write_text("body { background-color: #fff; }\nh1 { color: #000; }\n")
    (repo_path / "README.md").write_text("# Misc Repo\nThis is a test repository.\n")
    (repo_path / "Dockerfile").write_text("FROM alpine:latest\nCMD [\"echo\", \"Hello\"]\n")
    (repo_path / "config.yaml").write_text("version: 1\nservices:\n  app:\n    image: myapp\n")
    (repo_path / "data.json").write_text('{\n  "name": "misc",\n  "version": "1.0"\n}\n')
    
    repo.index.add(["index.html", "style.css", "README.md", "Dockerfile", "config.yaml", "data.json"])
    repo.index.commit("chore: Initial setup with various file types")
    time.sleep(1)

    # 2. Modify files
    (repo_path / "index.html").write_text("<!DOCTYPE html>\n<html><body><h1>Hello World</h1></body></html>\n")
    (repo_path / "style.css").write_text("body { background-color: #f0f0f0; }\nh1 { color: #333; }\n")
    (repo_path / "README.md").write_text("# Misc Repo\nThis is a test repository with changes.\n")
    (repo_path / "Dockerfile").write_text("FROM alpine:3.18\nCMD [\"echo\", \"Hello World\"]\n")
    (repo_path / "config.yaml").write_text("version: 2\nservices:\n  app:\n    image: myapp:v2\n")
    (repo_path / "data.json").write_text('{\n  "name": "misc",\n  "version": "2.0"\n}\n')
    
    repo.index.add(["index.html", "style.css", "README.md", "Dockerfile", "config.yaml", "data.json"])
    repo.index.commit("feat: Update files")
    time.sleep(1)
    
    console.print(f"[bold]Created Misc repo at: {repo_path}[/bold]")

if __name__ == "__main__":
    app()
