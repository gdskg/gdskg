import pytest
from pathlib import Path
import sqlite3
from core.graph_store import GraphStore
from core.extractor import GraphExtractor
import git

@pytest.fixture
def import_repo(tmp_path):
    repo_dir = tmp_path / "import_repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)
    return repo_dir, repo

def test_imports_and_hunk_filtering(import_repo, tmp_path):
    repo_dir, repo = import_repo
    
    # 1. Create a library file with a function
    lib_file = repo_dir / "lib.py"
    lib_file.write_text("def shared_func():\n    pass\n\ndef unused_func():\n    pass\n")
    repo.index.add([str(lib_file)])
    repo.index.commit("Add library")
    
    # 2. Create a main file that imports it
    main_file = repo_dir / "main.py"
    # Commit 1: Initial (Calls shared_func)
    main_file.write_text("from lib import shared_func\n\ndef main():\n    shared_func()\n")
    repo.index.add([str(main_file)])
    repo.index.commit("Add main using shared_func")
    
    # 3. Commit 2: Add a new call (Modify main)
    # We add a call to 'print' (builtin) and 'shared_func' again?
    # Let's add a new import and usage.
    main_file.write_text("from lib import shared_func, unused_func\n\ndef main():\n    shared_func()\n    unused_func()\n")
    repo.index.add([str(main_file)])
    c2 = repo.index.commit("Add call to unused_func")
    
    # Run Exractor
    db_path = tmp_path / "graph.db"
    store = GraphStore(db_path)
    extractor = GraphExtractor(repo_dir, store)
    extractor.process_repo()
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check Commit 2 (Add call to unused_func)
    # The diff hunk should contain "unused_func" and "from lib import ... unused_func"
    # So 'unused_func' should be linked.
    # 'shared_func' is in the file but NOT in the hunk (except maybe context). 
    # If it's context, it shouldn't be linked.
    
    commit_id = c2.hexsha
    
    c.execute("""
        SELECT source_id FROM edges 
        WHERE target_id=? AND type='MODIFIED_SYMBOL'
    """, (commit_id,))
    
    symbols = [r[0] for r in c.fetchall()]
    print(f"DEBUG: Symbols for commit {commit_id}: {symbols}")
    
    # 'unused_func' should be there (Added usage + Added import)
    assert "lib.unused_func" in symbols or "unused_func" in symbols
    
    # 'shared_func' IS in the modified line (Line 1 changed from 'import shared_func' to 'import shared_func, unused_func')
    # So it SHOULD be linked.
    assert "lib.shared_func" in symbols or "shared_func" in symbols

def test_ts_import_usage(import_repo, tmp_path):
    repo_dir, repo = import_repo
    
    # 1. Create component
    comp_file = repo_dir / "Button.tsx"
    comp_file.write_text("export const Button = () => {};")
    repo.index.add([str(comp_file)])
    repo.index.commit("Add Button")
    
    # 2. App using it
    app_file = repo_dir / "App.tsx"
    app_file.write_text("import { Button } from './Button';\n\nexport const App = () => {\n  Button();\n};")
    repo.index.add([str(app_file)])
    c2 = repo.index.commit("Add App using Button")
    
    db_path = tmp_path / "ts_graph.db"
    store = GraphStore(db_path)
    extractor = GraphExtractor(repo_dir, store)
    extractor.process_repo()
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Commit 2 should link 'Button' (lib.Button or similar) and 'App' (Defined)
    symbols = [r[0] for r in c.execute("SELECT source_id FROM edges WHERE target_id=? AND type='MODIFIED_SYMBOL'", (c2.hexsha,)).fetchall()]
    
    assert any("Button" in s for s in symbols), f"Button not found in {symbols}"
    assert any("App" in s for s in symbols), f"App not found in {symbols}"
