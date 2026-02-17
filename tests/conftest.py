import pytest
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

@pytest.fixture(scope="session", autouse=True)
def setup_test_repos():
    """
    Session-scoped fixture to create test repos before any tests run,
    and destroy them after all tests complete.
    """
    print("\n[setup] Generating test repositories...")
    
    # Run creation scripts
    # We assume the scripts are runnable via python -m or direct file execution
    # Using subprocess to call the CLI commands defined in generate_test_repo.py
    
    gen_script = PROJECT_ROOT / "scripts" / "generate_test_repo.py"
    
    # Generate Python Repo
    subprocess.run([sys.executable, str(gen_script), "python-repo"], check=True)
    
    # Generate TypeScript Repo
    subprocess.run([sys.executable, str(gen_script), "typescript-repo"], check=True)
    
    yield
    
    print("\n[teardown] Destroying test repositories...")
    dest_script = PROJECT_ROOT / "scripts" / "destroy_test_repos.py"
    subprocess.run([sys.executable, str(dest_script)], check=True)
