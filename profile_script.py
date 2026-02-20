import cProfile
import pstats
import os
from pathlib import Path
from core.extractor import GraphExtractor
from core.graph_store import GraphStore
from core.schema import Node, NodeType

def run_subset():
    repo_path = Path("./test_repos/sequelize")
    db_path = Path("./test_graph_prof")
    os.makedirs(db_path, exist_ok=True)
    store = GraphStore(db_path / "test.db")
    extractor = GraphExtractor(repo_path, store)
    
    commits = list(extractor.repo.iter_commits(topo_order=True, reverse=True))[:200]
    repo_node = Node(
        id=str(extractor.repo_path),
        type=NodeType.REPOSITORY,
        attributes={
            "name": extractor.repo_path.name,
            "remotes": [r.url for r in extractor.repo.remotes]
        }
    )
    extractor.store.upsert_node(repo_node)
    
    for commit in commits:
        extractor._process_commit(commit, repo_node)
        extractor.store.commit()
        
    extractor._process_keywords()
    extractor.store.commit()

if __name__ == "__main__":
    cProfile.run('run_subset()', 'profile_100.prof')
    p = pstats.Stats('profile_100.prof')
    p.sort_stats('tottime').print_stats(40)
