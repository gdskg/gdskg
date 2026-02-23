from typing import List, Dict, Optional
import os
import uuid
import logging
import numpy as np
from tree_sitter import Node as TSNode

from analysis.tree_sitter_utils import TreeSitterUtils
from analysis.embedder import ONNXEmbedder

logger = logging.getLogger(__name__)

class AstExtractor:
    """
    Extracts the Abstract Syntax Tree (AST) of a given source string in-memory
    and returns it as a list of dictionary nodes formatted for GDSKG search results.
    """

    @staticmethod
    def _create_ast_node_recursive(ts_node: TSNode, file_content: str, all_nodes: List[Dict], max_depth: int, current_depth: int, parent_id: Optional[str] = None) -> str:
        if max_depth > 0 and current_depth > max_depth:
            return ""

        node_id = f"AST:{uuid.uuid4().hex[:12]}"
        
        text = ""
        if ts_node.is_named and (len(ts_node.children) == 0 or ts_node.type in ['identifier', 'string', 'number']):
            text = ts_node.text.decode('utf-8', errors='ignore') if ts_node.text else ""
            
        attributes = {
            "ast_type": ts_node.type,
            "start_point": [ts_node.start_point[0], ts_node.start_point[1]],
            "end_point": [ts_node.end_point[0], ts_node.end_point[1]],
            "is_named": ts_node.is_named,
        }
        if text:
            attributes["text"] = text
            
        all_nodes.append({
            "id": node_id,
            "type": "AST_NODE",
            "attributes": attributes,
            "depth": current_depth
        })
            
        for child in ts_node.children:
            AstExtractor._create_ast_node_recursive(child, file_content, all_nodes, max_depth, current_depth + 1, node_id)
            
        return node_id

    @staticmethod
    def get_ast_nodes(file_content: str, language: str, max_depth: int = 0, query: str = None) -> List[Dict]:
        """
        Parses the given file content and creates an in-memory list of AST node dictionaries.
        If a query is provided, it extracts all nodes and then filters them by semantic similarity.
        """
        parser = TreeSitterUtils.get_parser(language)
        if not parser:
            return []
            
        tree = parser.parse(file_content.encode('utf-8'))
        if not tree or not tree.root_node:
            return []
            
        all_nodes = []
        AstExtractor._create_ast_node_recursive(tree.root_node, file_content, all_nodes, max_depth if not query else 0, 0)
        
        if not query:
            return all_nodes
            
        embedder = ONNXEmbedder()
        q_embs = embedder.embed([query])
        
        if not q_embs or len(q_embs[0]) == 0:
            return []
            
        query_embedding = q_embs[0][0].astype(np.float32)
        norm_q = np.linalg.norm(query_embedding)
        if norm_q == 0:
            return []
            
        texts = [n["attributes"].get("text", "") + " " + n["attributes"].get("ast_type", "") for n in all_nodes]
        n_embs_list = embedder.embed(texts)
        
        filtered_nodes = []
        for i, node in enumerate(all_nodes):
            n_embs = n_embs_list[i]
            if len(n_embs) == 0:
                continue
                
            node_emb = n_embs[0].astype(np.float32)
            norm_n = np.linalg.norm(node_emb)
            if norm_n > 0:
                sim = float(np.dot(query_embedding, node_emb) / (norm_q * norm_n))
                if sim >= 0.20:
                    node["similarity"] = sim
                    filtered_nodes.append(node)
                    
        # Sort by similarity
        filtered_nodes.sort(key=lambda x: x["similarity"], reverse=True)
        
        # If max_depth was specified but we ignored it for the query structure, we should
        # probably respect it for the output, though usually query implies doing a full search.
        if max_depth > 0:
             filtered_nodes = [n for n in filtered_nodes if n["depth"] <= max_depth]
             
        return filtered_nodes
