class AdvancedRetrieval:
    def __init__(self):
        self.vector_db = ChromaDB()
        self.bm25_index = BM25Index()  # Keyword search index
        self.embedding_model = InstructorEmbedding("hkunlp/instructor-xl")
    
    def retrieve(self, query_optimizations, top_k=20):
        """
        Apply retrieval techniques to get candidate documents
        """
        all_candidates = []
        
        # Get optimized query components
        original_query = query_optimizations['original_query']
        route = query_optimizations['route']
        filters = query_optimizations['filters']
        rewritten_queries = query_optimizations['rewritten_queries']
        hypothetical_answer = query_optimizations['hypothetical_answer']
        
        # ===== TECHNIQUE 1: Hybrid Search =====
        # Search with original query
        hybrid_results = self.hybrid_search(
            query=original_query,
            filters=filters,
            alpha=0.7,  # 70% semantic, 30% keyword
            top_k=top_k
        )
        all_candidates.extend(hybrid_results)
        
        # ===== TECHNIQUE 2: Multi-Query Retrieval =====
        # Search with rewritten queries
        for rewritten in rewritten_queries:
            results = self.hybrid_search(
                query=rewritten,
                filters=filters,
                alpha=0.7,
                top_k=top_k // 2
            )
            all_candidates.extend(results)
        
        # ===== TECHNIQUE 3: HyDE Search =====
        # Search with hypothetical answer
        hyde_results = self.semantic_search(
            query=hypothetical_answer,
            filters=filters,
            top_k=top_k // 2
        )
        all_candidates.extend(hyde_results)
        
        # ===== TECHNIQUE 4: Filtered Vector Search =====
        # Apply route-based filtering
        if route['source'] != 'multiple':
            all_candidates = [
                c for c in all_candidates 
                if c['metadata']['type'] == route['source']
            ]
        
        # Deduplicate candidates
        unique_candidates = self.deduplicate(all_candidates)
        
        return unique_candidates
    
    def hybrid_search(self, query, filters, alpha=0.7, top_k=20):
        """
        Hybrid Search: Combine semantic (vector) + keyword (BM25) search
        """
        # ===== A. Semantic Search (Vector) =====
        query_embedding = self.embedding_model.encode([
            ["Represent the customer question for retrieval:", query]
        ])[0]
        
        semantic_results = self.vector_db.similarity_search(
            embedding=query_embedding,
            filters=filters,
            top_k=top_k * 2  # Get more candidates
        )
        
        # Normalize scores to 0-1
        semantic_scores = self.normalize_scores(semantic_results)
        
        # ===== B. Keyword Search (BM25) =====
        keyword_results = self.bm25_index.search(
            query=query,
            filters=filters,
            top_k=top_k * 2
        )
        
        # Normalize scores to 0-1
        keyword_scores = self.normalize_scores(keyword_results)
        
        # ===== C. Fusion: Combine Scores =====
        combined_scores = {}
        
        for doc_id, score in semantic_scores.items():
            combined_scores[doc_id] = alpha * score
        
        for doc_id, score in keyword_scores.items():
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * score
            else:
                combined_scores[doc_id] = (1 - alpha) * score
        
        # Sort by combined score
        ranked_ids = sorted(
            combined_scores.keys(), 
            key=lambda x: combined_scores[x], 
            reverse=True
        )[:top_k]
        
        # Retrieve full documents
        results = [
            {
                'doc_id': doc_id,
                'score': combined_scores[doc_id],
                'document': self.vector_db.get(doc_id),
                'metadata': self.vector_db.get_metadata(doc_id)
            }
            for doc_id in ranked_ids
        ]
        
        return results
    
    def semantic_search(self, query, filters, top_k=20):
        """Pure semantic search (for HyDE)"""
        query_embedding = self.embedding_model.encode([
            ["Represent the text for retrieval:", query]
        ])[0]
        
        results = self.vector_db.similarity_search(
            embedding=query_embedding,
            filters=filters,
            top_k=top_k
        )
        
        return results
    
    def normalize_scores(self, results):
        """Normalize scores to 0-1 range"""
        if not results:
            return {}
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        if score_range == 0:
            return {r['doc_id']: 0.5 for r in results}
        
        normalized = {
            r['doc_id']: (r['score'] - min_score) / score_range
            for r in results
        }
        
        return normalized
    
    def deduplicate(self, candidates):
        """Remove duplicate documents"""
        seen_ids = set()
        unique = []
        
        for candidate in candidates:
            doc_id = candidate['doc_id']
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique.append(candidate)
        
        return unique