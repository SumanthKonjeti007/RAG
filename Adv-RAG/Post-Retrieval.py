from sentence_transformers import CrossEncoder

class PostRetrievalOptimizer:
    def __init__(self):
        # Cross-encoder for re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.llm = Anthropic(model="claude-3-5-sonnet-20241022")
    
    def optimize(self, query, candidates, max_context_length=4000):
        """
        Apply post-retrieval optimizations
        """
        # ===== TECHNIQUE 1: Re-ranking with Cross-Encoder =====
        reranked = self.rerank(query, candidates)
        
        # ===== TECHNIQUE 2: Diversity Filtering =====
        diverse_results = self.ensure_diversity(reranked, top_k=10)
        
        # ===== TECHNIQUE 3: Prompt Compression =====
        compressed_context = self.compress_context(
            diverse_results,
            max_length=max_context_length
        )
        
        return compressed_context
    
    def rerank(self, query, candidates):
        """
        Re-ranking: Use cross-encoder for more accurate relevance scoring
        """
        print(f"Re-ranking {len(candidates)} candidates...")
        
        # Prepare pairs for cross-encoder
        pairs = []
        for candidate in candidates:
            # Use small_text for re-ranking (more focused)
            text = candidate['document']['small_text']
            pairs.append([query, text])
        
        # Get cross-encoder scores (sees query + doc together)
        scores = self.reranker.predict(pairs)
        
        # Attach scores to candidates
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(
            candidates,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        print(f"Top 3 reranked scores: {[r['rerank_score'] for r in reranked[:3]]}")
        
        return reranked
    
    def ensure_diversity(self, ranked_results, top_k=10):
        """
        Diversity Filtering: Avoid redundant similar documents
        """
        diverse = []
        seen_topics = set()
        
        for result in ranked_results:
            topic = result['metadata'].get('topic', '')
            section = result['metadata'].get('section', '')
            
            # Create diversity key
            diversity_key = f"{topic}_{section}"
            
            # Skip if too similar to already selected
            if diversity_key in seen_topics:
                # Only include if score is significantly higher
                if result['rerank_score'] > diverse[-1]['rerank_score'] + 0.1:
                    diverse.append(result)
            else:
                diverse.append(result)
                seen_topics.add(diversity_key)
            
            if len(diverse) >= top_k:
                break
        
        return diverse
    
    def compress_context(self, results, max_length=4000):
        """
        Prompt Compression: Remove redundancy, keep essential info
        """
        # Extract large_context from each result (remember small-to-big?)
        contexts = [r['document']['large_context'] for r in results]
        
        # Combine contexts
        combined = "\n\n---\n\n".join(contexts)
        
        # If within limit, return as-is
        if len(combined) <= max_length:
            return combined
        
        # Otherwise, compress using LLM
        compression_prompt = f"""
        The following contexts contain information relevant to a user's question.
        Remove redundant information and keep only unique, essential details.
        Maintain all specific facts, steps, and technical details.
        
        Target length: ~{max_length} characters
        
        Contexts:
        {combined}
        
        Compressed version:
        """
        
        compressed = self.llm.generate(
            compression_prompt,
            max_tokens=max_length // 4  # Rough token estimate
        )
        
        return compressed