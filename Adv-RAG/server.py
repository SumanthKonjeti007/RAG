# Boilderplate code for Advanced RAG System
class AdvancedRAGSystem:
    def __init__(self):
        self.indexer = AdvancedDataIndexer()
        self.query_optimizer = QueryOptimizer()
        self.retriever = AdvancedRetrieval()
        self.post_processor = PostRetrievalOptimizer()
        self.llm = Anthropic(model="claude-3-5-sonnet-20241022")
    
    def answer_question(self, user_query):
        """
        Complete RAG pipeline with all optimizations
        """
        print(f"\n{'='*60}")
        print(f"USER QUERY: {user_query}")
        print(f"{'='*60}\n")
        
        # ===== STAGE 1: PRE-RETRIEVAL =====
        print("STAGE 1: Pre-Retrieval Optimization")
        print("-" * 60)
        
        # Query optimization
        query_opts = self.query_optimizer.optimize_query(user_query)
        query_opts['original_query'] = user_query
        
        print(f"✓ Route: {query_opts['route']}")
        print(f"✓ Filters: {query_opts['filters']}")
        print(f"✓ Rewrites: {len(query_opts['rewritten_queries'])} variations")
        print(f"✓ HyDE answer generated: {len(query_opts['hypothetical_answer'])} chars")
        
        # Check if RAG is needed
        if not query_opts['route']['needs_rag']:
            print("\n→ No RAG needed, responding directly")
            return self.llm.generate(user_query)
        
        # ===== STAGE 2: RETRIEVAL =====
        print(f"\nSTAGE 2: Retrieval Optimization")
        print("-" * 60)
        
        # Retrieve candidates
        candidates = self.retriever.retrieve(query_opts, top_k=50)
        print(f"✓ Retrieved {len(candidates)} candidates")
        print(f"✓ Hybrid search: semantic + keyword fusion")
        print(f"✓ Multi-query search: {len(query_opts['rewritten_queries']) + 1} queries")
        print(f"✓ HyDE search included")
        
        # ===== STAGE 3: POST-RETRIEVAL =====
        print(f"\nSTAGE 3: Post-Retrieval Optimization")
        print("-" * 60)
        
        # Re-rank and compress
        final_context = self.post_processor.optimize(
            query=user_query,
            candidates=candidates,
            max_context_length=4000
        )
        
        print(f"✓ Re-ranked using cross-encoder")
        print(f"✓ Diversity filtering applied")
        print(f"✓ Compressed context: {len(final_context)} chars")
        
        # ===== GENERATION =====
        print(f"\nSTAGE 4: Answer Generation")
        print("-" * 60)
        
        # Build final prompt
        final_prompt = f"""
        You are a helpful customer support assistant.
        
        Context (from documentation and support history):
        {final_context}
        
        Customer Question: {user_query}
        
        Instructions:
        - Provide a clear, step-by-step answer
        - Reference specific sections when relevant
        - If information is version-specific, mention it
        - If you're not certain, say so
        
        Answer:
        """
        
        answer = self.llm.generate(final_prompt)
        
        print(f"✓ Answer generated")
        print(f"\n{'='*60}\n")
        
        return {
            'answer': answer,
            'sources': [
                {
                    'title': c['metadata']['title'],
                    'type': c['metadata']['type'],
                    'score': c['rerank_score']
                }
                for c in candidates[:5]
            ],
            'metadata': {
                'route': query_opts['route'],
                'filters_used': query_opts['filters'],
                'num_candidates': len(candidates)
            }
        }

# ===== USAGE EXAMPLE =====

# 1. Index documents (one-time setup)
rag_system = AdvancedRAGSystem()

documents = [
    {
        'id': 'doc1',
        'title': 'SSO Setup Guide',
        'type': 'docs',
        'text': '... documentation content ...',
        'date': '2024-06-15',
        'version': 'v2.5'
    },
    # ... more documents
]

print("Indexing documents...")
rag_system.indexer.ingest_documents(documents)
print("✓ Indexing complete\n")

# 2. Answer user query
user_query = "How do I set up SSO authentication for enterprise users added after the May 2024 update?"

result = rag_system.answer_question(user_query)

# 3. Display result
print("FINAL ANSWER:")
print(result['answer'])
print("\nSOURCES:")
for source in result['sources']:
    print(f"  - {source['title']} ({source['type']}) - Score: {source['score']:.3f}")
# ```

# ---

# ## 📊 Example Output
# ```
# ============================================================
# USER QUERY: How do I set up SSO authentication for enterprise users added after the May 2024 update?
# ============================================================

# STAGE 1: Pre-Retrieval Optimization
# ------------------------------------------------------------
# ✓ Route: {'source': 'multiple', 'needs_rag': True, 'complexity': 'moderate'}
# ✓ Filters: {'date_after': '2024-05-01', 'topics': ['SSO', 'authentication', 'enterprise']}
# ✓ Rewrites: 3 variations
# ✓ HyDE answer generated: 450 chars

# STAGE 2: Retrieval Optimization
# ------------------------------------------------------------
# ✓ Retrieved 50 candidates
# ✓ Hybrid search: semantic + keyword fusion
# ✓ Multi-query search: 4 queries
# ✓ HyDE search included

# STAGE 3: Post-Retrieval Optimization
# ------------------------------------------------------------
# Re-ranking 50 candidates...
# Top 3 reranked scores: [0.892, 0.845, 0.801]
# ✓ Re-ranked using cross-encoder
# ✓ Diversity filtering applied
# ✓ Compressed context: 3850 chars

# STAGE 4: Answer Generation
# ------------------------------------------------------------
# ✓ Answer generated

# ============================================================

# FINAL ANSWER:
# To set up SSO authentication for enterprise users added after May 2024:

# 1. Navigate to Settings > Enterprise > Authentication
# 2. Enable "SSO for New Users" (added in v2.5, May 2024 update)
# 3. Configure your identity provider:
#    - SAML 2.0 or OAuth 2.0 supported
#    - Upload your IdP metadata XML
# 4. Set the user attribute mapping for new users
# 5. Test with a new enterprise user account

# Note: Users added before May 2024 need to be migrated separately using the 
# "Bulk User Migration" tool in the Admin panel.

# SOURCES:
#   - SSO Setup Guide (docs) - Score: 0.892
#   - May 2024 Release Notes (release_notes) - Score: 0.845
#   - Enterprise Authentication FAQ (docs) - Score: 0.801
#   - Ticket #5847: SSO for new users (ticket) - Score: 0.756
#   - API: Update User Auth Method (api) - Score: 0.723