class QueryOptimizer:
    def __init__(self):
        self.llm = Anthropic(model="claude-3-5-sonnet-20241022")
        self.embedding_model = InstructorEmbedding("hkunlp/instructor-xl")
    
    def optimize_query(self, user_query):
        """
        Apply multiple query optimization techniques
        """
        optimizations = {}
        
        # ===== TECHNIQUE 1: Query Routing =====
        route = self.route_query(user_query)
        optimizations['route'] = route
        
        # ===== TECHNIQUE 2: Self-Query (Extract Filters) =====
        filters = self.extract_filters(user_query)
        optimizations['filters'] = filters
        
        # ===== TECHNIQUE 3: Query Rewriting =====
        rewritten_queries = self.rewrite_query(user_query)
        optimizations['rewritten_queries'] = rewritten_queries
        
        # ===== TECHNIQUE 4: HyDE (Hypothetical Document Embeddings) =====
        hypothetical_answer = self.generate_hypothetical_answer(user_query)
        optimizations['hypothetical_answer'] = hypothetical_answer
        
        # ===== TECHNIQUE 5: Query Expansion =====
        expanded_query = self.expand_query(user_query)
        optimizations['expanded_query'] = expanded_query
        
        return optimizations
    
    def route_query(self, query):
        """Query Routing: Determine best data source and strategy"""
        routing_prompt = f"""
        Analyze this customer query: "{query}"
        
        Determine:
        1. Primary data source needed:
           - product_docs (general product documentation)
           - support_tickets (solved customer issues)
           - release_notes (version-specific features)
           - api_docs (API technical reference)
           - multiple (needs multiple sources)
        
        2. Is RAG context needed? (yes/no)
           - "no" for greetings, thank you, simple acknowledgments
        
        3. Query complexity:
           - simple (single concept)
           - moderate (multiple related concepts)
           - complex (requires multiple searches)
        
        Return JSON format:
        {{"source": "...", "needs_rag": true/false, "complexity": "..."}}
        """
        
        response = self.llm.generate(routing_prompt)
        route_info = json.loads(response)
        
        return route_info
    
    def extract_filters(self, query):
        """Self-Query: Extract structured filters from natural language"""
        filter_prompt = f"""
        Extract metadata filters from this query: "{query}"
        
        Identify:
        - Date/version mentions (e.g., "May 2024", "after v2.0")
        - Document types (e.g., "documentation", "tutorial", "release notes")
        - Difficulty level (e.g., "advanced", "beginner guide")
        - Specific topics (e.g., "SSO", "authentication", "API")
        - Product features (e.g., "enterprise features")
        
        Return JSON:
        {{
          "date_after": "YYYY-MM-DD" or null,
          "date_before": "YYYY-MM-DD" or null,
          "doc_types": [...] or null,
          "topics": [...] or null,
          "difficulty": "..." or null,
          "version_min": "..." or null
        }}
        """
        
        response = self.llm.generate(filter_prompt)
        filters = json.loads(response)
        
        # Clean up null values
        filters = {k: v for k, v in filters.items() if v is not None}
        
        return filters
    
    def rewrite_query(self, query):
        """Query Rewriting: Create multiple variations"""
        rewrite_prompt = f"""
        Original query: "{query}"
        
        Create 3 alternative formulations:
        1. More technical/formal version
        2. Simpler/beginner-friendly version
        3. Different phrasing with synonyms
        
        Return as JSON array: ["query1", "query2", "query3"]
        """
        
        response = self.llm.generate(rewrite_prompt)
        rewrites = json.loads(response)
        
        return rewrites
    
    def generate_hypothetical_answer(self, query):
        """HyDE: Generate hypothetical answer to embed"""
        hyde_prompt = f"""
        Question: {query}
        
        Write a detailed, technical answer as it would appear in official 
        documentation. Include specific steps, terminology, and technical details.
        (This is hypothetical - we'll use it to find similar real documentation)
        """
        
        hypothetical = self.llm.generate(hyde_prompt)
        return hypothetical
    
    def expand_query(self, query):
        """Query Expansion: Add synonyms and related terms"""
        expansion_prompt = f"""
        Query: "{query}"
        
        List related terms, synonyms, and alternative phrasings:
        - Technical synonyms
        - Common abbreviations
        - Related concepts
        
        Return as comma-separated list.
        """
        
        response = self.llm.generate(expansion_prompt)
        expanded = query + " " + response
        
        return expanded