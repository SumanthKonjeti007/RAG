from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

class AdvancedDataIndexer:
    def __init__(self):
        self.vector_db = ChromaDB()
        self.embedding_model = InstructorEmbedding("hkunlp/instructor-xl")
        self.llm = Anthropic(model="claude-3-5-sonnet-20241022")
    
    def ingest_documents(self, documents):
        """
        Apply best pre-retrieval indexing techniques
        """
        for doc in documents:
            # ===== TECHNIQUE 1: Enhanced Data Granularity =====
            cleaned_doc = self.clean_document(doc)
            
            # ===== TECHNIQUE 2: Sliding Window Chunking =====
            chunks = self.sliding_window_chunk(
                cleaned_doc,
                chunk_size=512,
                overlap=128  # 25% overlap
            )
            
            for i, chunk in enumerate(chunks):
                # ===== TECHNIQUE 3: Small-to-Big Retrieval =====
                # Create small chunk for embedding
                small_chunk = chunk['text'][:256]  # First 256 chars
                
                # Create large context (small chunk + surrounding context)
                large_context = self.create_large_context(
                    current_chunk=chunk['text'],
                    previous_chunk=chunks[i-1]['text'] if i > 0 else "",
                    next_chunk=chunks[i+1]['text'] if i < len(chunks)-1 else "",
                    document_title=doc['title'],
                    document_type=doc['type']
                )
                
                # ===== TECHNIQUE 4: Add Rich Metadata =====
                metadata = {
                    'document_id': doc['id'],
                    'title': doc['title'],
                    'type': doc['type'],  # 'docs', 'ticket', 'release_notes', 'api'
                    'date': doc['date'],
                    'section': chunk['section'],
                    'page': chunk['page'],
                    'topic': self.extract_topic(chunk['text']),
                    'keywords': self.extract_keywords(chunk['text']),
                    'product_version': doc.get('version', 'all'),
                    'difficulty_level': self.classify_difficulty(chunk['text'])
                }
                
                # ===== TECHNIQUE 5: Instructor Embeddings =====
                # Use instruction to guide embedding
                instruction = self.get_instruction_for_type(doc['type'])
                embedding = self.embedding_model.encode([
                    [instruction, small_chunk]  # Embed small chunk only
                ])[0]
                
                # Store in vector DB
                self.vector_db.add(
                    embedding=embedding,
                    small_text=small_chunk,      # For display
                    large_context=large_context,  # For LLM prompt
                    metadata=metadata
                )
    
    def clean_document(self, doc):
        """Enhanced Data Granularity"""
        text = doc['text']
        
        # Remove irrelevant sections
        text = self.remove_boilerplate(text)
        
        # Update outdated information
        text = self.update_version_references(text)
        
        # Fix broken links
        text = self.fix_links(text)
        
        # Remove duplicate information
        text = self.deduplicate_content(text)
        
        return text
    
    def sliding_window_chunk(self, doc, chunk_size=512, overlap=128):
        """Sliding Window Chunking"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = splitter.split_text(doc['text'])
        
        return [
            {
                'text': chunk,
                'section': self.detect_section(chunk, doc),
                'page': self.estimate_page(i, chunk_size)
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def create_large_context(self, current_chunk, previous_chunk, 
                            next_chunk, document_title, document_type):
        """Small-to-Big: Create enriched context"""
        
        # Use LLM to generate document context
        context_prompt = f"""
        Document Title: {document_title}
        Document Type: {document_type}
        
        Here is a chunk from this document:
        {current_chunk}
        
        Previous context: {previous_chunk[:100]}...
        Next context: {next_chunk[:100]}...
        
        Provide a 2-3 sentence context that situates this chunk within 
        the larger document, so it can be understood independently.
        """
        
        contextual_intro = self.llm.generate(context_prompt)
        
        # Combine everything
        large_context = f"""
        [Document: {document_title} | Type: {document_type}]
        
        Context: {contextual_intro}
        
        Content:
        {previous_chunk[-200:] if previous_chunk else ''}
        
        >>> {current_chunk} <
        
        {next_chunk[:200] if next_chunk else ''}
        """
        
        return large_context
    
    def get_instruction_for_type(self, doc_type):
        """Different instructions for different document types"""
        instructions = {
            'docs': "Represent the technical documentation for retrieval:",
            'ticket': "Represent the customer support ticket for retrieval:",
            'release_notes': "Represent the product update information for retrieval:",
            'api': "Represent the API reference documentation for retrieval:"
        }
        return instructions.get(doc_type, "Represent the document for retrieval:")
    
    def extract_topic(self, text):
        """Extract main topic using LLM"""
        return self.llm.generate(f"What is the main topic of: {text[:200]}... "
                                f"Answer in 2-3 words.")
    
    def extract_keywords(self, text):
        """Extract key terms"""
        # Simple keyword extraction (in production, use spaCy or LLM)
        return ['SSO', 'authentication', 'enterprise', 'setup']  # example
    
    def classify_difficulty(self, text):
        """Classify content difficulty"""
        return self.llm.generate(f"Rate this content difficulty (beginner/intermediate/advanced): {text[:200]}")