"""
Document Processor - Handles document loading and indexing
Supports PDF, TXT, MD files (DOCX, PPT as nice-to-have)
"""

import os
from pathlib import Path
from typing import List, Optional


class DocumentProcessor:
    """
    Document Processor for loading and indexing documents.
    
    Supported formats (MVP):
    - PDF
    - TXT
    - MD (Markdown)
    
    Nice-to-have:
    - DOCX
    - PPT
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.textbooks_dir = self.data_dir / "textbooks"
        self.uploads_dir = self.data_dir / "user_uploads"
        self.index = None
        self.embed_model = None
        
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.textbooks_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        
    def load_documents(self, directory: Optional[Path] = None) -> List:
        """
        Load documents from the specified directory.
        
        Args:
            directory: Path to load documents from. Defaults to textbooks_dir.
            
        Returns:
            List of loaded documents.
        """
        from llama_index.core import SimpleDirectoryReader
        
        load_dir = directory or self.textbooks_dir
        
        if not load_dir.exists():
            print(f"Directory {load_dir} does not exist.")
            return []
        
        # Define file extensions to load
        required_exts = [".pdf", ".txt", ".md"]
        
        try:
            reader = SimpleDirectoryReader(
                input_dir=str(load_dir),
                required_exts=required_exts,
                recursive=True
            )
            documents = reader.load_data()
            print(f"Loaded {len(documents)} documents from {load_dir}")
            return documents
            
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return []
    
    def create_index(self, documents: List):
        """
        Create a vector index from documents.
        
        Args:
            documents: List of documents to index.
            
        Returns:
            VectorStoreIndex instance.
        """
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        # Set up embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        Settings.embed_model = self.embed_model
        
        # Create index
        self.index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        
        print("Index created successfully!")
        return self.index
    
    def process_uploaded_file(self, file_path: str) -> bool:
        """
        Process a single uploaded file and add to index.
        
        Args:
            file_path: Path to the uploaded file.
            
        Returns:
            True if successful, False otherwise.
        """
        from llama_index.core import SimpleDirectoryReader
        
        try:
            # Load the single file
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            
            if self.index is None:
                self.create_index(documents)
            else:
                # Insert into existing index
                for doc in documents:
                    self.index.insert(doc)
                    
            print(f"Processed and indexed: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return False
    
    def get_index(self):
        """Return the current index."""
        return self.index
