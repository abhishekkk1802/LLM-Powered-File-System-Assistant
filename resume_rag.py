"""
Resume RAG System
================

Part A of the RAG Job Matching Assignment (50% of total grade).
Document processing pipeline with intelligent chunking and vector storage.

This module provides:
- Intelligent document chunking preserving resume sections
- Embeddings generation using multiple providers
- Vector database storage with metadata
- Semantic search capabilities

Usage:
    from resume_rag import ResumeRAG
    
    # Initialize RAG system
    rag = ResumeRAG(vector_db="chromadb")
    
    # Process resume directory
    rag.process_resume_directory("examples/resumes")
    
    # Search for candidates
    results = rag.semantic_search("Python machine learning experience", k=5)
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import hashlib
from datetime import datetime

# Vector database and embeddings imports
import chromadb
from chromadb.config import Settings
import openai
from dotenv import load_dotenv

# Optional import for sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Import our existing file system tools
from fs_tools import read_file, list_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResumeMetadata:
    """Structured metadata for resume documents."""
    name: str
    file_path: str
    skills: List[str]
    experience_years: Optional[int]
    education: List[str]
    job_titles: List[str]
    companies: List[str]
    certifications: List[str]
    programming_languages: List[str]
    file_hash: str
    processed_date: str


@dataclass
class DocumentChunk:
    """Represents a semantically meaningful chunk of a resume."""
    id: str
    content: str
    section_type: str  # e.g., "summary", "experience", "education", "skills"
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class ResumeRAG:
    """
    Resume Retrieval-Augmented Generation system.
    
    Handles document processing, chunking, embedding generation,
    and vector storage for semantic resume search.
    """
    
    def __init__(self, 
                 vector_db: str = "chromadb",
                 embedding_model: str = "openai",
                 collection_name: str = "resume_collection"):
        """
        Initialize the Resume RAG system.
        
        Args:
            vector_db (str): Vector database type ("chromadb", "pinecone", "weaviate")
            embedding_model (str): Embedding model ("openai", "sentence-transformers", "cohere")
            collection_name (str): Name for the vector collection
        """
        self.vector_db_type = vector_db
        self.embedding_model_type = embedding_model
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = self._initialize_embedding_model()
        
        # Initialize vector database
        self.vector_db = self._initialize_vector_db()
        
        # Resume processing patterns
        self.section_patterns = {
            "contact": r"(contact|email|phone|linkedin|address)",
            "summary": r"(summary|profile|objective|about)",
            "experience": r"(experience|work|employment|career|professional)",
            "education": r"(education|academic|degree|university|college)",
            "skills": r"(skills|technical|technologies|tools|programming)",
            "projects": r"(projects|portfolio|achievements)",
            "certifications": r"(certifications|certificates|licensed)"
        }
        
        # Skill extraction patterns
        self.skill_patterns = {
            "programming_languages": [
                "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
                "scala", "r", "matlab", "sql", "html", "css", "php", "swift", "kotlin"
            ],
            "frameworks": [
                "react", "angular", "vue", "django", "flask", "spring", "node.js",
                "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"
            ],
            "tools": [
                "docker", "kubernetes", "git", "jenkins", "aws", "azure", "gcp",
                "mongodb", "postgresql", "redis", "elasticsearch"
            ]
        }
        
        logger.info(f"Initialized Resume RAG with {vector_db} and {embedding_model}")
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        if self.embedding_model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            
            return openai.OpenAI(**client_kwargs)
            
        elif self.embedding_model_type == "sentence-transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is not available. Please install it with:\n"
                    "pip install sentence-transformers\n"
                    "Or use 'openai' embedding model instead."
                )
            return SentenceTransformer('all-MiniLM-L6-v2')
            
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model_type}")
    
    def _initialize_vector_db(self):
        """Initialize the vector database."""
        if self.vector_db_type == "chromadb":
            # Create persistent ChromaDB client
            client = chromadb.PersistentClient(
                path="./vector_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                collection = client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection '{self.collection_name}'")
            except Exception:
                collection = client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Resume embeddings for job matching"}
                )
                logger.info(f"Created new collection '{self.collection_name}'")
            
            return collection
            
        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db_type}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if self.embedding_model_type == "openai":
            response = self.embedding_model.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            return [embedding.embedding for embedding in response.data]
            
        elif self.embedding_model_type == "sentence-transformers":
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
            
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model_type}")
    
    def extract_metadata(self, content: str, file_path: str) -> ResumeMetadata:
        """
        Extract structured metadata from resume content.
        
        Args:
            content (str): Resume text content
            file_path (str): Path to the resume file
            
        Returns:
            ResumeMetadata: Extracted metadata
        """
        content_lower = content.lower()
        
        # Extract name (first line typically)
        lines = content.strip().split('\n')
        name = lines[0].strip() if lines else "Unknown"
        
        # Clean name (remove titles, contact info)
        name = re.sub(r'\b(resume|cv|curriculum vitae)\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[^\w\s]', ' ', name).strip()
        
        # Extract skills
        skills = self._extract_skills(content)
        programming_languages = self._extract_programming_languages(content)
        
        # Extract experience years
        experience_years = self._extract_experience_years(content)
        
        # Extract education
        education = self._extract_education(content)
        
        # Extract job titles and companies
        job_titles, companies = self._extract_work_history(content)
        
        # Extract certifications
        certifications = self._extract_certifications(content)
        
        # Create file hash for change detection
        file_hash = hashlib.md5(content.encode()).hexdigest()
        
        return ResumeMetadata(
            name=name,
            file_path=file_path,
            skills=skills,
            experience_years=experience_years,
            education=education,
            job_titles=job_titles,
            companies=companies,
            certifications=certifications,
            programming_languages=programming_languages,
            file_hash=file_hash,
            processed_date=datetime.now().isoformat()
        )
    
    def _extract_skills(self, content: str) -> List[str]:
        """Extract technical skills from resume content."""
        skills = []
        content_lower = content.lower()
        
        # Extract from all skill categories
        for category, skill_list in self.skill_patterns.items():
            for skill in skill_list:
                if skill in content_lower:
                    skills.append(skill)
        
        return list(set(skills))  # Remove duplicates
    
    def _extract_programming_languages(self, content: str) -> List[str]:
        """Extract programming languages specifically."""
        content_lower = content.lower()
        languages = []
        
        for lang in self.skill_patterns["programming_languages"]:
            if lang in content_lower:
                languages.append(lang)
        
        return languages
    
    def _extract_experience_years(self, content: str) -> Optional[int]:
        """Extract years of experience from resume."""
        # Look for patterns like "5+ years", "3 years experience", etc.
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'over\s*(\d+)\s*years?',
            r'more\s*than\s*(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return max(int(match) for match in matches)
        
        return None
    
    def _extract_education(self, content: str) -> List[str]:
        """Extract education information."""
        education = []
        
        # Common degree patterns
        degree_patterns = [
            r'(bachelor|master|phd|doctorate|associate).*?(?:degree|of|in)\s*([^\n\r.]+)',
            r'(b\.?\s*[as]\.?|m\.?\s*[as]\.?|ph\.?d\.?|m\.?b\.?a\.?)\s*([^\n\r.,]+)',
            r'(university|college|institute)\s*([^\n\r.,]+)'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    education.append(' '.join(match).strip())
                else:
                    education.append(match.strip())
        
        return education[:5]  # Limit to 5 entries
    
    def _extract_work_history(self, content: str) -> Tuple[List[str], List[str]]:
        """Extract job titles and companies."""
        job_titles = []
        companies = []
        
        # Common title patterns
        title_patterns = [
            r'(senior|lead|principal|junior)?\s*(engineer|developer|analyst|manager|director|specialist|consultant|architect)',
            r'(software|data|product|project|marketing|sales|finance)\s*(engineer|manager|analyst|director)',
        ]
        
        for pattern in title_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    title = ' '.join(match).strip()
                else:
                    title = match.strip()
                if title and title not in job_titles:
                    job_titles.append(title)
        
        # Extract companies (lines with "Inc", "Corp", "LLC", etc.)
        company_patterns = [
            r'([A-Z][a-zA-Z\s&]+(?:Inc|Corp|LLC|Ltd|Company|Technologies|Solutions)\.?)',
            r'([A-Z][a-zA-Z\s&]+(?:University|College|Institute))'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                company = match.strip()
                if company and len(company) > 2 and company not in companies:
                    companies.append(company)
        
        return job_titles[:5], companies[:5]  # Limit results
    
    def _extract_certifications(self, content: str) -> List[str]:
        """Extract certifications and licenses."""
        cert_patterns = [
            r'(certified|certification)\s*([^\n\r.,]+)',
            r'([A-Z]{2,})\s*certified',
            r'(aws|azure|google cloud|oracle|cisco|microsoft)\s*([^\n\r.,]+)'
        ]
        
        certifications = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    cert = ' '.join(match).strip()
                else:
                    cert = match.strip()
                if cert and len(cert) > 3:
                    certifications.append(cert)
        
        return certifications[:5]  # Limit results
    
    def chunk_resume(self, content: str, metadata: ResumeMetadata) -> List[DocumentChunk]:
        """
        Intelligently chunk resume content preserving semantic sections.
        
        Args:
            content (str): Resume text content
            metadata (ResumeMetadata): Resume metadata
            
        Returns:
            List[DocumentChunk]: List of document chunks
        """
        chunks = []
        
        # Split content into lines for section detection
        lines = content.split('\n')
        current_section = "summary"
        current_content = []
        section_start_idx = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip().lower()
            
            # Detect section headers
            detected_section = None
            for section, pattern in self.section_patterns.items():
                if re.search(pattern, line_stripped) and len(line_stripped) < 100:
                    detected_section = section
                    break
            
            # If new section detected, save current chunk
            if detected_section and current_content:
                chunk_content = '\n'.join(current_content).strip()
                if chunk_content:
                    chunk_id = f"{metadata.name}_{current_section}_{section_start_idx}"
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_content,
                    section_type=current_section,
                    metadata={
                        "resume_name": metadata.name,
                        "file_path": metadata.file_path,
                        "section": current_section,
                        "chunk_index": len(chunks),
                        "skills": ",".join(metadata.skills) if metadata.skills else "",
                        "programming_languages": ",".join(metadata.programming_languages) if metadata.programming_languages else "",
                        "experience_years": metadata.experience_years if metadata.experience_years is not None else 0
                    }
                )
                chunks.append(chunk)
                
                # Start new section
                current_section = detected_section
                current_content = [line]
                section_start_idx = i
            else:
                current_content.append(line)
        
        # Add final chunk
        if current_content:
            chunk_content = '\n'.join(current_content).strip()
            if chunk_content:
                chunk_id = f"{metadata.name}_{current_section}_{section_start_idx}"
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_content,
                    section_type=current_section,
                    metadata={
                        "resume_name": metadata.name,
                        "file_path": metadata.file_path,
                        "section": current_section,
                        "chunk_index": len(chunks),
                        "skills": ",".join(metadata.skills) if metadata.skills else "",
                        "programming_languages": ",".join(metadata.programming_languages) if metadata.programming_languages else "",
                        "experience_years": metadata.experience_years if metadata.experience_years is not None else 0
                    }
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for {metadata.name}")
        return chunks
    
    def store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Store document chunks in the vector database.
        
        Args:
            chunks (List[DocumentChunk]): List of chunks to store
        """
        if not chunks:
            return
        
        # Generate embeddings for all chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Store in vector database
        self.vector_db.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Stored {len(chunks)} chunks in vector database")
    
    def process_resume(self, file_path: str) -> ResumeMetadata:
        """
        Process a single resume file.
        
        Args:
            file_path (str): Path to resume file
            
        Returns:
            ResumeMetadata: Processed resume metadata
        """
        logger.info(f"Processing resume: {file_path}")
        
        # Read file content
        result = read_file(file_path)
        if not result["success"]:
            logger.error(f"Failed to read {file_path}: {result['error']}")
            return None
        
        content = result["content"]
        
        # Extract metadata
        metadata = self.extract_metadata(content, file_path)
        
        # Chunk the document
        chunks = self.chunk_resume(content, metadata)
        
        # Store chunks in vector database
        self.store_chunks(chunks)
        
        logger.info(f"Successfully processed {metadata.name}")
        return metadata
    
    def process_resume_directory(self, directory: str) -> List[ResumeMetadata]:
        """
        Process all resume files in a directory.
        
        Args:
            directory (str): Directory containing resume files
            
        Returns:
            List[ResumeMetadata]: List of processed resume metadata
        """
        logger.info(f"Processing resume directory: {directory}")
        
        # List all resume files
        files = list_files(directory, ".txt")  # Add other extensions as needed
        
        processed_resumes = []
        for file in files:
            metadata = self.process_resume(file["filepath"])
            if metadata:
                processed_resumes.append(metadata)
        
        logger.info(f"Processed {len(processed_resumes)} resumes from {directory}")
        return processed_resumes
    
    def semantic_search(self, query: str, k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform semantic search on resume chunks.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            filters (Optional[Dict]): Metadata filters
            
        Returns:
            List[Dict]: Search results with scores and metadata
        """
        # Generate embedding for query
        query_embedding = self.generate_embeddings([query])[0]
        
        # Perform vector search
        results = self.vector_db.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": (1 - results["distances"][0][i]) * 100,  # Convert distance to similarity (0-100 scale)
                "query": query
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector collection."""
        try:
            count = self.vector_db.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_type,
                "vector_db": self.vector_db_type
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    print("🔍 Resume RAG System - Example Usage")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        rag = ResumeRAG(
            vector_db="chromadb",
            embedding_model="openai",  # or "sentence-transformers"
            collection_name="resume_collection"
        )
        
        # Process resume directory
        print("📂 Processing resume directory...")
        processed = rag.process_resume_directory("examples/resumes")
        
        print(f"✅ Processed {len(processed)} resumes")
        
        # Show collection stats
        stats = rag.get_collection_stats()
        print(f"📊 Collection stats: {stats}")
        
        # Example semantic search
        print("\n🔍 Example semantic search:")
        results = rag.semantic_search("Python machine learning data science", k=5)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['metadata']['resume_name']} (Score: {result['similarity_score']:.1f}/100)")
            print(f"   Section: {result['metadata']['section']}")
            print(f"   Content preview: {result['content'][:100]}...")
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure to:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Install dependencies: pip install -r requirements.txt")