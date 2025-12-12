"""
Resume Embedder for Interview Coach
Creates vector embeddings for resume chunks using sentence-transformers.
"""

from typing import List, Dict, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from rag.chunker import Chunk


class ResumeEmbedder:
    """
    Embed resume chunks and store in ChromaDB.

    Uses sentence-transformers for embeddings and ChromaDB for vector storage.
    """

    def __init__(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            persist_dir: str = "./data/chromadb",
            collection_name: str = "resumes"
    ):
        """
        Initialize embedder.

        Args:
            model_name: Sentence transformer model to use
            persist_dir: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
        """
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Initialize ChromaDB
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Resume chunks for interview coach"}
        )

        print(f"ChromaDB collection '{collection_name}' ready")
        print(f"Current document count: {self.collection.count()}")

    def embed_chunks(
            self,
            chunks: List[Chunk],
            resume_id: str,
            clear_existing: bool = False
    ) -> int:
        """
        Embed and store chunks for a resume.

        Args:
            chunks: List of Chunk objects to embed
            resume_id: Unique identifier for this resume
            clear_existing: If True, remove existing chunks for this resume_id

        Returns:
            Number of chunks embedded
        """
        if clear_existing:
            self._delete_resume(resume_id)

        if not chunks:
            return 0

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{resume_id}_chunk_{i}"

            documents.append(chunk.content)

            # Flatten metadata for ChromaDB (only strings, ints, floats, bools)
            flat_metadata = {
                "resume_id": resume_id,
                "section": chunk.section,
                "chunk_index": i,
            }
            # Add chunk metadata (flatten nested dicts)
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    flat_metadata[key] = value
                elif isinstance(value, list):
                    flat_metadata[key] = ", ".join(str(v) for v in value[:10])

            metadatas.append(flat_metadata)
            ids.append(chunk_id)

        # Generate embeddings
        embeddings = self.model.encode(documents, show_progress_bar=True)

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

        print(f"Embedded {len(chunks)} chunks for resume: {resume_id}")
        return len(chunks)

    def _delete_resume(self, resume_id: str) -> None:
        """Delete all chunks for a resume."""
        try:
            # Get all chunks for this resume
            results = self.collection.get(
                where={"resume_id": resume_id}
            )
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                print(f"Deleted {len(results['ids'])} existing chunks for {resume_id}")
        except Exception as e:
            print(f"Warning: Could not delete existing chunks: {e}")

    def search(
            self,
            query: str,
            n_results: int = 5,
            resume_id: Optional[str] = None,
            section_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            n_results: Number of results to return
            resume_id: Filter by specific resume
            section_filter: Filter by section (skills, experience, etc.)

        Returns:
            List of matching chunks with scores
        """
        # Build where filter
        where_filter = None
        if resume_id or section_filter:
            conditions = []
            if resume_id:
                conditions.append({"resume_id": resume_id})
            if section_filter:
                conditions.append({"section": section_filter})

            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}

        # Query
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance": 1 - results["distances"][0][i]  # Convert distance to similarity
                })

        return formatted

    def get_all_chunks(self, resume_id: str) -> List[Dict]:
        """Get all chunks for a specific resume."""
        results = self.collection.get(
            where={"resume_id": resume_id},
            include=["documents", "metadatas"]
        )

        chunks = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                chunks.append({
                    "content": doc,
                    "metadata": results["metadatas"][i]
                })

        return chunks

    def clear_collection(self) -> None:
        """Clear all data from collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Resume chunks for interview coach"}
        )
        print("Collection cleared")

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection_name,
            "embedding_model": self.model_name
        }


def main():
    """Test embedder with sample resume."""
    from rag.resume_parser import ResumeParser
    from rag.chunker import ResumeChunker
    from pathlib import Path

    # Initialize components
    parser = ResumeParser()
    chunker = ResumeChunker()
    embedder = ResumeEmbedder()

    # Find test file
    sample_dir = Path("data/sample_resumes")
    files = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.txt"))

    if not files:
        print("No resume files found")
        return

    test_file = files[0]
    print(f"\n{'=' * 60}")
    print(f"Testing with: {test_file.name}")
    print("=" * 60)

    # Parse → Chunk → Embed
    parsed = parser.parse(str(test_file))
    chunks = chunker.chunk(parsed)

    resume_id = test_file.stem  # filename without extension
    embedder.embed_chunks(chunks, resume_id, clear_existing=True)

    # Test search
    print(f"\n{'=' * 60}")
    print("SEARCH TESTS")
    print("=" * 60)

    test_queries = [
        "Python programming experience",
        "machine learning projects",
        "leadership and team management",
        "education background",
        "cloud computing AWS"
    ]

    for query in test_queries:
        print(f"\n Query: '{query}'")
        results = embedder.search(query, n_results=2, resume_id=resume_id)

        for i, result in enumerate(results):
            print(f"\n  Result {i + 1} (relevance: {result['relevance']:.3f}):")
            print(f"  Section: {result['metadata']['section']}")
            print(f"  Content: {result['content'][:150]}...")

    # Print stats
    print(f"\n{'=' * 60}")
    print("STATS")
    print("=" * 60)
    print(embedder.get_stats())


if __name__ == "__main__":
    main()