"""
Retriever for Interview Coach
Retrieves relevant resume chunks and formats prompts for the LLM.
"""

from typing import List, Dict, Optional
from rag.resume_parser import ResumeParser, ParsedResume
from rag.chunker import ResumeChunker, Chunk
from rag.embedder import ResumeEmbedder


class InterviewRetriever:
    """
    End-to-end retrieval pipeline.

    Handles: PDF → Parse → Chunk → Embed → Retrieve → Format Prompt
    """

    def __init__(self, embedder: Optional[ResumeEmbedder] = None):
        self.parser = ResumeParser()
        self.chunker = ResumeChunker()
        self.embedder = embedder or ResumeEmbedder()

    def process_resume(self, file_path: str, resume_id: Optional[str] = None) -> str:
        """
        Process a resume file end-to-end.

        Returns resume_id for future queries.
        """
        from pathlib import Path

        if resume_id is None:
            resume_id = Path(file_path).stem

        # Parse
        parsed = self.parser.parse(file_path)

        # Chunk
        chunks = self.chunker.chunk(parsed)

        # Embed
        self.embedder.embed_chunks(chunks, resume_id, clear_existing=True)

        print(f"Processed resume: {resume_id} ({len(chunks)} chunks)")
        return resume_id

    def retrieve(
            self,
            resume_id: str,
            focus_area: Optional[str] = None,
            n_chunks: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for question generation.

        Args:
            resume_id: ID of processed resume
            focus_area: Optional focus (e.g., "technical skills", "leadership")
            n_chunks: Number of chunks to retrieve

        Returns:
            List of relevant chunks
        """
        if focus_area:
            # Search with focus area as query
            query = f"{focus_area} experience and skills"
            results = self.embedder.search(
                query=query,
                n_results=n_chunks,
                resume_id=resume_id
            )
        else:
            # Get all chunks for this resume
            results = self.embedder.get_all_chunks(resume_id)

        return results

    def build_prompt(
            self,
            resume_id: str,
            focus_area: Optional[str] = None,
            question_type: str = "mixed",
            n_questions: int = 5
    ) -> str:
        """
        Build a prompt for the fine-tuned model.

        Args:
            resume_id: ID of processed resume
            focus_area: Optional focus area
            question_type: "technical", "behavioral", or "mixed"
            n_questions: Number of questions to generate

        Returns:
            Formatted prompt for the LLM
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(resume_id, focus_area)

        # Build context from chunks
        context_parts = []
        for chunk in chunks:
            content = chunk.get("content", "")
            section = chunk.get("metadata", {}).get("section", "")
            context_parts.append(f"[{section.upper()}]\n{content}")

        context = "\n\n".join(context_parts)

        # Build instruction based on question type
        # Support both naming conventions: "tech"/"technical" and "hr"/"behavioral"
        if question_type in ["technical", "tech"]:
            instruction = f"Generate {n_questions} TECHNICAL interview questions focusing on the candidate's skills, projects, architecture decisions, and technical implementation details."
        elif question_type in ["behavioral", "hr"]:
            instruction = f"Generate {n_questions} BEHAVIORAL interview questions using STAR format (Tell me about a time...). Focus on teamwork, leadership, challenges, and soft skills. Do NOT ask technical implementation questions."
        else:
            instruction = f"Generate {n_questions} interview questions (mix of technical and behavioral) based on the candidate's resume."

        if focus_area:
            instruction += f" Focus specifically on: {focus_area}."

        # Format final prompt
        prompt = f"""You are an expert interviewer. {instruction}

Here is the candidate's resume information:

{context}

Generate specific, relevant interview questions based on this resume."""

        return prompt

    def get_resume_summary(self, resume_id: str) -> Dict:
        """Get a summary of the processed resume."""
        chunks = self.embedder.get_all_chunks(resume_id)

        sections = {}
        for chunk in chunks:
            section = chunk.get("metadata", {}).get("section", "unknown")
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk.get("content", "")[:100])

        return {
            "resume_id": resume_id,
            "total_chunks": len(chunks),
            "sections": list(sections.keys()),
            "preview": sections
        }

    def get_full_resume_text(self, resume_id: str) -> str:
        """
        Get the full text of a resume by combining all chunks.
        
        Used by HybridModelService for preprocessing.
        
        Args:
            resume_id: The resume ID
            
        Returns:
            Full resume text as a single string
        """
        try:
            # Get all chunks using the embedder
            chunks = self.embedder.get_all_chunks(resume_id)
            
            if not chunks:
                print(f"[Retriever] No chunks found for resume: {resume_id}")
                return ""
            
            # Sort by chunk_index if available
            def get_chunk_index(chunk):
                metadata = chunk.get("metadata", {})
                return metadata.get("chunk_index", 0)
            
            sorted_chunks = sorted(chunks, key=get_chunk_index)
            
            # Combine all text with section headers
            text_parts = []
            for chunk in sorted_chunks:
                content = chunk.get("content", "")
                section = chunk.get("metadata", {}).get("section", "")
                
                if section:
                    text_parts.append(f"[{section.upper()}]\n{content}")
                else:
                    text_parts.append(content)
            
            full_text = "\n\n".join(text_parts)
            print(f"[Retriever] Got full resume text: {len(full_text)} chars, {len(sorted_chunks)} chunks")
            
            return full_text
            
        except Exception as e:
            print(f"[Retriever] Error getting full resume text: {e}")
            return ""


def main():
    """Test the full retrieval pipeline."""
    from pathlib import Path

    retriever = InterviewRetriever()

    # Find test file
    sample_dir = Path("data/sample_resumes")
    files = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.txt"))

    if not files:
        print("No resume files found")
        return

    test_file = files[0]

    print("=" * 60)
    print("INTERVIEW RETRIEVER TEST")
    print("=" * 60)

    # Process resume
    print(f"\n1. Processing: {test_file.name}")
    resume_id = retriever.process_resume(str(test_file))

    # Get summary
    print(f"\n2. Resume Summary:")
    summary = retriever.get_resume_summary(resume_id)
    print(f"   Chunks: {summary['total_chunks']}")
    print(f"   Sections: {summary['sections']}")

    # Test get_full_resume_text
    print(f"\n3. Full Resume Text:")
    full_text = retriever.get_full_resume_text(resume_id)
    print(f"   Length: {len(full_text)} chars")
    print(f"   Preview: {full_text[:200]}...")

    # Build prompts for different scenarios
    print(f"\n4. Generated Prompts:")

    # General prompt
    print(f"\n{'=' * 40}")
    print("GENERAL INTERVIEW PROMPT:")
    print("=" * 40)
    prompt = retriever.build_prompt(resume_id, question_type="mixed")
    print(prompt[:800] + "..." if len(prompt) > 800 else prompt)

    # Technical focused
    print(f"\n{'=' * 40}")
    print("TECHNICAL FOCUSED (Python):")
    print("=" * 40)
    prompt = retriever.build_prompt(
        resume_id,
        focus_area="Python programming",
        question_type="technical",
        n_questions=3
    )
    print(prompt[:800] + "..." if len(prompt) > 800 else prompt)

    # Behavioral focused
    print(f"\n{'=' * 40}")
    print("BEHAVIORAL FOCUSED:")
    print("=" * 40)
    prompt = retriever.build_prompt(
        resume_id,
        question_type="behavioral",
        n_questions=3
    )
    print(prompt[:800] + "..." if len(prompt) > 800 else prompt)


if __name__ == "__main__":
    main()