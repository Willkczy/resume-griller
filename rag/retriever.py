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
        Build a prompt for the LLM with clear question type guidance.

        Args:
            resume_id: ID of processed resume
            focus_area: Optional focus area
            question_type: "hr", "tech", or "mixed" (converted from mode)
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

        # Build DETAILED instruction based on question type
        if question_type == "tech" or question_type == "technical":
            instruction = f"""Generate {n_questions} TECHNICAL interview questions.

    FOCUS ON:
    - Specific technologies, frameworks, and tools mentioned in the resume
    - Architecture and design decisions in their projects
    - Implementation details and technical trade-offs
    - Performance optimization and debugging approaches
    - Code quality, testing strategies, and best practices
    - System design and scalability considerations

    QUESTION STYLE:
    - "Walk me through the architecture of [specific project]..."
    - "How did you implement/optimize [specific feature]..."
    - "What technical challenges did you face in [project]..."
    - "Why did you choose [technology] over alternatives..."

    DO NOT ask behavioral or STAR-method questions."""

        elif question_type == "hr" or question_type == "behavioral":
            instruction = f"""Generate {n_questions} BEHAVIORAL (HR) interview questions using the STAR method.

    FOCUS ON:
    - Leadership and team collaboration experiences
    - Conflict resolution and difficult situations
    - Time management and handling pressure
    - Communication with stakeholders
    - Career growth, learning, and adaptability
    - Achievements and their impact on the team/company

    QUESTION STYLE:
    - "Tell me about a time when you [situation]..."
    - "Describe a situation where you had to [challenge]..."
    - "Give me an example of when you [behavior]..."
    - "How did you handle [difficult situation]..."

    DO NOT ask for technical implementation details. Focus on experiences, behaviors, and soft skills."""

        else:  # mixed
            instruction = f"""Generate {n_questions} interview questions (BALANCED mix of technical and behavioral).

    SPLIT: {n_questions // 2} technical + {n_questions - (n_questions // 2)} behavioral questions.

    TECHNICAL QUESTIONS should ask about:
    - Project architecture, technical decisions, implementation
    - Specific technologies and tools used
    - Challenges and problem-solving approaches

    BEHAVIORAL QUESTIONS should ask about:
    - Team collaboration, leadership experiences
    - Handling difficult situations, conflict resolution
    - Time management, communication skills

    Ensure clear distinction between technical and behavioral questions."""

        if focus_area:
            instruction += f"\n\nSPECIAL FOCUS: Prioritize questions related to {focus_area}."

        # Format final prompt with stronger guidance
        prompt = f"""You are an expert {question_type.upper()} interviewer preparing for a rigorous mock interview.

    {instruction}

    CANDIDATE'S RESUME:
    {context}

    REQUIREMENTS:
    1. Questions must be SPECIFIC to this candidate's actual experience
    2. Reference actual projects, technologies, or experiences from the resume
    3. Avoid generic questions that could apply to any candidate
    4. Each question should probe for depth and specific details
    5. Questions should be challenging but fair

    Generate exactly {n_questions} questions, numbered 1-{n_questions}."""

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

    # Build prompts for different scenarios
    print(f"\n3. Generated Prompts:")

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