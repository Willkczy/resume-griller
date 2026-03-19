"""
Retriever for Interview Coach.

Handles resume processing (LLM-based parsing) and prompt building.
At upload time, uses Groq to parse raw PDF text into a structured markdown format.
During interviews, the full parsed resume is passed in state — no per-question retrieval.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber

from backend.app.config import settings


# ============== LLM Resume Parsing Prompt ==============

RESUME_PARSE_PROMPT = """You are a resume parser. Convert the raw resume text below into a clean, structured markdown format.

RULES:
- Preserve ALL information from the original resume — do not omit anything
- Use the exact template below
- If a section has no content, omit it entirely
- For experience entries, include metrics/numbers where available
- Keep bullet points concise but complete

TEMPLATE:
# CANDIDATE: [Full Name]
Contact: [email] | [phone] | [linkedin] | [github/portfolio]

## SUMMARY
[2-3 sentence professional summary]

## SKILLS
[Comma-separated list of all technical and soft skills]

## EXPERIENCE
### [Title] at [Company] ([Start] - [End])
- [bullet point with metrics where available]
...

## EDUCATION
### [Degree], [Institution] ([Year])
...

## PROJECTS
### [Project Name]
- [description]
...

## CERTIFICATIONS
- [cert name]
...

RAW RESUME TEXT:
{raw_text}

Output ONLY the structured markdown. No explanations or commentary."""


class InterviewRetriever:
    """
    Resume processing and retrieval pipeline.

    Upload flow: PDF → extract text → LLM parse → save structured markdown
    Interview flow: read saved markdown from disk (full resume in state)
    """

    def __init__(self, embedder=None):
        # embedder kept for backward compat (old resumes fallback)
        self.embedder = embedder
        self._parsed_dir = Path(settings.PARSED_RESUME_DIR)
        self._parsed_dir.mkdir(parents=True, exist_ok=True)

    async def process_resume(self, file_path: str, resume_id: Optional[str] = None) -> str:
        """
        Process a resume: extract text, parse with LLM, save structured output.

        Returns resume_id for future queries.
        """
        if resume_id is None:
            resume_id = Path(file_path).stem

        # Step 1: Extract raw text from PDF/TXT
        raw_text = self._extract_text(file_path)
        if not raw_text.strip():
            raise ValueError(f"Could not extract text from {file_path}")

        # Step 2: Parse with LLM (Groq)
        parsed_text = await self._parse_with_llm(raw_text)

        # Step 3: Save to disk
        output_path = self._parsed_dir / f"{resume_id}.md"
        output_path.write_text(parsed_text, encoding="utf-8")

        print(f"[Retriever] Processed resume: {resume_id} ({len(parsed_text)} chars) -> {output_path}")
        return resume_id

    def _extract_text(self, file_path: str) -> str:
        """Extract raw text from PDF or text file."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        else:
            return path.read_text(encoding="utf-8")

    async def _parse_with_llm(self, raw_text: str) -> str:
        """Call Groq LLM to parse raw resume text into structured markdown."""
        from backend.app.services.llm_service import LLMServiceFactory

        llm = LLMServiceFactory.get_service(provider="groq")
        prompt = RESUME_PARSE_PROMPT.format(raw_text=raw_text)

        response = await llm.generate(
            prompt=prompt,
            system_prompt="You are a precise resume parser. Output only structured markdown.",
            temperature=0.1,
            max_tokens=3000,
        )
        return response.strip()

    def get_full_resume_text(self, resume_id: str) -> str:
        """
        Get the full LLM-parsed resume text.

        Primary: read from data/parsed_resumes/{resume_id}.md
        Fallback: reassemble from ChromaDB chunks (old resumes)
        """
        # Try LLM-parsed file first
        parsed_path = self._parsed_dir / f"{resume_id}.md"
        if parsed_path.exists():
            text = parsed_path.read_text(encoding="utf-8")
            print(f"[Retriever] Loaded parsed resume: {resume_id} ({len(text)} chars)")
            return text

        # Fallback: old ChromaDB-based reassembly
        if self.embedder:
            return self._get_full_text_from_chunks(resume_id)

        print(f"[Retriever] No parsed resume found for: {resume_id}")
        return ""

    def _get_full_text_from_chunks(self, resume_id: str) -> str:
        """Fallback: reassemble full text from ChromaDB chunks (old resumes)."""
        try:
            chunks = self.embedder.get_all_chunks(resume_id)
            if not chunks:
                return ""

            sorted_chunks = sorted(
                chunks,
                key=lambda c: c.get("metadata", {}).get("chunk_index", 0),
            )
            text_parts = []
            for chunk in sorted_chunks:
                content = chunk.get("content", "")
                section = chunk.get("metadata", {}).get("section", "")
                if section:
                    text_parts.append(f"[{section.upper()}]\n{content}")
                else:
                    text_parts.append(content)

            full_text = "\n\n".join(text_parts)
            print(f"[Retriever] Fallback: reassembled from {len(sorted_chunks)} chunks")
            return full_text
        except Exception as e:
            print(f"[Retriever] Error in chunk fallback: {e}")
            return ""

    def get_resume_summary(self, resume_id: str) -> Dict:
        """Get a summary of the processed resume."""
        parsed_path = self._parsed_dir / f"{resume_id}.md"
        if parsed_path.exists():
            return self._summary_from_parsed(resume_id, parsed_path)

        # Fallback: old ChromaDB-based summary
        if self.embedder:
            return self._summary_from_chunks(resume_id)

        return {
            "resume_id": resume_id,
            "total_chunks": 0,
            "sections": [],
            "preview": {},
        }

    def _summary_from_parsed(self, resume_id: str, parsed_path: Path) -> Dict:
        """Build summary from LLM-parsed markdown file."""
        text = parsed_path.read_text(encoding="utf-8")

        # Extract sections from markdown headers
        sections = []
        for match in re.finditer(r"^## (\w+)", text, re.MULTILINE):
            sections.append(match.group(1).lower())

        # Extract name
        name = None
        name_match = re.search(r"^# CANDIDATE:\s*(.+)", text, re.MULTILINE)
        if name_match:
            name = name_match.group(1).strip()

        # Extract skills
        skills = []
        skills_match = re.search(r"## SKILLS\n(.+?)(?:\n##|\Z)", text, re.DOTALL)
        if skills_match:
            skills_text = skills_match.group(1).strip()
            skills = [s.strip() for s in skills_text.split(",") if s.strip()]

        # Count experience entries
        experience_count = len(re.findall(r"^### .+ at .+", text, re.MULTILINE))

        # Count education entries
        education_count = len(
            re.findall(r"(?<=## EDUCATION\n)(?:### .+\n?)+", text, re.MULTILINE)
        )
        # Simpler: count ### under EDUCATION
        edu_section = re.search(r"## EDUCATION\n(.*?)(?:\n##|\Z)", text, re.DOTALL)
        education_count = len(re.findall(r"^###", edu_section.group(1), re.MULTILINE)) if edu_section else 0

        return {
            "resume_id": resume_id,
            "total_chunks": 1,  # Signal that resume exists (not chunk-based anymore)
            "sections": sections,
            "preview": {},
            "name": name,
            "skills": skills[:20],
            "experience_count": experience_count,
            "education_count": education_count,
        }

    def _summary_from_chunks(self, resume_id: str) -> Dict:
        """Fallback: summary from ChromaDB chunks."""
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
            "preview": sections,
        }

    def build_prompt(
        self,
        resume_id: str,
        focus_area: Optional[str] = None,
        question_type: str = "mixed",
        n_questions: int = 5,
    ) -> str:
        """Build a prompt for question generation using the full resume text."""
        context = self.get_full_resume_text(resume_id)

        if question_type in ["technical", "tech"]:
            instruction = f"Generate {n_questions} TECHNICAL interview questions focusing on the candidate's skills, projects, architecture decisions, and technical implementation details."
        elif question_type in ["behavioral", "hr"]:
            instruction = f"Generate {n_questions} BEHAVIORAL interview questions using STAR format (Tell me about a time...). Focus on teamwork, leadership, challenges, and soft skills. Do NOT ask technical implementation questions."
        else:
            instruction = f"Generate {n_questions} interview questions (mix of technical and behavioral) based on the candidate's resume."

        if focus_area:
            instruction += f" Focus specifically on: {focus_area}."

        return f"""You are an expert interviewer. {instruction}

Here is the candidate's resume information:

{context}

Generate specific, relevant interview questions based on this resume."""
