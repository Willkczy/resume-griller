"""
Resume Chunker for Interview Coach
Splits parsed resume into semantic chunks for RAG retrieval.
"""

from typing import List, Dict
from dataclasses import dataclass
from rag.resume_parser import ParsedResume


@dataclass
class Chunk:
    """A single chunk of resume content."""
    content: str
    section: str  # skills, experience, education, etc.
    metadata: Dict  # additional info (company, dates, etc.)

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "section": self.section,
            "metadata": self.metadata,
        }


class ResumeChunker:
    """
    Chunk resumes by semantic sections.

    Strategy: Create separate chunks for each meaningful unit:
    - One chunk per skill category
    - One chunk per job experience
    - One chunk per education entry
    - One chunk per project
    """

    def __init__(self, include_summary_in_all: bool = True):
        self.include_summary = include_summary_in_all

    def chunk(self, parsed: ParsedResume) -> List[Chunk]:
        """Main entry - chunk a parsed resume."""
        chunks = []

        # Contact & Summary chunk
        if parsed.contact or parsed.summary:
            chunks.append(self._create_overview_chunk(parsed))

        # Skills chunk
        if parsed.skills:
            chunks.append(self._create_skills_chunk(parsed.skills))

        # Experience chunks (one per job)
        for exp in parsed.experience:
            chunks.append(self._create_experience_chunk(exp))

        # Education chunks
        for edu in parsed.education:
            chunks.append(self._create_education_chunk(edu))

        # Project chunks
        for proj in parsed.projects:
            chunks.append(self._create_project_chunk(proj))

        # Certifications chunk
        if parsed.certifications:
            chunks.append(self._create_certifications_chunk(parsed.certifications))

        return chunks

    def _create_overview_chunk(self, parsed: ParsedResume) -> Chunk:
        """Create overview chunk with contact and summary."""
        parts = []

        if parsed.contact.get("name"):
            parts.append(f"Name: {parsed.contact['name']}")

        if parsed.summary:
            parts.append(f"Summary: {parsed.summary}")

        return Chunk(
            content="\n".join(parts),
            section="overview",
            metadata={"contact": parsed.contact}
        )

    def _create_skills_chunk(self, skills: List[str]) -> Chunk:
        """Create skills chunk."""
        content = f"Technical Skills: {', '.join(skills)}"

        return Chunk(
            content=content,
            section="skills",
            metadata={"skill_count": len(skills), "skills_list": skills}
        )

    def _create_experience_chunk(self, exp: Dict) -> Chunk:
        """Create chunk for single job experience."""
        parts = []

        # Header
        title = exp.get("title", "")
        company = exp.get("company", "")
        dates = exp.get("dates", "")

        if title and company:
            parts.append(f"{title} at {company}")
        elif title:
            parts.append(title)

        if dates:
            parts.append(f"Period: {dates}")

        # Bullets
        bullets = exp.get("bullets", [])
        if bullets:
            parts.append("Responsibilities and achievements:")
            for bullet in bullets:
                parts.append(f"- {bullet}")

        return Chunk(
            content="\n".join(parts),
            section="experience",
            metadata={
                "title": title,
                "company": company,
                "dates": dates,
                "bullet_count": len(bullets)
            }
        )

    def _create_education_chunk(self, edu: Dict) -> Chunk:
        """Create chunk for education entry."""
        parts = []

        degree = edu.get("degree", "")
        institution = edu.get("institution", "")
        year = edu.get("year", "")
        details = edu.get("details", "")

        if degree:
            parts.append(degree)
        if institution:
            parts.append(f"Institution: {institution}")
        if year:
            parts.append(f"Year: {year}")
        if details:
            parts.append(details.strip())

        return Chunk(
            content="\n".join(parts),
            section="education",
            metadata={"degree": degree, "year": year}
        )

    def _create_project_chunk(self, proj: Dict) -> Chunk:
        """Create chunk for a project."""
        parts = []

        name = proj.get("name", "")
        bullets = proj.get("bullets", [])

        if name:
            parts.append(f"Project: {name}")

        if bullets:
            for bullet in bullets:
                parts.append(f"- {bullet}")

        return Chunk(
            content="\n".join(parts),
            section="project",
            metadata={"project_name": name}
        )

    def _create_certifications_chunk(self, certs: List[str]) -> Chunk:
        """Create certifications chunk."""
        content = "Certifications:\n" + "\n".join(f"- {c}" for c in certs)

        return Chunk(
            content=content,
            section="certifications",
            metadata={"cert_count": len(certs)}
        )


def main():
    """Test chunker with sample resume."""
    from rag.resume_parser import ResumeParser
    from pathlib import Path

    parser = ResumeParser()
    chunker = ResumeChunker()

    sample_dir = Path("data/sample_resumes")

    # Test with first available file
    files = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.txt"))

    if not files:
        print("No resume files found")
        return

    test_file = files[0]
    print(f"Testing with: {test_file.name}\n")

    # Parse
    parsed = parser.parse(str(test_file))

    # Chunk
    chunks = chunker.chunk(parsed)

    print(f"Created {len(chunks)} chunks:\n")

    for i, chunk in enumerate(chunks):
        print(f"{'=' * 50}")
        print(f"CHUNK {i + 1}: [{chunk.section}]")
        print(f"{'=' * 50}")
        print(chunk.content[:300])
        if len(chunk.content) > 300:
            print("...")
        print(f"\nMetadata: {chunk.metadata}\n")


if __name__ == "__main__":
    main()