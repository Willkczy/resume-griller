"""
Resume Parser for Interview Coach
Extracts structured sections from PDF and text resumes.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pdfplumber


@dataclass
class ParsedResume:
    """Structured resume data."""
    raw_text: str = ""
    contact: Dict[str, str] = field(default_factory=dict)
    summary: str = ""
    skills: List[str] = field(default_factory=list)
    experience: List[Dict] = field(default_factory=list)
    education: List[Dict] = field(default_factory=list)
    projects: List[Dict] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "contact": self.contact,
            "summary": self.summary,
            "skills": self.skills,
            "experience": self.experience,
            "education": self.education,
            "projects": self.projects,
            "certifications": self.certifications,
        }


class ResumeParser:
    """Parse resumes from PDF or text files."""

    # Section header patterns
    SECTION_PATTERNS = {
        "contact": r"^(contact|personal\s+info)",
        "summary": r"^(summary|profile|objective|about)",
        "skills": r"^(skills|technical\s+skills|core\s+competencies|technologies|tech\s+stack)",
        "experience": r"^(experience|work\s+experience|professional\s+experience|employment)",
        "education": r"^(education|academic|qualification)",
        "projects": r"^(projects|personal\s+projects|portfolio)",
        "certifications": r"^(certifications?|certificates?|licenses?)",
    }

    # Contact extraction patterns
    EMAIL_PATTERN = r"[\w\.-]+@[\w\.-]+\.\w+"
    PHONE_PATTERN = r"[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}"
    LINKEDIN_PATTERN = r"linkedin\.com/in/[\w-]+"
    GITHUB_PATTERN = r"github\.com/[\w-]+"

    def __init__(self):
        self.parsed = ParsedResume()

    def parse(self, file_path: str) -> ParsedResume:
        """Main entry point - parse a resume file."""
        path = Path(file_path)

        if path.suffix.lower() == ".pdf":
            text = self._extract_pdf_text(file_path)
        elif path.suffix.lower() in [".txt", ".md"]:
            text = self._extract_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return self.parse_text(text)

    def parse_text(self, text: str) -> ParsedResume:
        """Parse resume from raw text."""
        self.parsed = ParsedResume(raw_text=text)

        # Clean text
        text = self._clean_text(text)

        # Extract contact info (usually at top)
        self._extract_contact(text)

        # Split into sections and parse each
        sections = self._identify_sections(text)

        for section_name, content in sections.items():
            self._parse_section(section_name, content)

        return self.parsed

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text_parts = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n".join(text_parts)

    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_contact(self, text: str) -> None:
        """Extract contact information."""
        # Get first portion of resume (usually contains contact)
        header = text[:1000]

        # Email
        email_match = re.search(self.EMAIL_PATTERN, header)
        if email_match:
            self.parsed.contact["email"] = email_match.group()

        # Phone
        phone_match = re.search(self.PHONE_PATTERN, header)
        if phone_match:
            self.parsed.contact["phone"] = phone_match.group()

        # LinkedIn
        linkedin_match = re.search(self.LINKEDIN_PATTERN, header, re.IGNORECASE)
        if linkedin_match:
            self.parsed.contact["linkedin"] = linkedin_match.group()

        # GitHub
        github_match = re.search(self.GITHUB_PATTERN, header, re.IGNORECASE)
        if github_match:
            self.parsed.contact["github"] = github_match.group()

        # Name (usually first line)
        first_line = text.split("\n")[0].strip()
        if first_line and not re.search(self.EMAIL_PATTERN, first_line):
            # Clean up name (remove titles, etc.)
            name = re.sub(r"(resume|cv|curriculum vitae)", "", first_line, flags=re.IGNORECASE)
            name = name.strip(" -|:")
            if len(name) < 50:  # Reasonable name length
                self.parsed.contact["name"] = name

    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and split text into sections."""
        sections = {}
        lines = text.split("\n")

        current_section = "header"
        current_content = []

        for line in lines:
            line_clean = line.strip().lower()
            line_clean = re.sub(r"[=\-#*_]{2,}", "", line_clean).strip()

            # Check if this line is a section header
            matched_section = None
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if re.match(pattern, line_clean, re.IGNORECASE):
                    matched_section = section_name
                    break

            if matched_section:
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)

                current_section = matched_section
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _parse_section(self, section_name: str, content: str) -> None:
        """Parse a specific section."""
        content = content.strip()

        if section_name == "summary":
            self.parsed.summary = self._parse_summary(content)
        elif section_name == "skills":
            self.parsed.skills = self._parse_skills(content)
        elif section_name == "experience":
            self.parsed.experience = self._parse_experience(content)
        elif section_name == "education":
            self.parsed.education = self._parse_education(content)
        elif section_name == "projects":
            self.parsed.projects = self._parse_projects(content)
        elif section_name == "certifications":
            self.parsed.certifications = self._parse_certifications(content)

    def _parse_summary(self, content: str) -> str:
        """Parse summary section."""
        # Remove empty lines and join
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        return " ".join(lines)

    def _parse_skills(self, content: str) -> List[str]:
        """Parse skills section into list."""
        skills = []

        # Common skill delimiters
        # Handle formats like "Languages: Python, Java, C++"
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove category labels like "Languages:", "Frameworks:"
            if ":" in line:
                line = line.split(":", 1)[1]

            # Split by common delimiters
            parts = re.split(r"[,|•·]", line)

            for part in parts:
                skill = part.strip(" -•·")
                # Clean up skill
                skill = re.sub(r"\([^)]*\)", "", skill).strip()

                if skill and len(skill) < 50:  # Reasonable skill length
                    skills.append(skill)

        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen and skill:
                seen.add(skill_lower)
                unique_skills.append(skill)

        return unique_skills

    def _parse_experience(self, content: str) -> List[Dict]:
        """Parse experience section."""
        experiences = []
        lines = content.split("\n")

        current_exp = None
        current_bullets = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a new job entry (usually has company/date pattern)
            is_new_entry = self._is_job_entry(line)

            if is_new_entry:
                # Save previous experience
                if current_exp:
                    current_exp["bullets"] = current_bullets
                    experiences.append(current_exp)

                current_exp = self._parse_job_line(line)
                current_bullets = []

            elif current_exp and self._is_bullet_point(line):
                bullet = self._clean_bullet(line)
                if bullet:
                    current_bullets.append(bullet)

        # Save last experience
        if current_exp:
            current_exp["bullets"] = current_bullets
            experiences.append(current_exp)

        return experiences

    def _is_job_entry(self, line: str) -> bool:
        """Check if line is a job entry header."""
        # Look for date patterns
        date_pattern = r"(19|20)\d{2}|present|current"
        has_date = re.search(date_pattern, line, re.IGNORECASE)

        # Look for separator patterns (| or -)
        has_separator = "|" in line or " - " in line or "–" in line

        return has_date or has_separator

    def _parse_job_line(self, line: str) -> Dict:
        """Parse a job entry line."""
        job = {"title": "", "company": "", "dates": "", "location": ""}

        # Extract dates
        date_pattern = r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{0,4}\s*[-–]\s*(?:Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{0,4})|(?:19|20)\d{2}\s*[-–]\s*(?:Present|Current|(?:19|20)\d{2}))"
        date_match = re.search(date_pattern, line, re.IGNORECASE)
        if date_match:
            job["dates"] = date_match.group().strip()
            line = line.replace(date_match.group(), "")

        # Split remaining by common separators
        parts = re.split(r"[|]|(?:\s[-–]\s)", line)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) >= 2:
            job["title"] = parts[0]
            job["company"] = parts[1]
        elif len(parts) == 1:
            job["title"] = parts[0]

        return job

    def _is_bullet_point(self, line: str) -> bool:
        """Check if line is a bullet point."""
        bullet_chars = ["•", "-", "–", "*", "▪", "○", "●"]
        return any(line.startswith(c) for c in bullet_chars) or line.startswith(("- ", "* "))

    def _clean_bullet(self, line: str) -> str:
        """Clean bullet point text."""
        return re.sub(r"^[\s\-•–*▪○●]+", "", line).strip()

    def _parse_education(self, content: str) -> List[Dict]:
        """Parse education section."""
        education = []
        lines = content.split("\n")

        current_edu = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for degree patterns
            degree_pattern = r"(bachelor|master|ph\.?d|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?b\.?a\.?)"
            has_degree = re.search(degree_pattern, line, re.IGNORECASE)

            # Check for university patterns
            uni_pattern = r"(university|college|institute|school)"
            has_uni = re.search(uni_pattern, line, re.IGNORECASE)

            if has_degree or has_uni:
                if current_edu:
                    education.append(current_edu)

                current_edu = {"degree": "", "institution": "", "year": "", "details": ""}

                # Extract year
                year_match = re.search(r"(19|20)\d{2}", line)
                if year_match:
                    current_edu["year"] = year_match.group()

                # Simple split for degree/institution
                current_edu["degree"] = line

            elif current_edu and line:
                current_edu["details"] += " " + line

        if current_edu:
            education.append(current_edu)

        return education

    def _parse_projects(self, content: str) -> List[Dict]:
        """Parse projects section."""
        projects = []
        lines = content.split("\n")

        current_project = None
        current_bullets = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Project headers usually have year or tech stack
            is_header = (
                    re.search(r"(19|20)\d{2}", line) or
                    "|" in line or
                    (not self._is_bullet_point(line) and len(line) < 80)
            )

            if is_header and not self._is_bullet_point(line):
                if current_project:
                    current_project["bullets"] = current_bullets
                    projects.append(current_project)

                current_project = {"name": line, "bullets": []}
                current_bullets = []

            elif current_project and self._is_bullet_point(line):
                bullet = self._clean_bullet(line)
                if bullet:
                    current_bullets.append(bullet)

        if current_project:
            current_project["bullets"] = current_bullets
            projects.append(current_project)

        return projects

    def _parse_certifications(self, content: str) -> List[str]:
        """Parse certifications section."""
        certs = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if line:
                cert = self._clean_bullet(line)
                if cert and len(cert) < 200:
                    certs.append(cert)

        return certs


def main():
    """Test the parser with sample resumes."""
    import json

    parser = ResumeParser()

    # Test with sample resumes
    sample_dir = Path("data/sample_resumes")

    if not sample_dir.exists():
        print(f"Sample directory not found: {sample_dir}")
        return

    for resume_file in sample_dir.glob("*.txt"):
        print(f"\n{'=' * 60}")
        print(f"Parsing: {resume_file.name}")
        print("=" * 60)

        try:
            result = parser.parse(str(resume_file))

            print(f"\nContact: {result.contact}")
            print(f"Skills ({len(result.skills)}): {result.skills[:10]}...")
            print(f"Experience entries: {len(result.experience)}")
            print(f"Education entries: {len(result.education)}")
            print(f"Projects: {len(result.projects)}")
            print(f"Certifications: {len(result.certifications)}")

            # Print detailed experience
            if result.experience:
                print(f"\nFirst experience:")
                exp = result.experience[0]
                print(f"  Title: {exp.get('title')}")
                print(f"  Company: {exp.get('company')}")
                print(f"  Dates: {exp.get('dates')}")
                print(f"  Bullets: {len(exp.get('bullets', []))}")

        except Exception as e:
            print(f"Error parsing {resume_file.name}: {e}")


if __name__ == "__main__":
    main()