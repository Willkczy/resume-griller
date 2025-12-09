"""
Resume Parser Module

Extracts structured information from PDF and DOCX resumes.

Author: [Your Name]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class WorkExperience:
    """Single work experience entry."""
    company: str
    role: str
    duration: str
    achievements: list[str] = field(default_factory=list)


@dataclass
class Education:
    """Single education entry."""
    school: str
    degree: str
    year: str
    gpa: Optional[str] = None


@dataclass
class Project:
    """Single project entry."""
    name: str
    description: str
    tech_stack: list[str] = field(default_factory=list)


@dataclass
class ResumeData:
    """Structured resume data."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    work_experience: list[WorkExperience] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    raw_text: str = ""


class ResumeParser:
    """
    Parse PDF/DOCX resumes into structured data.
    
    Example:
        parser = ResumeParser()
        
        # From file path
        resume = parser.parse_file("resume.pdf")
        
        # From bytes
        resume = parser.parse_bytes(file_content, "pdf")
        
        print(resume.name)
        print(resume.skills)
    """
    
    def parse_file(self, file_path: str | Path) -> ResumeData:
        """
        Parse resume from file path.
        
        Args:
            file_path: Path to PDF or DOCX file
            
        Returns:
            ResumeData with extracted information
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        content = path.read_bytes()
        file_type = path.suffix.lower().lstrip(".")
        
        return self.parse_bytes(content, file_type)
    
    def parse_bytes(self, content: bytes, file_type: str) -> ResumeData:
        """
        Parse resume from bytes.
        
        Args:
            content: File content as bytes
            file_type: "pdf" or "docx"
            
        Returns:
            ResumeData with extracted information
        """
        # Step 1: Extract raw text
        if file_type == "pdf":
            raw_text = self._extract_pdf(content)
        elif file_type in ("docx", "doc"):
            raw_text = self._extract_docx(content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Step 2: Parse into structured format
        resume_data = self._parse_text(raw_text)
        resume_data.raw_text = raw_text
        
        return resume_data
    
    def _extract_pdf(self, content: bytes) -> str:
        """
        Extract text from PDF.
        
        TODO: Implement using PyMuPDF (fitz)
        """
        # Option 1: PyMuPDF (recommended - fast and accurate)
        # import fitz
        # doc = fitz.open(stream=content, filetype="pdf")
        # text = ""
        # for page in doc:
        #     text += page.get_text()
        # return text
        
        # Option 2: pdfplumber (better for tables)
        # import pdfplumber
        # import io
        # with pdfplumber.open(io.BytesIO(content)) as pdf:
        #     text = ""
        #     for page in pdf.pages:
        #         text += page.extract_text() or ""
        # return text
        
        raise NotImplementedError("PDF extraction not implemented")
    
    def _extract_docx(self, content: bytes) -> str:
        """
        Extract text from DOCX.
        
        TODO: Implement using python-docx
        """
        # import io
        # from docx import Document
        # doc = Document(io.BytesIO(content))
        # text = "\n".join([para.text for para in doc.paragraphs])
        # return text
        
        raise NotImplementedError("DOCX extraction not implemented")
    
    def _parse_text(self, text: str) -> ResumeData:
        """
        Parse raw text into structured ResumeData.
        
        TODO: Implement parsing logic
        
        Options:
        1. Rule-based (regex patterns) - Fast, no API cost
        2. LLM-based (Claude/GPT) - More accurate, costs money
        3. Hybrid - Rules for simple fields, LLM for complex
        """
        # For now, return empty structure
        # TODO: Implement actual parsing
        return ResumeData()


# Convenience function
def parse_resume(file_path: str | Path) -> ResumeData:
    """Quick function to parse a resume file."""
    parser = ResumeParser()
    return parser.parse_file(file_path)
