"""Legacy extraction coverage using only newly authored synthetic data."""

import fitz

from rag.resume_parser import ResumeParser


def test_pdf(tmp_path):
    path = tmp_path / "synthetic.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Synthetic Candidate\ncandidate@example.invalid\nSkills\nPython, SQL",
        )
        document.save(path)

    result = ResumeParser().parse(str(path))

    assert "Synthetic Candidate" in result.raw_text
    assert result.contact["email"] == "candidate@example.invalid"
    assert "Python" in result.skills
