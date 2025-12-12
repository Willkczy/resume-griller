import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.resume_parser import ResumeParser
import json


def test_pdf():
    parser = ResumeParser()
    sample_dir = Path("data/sample_resumes")

    # Find all PDFs
    pdf_files = list(sample_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/sample_resumes/")
        print("Add some PDF resumes to test.")
        return

    for pdf_file in pdf_files:
        print(f"\n{'=' * 60}")
        print(f"Parsing: {pdf_file.name}")
        print("=" * 60)

        result = parser.parse(str(pdf_file))

        print(f"\n📋 RAW TEXT (first 500 chars):")
        print(result.raw_text[:500])

        print(f"\n👤 CONTACT:")
        print(json.dumps(result.contact, indent=2))

        print(f"\n🛠️ SKILLS ({len(result.skills)}):")
        print(result.skills)

        print(f"\n💼 EXPERIENCE ({len(result.experience)}):")
        for exp in result.experience:
            print(f"  - {exp.get('title')} @ {exp.get('company')}")

        print(f"\n🎓 EDUCATION ({len(result.education)}):")
        for edu in result.education:
            print(f"  - {edu.get('degree')}")


if __name__ == "__main__":
    test_pdf()