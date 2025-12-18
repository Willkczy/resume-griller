"""
Export prompts from RAG pipeline for Colab inference.
Saves prompts to JSON that can be uploaded to Colab.
"""

import json
from pathlib import Path
from datetime import datetime
from rag.retriever import InterviewRetriever


def export_prompts(
        resume_path: str,
        output_path: str = "data/exported_prompts.json",
        focus_areas: list = None
):
    """
    Process resume and export prompts for Colab inference.

    Args:
        resume_path: Path to resume PDF/TXT
        output_path: Where to save exported prompts
        focus_areas: Optional list of focus areas
    """
    if focus_areas is None:
        focus_areas = [None, "technical skills", "leadership", "projects"]

    retriever = InterviewRetriever()

    # Process resume
    print(f"Processing: {resume_path}")
    resume_id = retriever.process_resume(resume_path)

    # Get summary
    summary = retriever.get_resume_summary(resume_id)

    # Generate prompts for different scenarios
    prompts = []

    for focus in focus_areas:
        for q_type in ["mixed", "technical", "behavioral"]:
            prompt = retriever.build_prompt(
                resume_id=resume_id,
                focus_area=focus,
                question_type=q_type,
                n_questions=5
            )

            prompts.append({
                "id": f"{resume_id}_{q_type}_{focus or 'general'}",
                "focus_area": focus,
                "question_type": q_type,
                "prompt": prompt
            })

    # Export
    export_data = {
        "resume_id": resume_id,
        "resume_file": str(resume_path),
        "exported_at": datetime.now().isoformat(),
        "summary": summary,
        "prompts": prompts
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"\n Exported {len(prompts)} prompts to: {output_path}")
    print(f"   Upload this file to Colab for inference.")

    return export_data


def main():
    from pathlib import Path

    print("=" * 60)
    print("EXPORT PROMPTS FOR COLAB")
    print("=" * 60)

    # Find resume
    sample_dir = Path("data/sample_resumes")
    files = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.txt"))

    if not files:
        print("No resume files found in data/sample_resumes/")
        return

    # Process first resume
    resume_path = files[0]
    export_prompts(str(resume_path))


if __name__ == "__main__":
    main()