import json
from io import BytesIO
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, UploadFile

from backend.app.api.routes import resume
from backend.app.config import settings
from backend.app.models.schemas import GenerateQuestionsRequest, QuestionType
from rag.retriever import InterviewRetriever


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    path = tmp_path / "uploads"
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(path))
    monkeypatch.setattr(settings, "PARSED_RESUME_DIR", str(tmp_path / "parsed"))
    return path


def upload(filename: str, content: bytes = b"resume text") -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(content))


@pytest.mark.asyncio
async def test_upload_resume_success(upload_dir, monkeypatch):
    retriever = Mock()
    retriever.process_resume = AsyncMock()
    retriever.get_resume_summary.return_value = {
        "total_chunks": 1,
        "sections": ["skills", "experience"],
    }
    monkeypatch.setattr(resume, "generate_resume_id", Mock(return_value="resume-1"))

    result = await resume.upload_resume(upload("resume.txt"), retriever)

    assert result.resume_id == "resume-1"
    assert result.chunks_created == 1
    retriever.process_resume.assert_awaited_once()
    assert (upload_dir / "resume-1.txt").exists()


@pytest.mark.asyncio
async def test_upload_rejects_type_and_normalizes_parser_failure(
    upload_dir, monkeypatch
):
    retriever = Mock()
    with pytest.raises(HTTPException) as invalid:
        await resume.upload_resume(upload("resume.docx"), retriever)
    assert invalid.value.status_code == 400

    retriever.process_resume = AsyncMock(side_effect=RuntimeError("provider down"))
    monkeypatch.setattr(resume, "generate_resume_id", Mock(return_value="resume-2"))
    with pytest.raises(HTTPException) as failed:
        await resume.upload_resume(upload("resume.txt"), retriever)
    assert failed.value.status_code == 502
    assert not (upload_dir / "resume-2.txt").exists()


@pytest.mark.asyncio
async def test_resume_summary_success_and_missing():
    retriever = Mock()
    retriever.get_resume_summary.return_value = {
        "name": "Ada",
        "total_chunks": 1,
        "sections": ["skills"],
        "skills": ["Python"],
        "experience_count": 2,
        "education_count": 1,
    }
    result = await resume.get_resume_summary("resume-1", retriever)
    assert result.name == "Ada"
    assert result.skills == ["Python"]

    retriever.get_resume_summary.return_value = {
        "total_chunks": 0,
        "sections": [],
    }
    with pytest.raises(HTTPException) as missing:
        await resume.get_resume_summary("missing", retriever)
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_markdown_resume_without_initializing_legacy(upload_dir, tmp_path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    parsed = parsed_dir / "resume-1.md"
    parsed.write_text("resume", encoding="utf-8")
    (upload_dir / "resume-1.txt").parent.mkdir(parents=True, exist_ok=True)
    (upload_dir / "resume-1.txt").write_text("resume", encoding="utf-8")
    retriever = Mock()

    result = await resume.delete_resume("resume-1", retriever)
    assert result["message"].startswith("Resume resume-1")
    assert not parsed.exists()
    assert not (upload_dir / "resume-1.txt").exists()


@pytest.mark.asyncio
async def test_delete_rejects_external_symlink_before_removing_any_file(
    upload_dir, tmp_path
):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    parsed = parsed_dir / "resume-1.md"
    parsed.write_text("Synthetic resume", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("Synthetic unrelated file", encoding="utf-8")
    upload_dir.mkdir()
    (upload_dir / "resume-1.txt").symlink_to(outside)
    retriever = Mock()
    with pytest.raises(HTTPException) as rejected:
        await resume.delete_resume("resume-1", retriever)
    assert rejected.value.status_code == 422
    assert parsed.read_text() == "Synthetic resume"
    assert outside.read_text() == "Synthetic unrelated file"
    retriever.embedder._delete_resume.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_summary_route_retains_skills_and_counts(upload_dir):
    embedder = Mock()
    embedder.get_all_chunks.return_value = [
        {
            "content": "Skills",
            "metadata": {"section": "skills", "skills_list": "Python, FastAPI"},
        },
        {"content": "Job", "metadata": {"section": "experience"}},
        {"content": "Degree", "metadata": {"section": "education"}},
    ]
    result = await resume.get_resume_summary(
        "legacy", InterviewRetriever(embedder=embedder)
    )
    assert result.skills == ["Python", "FastAPI"]
    assert result.experience_count == result.education_count == 1


@pytest.mark.asyncio
async def test_generate_questions_and_parse_response():
    retriever = Mock()
    retriever.get_resume_summary.return_value = {"total_chunks": 1}
    retriever.build_prompt.return_value = "prompt"
    llm = AsyncMock()
    llm.generate.return_value = json.dumps(
        {
            "questions": [
                "How did you design the service?",
                "What measurable outcome did it achieve?",
            ]
        }
    )
    request = GenerateQuestionsRequest(
        resume_id="resume-1",
        question_type=QuestionType.TECHNICAL,
        num_questions=2,
    )

    result = await resume.generate_questions("resume-1", request, retriever, llm)
    assert len(result.questions) == 2
    assert result.questions[0].type == QuestionType.TECHNICAL


@pytest.mark.parametrize(
    "output,code",
    [
        ('{"questions": []}', 502),
        ('{"questions": [" "]}', 502),
        ('{"questions": ["One?", "Two?"]}', 502),
        ("1. An old numbered question?", 502),
        (TimeoutError("synthetic sentinel"), 503),
    ],
)
async def test_standalone_questions_validate_count_and_hide_failure_payload(
    output, code
):
    retriever = Mock()
    retriever.get_resume_summary.return_value = {"total_chunks": 1}
    retriever.build_prompt.return_value = "synthetic prompt"
    llm = AsyncMock()
    llm.generate.side_effect = [output]
    request = GenerateQuestionsRequest(resume_id="synthetic", num_questions=1)
    with pytest.raises(HTTPException) as exc:
        await resume.generate_questions("synthetic", request, retriever, llm)
    assert exc.value.status_code == code
    assert "sentinel" not in exc.value.detail
