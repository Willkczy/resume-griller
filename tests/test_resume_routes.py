from io import BytesIO
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, UploadFile

from backend.app.api.routes import resume
from backend.app.config import settings
from backend.app.models.schemas import GenerateQuestionsRequest, QuestionType


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
async def test_generate_questions_and_parse_response():
    retriever = Mock()
    retriever.get_resume_summary.return_value = {"total_chunks": 1}
    retriever.build_prompt.return_value = "prompt"
    llm = AsyncMock()
    llm.generate.return_value = (
        "1. How did you design the service?\n"
        "2. What measurable outcome did it achieve?"
    )
    request = GenerateQuestionsRequest(
        resume_id="resume-1",
        question_type=QuestionType.TECHNICAL,
        num_questions=2,
    )

    result = await resume.generate_questions("resume-1", request, retriever, llm)
    assert len(result.questions) == 2
    assert result.questions[0].type == QuestionType.TECHNICAL
