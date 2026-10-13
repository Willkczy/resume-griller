from unittest.mock import AsyncMock, Mock

import pytest

from backend.app.config import settings
from backend.app.models.resume_id import InvalidResumeId
from rag.retriever import InterviewRetriever


@pytest.fixture
def parsed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PARSED_RESUME_DIR", str(tmp_path / "parsed"))
    return tmp_path / "parsed"


def test_markdown_is_primary_and_does_not_initialize_embedder(parsed_dir):
    factory = Mock()
    retriever = InterviewRetriever(embedder_factory=factory)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / "resume-1.md").write_text(
        "# CANDIDATE: Ada\n\n## SKILLS\nPython, FastAPI", encoding="utf-8"
    )

    assert "Ada" in retriever.get_full_resume_text("resume-1")
    summary = retriever.get_resume_summary("resume-1")
    assert summary["name"] == "Ada"
    assert summary["skills"] == ["Python", "FastAPI"]
    factory.assert_not_called()


def test_legacy_chroma_fallback_is_lazy(parsed_dir):
    embedder = Mock()
    embedder.get_all_chunks.return_value = [
        {"content": "second", "metadata": {"chunk_index": 1, "section": "skills"}},
        {"content": "first", "metadata": {"chunk_index": 0, "section": "summary"}},
    ]
    factory = Mock(return_value=embedder)
    retriever = InterviewRetriever(embedder_factory=factory)

    assert factory.call_count == 0
    text = retriever.get_full_resume_text("legacy")
    assert text.index("first") < text.index("second")
    assert factory.call_count == 1


@pytest.mark.asyncio
async def test_process_txt_writes_parsed_markdown(parsed_dir, tmp_path, monkeypatch):
    source = tmp_path / "resume.txt"
    source.write_text("Ada Lovelace, software engineer", encoding="utf-8")
    retriever = InterviewRetriever()
    monkeypatch.setattr(
        retriever,
        "_parse_with_llm",
        AsyncMock(return_value="# CANDIDATE: Ada"),
    )

    result = await retriever.process_resume(str(source), "resume-1")
    assert result == "resume-1"
    assert (parsed_dir / "resume-1.md").read_text() == "# CANDIDATE: Ada"


@pytest.mark.asyncio
async def test_process_rejects_empty_and_unsupported_files(parsed_dir, tmp_path):
    retriever = InterviewRetriever()
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    unsupported = tmp_path / "resume.docx"
    unsupported.write_bytes(b"not a docx")

    with pytest.raises(ValueError, match="no extractable text"):
        await retriever.process_resume(str(empty), "empty")
    with pytest.raises(ValueError, match="Unsupported resume type"):
        retriever._extract_text(str(unsupported))


@pytest.mark.asyncio
async def test_llm_failures_are_normalized(parsed_dir, monkeypatch):
    llm = AsyncMock()
    llm.generate.side_effect = ConnectionError("provider down")
    monkeypatch.setattr(
        "backend.app.services.llm_service.LLMServiceFactory.get_service",
        Mock(return_value=llm),
    )
    retriever = InterviewRetriever()

    with pytest.raises(RuntimeError, match="service failed"):
        await retriever._parse_with_llm("resume text")


# Synthetic storage regressions: resume-storage-review-v1.
@pytest.mark.parametrize(
    "resume_id", ["../outside", "/tmp/outside", r"..\outside", "C:outside", "a\x00b"]
)
def test_resume_ids_cannot_be_paths(parsed_dir, resume_id):
    factory = Mock()
    retriever = InterviewRetriever(embedder_factory=factory)
    for read in (retriever.get_full_resume_text, retriever.get_resume_summary):
        with pytest.raises(InvalidResumeId):
            read(resume_id)
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_symlink_escape_rejected_before_read_write_or_provider(
    parsed_dir, tmp_path, monkeypatch
):
    factory = Mock()
    retriever = InterviewRetriever(embedder_factory=factory)
    outside = tmp_path / "outside.md"
    outside.write_text("Synthetic private note", encoding="utf-8")
    (parsed_dir / "link.md").symlink_to(outside)
    parse = AsyncMock()
    monkeypatch.setattr(retriever, "_parse_with_llm", parse)
    for read in (retriever.get_full_resume_text, retriever.get_resume_summary):
        with pytest.raises(InvalidResumeId):
            read("link")
    with pytest.raises(InvalidResumeId):
        await retriever.process_resume(str(tmp_path / "unused.txt"), "link")
    parse.assert_not_awaited()
    factory.assert_not_called()
    assert outside.read_text() == "Synthetic private note"


@pytest.mark.parametrize("resume_id", ["legacy resume.v1", "履歷_2026-10-13"])
def test_safe_legacy_filename_stems_remain_supported(parsed_dir, resume_id):
    retriever = InterviewRetriever()
    (parsed_dir / f"{resume_id}.md").write_text("Synthetic resume", encoding="utf-8")
    assert retriever.get_full_resume_text(resume_id) == "Synthetic resume"


@pytest.mark.parametrize("spacing", ["", "\n"])
def test_markdown_counts_all_degrees_until_next_section(parsed_dir, spacing):
    retriever = InterviewRetriever()
    (parsed_dir / "synthetic.md").write_text(
        "# CANDIDATE: Synthetic Candidate\n\n## EDUCATION\n"
        + spacing
        + "### BS, Synthetic University (2020)\n"
        "### MS, Synthetic University (2022)\n\n"
        "## PROJECTS\n### Synthetic queue\n",
        encoding="utf-8",
    )
    assert retriever.get_resume_summary("synthetic")["education_count"] == 2


def test_legacy_summary_preserves_flat_chroma_metadata(parsed_dir):
    embedder = Mock()
    embedder.get_all_chunks.return_value = [
        {
            "content": "Skills",
            "metadata": {"section": "skills", "skills_list": "Python, FastAPI"},
        },
        {"content": "Job", "metadata": {"section": "experience"}},
        {"content": "Degree", "metadata": {"section": "education"}},
        {"content": "Degree 2", "metadata": {"section": "education"}},
    ]
    summary = InterviewRetriever(embedder=embedder).get_resume_summary("legacy")
    assert summary["skills"] == ["Python", "FastAPI"]
    assert summary["experience_count"] == 1
    assert summary["education_count"] == 2
    assert summary["total_chunks"] == 4
