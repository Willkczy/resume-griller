"""Regressions for the CI boundary, including errors swallowed by provider adapters."""

import os
from pathlib import Path

import pytest

from backend.app.config import Settings, settings

pytest_plugins = ["pytester"]


def test_settings_ignore_dotenv_even_when_explicitly_requested(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("GROQ_API_KEY=synthetic-dotenv-key\nVOICE_ENABLED=true\n")
    isolated = Settings(_env_file=dotenv)
    assert isolated.GROQ_API_KEY is None
    assert isolated.VOICE_ENABLED is False


def test_test_defaults_are_credential_free_and_use_temporary_storage():
    assert settings.GROQ_API_KEY is None
    assert settings.OPENAI_API_KEY is None
    assert settings.GOOGLE_API_KEY is None
    assert settings.ANTHROPIC_API_KEY is None
    assert settings.DEEPGRAM_API_KEY is None
    assert settings.ELEVENLABS_API_KEY is None
    assert settings.VOICE_ENABLED is False
    assert settings.CUSTOM_MODEL_ENABLED is False
    paths = [
        settings.UPLOAD_DIR,
        settings.PARSED_RESUME_DIR,
        settings.CHROMA_PERSIST_DIR,
        settings.CHECKPOINT_DB_PATH,
    ]
    parent = Path(paths[0]).parent
    assert parent.name.startswith("resume-griller-tests-")
    assert all(Path(path).parent == parent for path in paths)
    assert os.environ["HF_HOME"].startswith(str(parent))
    assert os.environ["HF_HUB_OFFLINE"] == "1"


@pytest.mark.parametrize(
    "operation", ["dns", "connect", "connect_ex", "create_connection"]
)
def test_network_attempt_fails_even_when_caught(pytester, operation):
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text())
    pytester.makepyfile(
        test_guard=f"""
import socket

def test_swallowed_provider_error():
    try:
        if {operation!r} == "dns":
            socket.getaddrinfo("synthetic.invalid", 443)
        elif {operation!r} == "create_connection":
            socket.create_connection(("192.0.2.1", 443))
        else:
            with socket.socket(socket.AF_INET) as sock:
                getattr(sock, {operation!r})(("192.0.2.1", 443))
    except Exception:
        pass
"""
    )
    result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*Test attempted network access; mock provider I/O*"])
