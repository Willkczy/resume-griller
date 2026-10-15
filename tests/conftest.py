"""Hermetic test defaults: no local dotenv, runtime data, downloads or live providers."""

import socket
from tempfile import TemporaryDirectory

import pytest
from pydantic_settings import BaseSettings

_ISOLATION = pytest.StashKey()


def pytest_configure(config):
    # Configure before collection imports the application/settings singleton.
    patch = pytest.MonkeyPatch()
    directory = TemporaryDirectory(prefix="resume-griller-tests-")
    config.stash[_ISOLATION] = (patch, directory)
    original_init = BaseSettings.__init__

    def without_dotenv(self, **values):
        original_init(self, **{**values, "_env_file": None})

    # _env_file is the public Pydantic settings override. Never load a developer's
    # .env merely to discard its secrets after collection.
    patch.setattr(BaseSettings, "__init__", without_dotenv)
    for name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "DEEPGRAM_API_KEY",
        "ELEVENLABS_API_KEY",
        "HF_TOKEN",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
    ):
        patch.delenv(name, raising=False)
    defaults = {
        "DEBUG": "false",
        "LLM_MODE": "api",
        "LLM_PROVIDER": "groq",
        "VOICE_ENABLED": "false",
        "CUSTOM_MODEL_ENABLED": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "UPLOAD_DIR": f"{directory.name}/uploads",
        "PARSED_RESUME_DIR": f"{directory.name}/parsed",
        "CHROMA_PERSIST_DIR": f"{directory.name}/chroma",
        "CHECKPOINT_DB_PATH": f"{directory.name}/checkpoints.db",
        "DATABASE_URL": f"sqlite+aiosqlite:///{directory.name}/sessions.db",
        "HF_HOME": f"{directory.name}/huggingface",
        "TORCH_HOME": f"{directory.name}/torch",
    }
    for name, value in defaults.items():
        patch.setenv(name, value)


def pytest_unconfigure(config):
    isolation = config.stash.get(_ISOLATION, None)
    if isolation:
        patch, directory = isolation
        patch.undo()
        directory.cleanup()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Permit asyncio's local Unix sockets; fail even on swallowed network errors."""
    attempts = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def deny(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Network access is forbidden in tests; mock provider I/O")

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            return deny()
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            return deny()
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    yield
    if attempts:
        pytest.fail("Test attempted network access; mock provider I/O", pytrace=False)
