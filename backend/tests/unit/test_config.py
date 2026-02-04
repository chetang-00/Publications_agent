import pytest
from pydantic import ValidationError

from app.config import Settings


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_missing_portkey_key_names_the_field():
    with pytest.raises(ValidationError) as exc:
        make()
    assert "portkey_api_key" in str(exc.value)


def test_blank_portkey_key_is_rejected():
    with pytest.raises(ValidationError) as exc:
        make(portkey_api_key="   ")
    assert "PORTKEY_API_KEY" in str(exc.value)


def test_embedding_credentials_fall_back_to_chat_credentials():
    s = make(portkey_api_key="chat-key", portkey_virtual_key="chat-vk")
    assert s.embedding_api_key == "chat-key"
    assert s.embedding_virtual_key == "chat-vk"


def test_embedding_credentials_can_be_overridden():
    s = make(
        portkey_api_key="chat-key",
        embedding_portkey_api_key="emb-key",
        embedding_portkey_virtual_key="emb-vk",
    )
    assert s.embedding_api_key == "emb-key"
    assert s.embedding_virtual_key == "emb-vk"


def test_blank_optional_keys_mean_unset(monkeypatch):
    monkeypatch.setenv("PORTKEY_API_KEY", "k")
    monkeypatch.setenv("PORTKEY_VIRTUAL_KEY", "")
    monkeypatch.setenv("EMBEDDING_PORTKEY_API_KEY", "  ")
    s = Settings(_env_file=None)
    assert s.chat_virtual_key is None
    assert s.embedding_api_key == "k"


def test_embedding_batch_size_upper_bound():
    with pytest.raises(ValidationError):
        make(portkey_api_key="k", embedding_batch_size=4096)


def test_reads_environment_variables(monkeypatch):
    monkeypatch.setenv("PORTKEY_API_KEY", "env-key")
    monkeypatch.setenv("CHAT_MODEL", "@openai/gpt-4o")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173, http://127.0.0.1:5173")
    s = Settings(_env_file=None)
    assert s.portkey_api_key.get_secret_value() == "env-key"
    assert s.chat_model == "@openai/gpt-4o"
    assert s.cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_base_url_trailing_slash_is_removed():
    s = make(portkey_api_key="k", portkey_base_url="https://gw.example/v1/")
    assert s.portkey_base_url == "https://gw.example/v1"


def test_secrets_are_not_printed():
    s = make(portkey_api_key="super-secret")
    assert "super-secret" not in repr(s)


def test_max_upload_bytes():
    assert make(portkey_api_key="k", max_upload_mb=2).max_upload_bytes == 2 * 1024 * 1024
