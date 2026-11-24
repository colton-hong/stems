"""Owner-supplied settings come from the environment, then from .env."""

from __future__ import annotations

from stems_common.env import get_env, parse_dotenv


def test_parse_dotenv_handles_comments_quotes_and_export() -> None:
    text = "\n".join(
        [
            "# a comment",
            "",
            "PLAIN=/data/musdb",
            'QUOTED="/data/with space"',
            "SINGLE='x=y'",
            "export EXPORTED=1",
            "EMPTY=",
            "not a setting",
            "  SPACED = padded  ",
        ]
    )
    assert parse_dotenv(text) == {
        "PLAIN": "/data/musdb",
        "QUOTED": "/data/with space",
        "SINGLE": "x=y",
        "EXPORTED": "1",
        "EMPTY": "",
        "SPACED": "padded",
    }


def test_environment_wins_then_dotenv_then_none(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("STEMS_TEST_VALUE=from-file\nSTEMS_TEST_EMPTY=\n")
    monkeypatch.delenv("STEMS_TEST_VALUE", raising=False)
    assert get_env("STEMS_TEST_VALUE", dotenv_path=dotenv) == "from-file"
    monkeypatch.setenv("STEMS_TEST_VALUE", "from-env")
    assert get_env("STEMS_TEST_VALUE", dotenv_path=dotenv) == "from-env"
    monkeypatch.setenv("STEMS_TEST_VALUE", "")  # set but empty counts as unset
    assert get_env("STEMS_TEST_VALUE", dotenv_path=dotenv) == "from-file"
    assert get_env("STEMS_TEST_EMPTY", dotenv_path=dotenv) is None
    assert get_env("STEMS_TEST_MISSING", dotenv_path=dotenv) is None
    assert get_env("STEMS_TEST_MISSING", dotenv_path=tmp_path / "absent") is None


def test_another_dotenv_file_can_be_named_and_the_tests_name_one_that_is_absent(
    tmp_path, monkeypatch
) -> None:
    from stems_common import env

    # tests/conftest.py points every test at a file that does not exist.
    assert not env.dotenv_file().exists()
    assert env.dotenv_file() != env.DOTENV_PATH
    for name in ("TYPESAFE_API_KEY", "CLOUDFLARE_AUTH_TOKEN", "MUSDB18HQ_PATH"):
        assert get_env(name) is None
    other = tmp_path / "other.env"
    other.write_text("STEMS_TEST_VALUE=from-other\n")
    monkeypatch.setenv("STEMS_DOTENV_PATH", str(other))
    monkeypatch.delenv("STEMS_TEST_VALUE", raising=False)
    assert get_env("STEMS_TEST_VALUE") == "from-other"
    monkeypatch.delenv("STEMS_DOTENV_PATH")
    assert env.dotenv_file() == env.DOTENV_PATH
