from pathlib import Path

import pytest

from app import app as flask_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    mpl_dir = tmp_path / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))

    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    flask_app.config.update(
        TESTING=True,
        BENCHMARK_EXPORT_DIR=str(benchmark_dir),
    )
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def app_context(app):
    with app.app_context():
        yield app
