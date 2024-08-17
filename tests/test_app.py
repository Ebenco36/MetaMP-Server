import os
import tempfile

import pytest
from src.MP import flask_app as app


@pytest.fixture
def client():
    db_fd, app.config['DATABASE'] = tempfile.mkstemp()

    app.config['TESTING'] = True

    client = app.test_client()

    yield client

    os.close(db_fd)
    os.unlink(app.config['DATABASE'])
