import pytest
from click.testing import CliRunner

from crystal_property_predictor.cli import cli


@pytest.fixture()
def runner():
    return CliRunner()


def test_cli_template(runner):
    result = runner.invoke(cli)
    assert result.exit_code == 0
