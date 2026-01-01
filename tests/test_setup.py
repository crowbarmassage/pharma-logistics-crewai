"""Setup validation tests - run after initial setup."""

import pytest
from pathlib import Path


class TestProjectStructure:
    """Validate project directory structure exists."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    def test_src_directory_exists(self, project_root: Path):
        """Verify src directory structure."""
        assert (project_root / "src" / "pharma_logistics").is_dir()
        assert (project_root / "src" / "pharma_logistics" / "__init__.py").is_file()

    def test_config_directory_exists(self, project_root: Path):
        """Verify config directory exists."""
        assert (project_root / "src" / "pharma_logistics" / "config").is_dir()

    def test_data_directory_exists(self, project_root: Path):
        """Verify data directory exists."""
        assert (project_root / "src" / "pharma_logistics" / "data").is_dir()

    def test_tests_directory_exists(self, project_root: Path):
        """Verify tests directory exists."""
        assert (project_root / "tests").is_dir()
        assert (project_root / "tests" / "__init__.py").is_file()

    def test_outputs_directory_exists(self, project_root: Path):
        """Verify outputs directory exists."""
        assert (project_root / "outputs").is_dir()

    def test_notebooks_directory_exists(self, project_root: Path):
        """Verify notebooks directory exists."""
        assert (project_root / "notebooks").is_dir()


class TestConfigurationFiles:
    """Validate configuration files exist and have content."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    def test_requirements_txt_exists(self, project_root: Path):
        """Verify requirements.txt exists with dependencies."""
        req_file = project_root / "requirements.txt"
        assert req_file.is_file()
        content = req_file.read_text()
        assert "crewai" in content
        assert "pydantic" in content
        assert "pytest" in content

    def test_pyproject_toml_exists(self, project_root: Path):
        """Verify pyproject.toml exists with correct project name."""
        pyproject = project_root / "pyproject.toml"
        assert pyproject.is_file()
        content = pyproject.read_text()
        assert "pharma-logistics-crew" in content
        assert "crewai" in content

    def test_gitignore_exists(self, project_root: Path):
        """Verify .gitignore exists with common patterns."""
        gitignore = project_root / ".gitignore"
        assert gitignore.is_file()
        content = gitignore.read_text()
        assert ".env" in content
        assert "__pycache__" in content
        assert "venv" in content

    def test_env_example_exists(self, project_root: Path):
        """Verify .env.example exists with required keys."""
        env_example = project_root / ".env.example"
        assert env_example.is_file()
        content = env_example.read_text()
        assert "OPENAI_API_KEY" in content


class TestDocumentation:
    """Validate documentation files exist."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    def test_readme_exists(self, project_root: Path):
        """Verify README.md exists."""
        assert (project_root / "README.md").is_file()

    def test_tech_specs_exists(self, project_root: Path):
        """Verify TECH_SPECS.md exists."""
        assert (project_root / "TECH_SPECS.md").is_file()

    def test_atomic_steps_exists(self, project_root: Path):
        """Verify ATOMIC_STEPS.md exists."""
        assert (project_root / "ATOMIC_STEPS.md").is_file()
