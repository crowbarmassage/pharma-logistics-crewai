"""Tests for configuration files."""

import json
from pathlib import Path

import pytest
import yaml


class TestAgentsConfig:
    """Tests for agents.yaml configuration."""

    @pytest.fixture
    def agents_config(self) -> dict:
        """Load agents configuration."""
        config_path = (
            Path(__file__).parent.parent
            / "src"
            / "pharma_logistics"
            / "config"
            / "agents.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_logistics_analyst_exists(self, agents_config: dict):
        """Verify logistics_analyst agent is defined."""
        assert "logistics_analyst" in agents_config

    def test_optimization_strategist_exists(self, agents_config: dict):
        """Verify optimization_strategist agent is defined."""
        assert "optimization_strategist" in agents_config

    def test_logistics_analyst_has_required_fields(self, agents_config: dict):
        """Verify logistics_analyst has all required fields."""
        analyst = agents_config["logistics_analyst"]
        assert "role" in analyst
        assert "goal" in analyst
        assert "backstory" in analyst
        assert "Pharmaceutical" in analyst["role"]

    def test_optimization_strategist_has_required_fields(self, agents_config: dict):
        """Verify optimization_strategist has all required fields."""
        strategist = agents_config["optimization_strategist"]
        assert "role" in strategist
        assert "goal" in strategist
        assert "backstory" in strategist
        assert "Optimization" in strategist["role"]


class TestTasksConfig:
    """Tests for tasks.yaml configuration."""

    @pytest.fixture
    def tasks_config(self) -> dict:
        """Load tasks configuration."""
        config_path = (
            Path(__file__).parent.parent
            / "src"
            / "pharma_logistics"
            / "config"
            / "tasks.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_logistics_analysis_task_exists(self, tasks_config: dict):
        """Verify logistics_analysis_task is defined."""
        assert "logistics_analysis_task" in tasks_config

    def test_optimization_strategy_task_exists(self, tasks_config: dict):
        """Verify optimization_strategy_task is defined."""
        assert "optimization_strategy_task" in tasks_config

    def test_logistics_analysis_task_has_required_fields(self, tasks_config: dict):
        """Verify logistics_analysis_task has required fields."""
        task = tasks_config["logistics_analysis_task"]
        assert "description" in task
        assert "expected_output" in task
        assert "agent" in task
        assert "{products}" in task["description"]

    def test_optimization_strategy_task_has_required_fields(self, tasks_config: dict):
        """Verify optimization_strategy_task has required fields."""
        task = tasks_config["optimization_strategy_task"]
        assert "description" in task
        assert "expected_output" in task
        assert "agent" in task
        assert "context" in task
        assert "{products}" in task["description"]

    def test_task_dependency(self, tasks_config: dict):
        """Verify optimization task depends on analysis task."""
        task = tasks_config["optimization_strategy_task"]
        assert "logistics_analysis_task" in task["context"]


class TestSampleData:
    """Tests for sample_products.json."""

    @pytest.fixture
    def sample_products(self) -> list:
        """Load sample products."""
        data_path = (
            Path(__file__).parent.parent
            / "src"
            / "pharma_logistics"
            / "data"
            / "sample_products.json"
        )
        with open(data_path) as f:
            return json.load(f)

    def test_sample_products_not_empty(self, sample_products: list):
        """Verify sample products list is not empty."""
        assert len(sample_products) >= 6

    def test_products_have_required_fields(self, sample_products: list):
        """Verify all products have required fields."""
        required_fields = [
            "product_id",
            "name",
            "temperature_requirement",
            "priority_tier",
            "destination_city",
            "destination_facility",
        ]
        for product in sample_products:
            for field in required_fields:
                assert field in product, f"Missing {field} in {product.get('name', 'unknown')}"

    def test_products_have_valid_temperature(self, sample_products: list):
        """Verify all products have valid temperature requirements."""
        valid_temps = ["ambient", "refrigerated", "frozen", "controlled"]
        for product in sample_products:
            assert product["temperature_requirement"] in valid_temps

    def test_products_have_valid_priority(self, sample_products: list):
        """Verify all products have valid priority tiers."""
        valid_priorities = ["critical", "urgent", "standard", "routine"]
        for product in sample_products:
            assert product["priority_tier"] in valid_priorities

    def test_has_critical_priority_products(self, sample_products: list):
        """Verify at least one critical priority product exists."""
        critical = [p for p in sample_products if p["priority_tier"] == "critical"]
        assert len(critical) >= 1

    def test_has_cold_chain_products(self, sample_products: list):
        """Verify at least one cold chain product exists."""
        cold_chain = [
            p
            for p in sample_products
            if p["temperature_requirement"] in ["refrigerated", "frozen"]
        ]
        assert len(cold_chain) >= 1
