"""Tests for Pydantic models."""

import pytest

from src.pharma_logistics.models import (
    PharmaceuticalProduct,
    PriorityTier,
    TemperatureRequirement,
    format_products_for_crew,
)


class TestPharmaceuticalProduct:
    """Tests for PharmaceuticalProduct model."""

    def test_create_product_with_required_fields(self):
        """Test creating a product with only required fields."""
        product = PharmaceuticalProduct(
            product_id="TEST-001",
            name="Test Medication",
            destination_city="Chicago",
            destination_facility="Test Hospital",
        )
        assert product.product_id == "TEST-001"
        assert product.name == "Test Medication"
        assert product.temperature_requirement == TemperatureRequirement.AMBIENT
        assert product.priority_tier == PriorityTier.STANDARD

    def test_create_product_with_all_fields(self):
        """Test creating a product with all fields specified."""
        product = PharmaceuticalProduct(
            product_id="INS-001",
            name="Insulin Glargine",
            temperature_requirement=TemperatureRequirement.REFRIGERATED,
            priority_tier=PriorityTier.CRITICAL,
            destination_city="Chicago",
            destination_facility="Northwestern Memorial Hospital",
            quantity_units=50,
            max_delivery_hours=4,
            requires_signature=True,
            hazmat_classification=None,
        )
        assert product.temperature_requirement == TemperatureRequirement.REFRIGERATED
        assert product.priority_tier == PriorityTier.CRITICAL
        assert product.quantity_units == 50
        assert product.max_delivery_hours == 4

    def test_format_for_prompt(self):
        """Test product formatting for LLM prompts."""
        product = PharmaceuticalProduct(
            product_id="INS-001",
            name="Insulin Glargine",
            temperature_requirement=TemperatureRequirement.REFRIGERATED,
            priority_tier=PriorityTier.CRITICAL,
            destination_city="Chicago",
            destination_facility="Northwestern Memorial",
            quantity_units=50,
        )
        formatted = product.format_for_prompt()
        assert "Insulin Glargine" in formatted
        assert "INS-001" in formatted
        assert "refrigerated" in formatted
        assert "critical" in formatted
        assert "Northwestern Memorial" in formatted
        assert "50 units" in formatted


class TestTemperatureRequirement:
    """Tests for TemperatureRequirement enum."""

    def test_all_temperature_values(self):
        """Verify all temperature requirement values exist."""
        assert TemperatureRequirement.AMBIENT.value == "ambient"
        assert TemperatureRequirement.REFRIGERATED.value == "refrigerated"
        assert TemperatureRequirement.FROZEN.value == "frozen"
        assert TemperatureRequirement.CONTROLLED.value == "controlled"


class TestPriorityTier:
    """Tests for PriorityTier enum."""

    def test_all_priority_values(self):
        """Verify all priority tier values exist."""
        assert PriorityTier.CRITICAL.value == "critical"
        assert PriorityTier.URGENT.value == "urgent"
        assert PriorityTier.STANDARD.value == "standard"
        assert PriorityTier.ROUTINE.value == "routine"


class TestFormatProductsForCrew:
    """Tests for format_products_for_crew function."""

    def test_empty_product_list(self):
        """Test formatting empty product list."""
        result = format_products_for_crew([])
        assert result == "No products provided."

    def test_products_grouped_by_priority(self):
        """Test that products are grouped by priority tier."""
        products = [
            PharmaceuticalProduct(
                product_id="P1",
                name="Critical Med",
                priority_tier=PriorityTier.CRITICAL,
                destination_city="Chicago",
                destination_facility="Hospital A",
            ),
            PharmaceuticalProduct(
                product_id="P2",
                name="Routine Med",
                priority_tier=PriorityTier.ROUTINE,
                destination_city="Chicago",
                destination_facility="Pharmacy B",
            ),
        ]
        result = format_products_for_crew(products)
        assert "CRITICAL PRIORITY" in result
        assert "ROUTINE PRIORITY" in result
        # Critical should appear before routine
        assert result.index("CRITICAL") < result.index("ROUTINE")

    def test_summary_statistics(self):
        """Test that summary statistics are included."""
        products = [
            PharmaceuticalProduct(
                product_id="P1",
                name="Cold Med",
                temperature_requirement=TemperatureRequirement.REFRIGERATED,
                priority_tier=PriorityTier.CRITICAL,
                destination_city="Chicago",
                destination_facility="Hospital A",
            ),
            PharmaceuticalProduct(
                product_id="P2",
                name="Frozen Med",
                temperature_requirement=TemperatureRequirement.FROZEN,
                destination_city="Detroit",
                destination_facility="Hospital B",
            ),
        ]
        result = format_products_for_crew(products)
        assert "Total Products: 2" in result
        assert "Critical Priority: 1" in result
        assert "Cold Chain Required: 2" in result
        assert "Unique Destinations: 2" in result
