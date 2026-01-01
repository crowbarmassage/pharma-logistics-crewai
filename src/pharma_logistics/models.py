"""Pydantic models for pharmaceutical logistics optimization."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TemperatureRequirement(str, Enum):
    """Temperature storage/transport requirements for pharmaceutical products."""

    AMBIENT = "ambient"  # 15-25°C
    REFRIGERATED = "refrigerated"  # 2-8°C
    FROZEN = "frozen"  # -20°C or below
    CONTROLLED = "controlled"  # Specific range required


class PriorityTier(str, Enum):
    """Delivery priority classifications."""

    CRITICAL = "critical"  # Life-saving, <4hr delivery
    URGENT = "urgent"  # Same-day required
    STANDARD = "standard"  # Next-day acceptable
    ROUTINE = "routine"  # 2-3 day window


class PharmaceuticalProduct(BaseModel):
    """Represents a pharmaceutical product for delivery optimization."""

    product_id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    temperature_requirement: TemperatureRequirement = Field(
        default=TemperatureRequirement.AMBIENT,
        description="Storage/transport temperature requirement",
    )
    priority_tier: PriorityTier = Field(
        default=PriorityTier.STANDARD,
        description="Delivery priority classification",
    )
    destination_city: str = Field(..., description="Delivery destination city")
    destination_facility: str = Field(..., description="Hospital/pharmacy name")
    quantity_units: int = Field(default=1, description="Number of units to deliver")
    max_delivery_hours: Optional[int] = Field(
        default=None,
        description="Maximum hours for delivery (overrides priority default)",
    )
    requires_signature: bool = Field(
        default=True,
        description="Requires signature on delivery",
    )
    hazmat_classification: Optional[str] = Field(
        default=None,
        description="Hazardous material classification if applicable",
    )

    def format_for_prompt(self) -> str:
        """Format product details for LLM prompt injection."""
        return (
            f"- {self.name} (ID: {self.product_id})\n"
            f"  Temperature: {self.temperature_requirement.value}\n"
            f"  Priority: {self.priority_tier.value}\n"
            f"  Destination: {self.destination_facility}, {self.destination_city}\n"
            f"  Quantity: {self.quantity_units} units"
        )


def format_products_for_crew(products: List[PharmaceuticalProduct]) -> str:
    """
    Formats a list of pharmaceutical products into a string
    suitable for injection into Crew AI task descriptions.
    """
    if not products:
        return "No products provided."

    formatted = []

    # Group by priority for clearer analysis
    critical = [p for p in products if p.priority_tier == PriorityTier.CRITICAL]
    urgent = [p for p in products if p.priority_tier == PriorityTier.URGENT]
    standard = [p for p in products if p.priority_tier == PriorityTier.STANDARD]
    routine = [p for p in products if p.priority_tier == PriorityTier.ROUTINE]

    if critical:
        formatted.append("## CRITICAL PRIORITY (Life-saving, <4hr)")
        formatted.extend([p.format_for_prompt() for p in critical])

    if urgent:
        formatted.append("\n## URGENT PRIORITY (Same-day)")
        formatted.extend([p.format_for_prompt() for p in urgent])

    if standard:
        formatted.append("\n## STANDARD PRIORITY (Next-day)")
        formatted.extend([p.format_for_prompt() for p in standard])

    if routine:
        formatted.append("\n## ROUTINE PRIORITY (2-3 day)")
        formatted.extend([p.format_for_prompt() for p in routine])

    # Add summary statistics
    cold_chain_count = len(
        [
            p
            for p in products
            if p.temperature_requirement
            in [TemperatureRequirement.REFRIGERATED, TemperatureRequirement.FROZEN]
        ]
    )

    summary = f"""
---
SUMMARY STATISTICS:
- Total Products: {len(products)}
- Critical Priority: {len(critical)}
- Cold Chain Required: {cold_chain_count}
- Unique Destinations: {len(set(p.destination_city for p in products))}
"""
    formatted.append(summary)

    return "\n".join(formatted)
