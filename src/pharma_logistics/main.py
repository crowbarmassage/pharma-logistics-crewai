"""Main entry point for pharmaceutical logistics optimization."""

import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from .crew import run_crew
from .models import PharmaceuticalProduct


def load_sample_products() -> List[PharmaceuticalProduct]:
    """Load sample pharmaceutical products from JSON file."""
    data_path = Path(__file__).parent / "data" / "sample_products.json"

    with open(data_path) as f:
        data = json.load(f)

    return [PharmaceuticalProduct(**item) for item in data]


def main():
    """Run the pharmaceutical logistics optimization crew."""
    # Load environment variables
    load_dotenv()

    # Verify API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your .env file "
            "or environment variables."
        )

    print("=" * 60)
    print("Pharmaceutical Logistics Optimization Crew")
    print("=" * 60)
    print()

    # Load sample products
    products = load_sample_products()
    print(f"Loaded {len(products)} pharmaceutical products for analysis")
    print()

    # Display product summary
    print("Products to analyze:")
    for product in products:
        print(f"  - {product.name} ({product.priority_tier.value}, "
              f"{product.temperature_requirement.value})")
    print()

    print("Starting crew execution...")
    print("-" * 60)
    print()

    # Run the crew
    result = run_crew(products)

    print()
    print("-" * 60)
    print("Crew execution complete!")
    print()
    print("Output saved to: outputs/optimization_strategy.md")
    print()

    return result


if __name__ == "__main__":
    main()
