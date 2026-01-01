"""Crew AI orchestration for pharmaceutical logistics optimization."""

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from .models import PharmaceuticalProduct, format_products_for_crew


@CrewBase
class PharmaLogisticsCrew:
    """Pharmaceutical Logistics Optimization Crew.

    This crew analyzes pharmaceutical delivery operations and develops
    optimization strategies using two collaborative agents.
    """

    # Path to config files relative to this module
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        """Initialize the crew with config file paths."""
        # Ensure config paths are absolute
        base_path = Path(__file__).parent
        self.agents_config = str(base_path / "config" / "agents.yaml")
        self.tasks_config = str(base_path / "config" / "tasks.yaml")

    @agent
    def logistics_analyst(self) -> Agent:
        """Create the Logistics Analyst agent."""
        return Agent(
            config=self.agents_config["logistics_analyst"],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def optimization_strategist(self) -> Agent:
        """Create the Optimization Strategist agent."""
        return Agent(
            config=self.agents_config["optimization_strategist"],
            verbose=True,
            allow_delegation=False,
        )

    @task
    def logistics_analysis_task(self) -> Task:
        """Create the logistics analysis task."""
        return Task(
            config=self.tasks_config["logistics_analysis_task"],
        )

    @task
    def optimization_strategy_task(self) -> Task:
        """Create the optimization strategy task."""
        return Task(
            config=self.tasks_config["optimization_strategy_task"],
            output_file="outputs/optimization_strategy.md",
        )

    @crew
    def crew(self) -> Crew:
        """Create the pharmaceutical logistics crew."""
        return Crew(
            agents=self.agents,  # Automatically populated by @agent decorators
            tasks=self.tasks,  # Automatically populated by @task decorators
            process=Process.sequential,
            verbose=True,
        )


def run_crew(products: list[PharmaceuticalProduct]) -> str:
    """
    Run the pharmaceutical logistics crew with the given products.

    Args:
        products: List of pharmaceutical products to analyze

    Returns:
        The crew's output as a string
    """
    # Format products for the crew
    formatted_products = format_products_for_crew(products)

    # Create and run the crew
    pharma_crew = PharmaLogisticsCrew()
    result = pharma_crew.crew().kickoff(inputs={"products": formatted_products})

    return result.raw
