# Pharmaceutical Logistics Optimization with Crew AI

A multi-agent AI system for analyzing pharmaceutical delivery routes and developing optimization strategies. Built with [Crew AI](https://www.crewai.com/) as part of the Analytics Vidhya GenAI Pinnacle Program.

---

## Overview

This project implements a two-agent Crew AI system that tackles pharmaceutical logistics optimization—a domain with meaningful complexity including cold chain requirements, priority-based delivery windows, and regulatory considerations.

**Agent 1: Logistics Analyst**  
Analyzes current delivery operations, identifies inefficiencies, and assesses cold chain compliance risks.

**Agent 2: Optimization Strategist**  
Develops actionable optimization strategies based on the analyst's findings, complete with implementation roadmaps and ROI projections.

---

## Features

- **Parameterized Tasks**: Supply any list of pharmaceutical products; the crew adapts its analysis accordingly
- **Cold Chain Awareness**: Special handling for refrigerated (2-8°C) and frozen (-20°C) medications
- **Priority-Based Scheduling**: Critical (life-saving), urgent (same-day), standard, and routine tiers
- **Structured Output**: Generates comprehensive markdown reports with actionable recommendations
- **Dual Environment Support**: Runs locally (GitHub development) and in Google Colab (notebook submission)

---

## Project Structure

```
pharma-logistics-crewai/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── src/
│   └── pharma_logistics/
│       ├── __init__.py
│       ├── main.py              # Entry point with CLI flags
│       ├── crew.py              # Crew AI orchestration
│       ├── models.py            # Pydantic data models
│       ├── config/
│       │   ├── agents.yaml      # Agent definitions
│       │   └── tasks.yaml       # Task definitions
│       └── data/
│           └── sample_products.json
│
├── notebooks/
│   └── pharma_logistics_crew.ipynb  # Colab-ready notebook
│
├── outputs/                     # Generated reports
│
└── tests/
    ├── test_setup.py            # Project structure validation
    ├── test_models.py           # Pydantic model tests
    └── test_config.py           # YAML config tests
```

---

## Quick Start

### Prerequisites

- Python 3.10 - 3.12
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/crowbarmassage/pharma-logistics-crewai.git
cd pharma-logistics-crewai

# Create virtual environment (Python 3.10-3.13)
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the package
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run

```bash
# Run with defaults (gpt-4o-mini, temperature 0.7)
python -m pharma_logistics.main

# Specify model and temperature
python -m pharma_logistics.main --model gpt-4o --temperature 0.5

# See all options
python -m pharma_logistics.main --help
```

**CLI Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt-4o-mini` | OpenAI model to use |
| `--temperature` | `0.7` | LLM temperature (0.0-1.0) |

The optimization strategy report will be generated at `outputs/optimization_strategy.md`.

---

## Sample Output

The crew processes pharmaceutical products like:

| Product | Temperature | Priority | Destination |
|---------|-------------|----------|-------------|
| Insulin Glargine | Refrigerated | Critical | Northwestern Memorial |
| COVID-19 Vaccine | Frozen | Urgent | Rush University Medical |
| Amoxicillin 500mg | Ambient | Standard | Oak Park Family Medicine |

And produces optimization strategies covering:
- Route restructuring recommendations
- Priority-based scheduling matrices
- Cold chain optimization protocols
- 90-day implementation roadmaps
- ROI projections

---

## Configuration

### Agents (`config/agents.yaml`)

Both agents have pharma-specific backstories:

```yaml
logistics_analyst:
  role: Pharmaceutical Logistics Analyst
  goal: Analyze delivery route operations with attention to cold chain and priority classifications
  backstory: 15 years experience with McKesson and Cardinal Health...

optimization_strategist:
  role: Delivery Route Optimization Strategist
  goal: Develop actionable strategies balancing efficiency with patient safety
  backstory: Fortune 500 pharma consultant, 20-35% cost reductions...
```

### Tasks (`config/tasks.yaml`)

Tasks accept a `{products}` parameter for flexibility:

```yaml
logistics_analysis_task:
  description: >
    Analyze delivery route operations for: {products}
    ...
  agent: logistics_analyst

optimization_strategy_task:
  description: >
    Based on the logistics analysis, develop optimization strategy for: {products}
    ...
  agent: optimization_strategist
  context:
    - logistics_analysis_task  # Receives analyst output
```

---

## Data Models

Products are validated with Pydantic:

```python
class PharmaceuticalProduct(BaseModel):
    product_id: str
    name: str
    temperature_requirement: TemperatureRequirement  # ambient|refrigerated|frozen|controlled
    priority_tier: PriorityTier                      # critical|urgent|standard|routine
    destination_city: str
    destination_facility: str
    quantity_units: int
```

---

## Notebook Submission

For Google Colab execution:

1. Open `notebooks/pharma_logistics_crew.ipynb`
2. Add `OPENAI_API_KEY` to Colab secrets
3. Run all cells

The notebook is self-contained with all dependencies and configurations embedded.

---

## Architecture

```
┌──────────────────┐
│  Product List    │
│  (Parameterized) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│ Logistics        │────▶│ Optimization     │
│ Analyst          │     │ Strategist       │
│                  │     │                  │
│ • Route analysis │     │ • Strategy dev   │
│ • Bottlenecks    │     │ • Implementation │
│ • Cold chain     │     │ • ROI projection │
└──────────────────┘     └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Optimization     │
                         │ Strategy Report  │
                         │ (.md file)       │
                         └──────────────────┘
```

**Process**: Sequential — the Optimization Strategist receives the Logistics Analyst's findings as context before generating recommendations.

---

## Assignment Context

**Course**: Analytics Vidhya GenAI Pinnacle Program  
**Module**: Building AI Agents from Scratch  
**Assignment**: Logistics Optimization Analysis with Crew AI

### Requirements Addressed

- [x] Two agents: Logistics Analyst and Optimization Strategist
- [x] Analyst researches current state of logistics operations
- [x] Strategist creates optimization strategy from analyst insights
- [x] Tasks parameterized for product list input
- [x] Each agent has clear goal and backstory aligned with logistics domain
- [x] Crew assembled with both agents and their tasks

---

## Future Enhancements

See [FutureFeatures.md](./FutureFeatures.md) for the full roadmap. Highlights:

- **Compliance Officer Agent**: FDA/DEA regulatory review
- **Geographic Visualization**: Route maps with Folium
- **Multi-Depot Optimization**: Enterprise-scale logistics
- **Web Dashboard**: Streamlit interface for non-technical users

---

## License

MIT

---

## Acknowledgments

- [Crew AI](https://www.crewai.com/) for the multi-agent orchestration framework
- [Analytics Vidhya](https://www.analyticsvidhya.com/) for the GenAI Pinnacle Program curriculum
