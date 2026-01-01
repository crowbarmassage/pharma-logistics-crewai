# Technical Specification: Pharmaceutical Logistics Optimization with Crew AI

**Project:** Logistics Optimization Analysis with Crew AI  
**Domain:** Pharmaceutical/Medical Supply Delivery Route Optimization  
**Version:** 1.0  
**Date:** December 2024

---

## 1. Executive Summary

This project implements a Crew AI multi-agent system for analyzing pharmaceutical logistics operations and developing optimized delivery route strategies. The system features two collaborative agents—a Logistics Analyst and an Optimization Strategist—working in sequence to analyze current delivery operations and propose evidence-based optimization strategies.

The domain of pharmaceutical logistics introduces meaningful complexity: temperature-controlled products (cold chain), priority tiers (critical medications vs. routine supplies), regulatory compliance considerations, and time-sensitive delivery windows.

---

## 2. Assignment Requirements Mapping

| Requirement | Implementation |
|-------------|----------------|
| Two agents: Logistics Analyst + Optimization Strategist | ✅ Defined in `agents.yaml` |
| Logistics Analyst researches current state | ✅ Task 1: Analyze route efficiency |
| Optimization Strategist creates strategy from insights | ✅ Task 2: Consumes Task 1 output |
| Parameterized tasks (list of products) | ✅ `{products}` variable in YAML |
| Clear goal and backstory per agent | ✅ Pharma-specific personas |
| Build the Crew with both agents/tasks | ✅ `crew.py` orchestration |

---

## 3. System Architecture

### 3.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│  products: List[PharmaceuticalProduct]                          │
│  - name, temperature_requirement, priority, destination, etc.   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOGISTICS ANALYST AGENT                       │
│  Role: Pharmaceutical Logistics Analyst                         │
│  Goal: Analyze current delivery operations and identify         │
│        inefficiencies in route planning                         │
│  Output: Current State Analysis Report                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OPTIMIZATION STRATEGIST AGENT                   │
│  Role: Route Optimization Strategist                            │
│  Goal: Develop actionable delivery optimization strategy        │
│        based on analyst insights                                │
│  Input: Logistics Analyst's findings                            │
│  Output: Optimization Strategy Report                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                   │
│  - optimization_strategy.md (final report)                      │
│  - CrewOutput object with structured results                    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Crew AI Process Type

**Sequential Process**: Task 2 (Optimization Strategy) depends on Task 1 (Logistics Analysis) output. This is the natural workflow—strategy follows analysis.

---

## 4. Agent Specifications

### 4.1 Logistics Analyst Agent

```yaml
logistics_analyst:
  role: >
    Pharmaceutical Logistics Analyst
  goal: >
    Analyze current delivery route operations for pharmaceutical products,
    identifying inefficiencies, bottlenecks, and optimization opportunities
    with special attention to cold chain requirements and priority classifications.
  backstory: >
    You are a seasoned logistics analyst with 15 years of experience in
    pharmaceutical supply chain operations. You've worked with major distributors
    like McKesson and Cardinal Health, specializing in temperature-sensitive
    medication delivery. Your expertise includes route efficiency analysis,
    cold chain compliance auditing, and delivery window optimization. You're
    known for your meticulous data analysis and ability to identify hidden
    inefficiencies that cost companies millions annually.
  verbose: true
  allow_delegation: false
```

**Key Characteristics:**
- Domain expertise in pharmaceutical cold chain
- Analytical mindset focused on identifying problems
- Does not delegate—performs deep analysis independently
- Outputs structured findings for downstream consumption

### 4.2 Optimization Strategist Agent

```yaml
optimization_strategist:
  role: >
    Delivery Route Optimization Strategist
  goal: >
    Develop comprehensive, actionable optimization strategies for pharmaceutical
    delivery routes that maximize efficiency while maintaining cold chain integrity
    and meeting priority-based delivery windows.
  backstory: >
    You are a strategic logistics consultant who has helped Fortune 500
    pharmaceutical companies reduce delivery costs by 20-35% while improving
    on-time delivery rates. Your approach combines operations research principles
    with practical implementation experience. You specialize in translating
    analytical findings into executable strategies with clear ROI projections.
    Healthcare executives trust your recommendations because you balance
    efficiency gains with patient safety requirements.
  verbose: true
  allow_delegation: false
```

**Key Characteristics:**
- Strategic thinking focused on solutions
- Consumes analyst findings as context
- Produces actionable recommendations
- Balances efficiency with compliance/safety

---

## 5. Task Specifications

### 5.1 Logistics Analysis Task

```yaml
logistics_analysis_task:
  description: >
    Analyze the current delivery route operations for the following pharmaceutical
    products: {products}
    
    Your analysis must cover:
    1. CURRENT STATE ASSESSMENT
       - Map existing delivery routes and their characteristics
       - Identify which products require cold chain handling
       - Document current priority classifications and delivery windows
    
    2. EFFICIENCY METRICS
       - Estimate route utilization rates
       - Identify redundant or overlapping delivery paths
       - Assess vehicle capacity utilization
    
    3. COMPLIANCE CONSIDERATIONS
       - Cold chain maintenance risks on current routes
       - Time-sensitive delivery window compliance
       - Regulatory exposure points
    
    4. BOTTLENECK IDENTIFICATION
       - Geographic clustering inefficiencies
       - Peak demand timing conflicts
       - Priority tier conflicts (critical vs. routine)
    
    Provide specific, data-informed observations that can be acted upon.
  expected_output: >
    A comprehensive logistics analysis report in markdown format containing:
    - Executive summary of key findings
    - Detailed current state assessment
    - Quantified efficiency metrics (with estimates where exact data unavailable)
    - Prioritized list of identified bottlenecks and inefficiencies
    - Risk assessment for cold chain and compliance issues
    - Clear handoff points for optimization strategy development
  agent: logistics_analyst
```

### 5.2 Optimization Strategy Task

```yaml
optimization_strategy_task:
  description: >
    Based on the logistics analysis provided, develop a comprehensive
    optimization strategy for pharmaceutical delivery routes handling
    the following products: {products}
    
    Your strategy must address:
    1. ROUTE RESTRUCTURING RECOMMENDATIONS
       - Proposed route consolidations or splits
       - Geographic clustering optimizations
       - Hub-and-spoke vs. direct delivery trade-offs
    
    2. PRIORITY-BASED SCHEDULING
       - Critical medication prioritization protocols
       - Delivery window optimization
       - Buffer time allocations for high-priority items
    
    3. COLD CHAIN OPTIMIZATION
       - Temperature-controlled vehicle allocation
       - Route sequencing for temperature-sensitive products
       - Contingency protocols for cold chain breaks
    
    4. IMPLEMENTATION ROADMAP
       - Phased rollout plan (Quick wins → Medium-term → Long-term)
       - Resource requirements
       - Success metrics and KPIs
    
    5. ROI PROJECTION
       - Estimated cost savings
       - Efficiency gains (time, fuel, vehicle utilization)
       - Risk mitigation value
    
    Be specific and actionable. Avoid generic recommendations.
  expected_output: >
    A detailed optimization strategy report in markdown format containing:
    - Executive summary with top 3-5 recommendations
    - Detailed route restructuring plan with rationale
    - Priority-based scheduling matrix
    - Cold chain optimization protocols
    - 90-day implementation roadmap
    - ROI projections with assumptions stated
    - Success metrics and monitoring approach
  agent: optimization_strategist
  context:
    - logistics_analysis_task
  output_file: optimization_strategy.md
```

---

## 6. Data Models (Pydantic)

### 6.1 PharmaceuticalProduct Model

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class TemperatureRequirement(str, Enum):
    AMBIENT = "ambient"           # 15-25°C
    REFRIGERATED = "refrigerated" # 2-8°C
    FROZEN = "frozen"             # -20°C or below
    CONTROLLED = "controlled"     # Specific range required

class PriorityTier(str, Enum):
    CRITICAL = "critical"         # Life-saving, <4hr delivery
    URGENT = "urgent"             # Same-day required
    STANDARD = "standard"         # Next-day acceptable
    ROUTINE = "routine"           # 2-3 day window

class PharmaceuticalProduct(BaseModel):
    """Represents a pharmaceutical product for delivery optimization."""
    
    product_id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    temperature_requirement: TemperatureRequirement = Field(
        default=TemperatureRequirement.AMBIENT,
        description="Storage/transport temperature requirement"
    )
    priority_tier: PriorityTier = Field(
        default=PriorityTier.STANDARD,
        description="Delivery priority classification"
    )
    destination_city: str = Field(..., description="Delivery destination city")
    destination_facility: str = Field(..., description="Hospital/pharmacy name")
    quantity_units: int = Field(default=1, description="Number of units to deliver")
    max_delivery_hours: Optional[int] = Field(
        default=None,
        description="Maximum hours for delivery (overrides priority default)"
    )
    requires_signature: bool = Field(
        default=True,
        description="Requires signature on delivery"
    )
    hazmat_classification: Optional[str] = Field(
        default=None,
        description="Hazardous material classification if applicable"
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
```

### 6.2 ProductList Input Formatter

```python
from typing import List

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
    cold_chain_count = len([
        p for p in products 
        if p.temperature_requirement in [
            TemperatureRequirement.REFRIGERATED,
            TemperatureRequirement.FROZEN
        ]
    ])
    
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
```

---

## 7. Project Structure

```
pharma_logistics_crew/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   └── pharma_logistics/
│       ├── __init__.py
│       ├── main.py              # Entry point
│       ├── crew.py              # Crew definition with @CrewBase
│       ├── models.py            # Pydantic models
│       │
│       ├── config/
│       │   ├── agents.yaml      # Agent definitions
│       │   └── tasks.yaml       # Task definitions
│       │
│       └── data/
│           └── sample_products.json  # Test data
│
├── notebooks/
│   └── pharma_logistics_crew.ipynb   # Submission notebook
│
├── outputs/
│   └── .gitkeep                 # Generated reports go here
│
└── tests/
    ├── __init__.py
    ├── test_models.py
    └── test_crew.py
```

---

## 8. Environment Configuration

### 8.1 Local Development (.env)

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini

# Optional: Alternative providers
# ANTHROPIC_API_KEY=your_anthropic_key
# GROQ_API_KEY=your_groq_key

# Crew AI Settings
CREWAI_VERBOSE=true
```

### 8.2 Notebook Environment (Colab Secrets)

```python
# In notebook, use userdata for secrets
from google.colab import userdata
import os

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

### 8.3 Dual-Environment Configuration Pattern

```python
# config.py - Works in both environments
import os

def get_api_key(key_name: str) -> str:
    """
    Retrieves API key from environment.
    Works in both local (.env) and Colab (userdata) environments.
    """
    # Try environment variable first (local dev)
    value = os.environ.get(key_name)
    
    if value:
        return value
    
    # Try Colab userdata (notebook environment)
    try:
        from google.colab import userdata
        return userdata.get(key_name)
    except (ImportError, ModuleNotFoundError):
        pass
    
    raise ValueError(f"API key '{key_name}' not found in environment or Colab secrets")
```

---

## 9. Dependencies

### 9.1 requirements.txt

```
crewai>=0.86.0
crewai-tools>=0.17.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pyyaml>=6.0.0
```

### 9.2 pyproject.toml

```toml
[project]
name = "pharma-logistics-crew"
version = "1.0.0"
description = "Crew AI system for pharmaceutical logistics optimization"
requires-python = ">=3.10,<3.13"
dependencies = [
    "crewai>=0.86.0",
    "crewai-tools>=0.17.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
notebook = [
    "ipykernel>=6.0.0",
    "nbformat>=5.0.0",
]
```

---

## 10. LLM Configuration

### 10.1 Model Selection

| Provider | Model | Use Case |
|----------|-------|----------|
| OpenAI | gpt-4o-mini | **Default** - Cost-effective, good reasoning |
| OpenAI | gpt-4o | Higher quality, use if budget allows |
| Anthropic | claude-3-haiku | Alternative, similar cost profile |
| Groq | llama-3.1-70b | Fast inference, local testing |

### 10.2 Crew AI LLM Configuration

```python
from crewai import LLM

# Default configuration
llm = LLM(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=4096
)

# Pass to agents
logistics_analyst = Agent(
    config=self.agents_config['logistics_analyst'],
    llm=llm
)
```

---

## 11. Crew Execution Methods

### 11.1 Standard Execution

```python
# Single run with product list
inputs = {
    "products": format_products_for_crew(product_list)
}
result = crew.kickoff(inputs=inputs)
```

### 11.2 Batch Execution (Multiple Scenarios)

```python
# Test multiple product configurations
scenarios = [
    {"products": format_products_for_crew(critical_only)},
    {"products": format_products_for_crew(cold_chain_heavy)},
    {"products": format_products_for_crew(multi_destination)},
]
results = crew.kickoff_for_each(inputs=scenarios)
```

---

## 12. Output Specifications

### 12.1 CrewOutput Structure

```python
result = crew.kickoff(inputs=inputs)

# Access outputs
print(result.raw)           # Full text output
print(result.tasks_output)  # Individual task outputs
print(result.token_usage)   # Token consumption metrics

# Final report written to:
# outputs/optimization_strategy.md
```

### 12.2 Expected Report Structure

```markdown
# Pharmaceutical Delivery Route Optimization Strategy

## Executive Summary
[Top 3-5 recommendations with projected impact]

## Current State Analysis
[From Logistics Analyst findings]

## Route Restructuring Plan
### Geographic Clustering
### Hub-and-Spoke Analysis
### Consolidation Opportunities

## Priority-Based Scheduling Matrix
| Priority | Products | Delivery Window | Scheduling Rule |
|----------|----------|-----------------|-----------------|
| Critical | ...      | <4 hours        | First dispatch  |

## Cold Chain Optimization
### Vehicle Allocation
### Route Sequencing
### Contingency Protocols

## Implementation Roadmap
### Phase 1: Quick Wins (0-30 days)
### Phase 2: Medium-term (30-60 days)
### Phase 3: Long-term (60-90 days)

## ROI Projections
[Quantified estimates with assumptions]

## Success Metrics
[KPIs and monitoring approach]
```

---

## 13. Testing Strategy

### 13.1 Unit Tests

```python
# tests/test_models.py
def test_pharmaceutical_product_validation():
    """Test Pydantic model validation."""
    product = PharmaceuticalProduct(
        product_id="INS-001",
        name="Insulin Glargine",
        temperature_requirement=TemperatureRequirement.REFRIGERATED,
        priority_tier=PriorityTier.CRITICAL,
        destination_city="Chicago",
        destination_facility="Northwestern Memorial Hospital"
    )
    assert product.temperature_requirement == TemperatureRequirement.REFRIGERATED

def test_products_formatting():
    """Test product list formatting for prompts."""
    products = [...]
    formatted = format_products_for_crew(products)
    assert "CRITICAL PRIORITY" in formatted
    assert "Cold Chain Required:" in formatted
```

### 13.2 Integration Tests

```python
# tests/test_crew.py
def test_crew_execution_smoke():
    """Smoke test for crew execution."""
    crew = PharmaLogisticsCrew()
    result = crew.crew().kickoff(inputs={
        "products": "Test product: Insulin, refrigerated, critical priority"
    })
    assert result.raw is not None
    assert len(result.tasks_output) == 2
```

---

## 14. Notebook Conversion Strategy

Since the development happens in GitHub but submission requires a working notebook:

### 14.1 Notebook Structure

```
Cell 1: Environment Setup
- Install dependencies
- Configure API keys from Colab secrets

Cell 2: Model Definitions
- PharmaceuticalProduct Pydantic model
- Formatting utilities

Cell 3: Agent Configuration (YAML as string)
- agents_config dictionary

Cell 4: Task Configuration (YAML as string)
- tasks_config dictionary

Cell 5: Crew Definition
- PharmaLogisticsCrew class

Cell 6: Sample Data
- Example pharmaceutical products

Cell 7: Execution
- Run crew with sample data

Cell 8: Results Display
- Show outputs and generated report
```

### 14.2 YAML-in-Notebook Pattern

```python
# For notebook, embed YAML as Python dictionaries
agents_config = {
    "logistics_analyst": {
        "role": "Pharmaceutical Logistics Analyst",
        "goal": "Analyze current delivery route operations...",
        "backstory": "You are a seasoned logistics analyst..."
    },
    "optimization_strategist": {
        "role": "Delivery Route Optimization Strategist",
        "goal": "Develop comprehensive optimization strategies...",
        "backstory": "You are a strategic logistics consultant..."
    }
}
```

---

## 15. Risk Considerations

| Risk | Mitigation |
|------|------------|
| LLM rate limits | Implement retry logic with exponential backoff |
| Long execution time | Use gpt-4o-mini for faster inference |
| Inconsistent outputs | Set temperature=0.7, provide structured prompts |
| Token limits exceeded | Limit product list to ~20 items per run |
| Cold chain misunderstanding | Explicit temperature ranges in prompts |

---

## 16. Success Criteria

1. **Functional**: Crew executes end-to-end without errors
2. **Output Quality**: Generated strategy is specific, actionable, and pharma-relevant
3. **Parameterization**: Different product lists produce appropriately different strategies
4. **Task Dependency**: Optimization strategy clearly builds on analyst findings
5. **Deliverable**: Notebook runs successfully in fresh Colab environment

---

*End of Technical Specification*
