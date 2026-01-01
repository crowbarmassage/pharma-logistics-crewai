# Atomic Implementation Steps: Pharmaceutical Logistics Crew AI

**Project:** Logistics Optimization Analysis with Crew AI  
**Total Phases:** 8  
**Total Steps:** 32  
**Estimated Time:** 4-6 hours

---

## Phase 0: Environment Setup
**Objective:** Establish development environment with all dependencies

### Step 0.1: Create Project Directory Structure
```bash
mkdir -p pharma_logistics_crew/{src/pharma_logistics/{config,data},notebooks,outputs,tests}
cd pharma_logistics_crew
touch README.md requirements.txt .env.example .gitignore
touch src/pharma_logistics/__init__.py
touch src/pharma_logistics/{main.py,crew.py,models.py}
touch src/pharma_logistics/config/{agents.yaml,tasks.yaml}
touch tests/{__init__.py,test_models.py,test_crew.py}
```

**Validation:**
- [ ] Directory structure matches TECH_SPECS Section 7
- [ ] All placeholder files exist

### Step 0.2: Initialize Git Repository
```bash
git init
```

**Create .gitignore:**
```
# Environment
.env
.venv/
venv/
__pycache__/
*.pyc

# IDE
.vscode/
.idea/

# Outputs
outputs/*.md
outputs/*.json

# Notebook checkpoints
.ipynb_checkpoints/

# OS
.DS_Store
```

**Validation:**
- [ ] `git status` shows untracked files
- [ ] `.env` will be ignored

### Step 0.3: Create requirements.txt
```
crewai>=0.86.0
crewai-tools>=0.17.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pyyaml>=6.0.0
```

**Validation:**
- [ ] File contains all 5 dependencies
- [ ] Version constraints use `>=` for flexibility

### Step 0.4: Create Virtual Environment and Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Validation:**
```bash
python -c "import crewai; print(crewai.__version__)"
python -c "from pydantic import BaseModel; print('Pydantic OK')"
```
- [ ] CrewAI version prints (should be 0.86.0+)
- [ ] Pydantic import succeeds

### Step 0.5: Configure Environment Variables
**Create .env:**
```bash
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-4o-mini
CREWAI_VERBOSE=true
```

**Create .env.example (for GitHub):**
```bash
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
CREWAI_VERBOSE=true
```

**Validation:**
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('Key loaded:', bool(os.getenv('OPENAI_API_KEY')))"
```
- [ ] Output shows `Key loaded: True`

---

## Phase 1: Data Models
**Objective:** Define Pydantic models for pharmaceutical products

### Step 1.1: Create Enums for Product Attributes
**File:** `src/pharma_logistics/models.py`

```python
from enum import Enum

class TemperatureRequirement(str, Enum):
    """Temperature requirements for pharmaceutical storage/transport."""
    AMBIENT = "ambient"           # 15-25°C
    REFRIGERATED = "refrigerated" # 2-8°C
    FROZEN = "frozen"             # -20°C or below
    CONTROLLED = "controlled"     # Specific range required

class PriorityTier(str, Enum):
    """Delivery priority classifications."""
    CRITICAL = "critical"         # Life-saving, <4hr delivery
    URGENT = "urgent"             # Same-day required
    STANDARD = "standard"         # Next-day acceptable
    ROUTINE = "routine"           # 2-3 day window
```

**Validation:**
```python
from models import TemperatureRequirement, PriorityTier
assert TemperatureRequirement.REFRIGERATED.value == "refrigerated"
assert PriorityTier.CRITICAL.value == "critical"
```
- [ ] Both enums import correctly
- [ ] Values are lowercase strings

### Step 1.2: Create PharmaceuticalProduct Pydantic Model
**Append to:** `src/pharma_logistics/models.py`

```python
from pydantic import BaseModel, Field
from typing import Optional

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
```

**Validation:**
```python
product = PharmaceuticalProduct(
    product_id="INS-001",
    name="Insulin Glargine",
    temperature_requirement=TemperatureRequirement.REFRIGERATED,
    priority_tier=PriorityTier.CRITICAL,
    destination_city="Chicago",
    destination_facility="Northwestern Memorial Hospital"
)
print(product.model_dump())
```
- [ ] Model instantiates without errors
- [ ] `model_dump()` returns valid dictionary

### Step 1.3: Add Prompt Formatting Method
**Append method to PharmaceuticalProduct class:**

```python
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

**Validation:**
```python
print(product.format_for_prompt())
# Should output formatted multi-line string
```
- [ ] Output is human-readable
- [ ] Contains all key product attributes

### Step 1.4: Create Product List Formatter Function
**Append to:** `src/pharma_logistics/models.py`

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

**Validation:**
```python
products = [product]  # From step 1.2
formatted = format_products_for_crew(products)
print(formatted)
assert "CRITICAL PRIORITY" in formatted
assert "Cold Chain Required: 1" in formatted
```
- [ ] Output groups products by priority
- [ ] Summary statistics are accurate

### Step 1.5: Create Sample Data File
**File:** `src/pharma_logistics/data/sample_products.json`

```json
[
  {
    "product_id": "INS-001",
    "name": "Insulin Glargine 100U/mL",
    "temperature_requirement": "refrigerated",
    "priority_tier": "critical",
    "destination_city": "Chicago",
    "destination_facility": "Northwestern Memorial Hospital",
    "quantity_units": 50
  },
  {
    "product_id": "VAC-002",
    "name": "Moderna COVID-19 Vaccine",
    "temperature_requirement": "frozen",
    "priority_tier": "urgent",
    "destination_city": "Chicago",
    "destination_facility": "Rush University Medical Center",
    "quantity_units": 200
  },
  {
    "product_id": "CHM-003",
    "name": "Doxorubicin 50mg Injection",
    "temperature_requirement": "refrigerated",
    "priority_tier": "urgent",
    "destination_city": "Evanston",
    "destination_facility": "NorthShore University HealthSystem",
    "quantity_units": 25
  },
  {
    "product_id": "ANT-004",
    "name": "Amoxicillin 500mg Capsules",
    "temperature_requirement": "ambient",
    "priority_tier": "standard",
    "destination_city": "Oak Park",
    "destination_facility": "Oak Park Family Medicine",
    "quantity_units": 500
  },
  {
    "product_id": "VIT-005",
    "name": "Vitamin D3 5000IU Tablets",
    "temperature_requirement": "ambient",
    "priority_tier": "routine",
    "destination_city": "Naperville",
    "destination_facility": "Walgreens Pharmacy #1247",
    "quantity_units": 1000
  },
  {
    "product_id": "EPI-006",
    "name": "Epinephrine Auto-Injector",
    "temperature_requirement": "controlled",
    "priority_tier": "critical",
    "destination_city": "Schaumburg",
    "destination_facility": "AMITA Health Alexian Brothers",
    "quantity_units": 30
  }
]
```

**Validation:**
```python
import json
with open('src/pharma_logistics/data/sample_products.json') as f:
    data = json.load(f)
assert len(data) == 6
products = [PharmaceuticalProduct(**p) for p in data]
print(f"Loaded {len(products)} products")
```
- [ ] JSON parses without errors
- [ ] All 6 products validate as Pydantic models

---

## Phase 2: Agent Configuration
**Objective:** Define agents in YAML with pharma-specific personas

### Step 2.1: Create Logistics Analyst Agent Definition
**File:** `src/pharma_logistics/config/agents.yaml`

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

**Validation:**
```python
import yaml
with open('src/pharma_logistics/config/agents.yaml') as f:
    config = yaml.safe_load(f)
assert 'logistics_analyst' in config
assert 'cold chain' in config['logistics_analyst']['backstory'].lower()
```
- [ ] YAML parses correctly
- [ ] Agent has role, goal, backstory

### Step 2.2: Add Optimization Strategist Agent Definition
**Append to:** `src/pharma_logistics/config/agents.yaml`

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

**Validation:**
```python
import yaml
with open('src/pharma_logistics/config/agents.yaml') as f:
    config = yaml.safe_load(f)
assert len(config) == 2
assert 'optimization_strategist' in config
```
- [ ] Both agents defined
- [ ] Strategist has distinct goal focused on solutions

---

## Phase 3: Task Configuration
**Objective:** Define parameterized tasks in YAML

### Step 3.1: Create Logistics Analysis Task
**File:** `src/pharma_logistics/config/tasks.yaml`

```yaml
logistics_analysis_task:
  description: >
    Analyze the current delivery route operations for the following pharmaceutical
    products:
    
    {products}
    
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

**Validation:**
```python
import yaml
with open('src/pharma_logistics/config/tasks.yaml') as f:
    config = yaml.safe_load(f)
assert 'logistics_analysis_task' in config
assert '{products}' in config['logistics_analysis_task']['description']
```
- [ ] Task parses correctly
- [ ] Contains `{products}` placeholder for parameterization

### Step 3.2: Add Optimization Strategy Task
**Append to:** `src/pharma_logistics/config/tasks.yaml`

```yaml
optimization_strategy_task:
  description: >
    Based on the logistics analysis provided, develop a comprehensive
    optimization strategy for pharmaceutical delivery routes handling
    the following products:
    
    {products}
    
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
       - Phased rollout plan (Quick wins -> Medium-term -> Long-term)
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
  output_file: outputs/optimization_strategy.md
```

**Validation:**
```python
import yaml
with open('src/pharma_logistics/config/tasks.yaml') as f:
    config = yaml.safe_load(f)
assert len(config) == 2
assert config['optimization_strategy_task']['agent'] == 'optimization_strategist'
assert 'output_file' in config['optimization_strategy_task']
```
- [ ] Both tasks defined
- [ ] Strategy task has output_file specified
- [ ] Agent assignments match agent names

---

## Phase 4: Crew Implementation
**Objective:** Create the Crew class with proper orchestration

### Step 4.1: Create Basic Crew Structure
**File:** `src/pharma_logistics/crew.py`

```python
"""
Pharmaceutical Logistics Optimization Crew

This module defines the Crew AI system for analyzing pharmaceutical
delivery routes and developing optimization strategies.
"""

import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@CrewBase
class PharmaLogisticsCrew:
    """
    Pharmaceutical Logistics Optimization Crew
    
    A two-agent system that:
    1. Analyzes current delivery route operations (Logistics Analyst)
    2. Develops optimization strategies (Optimization Strategist)
    """
    
    # Path to configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
```

**Validation:**
```python
from crew import PharmaLogisticsCrew
# Should import without errors
```
- [ ] Module imports successfully
- [ ] @CrewBase decorator applied

### Step 4.2: Define Agent Methods
**Append to PharmaLogisticsCrew class:**

```python
    @agent
    def logistics_analyst(self) -> Agent:
        """
        Creates the Logistics Analyst agent.
        
        This agent analyzes current delivery operations and identifies
        inefficiencies, with expertise in cold chain compliance.
        """
        return Agent(
            config=self.agents_config['logistics_analyst'],
            verbose=True
        )
    
    @agent
    def optimization_strategist(self) -> Agent:
        """
        Creates the Optimization Strategist agent.
        
        This agent develops actionable optimization strategies based
        on the analyst's findings.
        """
        return Agent(
            config=self.agents_config['optimization_strategist'],
            verbose=True
        )
```

**Validation:**
```python
crew_instance = PharmaLogisticsCrew()
analyst = crew_instance.logistics_analyst()
strategist = crew_instance.optimization_strategist()
print(f"Analyst role: {analyst.role}")
print(f"Strategist role: {strategist.role}")
```
- [ ] Both agents instantiate
- [ ] Roles match YAML configuration

### Step 4.3: Define Task Methods
**Append to PharmaLogisticsCrew class:**

```python
    @task
    def logistics_analysis_task(self) -> Task:
        """
        Creates the logistics analysis task.
        
        Analyzes current delivery operations for the provided products.
        """
        return Task(
            config=self.tasks_config['logistics_analysis_task']
        )
    
    @task
    def optimization_strategy_task(self) -> Task:
        """
        Creates the optimization strategy task.
        
        Develops optimization strategy based on analysis findings.
        Depends on logistics_analysis_task output.
        """
        return Task(
            config=self.tasks_config['optimization_strategy_task'],
            context=[self.logistics_analysis_task()]
        )
```

**Validation:**
```python
task1 = crew_instance.logistics_analysis_task()
task2 = crew_instance.optimization_strategy_task()
print(f"Task 1 agent: {task1.agent}")
print(f"Task 2 has context: {bool(task2.context)}")
```
- [ ] Tasks instantiate correctly
- [ ] Task 2 has context dependency on Task 1

### Step 4.4: Define Crew Method
**Append to PharmaLogisticsCrew class:**

```python
    @crew
    def crew(self) -> Crew:
        """
        Creates and returns the Pharmaceutical Logistics Crew.
        
        Uses sequential process - strategy task depends on analysis task.
        """
        return Crew(
            agents=self.agents,  # Automatically populated by @agent decorators
            tasks=self.tasks,    # Automatically populated by @task decorators
            process=Process.sequential,
            verbose=True
        )
```

**Validation:**
```python
the_crew = crew_instance.crew()
print(f"Crew agents: {len(the_crew.agents)}")
print(f"Crew tasks: {len(the_crew.tasks)}")
print(f"Process: {the_crew.process}")
```
- [ ] Crew has 2 agents
- [ ] Crew has 2 tasks
- [ ] Process is sequential

---

## Phase 5: Main Entry Point
**Objective:** Create runnable entry point with sample data

### Step 5.1: Create Main Module
**File:** `src/pharma_logistics/main.py`

```python
"""
Main entry point for Pharmaceutical Logistics Optimization Crew

Usage:
    python -m pharma_logistics.main
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pharma_logistics.crew import PharmaLogisticsCrew
from pharma_logistics.models import (
    PharmaceuticalProduct,
    format_products_for_crew
)


def load_sample_products() -> list[PharmaceuticalProduct]:
    """Load sample products from JSON file."""
    data_path = Path(__file__).parent / 'data' / 'sample_products.json'
    
    with open(data_path) as f:
        data = json.load(f)
    
    return [PharmaceuticalProduct(**item) for item in data]


def run():
    """
    Main execution function.
    
    Loads sample products, formats them for the crew,
    and executes the optimization analysis.
    """
    print("=" * 60)
    print("Pharmaceutical Logistics Optimization Crew")
    print("=" * 60)
    
    # Load and format products
    products = load_sample_products()
    print(f"\nLoaded {len(products)} pharmaceutical products")
    
    formatted_products = format_products_for_crew(products)
    print("\nProduct Summary:")
    print("-" * 40)
    # Print just the summary stats
    summary_start = formatted_products.find("SUMMARY STATISTICS:")
    if summary_start != -1:
        print(formatted_products[summary_start:])
    
    # Prepare inputs
    inputs = {
        "products": formatted_products
    }
    
    # Create and run crew
    print("\n" + "=" * 60)
    print("Starting Crew Execution...")
    print("=" * 60 + "\n")
    
    crew_instance = PharmaLogisticsCrew()
    result = crew_instance.crew().kickoff(inputs=inputs)
    
    # Display results
    print("\n" + "=" * 60)
    print("Execution Complete!")
    print("=" * 60)
    print(f"\nToken Usage: {result.token_usage}")
    print(f"\nOutput file: outputs/optimization_strategy.md")
    
    return result


if __name__ == "__main__":
    run()
```

**Validation:**
- [ ] File created
- [ ] Imports resolve correctly
- [ ] Run will be tested in Phase 6

### Step 5.2: Create outputs Directory
```bash
mkdir -p outputs
touch outputs/.gitkeep
```

**Validation:**
- [ ] `outputs/` directory exists
- [ ] Will contain generated reports

---

## Phase 6: Integration Testing
**Objective:** Verify end-to-end execution

### Step 6.1: Test Model Imports
```bash
cd src
python -c "
from pharma_logistics.models import (
    PharmaceuticalProduct,
    TemperatureRequirement,
    PriorityTier,
    format_products_for_crew
)
print('All model imports successful')
"
```

**Validation:**
- [ ] No import errors
- [ ] All classes/functions available

### Step 6.2: Test Crew Instantiation
```bash
cd src
python -c "
from pharma_logistics.crew import PharmaLogisticsCrew
crew = PharmaLogisticsCrew()
print(f'Crew class instantiated')
print(f'Agents config: {crew.agents_config}')
print(f'Tasks config: {crew.tasks_config}')
"
```

**Validation:**
- [ ] Crew instantiates without API calls
- [ ] Config paths are correct

### Step 6.3: Dry Run Test (Minimal Product Set)
**Create test script:** `tests/test_dry_run.py`

```python
"""
Dry run test with minimal product set.
Tests full execution with real API calls.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
load_dotenv()

from pharma_logistics.crew import PharmaLogisticsCrew
from pharma_logistics.models import (
    PharmaceuticalProduct,
    TemperatureRequirement,
    PriorityTier,
    format_products_for_crew
)


def test_dry_run():
    """Execute crew with minimal test data."""
    
    # Minimal product set for faster execution
    products = [
        PharmaceuticalProduct(
            product_id="TEST-001",
            name="Test Insulin",
            temperature_requirement=TemperatureRequirement.REFRIGERATED,
            priority_tier=PriorityTier.CRITICAL,
            destination_city="Chicago",
            destination_facility="Test Hospital"
        ),
        PharmaceuticalProduct(
            product_id="TEST-002",
            name="Test Antibiotic",
            temperature_requirement=TemperatureRequirement.AMBIENT,
            priority_tier=PriorityTier.STANDARD,
            destination_city="Evanston",
            destination_facility="Test Pharmacy"
        )
    ]
    
    formatted = format_products_for_crew(products)
    inputs = {"products": formatted}
    
    print("Starting dry run with 2 test products...")
    
    crew = PharmaLogisticsCrew()
    result = crew.crew().kickoff(inputs=inputs)
    
    # Assertions
    assert result.raw is not None, "No output generated"
    assert len(result.tasks_output) == 2, f"Expected 2 task outputs, got {len(result.tasks_output)}"
    
    print("\n✓ Dry run successful!")
    print(f"Token usage: {result.token_usage}")
    
    return result


if __name__ == "__main__":
    test_dry_run()
```

**Run test:**
```bash
python tests/test_dry_run.py
```

**Validation:**
- [ ] Execution completes without errors
- [ ] Both tasks produce output
- [ ] `outputs/optimization_strategy.md` is created

### Step 6.4: Full Execution with Sample Data
```bash
cd src
python -m pharma_logistics.main
```

**Validation:**
- [ ] All 6 sample products processed
- [ ] Final report written to `outputs/optimization_strategy.md`
- [ ] Report contains pharma-specific recommendations

---

## Phase 7: Notebook Conversion
**Objective:** Create submission-ready Jupyter notebook

### Step 7.1: Create Notebook Structure
**File:** `notebooks/pharma_logistics_crew.ipynb`

Create notebook with the following cell structure:

**Cell 1: Title and Overview (Markdown)**
```markdown
# Pharmaceutical Logistics Optimization with Crew AI

## Assignment: Logistics Optimization Analysis

This notebook implements a Crew AI system for analyzing pharmaceutical delivery 
routes and developing optimization strategies.

**Agents:**
1. **Logistics Analyst** - Analyzes current delivery operations
2. **Optimization Strategist** - Develops optimization strategies

**Domain:** Pharmaceutical/Medical Supply Delivery Route Optimization
```

**Cell 2: Environment Setup (Code)**
```python
# Install dependencies
!pip install crewai crewai-tools pydantic python-dotenv pyyaml -q

# Configure API key
import os
try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("✓ API key loaded from Colab secrets")
except:
    # For local execution, ensure .env is configured
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ API key loaded from .env")
```

**Cell 3: Data Models (Code)**
```python
# [Paste entire models.py content here]
```

**Cell 4: Agent Configuration (Code)**
```python
# Agent configurations as Python dictionary
agents_config = {
    "logistics_analyst": {
        "role": "Pharmaceutical Logistics Analyst",
        "goal": """Analyze current delivery route operations for pharmaceutical products,
identifying inefficiencies, bottlenecks, and optimization opportunities
with special attention to cold chain requirements and priority classifications.""",
        "backstory": """You are a seasoned logistics analyst with 15 years of experience in
pharmaceutical supply chain operations. You've worked with major distributors
like McKesson and Cardinal Health, specializing in temperature-sensitive
medication delivery. Your expertise includes route efficiency analysis,
cold chain compliance auditing, and delivery window optimization. You're
known for your meticulous data analysis and ability to identify hidden
inefficiencies that cost companies millions annually."""
    },
    "optimization_strategist": {
        "role": "Delivery Route Optimization Strategist",
        "goal": """Develop comprehensive, actionable optimization strategies for pharmaceutical
delivery routes that maximize efficiency while maintaining cold chain integrity
and meeting priority-based delivery windows.""",
        "backstory": """You are a strategic logistics consultant who has helped Fortune 500
pharmaceutical companies reduce delivery costs by 20-35% while improving
on-time delivery rates. Your approach combines operations research principles
with practical implementation experience. You specialize in translating
analytical findings into executable strategies with clear ROI projections.
Healthcare executives trust your recommendations because you balance
efficiency gains with patient safety requirements."""
    }
}
```

**Cell 5: Task Configuration (Code)**
```python
# Task configurations as Python dictionary
# [Include full task descriptions with {products} placeholder]
```

**Cell 6: Crew Definition (Code)**
```python
from crewai import Agent, Crew, Process, Task

class PharmaLogisticsCrew:
    """Pharmaceutical Logistics Optimization Crew"""
    
    def __init__(self):
        self.agents_config = agents_config
        self.tasks_config = tasks_config
    
    def logistics_analyst(self) -> Agent:
        return Agent(
            role=self.agents_config['logistics_analyst']['role'],
            goal=self.agents_config['logistics_analyst']['goal'],
            backstory=self.agents_config['logistics_analyst']['backstory'],
            verbose=True
        )
    
    def optimization_strategist(self) -> Agent:
        return Agent(
            role=self.agents_config['optimization_strategist']['role'],
            goal=self.agents_config['optimization_strategist']['goal'],
            backstory=self.agents_config['optimization_strategist']['backstory'],
            verbose=True
        )
    
    def create_crew(self, products_input: str) -> Crew:
        # Create agents
        analyst = self.logistics_analyst()
        strategist = self.optimization_strategist()
        
        # Create tasks with product input
        analysis_task = Task(
            description=self.tasks_config['logistics_analysis_task']['description'].format(
                products=products_input
            ),
            expected_output=self.tasks_config['logistics_analysis_task']['expected_output'],
            agent=analyst
        )
        
        strategy_task = Task(
            description=self.tasks_config['optimization_strategy_task']['description'].format(
                products=products_input
            ),
            expected_output=self.tasks_config['optimization_strategy_task']['expected_output'],
            agent=strategist,
            context=[analysis_task]
        )
        
        return Crew(
            agents=[analyst, strategist],
            tasks=[analysis_task, strategy_task],
            process=Process.sequential,
            verbose=True
        )
```

**Cell 7: Sample Data (Code)**
```python
# Create sample pharmaceutical products
sample_products = [
    PharmaceuticalProduct(
        product_id="INS-001",
        name="Insulin Glargine 100U/mL",
        temperature_requirement=TemperatureRequirement.REFRIGERATED,
        priority_tier=PriorityTier.CRITICAL,
        destination_city="Chicago",
        destination_facility="Northwestern Memorial Hospital",
        quantity_units=50
    ),
    # [Add remaining products...]
]

# Format for crew input
formatted_products = format_products_for_crew(sample_products)
print("Product Summary:")
print(formatted_products)
```

**Cell 8: Execute Crew (Code)**
```python
# Initialize and run the crew
crew_instance = PharmaLogisticsCrew()
crew = crew_instance.create_crew(formatted_products)

print("Starting Crew Execution...")
print("=" * 60)

result = crew.kickoff()

print("\n" + "=" * 60)
print("Execution Complete!")
```

**Cell 9: Display Results (Code)**
```python
# Display final output
print("=" * 60)
print("OPTIMIZATION STRATEGY REPORT")
print("=" * 60)
print(result.raw)

# Token usage
print("\n" + "-" * 40)
print(f"Token Usage: {result.token_usage}")
```

**Cell 10: Explanation (Markdown)**
```markdown
## How the System Works

### Agent Roles

1. **Logistics Analyst Agent**
   - Analyzes current delivery operations
   - Identifies inefficiencies and bottlenecks
   - Assesses cold chain compliance risks
   - Produces detailed findings report

2. **Optimization Strategist Agent**
   - Receives analyst's findings as context
   - Develops actionable optimization strategies
   - Creates implementation roadmap
   - Projects ROI and defines success metrics

### Task Flow

The crew uses a **sequential process**:
1. First, the Logistics Analyst completes their analysis
2. Their output becomes context for the Optimization Strategist
3. The Strategist builds upon the analysis to create the final strategy

### Parameterization

Tasks accept a `{products}` variable, allowing the same crew to optimize
routes for different product sets without code changes.
```

**Validation:**
- [ ] Notebook runs end-to-end in fresh Colab environment
- [ ] All cells execute without errors
- [ ] Output demonstrates both agents working

### Step 7.2: Test Notebook in Colab
1. Upload notebook to Google Colab
2. Add OPENAI_API_KEY to Colab secrets
3. Run all cells
4. Verify output generation

**Validation:**
- [ ] Dependencies install correctly
- [ ] API key loads from secrets
- [ ] Crew executes successfully
- [ ] Results display properly

---

## Phase 8: Final Validation and Cleanup
**Objective:** Ensure submission readiness

### Step 8.1: Code Cleanup
```bash
# Format code with black
pip install black
black src/ tests/

# Check for issues with ruff
pip install ruff
ruff check src/
```

**Validation:**
- [ ] No formatting issues
- [ ] No linting errors

### Step 8.2: Documentation Review
- [ ] README.md explains project purpose and setup
- [ ] Code has docstrings
- [ ] YAML files have clear structure

### Step 8.3: Git Commit
```bash
git add .
git commit -m "Complete Pharma Logistics Crew AI implementation"
```

**Validation:**
- [ ] All files committed
- [ ] .env excluded from commit

### Step 8.4: Final Notebook Export
1. Run notebook from fresh state
2. Download as .ipynb
3. Verify file size < 100 MB (assignment limit)

**Validation:**
- [ ] Notebook file ready for submission
- [ ] Size within limits
- [ ] All outputs visible in notebook

---

## Completion Checklist

### Assignment Requirements
- [ ] Two agents defined (Logistics Analyst, Optimization Strategist)
- [ ] Analyst researches current logistics state
- [ ] Strategist creates strategy from analyst insights
- [ ] Tasks parameterized for product list input
- [ ] Each agent has clear goal and backstory
- [ ] Crew assembled with both agents and tasks

### Technical Quality
- [ ] Pydantic models validate product data
- [ ] YAML configuration separates concerns
- [ ] Code is well-documented
- [ ] Error handling in place

### Deliverables
- [ ] Working Jupyter notebook
- [ ] Generates optimization strategy report
- [ ] Notebook runs in Colab environment

---

*End of Atomic Implementation Steps*
