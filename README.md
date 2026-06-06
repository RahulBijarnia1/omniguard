# omniguard

LLM red-teaming and safety evaluation framework for generating adversarial prompts, running assessments, and reporting results.

## Overview

Omniguard helps generate adversarial attack vectors, apply transforms, score results, and export campaign reports for analysis.

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run a sample generator:

```powershell
python run.py
```

## Project Layout

- `omniguard/core`: shared data models
- `omniguard/generators`: adversarial prompt generation
- `omniguard/transforms`: prompt transformations and obfuscation helpers
- `omniguard/judge`: scoring and reporting logic
- `omniguard/reporting`: HTML report generation
- `results/`: generated campaign artifacts and reports

## Notes

- Generated reports and campaign exports are written to `results/`.
- The repository is intended for evaluation and safety testing workflows.
