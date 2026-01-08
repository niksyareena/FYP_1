# Automated and Explainable Data Cleaning Using Hybrid AI and LLM

**Final Year Project (FYP)**
**Academic Year**: 2024/2025

## Overview

This research project develops an intelligent data cleaning system that combines rule-based logic, machine learning, and large language models (LLMs) to automatically detect, correct, and explain data quality issues in tabular datasets.

## Research Objectives

- Evaluate the effectiveness of hybrid approaches (Rule + ML + LLM) for automated data cleaning
- Assess the quality of LLM-generated explanations for cleaning decisions
- Analyze the cost-effectiveness tradeoff of LLM integration in data pipelines

## Scope

**Data Quality Issues** (4 types):
- Missing values
- Duplicate rows
- Outliers
- Data type inconsistencies

**Methods**:
- Baseline A: Rule-based cleaning
- Baseline B: ML-based cleaning
- Proposed: Rule + ML + LLM explanations

**Datasets**: 4-5 benchmark datasets (UCI/Kaggle + synthetic)

## Timeline

### FYP1 (Weeks 1-13)
-Data profiling module
-Format correction module
-Duplicate Detection module

### FYP2 (Weeks 14-26)
-Missing value imputation module
-Outlier detection module
-LLM explanation module

## Setup

```bash
# Clone repository
git clone <repo-url>
cd FYP_1

# Install dependencies
pip install pandas numpy scikit-learn ydata-profiling

# (More dependencies to be added)
```

## Project Structure

```
FYP_1/
├── data/              # Datasets
├── src/               # Source code
├── experiments/       # Experiment results
├── notebooks/         # Jupyter notebooks
├── docs/              # Documentation & reports
└── README.md
```

## Current Status

**Phase**: Project Planning
**Last Updated**: 2025-10-31

## Deliverables

**FYP1**: Working prototype, preliminary results, interim report (22-28 pages)
**FYP2**: Complete system, comprehensive evaluation, final thesis (48-60 pages)

---

**Supervisor**: [Name]
**Student**: [Your Name]
