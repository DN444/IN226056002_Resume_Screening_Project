# IN226056002_Resume_Screening_Project


# AI Resume Screening System

This project implements an AI-based resume screening system using LangChain, Groq, and LangSmith.

## Pipeline

Resume → Extract → Match → Score → Explain

## Features

- 3 resumes: strong, average, weak
- 1 job description: Data Scientist
- Skill extraction
- Matching logic
- Fit score from 0 to 100
- Explanation of the assigned score
- LangSmith tracing enabled
- Intentional incorrect debug output included

## Project structure

- `prompts/` → prompt templates
- `chains/` → schemas and pipeline logic
- `data/` → resumes and job description
- `main.py` → entry point
- `.env` → API keys and tracing config

## Setup

```bash
pip install -r requirements.txt
```

Then configure `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ai-resume-screening-demo
```

## Run

```bash
python main.py
```

## Expected LangSmith traces

You should see at least 3 pipeline runs:

- strong
- average
- weak

Each run should show these visible child steps:

- extract_resume_facts
- match_resume_to_jd
- score_candidate
- explain_score

You should also see one separate debug run:

- debug_incorrect_extraction

## Prompt rule

The extraction step explicitly enforces:

**Do NOT assume skills not present in the resume.**
