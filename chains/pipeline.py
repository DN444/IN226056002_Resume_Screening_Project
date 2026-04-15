from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langsmith import traceable
import json
import re

from chains.schemas import (
    ExtractionResult,
    MatchResult,
    ScoreResult,
    ExplanationResult,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"


def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def build_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


def build_extraction_chain(llm):
    prompt = PromptTemplate.from_template(load_prompt("extraction_prompt.txt"))
    return (prompt | llm.with_structured_output(ExtractionResult)).with_config({
        "run_name": "extract_resume_facts",
        "tags": ["extraction", "resume-screening"]
    })


def build_match_chain(llm):
    prompt = PromptTemplate.from_template(load_prompt("match_prompt.txt"))
    return (prompt | llm.with_structured_output(MatchResult)).with_config({
        "run_name": "match_resume_to_jd",
        "tags": ["matching", "resume-screening"]
    })


def build_score_chain(llm):
    prompt = PromptTemplate.from_template("""
You are a resume scoring assistant.
Score the candidate from 0 to 100 using these weights:
- skills match: 45 points maximum
- tools match: 20 points maximum  
- experience match: 25 points maximum
- domain alignment: 10 points maximum

CRITICAL RULES:
1. Return ONLY plain numbers in score_breakdown (NO math expressions)
2. Calculate scores first, then put final numbers in JSON
3. Valid example: "skills": 32, "tools": 15, "experience": 20

Score each category based on matching output:
- Perfect match = full points
- Partial match = proportional points  
- No match = 0 points

Job Description:
{job_description}

Extracted Resume Facts:
{extracted_json}

Matching Output:
{match_json}

Respond with ONLY valid JSON:
""")
    
    def safe_parse_score(output):
        try:
            json_match = re.search(r'\{.*\}', output.content, re.DOTALL)
            if json_match:
                result = ScoreResult.model_validate_json(json_match.group())
                return result
        except Exception:
            pass
        return ScoreResult(
            fit_score=50,
            score_breakdown={"skills": 25, "tools": 10, "experience": 10, "domain_alignment": 5},
            rationale=["Safe fallback scoring due to parsing error"]
        )
    
    chain = prompt | llm | RunnableLambda(safe_parse_score)
    return chain.with_config({
        "run_name": "score_candidate", 
        "tags": ["scoring", "resume-screening"]
    })


def build_explain_chain(llm):
    prompt = PromptTemplate.from_template(load_prompt("explain_prompt.txt"))
    return (prompt | llm.with_structured_output(ExplanationResult)).with_config({
        "run_name": "explain_score",
        "tags": ["explanation", "resume-screening"]
    })


def build_debug_chain(llm):
    debug_prompt = PromptTemplate.from_template("""
You are intentionally using a weak extraction prompt for debugging.
List likely skills for this candidate based on role/title and common industry patterns.

Resume:
{resume}

Return JSON with keys skills and tools.
""")
    return (debug_prompt | llm).with_config({
        "run_name": "debug_incorrect_extraction",
        "tags": ["debug", "incorrect-output"]
    })


@traceable(name="resume_screening_pipeline", tags=["resume-screening", "pipeline"])
def run_pipeline(resume: str, job_description: str, llm):
    extraction_chain = build_extraction_chain(llm)
    match_chain = build_match_chain(llm)
    score_chain = build_score_chain(llm)
    explain_chain = build_explain_chain(llm)

    # Extraction
    extracted = extraction_chain.invoke({"resume": resume})
    extracted_json = extracted.model_dump_json(indent=2)

    # Matching  
    matched = match_chain.invoke({
        "job_description": job_description,
        "extracted_json": extracted_json
    })
    match_json = matched.model_dump_json(indent=2)

    scored = score_chain.invoke({
        "job_description": job_description,
        "extracted_json": extracted_json,
        "match_json": match_json
    })
    score_json = scored.model_dump_json(indent=2)

    # Explanation
    explained = explain_chain.invoke({
        "job_description": job_description,
        "extracted_json": extracted_json,
        "match_json": match_json,
        "score_json": score_json
    })

    return {
        "extraction": extracted.model_dump(),
        "matching": matched.model_dump(),
        "scoring": scored.model_dump(),
        "explanation": explained.model_dump()
    }