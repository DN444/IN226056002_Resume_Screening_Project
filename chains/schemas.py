from typing import List, Dict
from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    candidate_name: str = Field(description="Candidate full name")
    skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    experience_years: float = Field(description="Years of experience explicitly supported by resume")
    experience_summary: str = Field(description="Brief factual summary of experience")
    evidence: List[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    matched_tools: List[str] = Field(default_factory=list)
    missing_tools: List[str] = Field(default_factory=list)
    experience_match: str
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)


class ScoreResult(BaseModel):
    fit_score: int
    score_breakdown: Dict[str, int]
    rationale: List[str] = Field(default_factory=list)


class ExplanationResult(BaseModel):
    summary: str
    why_this_score: List[str] = Field(default_factory=list)
    hiring_recommendation: str