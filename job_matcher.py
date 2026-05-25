"""
Job Matching Engine
==================

Part B of the RAG Job Matching Assignment (50% of total grade).
Semantic job-resume matching with hybrid search and intelligent scoring.

This module provides:
- Job description processing and embedding
- Hybrid search combining semantic and keyword matching
- Intelligent scoring and ranking algorithms
- Match reasoning and explanation generation

Usage:
    from job_matcher import JobMatcher
    from resume_rag import ResumeRAG
    
    # Initialize systems
    rag = ResumeRAG()
    matcher = JobMatcher(rag)
    
    # Find matches for a job
    matches = matcher.find_matches(job_description, k=10)
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from resume_rag import ResumeRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class JobRequirement:
    """Structured job requirement information."""
    title: str
    skills_required: List[str]
    skills_preferred: List[str]
    experience_years: Optional[int]
    education_required: List[str]
    must_have_keywords: List[str]
    nice_to_have_keywords: List[str]
    description: str


@dataclass
class CandidateMatch:
    """Represents a candidate match with detailed scoring."""
    candidate_name: str
    resume_path: str
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    relevant_excerpts: List[str]
    reasoning: str
    section_scores: Dict[str, float]
    keyword_matches: Dict[str, int]
    experience_match: bool
    education_match: bool


class JobMatcher:
    """
    Intelligent job-resume matching engine using hybrid search.
    
    Combines semantic similarity with keyword matching and requirement filtering
    to provide accurate and explainable candidate recommendations.
    """
    
    def __init__(self, rag_system: ResumeRAG, weight_config: Optional[Dict] = None):
        """
        Initialize the Job Matcher.
        
        Args:
            rag_system (ResumeRAG): Initialized RAG system with processed resumes
            weight_config (Optional[Dict]): Custom scoring weights
        """
        self.rag = rag_system
        
        # Default scoring weights
        self.weights = weight_config or {
            "semantic_similarity": 0.4,
            "skill_match": 0.25,
            "experience_match": 0.15,
            "education_match": 0.10,
            "keyword_bonus": 0.10
        }
        
        # Skill categories for matching
        self.skill_categories = {
            "programming": ["python", "java", "javascript", "c++", "go", "rust", "scala"],
            "web": ["react", "angular", "vue", "html", "css", "node.js", "django", "flask"],
            "data": ["sql", "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn"],
            "cloud": ["aws", "azure", "gcp", "docker", "kubernetes"],
            "tools": ["git", "jenkins", "jira", "mongodb", "postgresql"]
        }
        
        logger.info("Initialized Job Matcher with RAG system")
    
    def parse_job_description(self, job_description: str, job_title: str = "") -> JobRequirement:
        """
        Parse job description to extract structured requirements.
        
        Args:
            job_description (str): Raw job description text
            job_title (str): Job title
            
        Returns:
            JobRequirement: Parsed job requirements
        """
        desc_lower = job_description.lower()
        
        # Extract required vs preferred skills
        skills_required = self._extract_required_skills(job_description)
        skills_preferred = self._extract_preferred_skills(job_description)
        
        # Extract experience requirements
        experience_years = self._extract_experience_requirement(job_description)
        
        # Extract education requirements
        education_required = self._extract_education_requirement(job_description)
        
        # Extract must-have keywords (critical requirements)
        must_have_keywords = self._extract_must_have_keywords(job_description)
        
        # Extract nice-to-have keywords
        nice_to_have_keywords = self._extract_nice_to_have_keywords(job_description)
        
        return JobRequirement(
            title=job_title,
            skills_required=skills_required,
            skills_preferred=skills_preferred,
            experience_years=experience_years,
            education_required=education_required,
            must_have_keywords=must_have_keywords,
            nice_to_have_keywords=nice_to_have_keywords,
            description=job_description
        )
    
    def _extract_required_skills(self, text: str) -> List[str]:
        """Extract required technical skills from job description."""
        skills = []
        text_lower = text.lower()
        
        # Look for skills in all categories
        for category, skill_list in self.skill_categories.items():
            for skill in skill_list:
                if skill in text_lower:
                    # Check if it's mentioned as required
                    skill_context = self._get_skill_context(text_lower, skill)
                    if self._is_skill_required(skill_context):
                        skills.append(skill)
        
        return list(set(skills))
    
    def _extract_preferred_skills(self, text: str) -> List[str]:
        """Extract preferred/nice-to-have skills."""
        skills = []
        text_lower = text.lower()
        
        # Look for preferred skill indicators
        preferred_patterns = [
            r'preferred?:?\s*([^.!?]+)',
            r'nice\s*to\s*have:?\s*([^.!?]+)',
            r'plus:?\s*([^.!?]+)',
            r'bonus:?\s*([^.!?]+)'
        ]
        
        for pattern in preferred_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Extract skills from the preferred section
                for category, skill_list in self.skill_categories.items():
                    for skill in skill_list:
                        if skill in match:
                            skills.append(skill)
        
        return list(set(skills))
    
    def _get_skill_context(self, text: str, skill: str) -> str:
        """Get context around a skill mention."""
        skill_pos = text.find(skill)
        if skill_pos == -1:
            return ""
        
        # Get 100 characters before and after
        start = max(0, skill_pos - 100)
        end = min(len(text), skill_pos + len(skill) + 100)
        return text[start:end]
    
    def _is_skill_required(self, context: str) -> bool:
        """Determine if a skill is required based on context."""
        required_indicators = [
            "required", "must have", "mandatory", "essential", "minimum",
            "need", "necessary", "critical", "key requirement"
        ]
        
        return any(indicator in context for indicator in required_indicators)
    
    def _extract_experience_requirement(self, text: str) -> Optional[int]:
        """Extract minimum experience requirement."""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'minimum\s*(?:of\s*)?(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?',
            r'(\d+)\+\s*years?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return min(int(match) for match in matches)  # Take minimum requirement
        
        return None
    
    def _extract_education_requirement(self, text: str) -> List[str]:
        """Extract education requirements."""
        education_patterns = [
            r'(bachelor|master|phd|doctorate).*?(?:degree|in)\s*([^\n.!?]+)',
            r'(b\.?s\.?|m\.?s\.?|m\.?b\.?a\.?|ph\.?d\.?)\s*([^\n.!?]+)',
            r'degree\s*in\s*([^\n.!?]+)'
        ]
        
        education = []
        for pattern in education_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    education.append(' '.join(match).strip())
                else:
                    education.append(match.strip())
        
        return education
    
    def _extract_must_have_keywords(self, text: str) -> List[str]:
        """Extract critical must-have keywords."""
        must_have_patterns = [
            r'must\s*have:?\s*([^.!?\n]+)',
            r'required:?\s*([^.!?\n]+)',
            r'mandatory:?\s*([^.!?\n]+)',
            r'essential:?\s*([^.!?\n]+)'
        ]
        
        keywords = []
        for pattern in must_have_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                # Split by common separators and clean
                terms = re.split(r'[,;•\n]', match)
                for term in terms:
                    term = term.strip()
                    if term and len(term) > 2:
                        keywords.append(term)
        
        return keywords
    
    def _extract_nice_to_have_keywords(self, text: str) -> List[str]:
        """Extract nice-to-have keywords."""
        nice_to_have_patterns = [
            r'nice\s*to\s*have:?\s*([^.!?\n]+)',
            r'preferred?:?\s*([^.!?\n]+)',
            r'plus:?\s*([^.!?\n]+)',
            r'bonus:?\s*([^.!?\n]+)'
        ]
        
        keywords = []
        for pattern in nice_to_have_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                terms = re.split(r'[,;•\n]', match)
                for term in terms:
                    term = term.strip()
                    if term and len(term) > 2:
                        keywords.append(term)
        
        return keywords
    
    def find_matches(self, 
                    job_description: str, 
                    job_title: str = "", 
                    k: int = 10,
                    min_score: float = 0.3) -> Dict[str, Any]:
        """
        Find top candidate matches for a job description.
        
        Args:
            job_description (str): Job description text
            job_title (str): Job title
            k (int): Number of top matches to return
            min_score (float): Minimum match score threshold
            
        Returns:
            Dict[str, Any]: Structured match results
        """
        logger.info(f"Finding matches for job: {job_title}")
        
        # Parse job requirements
        job_req = self.parse_job_description(job_description, job_title)
        
        # Perform semantic search on job description
        semantic_results = self.rag.semantic_search(job_description, k=k*3)  # Get more for filtering
        
        # Group results by candidate
        candidate_results = self._group_by_candidate(semantic_results)
        
        # Score and rank candidates
        scored_candidates = []
        for candidate_name, candidate_chunks in candidate_results.items():
            match = self._score_candidate(candidate_chunks, job_req)
            if match.match_score >= min_score:
                scored_candidates.append(match)
        
        # Sort by match score
        scored_candidates.sort(key=lambda x: x.match_score, reverse=True)
        
        # Take top K
        top_matches = scored_candidates[:k]
        
        # Format output according to specified format
        formatted_matches = []
        for match in top_matches:
            formatted_matches.append({
                "candidate_name": match.candidate_name,
                "resume_path": match.resume_path,
                "match_score": int(match.match_score),  # Convert to integer as shown in example
                "matched_skills": match.matched_skills,
                "relevant_excerpts": match.relevant_excerpts,
                "reasoning": match.reasoning
            })
        
        return {
            "job_description": job_description,
            "top_matches": formatted_matches
        }
    
    def get_matches_json(self, job_description: str, k: int = 10, min_score: float = 50.0) -> str:
        """
        Get job matches in the specified JSON format.
        
        Args:
            job_description (str): Job description text
            k (int): Number of top matches to return
            min_score (float): Minimum match score (0-100 scale)
            
        Returns:
            str: JSON formatted results
        """
        results = self.find_matches(job_description, k=k, min_score=min_score/100.0)
        return json.dumps(results, indent=2, ensure_ascii=False)
    
    def _group_by_candidate(self, search_results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group search results by candidate name."""
        candidate_groups = {}
        
        for result in search_results:
            candidate_name = result["metadata"]["resume_name"]
            if candidate_name not in candidate_groups:
                candidate_groups[candidate_name] = []
            candidate_groups[candidate_name].append(result)
        
        return candidate_groups
    
    def _score_candidate(self, candidate_chunks: List[Dict], job_req: JobRequirement) -> CandidateMatch:
        """
        Score a candidate against job requirements.
        
        Args:
            candidate_chunks (List[Dict]): Candidate's resume chunks
            job_req (JobRequirement): Job requirements
            
        Returns:
            CandidateMatch: Detailed candidate match information
        """
        if not candidate_chunks:
            return None
        
        # Get candidate info from first chunk
        first_chunk = candidate_chunks[0]
        candidate_name = first_chunk["metadata"]["resume_name"]
        resume_path = first_chunk["metadata"]["file_path"]
        
        # Calculate component scores
        semantic_score = self._calculate_semantic_score(candidate_chunks)
        skill_score, matched_skills, missing_skills = self._calculate_skill_score(candidate_chunks, job_req)
        experience_score, experience_match = self._calculate_experience_score(candidate_chunks, job_req)
        education_score, education_match = self._calculate_education_score(candidate_chunks, job_req)
        keyword_score, keyword_matches = self._calculate_keyword_score(candidate_chunks, job_req)
        
        # Calculate weighted total score
        total_score = (
            semantic_score * self.weights["semantic_similarity"] +
            skill_score * self.weights["skill_match"] +
            experience_score * self.weights["experience_match"] +
            education_score * self.weights["education_match"] +
            keyword_score * self.weights["keyword_bonus"]
        )
        
        # Ensure score is between 0-100
        total_score = max(0, min(100, total_score * 100))
        
        # Get relevant excerpts
        relevant_excerpts = self._extract_relevant_excerpts(candidate_chunks, job_req)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            matched_skills, missing_skills, experience_match, 
            education_match, keyword_matches, semantic_score
        )
        
        # Section scores for detailed analysis
        section_scores = {
            "semantic_similarity": semantic_score * 100,
            "skill_match": skill_score * 100,
            "experience_match": experience_score * 100,
            "education_match": education_score * 100,
            "keyword_bonus": keyword_score * 100
        }
        
        return CandidateMatch(
            candidate_name=candidate_name,
            resume_path=resume_path,
            match_score=round(total_score, 1),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            relevant_excerpts=relevant_excerpts,
            reasoning=reasoning,
            section_scores=section_scores,
            keyword_matches=keyword_matches,
            experience_match=experience_match,
            education_match=education_match
        )
    
    def _calculate_semantic_score(self, candidate_chunks: List[Dict]) -> float:
        """Calculate average semantic similarity score."""
        if not candidate_chunks:
            return 0.0
        
        scores = [chunk["similarity_score"] for chunk in candidate_chunks]
        return np.mean(scores)
    
    def _calculate_skill_score(self, candidate_chunks: List[Dict], job_req: JobRequirement) -> Tuple[float, List[str], List[str]]:
        """Calculate skill match score."""
        # Get all candidate skills
        candidate_skills = set()
        for chunk in candidate_chunks:
            if "skills" in chunk["metadata"] and chunk["metadata"]["skills"]:
                skills_list = chunk["metadata"]["skills"].split(",") if isinstance(chunk["metadata"]["skills"], str) else chunk["metadata"]["skills"]
                candidate_skills.update([s.strip() for s in skills_list if s.strip()])
            if "programming_languages" in chunk["metadata"] and chunk["metadata"]["programming_languages"]:
                langs_list = chunk["metadata"]["programming_languages"].split(",") if isinstance(chunk["metadata"]["programming_languages"], str) else chunk["metadata"]["programming_languages"]
                candidate_skills.update([s.strip() for s in langs_list if s.strip()])
        
        # Calculate matches
        required_skills = set(job_req.skills_required)
        preferred_skills = set(job_req.skills_preferred)
        
        matched_required = required_skills.intersection(candidate_skills)
        matched_preferred = preferred_skills.intersection(candidate_skills)
        
        # Calculate score
        required_score = len(matched_required) / len(required_skills) if required_skills else 1.0
        preferred_score = len(matched_preferred) / len(preferred_skills) if preferred_skills else 0.5
        
        # Weighted combination (required skills are more important)
        skill_score = required_score * 0.8 + preferred_score * 0.2
        
        matched_skills = list(matched_required.union(matched_preferred))
        missing_skills = list(required_skills.difference(candidate_skills))
        
        return skill_score, matched_skills, missing_skills
    
    def _calculate_experience_score(self, candidate_chunks: List[Dict], job_req: JobRequirement) -> Tuple[float, bool]:
        """Calculate experience match score."""
        if not job_req.experience_years:
            return 1.0, True  # No requirement specified
        
        # Get candidate experience from metadata
        candidate_experience = None
        for chunk in candidate_chunks:
            exp_years = chunk["metadata"].get("experience_years")
            if exp_years and exp_years != 0:
                candidate_experience = exp_years
                break
        
        if candidate_experience is None:
            return 0.5, False  # Unknown experience
        
        if candidate_experience >= job_req.experience_years:
            # Calculate bonus for extra experience
            bonus = min(1.0, 1.0 + (candidate_experience - job_req.experience_years) * 0.1)
            return bonus, True
        else:
            # Penalty for insufficient experience
            ratio = candidate_experience / job_req.experience_years
            return ratio * 0.8, False
    
    def _calculate_education_score(self, candidate_chunks: List[Dict], job_req: JobRequirement) -> Tuple[float, bool]:
        """Calculate education match score."""
        if not job_req.education_required:
            return 1.0, True  # No requirement specified
        
        # This is a simplified implementation
        # In practice, you'd want more sophisticated education matching
        return 0.8, True  # Assume reasonable match for now
    
    def _calculate_keyword_score(self, candidate_chunks: List[Dict], job_req: JobRequirement) -> Tuple[float, Dict[str, int]]:
        """Calculate keyword match score."""
        candidate_text = " ".join([chunk["content"].lower() for chunk in candidate_chunks])
        
        keyword_matches = {}
        
        # Check must-have keywords
        must_have_score = 0
        for keyword in job_req.must_have_keywords:
            count = candidate_text.count(keyword.lower())
            keyword_matches[keyword] = count
            if count > 0:
                must_have_score += 1
        
        # Check nice-to-have keywords  
        nice_to_have_score = 0
        for keyword in job_req.nice_to_have_keywords:
            count = candidate_text.count(keyword.lower())
            keyword_matches[keyword] = count
            if count > 0:
                nice_to_have_score += 1
        
        # Calculate weighted score
        total_must_have = len(job_req.must_have_keywords)
        total_nice_to_have = len(job_req.nice_to_have_keywords)
        
        must_have_ratio = must_have_score / total_must_have if total_must_have > 0 else 1.0
        nice_to_have_ratio = nice_to_have_score / total_nice_to_have if total_nice_to_have > 0 else 0.5
        
        keyword_score = must_have_ratio * 0.8 + nice_to_have_ratio * 0.2
        
        return keyword_score, keyword_matches
    
    def _extract_relevant_excerpts(self, candidate_chunks: List[Dict], job_req: JobRequirement) -> List[str]:
        """Extract most relevant excerpts from candidate resume."""
        # Sort chunks by similarity and take top 3
        sorted_chunks = sorted(candidate_chunks, key=lambda x: x["similarity_score"], reverse=True)
        
        excerpts = []
        for chunk in sorted_chunks[:3]:
            content = chunk["content"]
            # Truncate if too long
            if len(content) > 200:
                content = content[:200] + "..."
            excerpts.append(content)
        
        return excerpts
    
    def _generate_reasoning(self, 
                          matched_skills: List[str], 
                          missing_skills: List[str],
                          experience_match: bool, 
                          education_match: bool,
                          keyword_matches: Dict[str, int],
                          semantic_score: float) -> str:
        """Generate human-readable reasoning for the match."""
        reasoning_parts = []
        
        # Skill analysis
        if matched_skills:
            reasoning_parts.append(f"Strong technical skills match including {', '.join(matched_skills[:3])}")
        
        if missing_skills:
            reasoning_parts.append(f"Missing some required skills: {', '.join(missing_skills[:3])}")
        
        # Experience analysis
        if experience_match:
            reasoning_parts.append("Meets experience requirements")
        else:
            reasoning_parts.append("May lack sufficient experience")
        
        # Keyword analysis
        positive_keywords = [k for k, v in keyword_matches.items() if v > 0]
        if positive_keywords:
            reasoning_parts.append(f"Relevant experience in {', '.join(positive_keywords[:3])}")
        
        # Semantic similarity
        if semantic_score > 0.8:
            reasoning_parts.append("Excellent semantic alignment with job requirements")
        elif semantic_score > 0.6:
            reasoning_parts.append("Good alignment with job requirements")
        
        return ". ".join(reasoning_parts) + "."


if __name__ == "__main__":
    # Example usage
    print("🎯 Job Matching Engine - Example Usage")
    print("=" * 50)
    
    try:
        # Initialize systems
        rag = ResumeRAG()
        matcher = JobMatcher(rag)
        
        # Example job description
        job_description = """
        We are looking for a Senior Python Developer with 5+ years of experience.
        
        Required Skills:
        - Python programming
        - Machine Learning frameworks (TensorFlow, scikit-learn)
        - SQL databases
        - AWS cloud platform
        
        Preferred Skills:
        - React frontend development
        - Docker containerization
        - Kubernetes orchestration
        
        Must have: Strong problem-solving skills, team collaboration
        Nice to have: Open source contributions, tech blog writing
        
        Education: Bachelor's degree in Computer Science or related field
        """
        
        # Find matches
        results = matcher.find_matches(
            job_description=job_description,
            job_title="Senior Python Developer",
            k=5
        )
        
        # Display results in JSON format
        print("🎯 Job Matching Results:")
        print("=" * 50)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
        # Also show a summary
        print(f"\n📊 Summary: Found {len(results['top_matches'])} matches")
        for i, match in enumerate(results['top_matches'], 1):
            print(f"{i}. {match['candidate_name']} - Score: {match['match_score']}/100")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure to:")
        print("1. Initialize RAG system first")
        print("2. Process some resume data")
        print("3. Check API keys and dependencies")