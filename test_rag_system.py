#!/usr/bin/env python3
"""
RAG System Comprehensive Test Script
===================================

Tests the complete RAG system for job-resume matching including:
- Document processing and chunking
- Vector embedding and storage
- Semantic search and hybrid matching
- Performance evaluation

Author: Assignment Submission
Course: RAG Systems and Semantic Search
Date: 2024
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List

# Import RAG modules
from resume_rag import ResumeRAG
from job_matcher import JobMatcher
from fs_tools import list_files, read_file


def test_rag_system():
    """Comprehensive test of the RAG system."""
    print("🚀 RAG System Comprehensive Test")
    print("=" * 60)
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Initialize RAG System
        print("\n🔧 Step 1: Initializing RAG System...")
        start_time = time.time()
        
        rag = ResumeRAG(
            vector_db="chromadb",
            embedding_model="openai",  # Change to "sentence-transformers" if no OpenAI key
            collection_name="test_resume_collection"
        )
        
        init_time = time.time() - start_time
        print(f"✅ RAG system initialized in {init_time:.3f} seconds")
        
        # 2. Process Resume Documents
        print("\n📂 Step 2: Processing Resume Documents...")
        process_start = time.time()
        
        processed_resumes = rag.process_resume_directory("examples/resumes")
        
        process_time = time.time() - process_start
        print(f"✅ Processed {len(processed_resumes)} resumes in {process_time:.3f} seconds")
        print(f"⏱️  Average processing time: {process_time/len(processed_resumes):.3f}s per resume")
        
        # Display processed resume info
        print(f"\n📋 Processed Resumes ({len(processed_resumes)} total):")
        for resume in processed_resumes[:10]:  # Show first 10 (change to [:] to show all 30)
            print(f"  • {resume.name}")
            print(f"    Skills: {len(resume.skills)} | Languages: {len(resume.programming_languages)}")
            print(f"    Experience: {resume.experience_years} years | Education: {len(resume.education)}")
        
        # 3. Get Collection Statistics
        print("\n📊 Step 3: Vector Database Statistics...")
        stats = rag.get_collection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 4. Test Semantic Search
        print("\n🔍 Step 4: Testing Semantic Search...")
        search_queries = [
            "Python machine learning experience",
            "DevOps cloud infrastructure",
            "Product management UX design",
            "Cybersecurity network security",
            "Financial analysis data science"
        ]
        
        search_results = {}
        total_search_time = 0
        
        for query in search_queries:
            search_start = time.time()
            results = rag.semantic_search(query, k=5)
            search_time = time.time() - search_start
            total_search_time += search_time
            
            search_results[query] = {
                "results": results,
                "time": search_time,
                "matches": len(results)
            }
            
            print(f"  Query: '{query}'")
            print(f"    Found: {len(results)} matches in {search_time:.3f}s")
            if results:
                top_result = results[0]
                print(f"    Top match: {top_result['metadata']['resume_name']} (Score: {top_result['similarity_score']:.3f})")
        
        avg_search_time = total_search_time / len(search_queries)
        print(f"\n⏱️  Average search time: {avg_search_time:.3f} seconds")
        
        # 5. Initialize Job Matcher
        print("\n🎯 Step 5: Initializing Job Matcher...")
        matcher = JobMatcher(rag)
        print("✅ Job matcher initialized")
        
        # 6. Load Job Descriptions
        print("\n📋 Step 6: Loading Job Descriptions...")
        job_files = list_files("examples/job_descriptions", ".txt")
        job_descriptions = {}
        
        for job_file in job_files:
            result = read_file(job_file["filepath"])
            if result["success"]:
                job_name = job_file["name"].replace(".txt", "").replace("_", " ").title()
                job_descriptions[job_name] = result["content"]
        
        print(f"✅ Loaded {len(job_descriptions)} job descriptions:")
        for job_name in job_descriptions.keys():
            print(f"  • {job_name}")
        
        # 7. Test Job Matching
        print("\n🔄 Step 7: Testing Job-Resume Matching...")
        matching_results = {}
        total_matching_time = 0
        
        for job_name, job_desc in job_descriptions.items():
            print(f"\n  Testing: {job_name}")
            
            match_start = time.time()
            matches = matcher.find_matches(
                job_description=job_desc,
                job_title=job_name,
                k=5,
                min_score=30.0
            )
            match_time = time.time() - match_start
            total_matching_time += match_time
            
            matching_results[job_name] = {
                "matches": matches,
                "time": match_time,
                "candidates_found": len(matches["top_matches"])
            }
            
            print(f"    Candidates found: {len(matches['top_matches'])}")
            print(f"    Matching time: {match_time:.3f}s")
            
            # Show top match
            if matches["top_matches"]:
                top_match = matches["top_matches"][0]
                print(f"    Top candidate: {top_match['candidate_name']} ({top_match['match_score']:.1f}%)")
                print(f"    Reasoning: {top_match['reasoning'][:80]}...")
        
        avg_matching_time = total_matching_time / len(job_descriptions)
        print(f"\n⏱️  Average matching time: {avg_matching_time:.3f} seconds")
        
        # 8. Performance Analysis
        print("\n📈 Step 8: Performance Analysis...")
        
        # Calculate metrics
        total_matches_found = sum(result["candidates_found"] for result in matching_results.values())
        successful_jobs = sum(1 for result in matching_results.values() if result["candidates_found"] > 0)
        
        all_scores = []
        for result in matching_results.values():
            if result["matches"]["top_matches"]:
                all_scores.extend([m["match_score"] for m in result["matches"]["top_matches"]])
        
        performance_metrics = {
            "processing_metrics": {
                "resumes_processed": len(processed_resumes),
                "total_processing_time": process_time,
                "avg_processing_time_per_resume": process_time / len(processed_resumes)
            },
            "search_metrics": {
                "total_search_queries": len(search_queries),
                "avg_search_time": avg_search_time,
                "total_search_results": sum(r["matches"] for r in search_results.values())
            },
            "matching_metrics": {
                "jobs_tested": len(job_descriptions),
                "successful_matches": successful_jobs,
                "success_rate": successful_jobs / len(job_descriptions),
                "total_candidates_found": total_matches_found,
                "avg_matching_time": avg_matching_time
            },
            "quality_metrics": {
                "avg_match_score": sum(all_scores) / len(all_scores) if all_scores else 0,
                "max_match_score": max(all_scores) if all_scores else 0,
                "min_match_score": min(all_scores) if all_scores else 0,
                "score_std": (sum((s - sum(all_scores)/len(all_scores))**2 for s in all_scores) / len(all_scores))**0.5 if all_scores else 0
            }
        }
        
        print("📊 Performance Metrics Summary:")
        for category, metrics in performance_metrics.items():
            print(f"\n  {category.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if "time" in metric:
                        print(f"    {metric}: {value:.3f}s")
                    elif "score" in metric or "rate" in metric:
                        print(f"    {metric}: {value:.2f}")
                    else:
                        print(f"    {metric}: {value:.1f}")
                else:
                    print(f"    {metric}: {value}")
        
        # 9. Detailed Match Analysis
        print("\n🔍 Step 9: Detailed Match Analysis...")
        
        # Find best performing job
        best_job = max(matching_results.items(), 
                      key=lambda x: len(x[1]["matches"]["top_matches"]))
        
        print(f"Best performing job: {best_job[0]}")
        print(f"Candidates found: {len(best_job[1]['matches']['top_matches'])}")
        
        if best_job[1]["matches"]["top_matches"]:
            print("\nTop 3 candidates:")
            for i, match in enumerate(best_job[1]["matches"]["top_matches"][:3], 1):
                print(f"  {i}. {match['candidate_name']} - {match['match_score']}%")
                print(f"     Skills: {', '.join(match['matched_skills'][:3]) if match['matched_skills'] else 'No skills listed'}...")
                print(f"     Reasoning: {match['reasoning'][:100]}..." if len(match['reasoning']) > 100 else f"     Reasoning: {match['reasoning']}")
        
        # 10. Save Test Results
        print("\n💾 Step 10: Saving Test Results...")
        
        test_results = {
            "test_metadata": {
                "test_date": datetime.now().isoformat(),
                "system_config": {
                    "vector_db": rag.vector_db_type,
                    "embedding_model": rag.embedding_model_type,
                    "collection_name": rag.collection_name
                }
            },
            "performance_metrics": performance_metrics,
            "detailed_results": {
                "search_results": {k: {"matches": v["matches"], "time": v["time"]} 
                                 for k, v in search_results.items()},
                "matching_results": {k: {"candidates_found": v["candidates_found"], 
                                       "time": v["time"]} 
                                   for k, v in matching_results.items()}
            }
        }
        
        with open("rag_test_results.json", "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print("✅ Test results saved to 'rag_test_results.json'")
        
        # 11. Final Assessment
        print("\n🎯 Step 11: Final Assessment...")
        
        overall_score = (
            min(100, (successful_jobs / len(job_descriptions)) * 100) * 0.3 +  # Success rate
            min(100, (1 / avg_matching_time) * 50) * 0.2 +                     # Speed (2s = 100%)
            min(100, performance_metrics["quality_metrics"]["avg_match_score"]) * 0.3 +  # Quality
            min(100, (total_matches_found / (len(job_descriptions) * 5)) * 100) * 0.2    # Coverage
        )
        
        print(f"Overall System Score: {overall_score:.1f}/100")
        
        if overall_score >= 80:
            print("🟢 Excellent - System ready for production")
        elif overall_score >= 60:
            print("🟡 Good - Minor improvements recommended")
        elif overall_score >= 40:
            print("🟠 Fair - Significant improvements needed")
        else:
            print("🔴 Poor - Major rework required")
        
        print(f"\n🎉 RAG System Test Completed Successfully!")
        print(f"⏱️  Total test time: {time.time() - start_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """Test individual components separately."""
    print("\n🔧 Testing Individual Components...")
    print("-" * 40)
    
    # Test 1: Basic file operations
    print("1. Testing file operations...")
    try:
        resumes = list_files("examples/resumes", ".txt")
        jobs = list_files("examples/job_descriptions", ".txt")
        print(f"   ✅ Found {len(resumes)} resumes and {len(jobs)} job descriptions")
    except Exception as e:
        print(f"   ❌ File operations failed: {e}")
        return False
    
    # Test 2: RAG initialization
    print("2. Testing RAG initialization...")
    try:
        rag = ResumeRAG(
            vector_db="chromadb",
            embedding_model="sentence-transformers",  # Use local model for safety
            collection_name="test_collection"
        )
        print("   ✅ RAG system initialized successfully")
    except Exception as e:
        print(f"   ❌ RAG initialization failed: {e}")
        print("   Make sure ChromaDB and sentence-transformers are installed")
        return False
    
    # Test 3: Document processing
    print("3. Testing document processing...")
    try:
        if resumes:
            metadata = rag.process_resume(resumes[0]["filepath"])
            print(f"   ✅ Processed resume: {metadata.name}")
            print(f"      Skills found: {len(metadata.skills)}")
    except Exception as e:
        print(f"   ❌ Document processing failed: {e}")
        return False
    
    print("✅ All individual components working correctly")
    return True


if __name__ == "__main__":
    print("🧪 RAG System Test Suite")
    print("=" * 60)
    
    # Check if we should run individual tests first
    if os.getenv("TEST_MODE") == "individual":
        success = test_individual_components()
    else:
        # Run comprehensive test
        success = test_rag_system()
    
    if success:
        print("\n🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Review the test results in 'rag_test_results.json'")
        print("2. Analyze performance metrics in the Jupyter notebook")
        print("3. Experiment with different configurations")
        print("4. Add more resume and job description samples")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Set OPENAI_API_KEY environment variable if using OpenAI embeddings")
        print("3. Check that example files exist in examples/resumes and examples/job_descriptions")
        print("4. Run individual tests: TEST_MODE=individual python test_rag_system.py")