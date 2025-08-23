#!/usr/bin/env python3
"""
JEE Benchmark Script for Math Routing Agent
Evaluates the agent's performance on JEE-level mathematics problems
"""

import asyncio
import json
import time
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import httpx
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Single benchmark test result"""
    question_id: str
    question: str
    expected_answer: str
    agent_answer: str
    source: str  # 'kb' or 'web'
    response_time: float
    is_correct: bool
    similarity_score: float
    error: Optional[str] = None

@dataclass
class BenchmarkSummary:
    """Overall benchmark summary"""
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_response_time: float
    kb_usage_rate: float
    web_usage_rate: float
    total_runtime: float
    errors: int
    
class JEEBenchmarkSuite:
    """JEE Benchmark Suite for Math Agent"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.results: List[BenchmarkResult] = []
        
    async def load_jee_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load JEE dataset from file"""
        try:
            if dataset_path.endswith('.json'):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif dataset_path.endswith('.jsonl'):
                data = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                raise ValueError("Unsupported file format. Use .json or .jsonl")
                
            logger.info(f"Loaded {len(data)} questions from JEE dataset")
            return data
            
        except FileNotFoundError:
            # Create sample JEE dataset if file doesn't exist
            logger.warning(f"Dataset file {dataset_path} not found. Creating sample dataset...")
            return self.create_sample_jee_dataset()
    
    def create_sample_jee_dataset(self) -> List[Dict[str, Any]]:
        """Create a sample JEE dataset for testing"""
        sample_data = [
            {
                "id": "jee_001",
                "question": "Find the derivative of f(x) = 3x^4 + 2x^3 - x^2 + 5x - 1",
                "answer": "12x^3 + 6x^2 - 2x + 5",
                "topic": "calculus",
                "difficulty": "medium"
            },
            {
                "id": "jee_002", 
                "question": "Solve the equation: 2x^2 - 8x + 6 = 0",
                "answer": "x = 1, x = 3",
                "topic": "algebra",
                "difficulty": "easy"
            },
            {
                "id": "jee_003",
                "question": "Find the integral of ∫(3x^2 + 4x - 2)dx",
                "answer": "x^3 + 2x^2 - 2x + C",
                "topic": "calculus", 
                "difficulty": "medium"
            },
            {
                "id": "jee_004",
                "question": "Calculate the determinant of the 2x2 matrix [[3, 4], [1, 2]]",
                "answer": "2",
                "topic": "linear_algebra",
                "difficulty": "easy"
            },
            {
                "id": "jee_005",
                "question": "Find the value of sin(π/6) + cos(π/3)",
                "answer": "1",
                "topic": "trigonometry",
                "difficulty": "easy"
            },
            {
                "id": "jee_006",
                "question": "If log₂(x) = 3, find the value of x",
                "answer": "8",
                "topic": "logarithms",
                "difficulty": "easy"
            },
            {
                "id": "jee_007",
                "question": "Find the limit: lim(x→0) (sin(x)/x)",
                "answer": "1",
                "topic": "limits",
                "difficulty": "medium"
            },
            {
                "id": "jee_008",
                "question": "Calculate the area under the curve y = x^2 from x = 0 to x = 2",
                "answer": "8/3",
                "topic": "integration",
                "difficulty": "medium"
            },
            {
                "id": "jee_009",
                "question": "Solve for x: e^(2x) - 3e^x + 2 = 0",
                "answer": "x = 0, x = ln(2)",
                "topic": "exponentials",
                "difficulty": "hard"
            },
            {
                "id": "jee_010",
                "question": "Find the equation of the tangent line to y = x^3 at x = 2",
                "answer": "y = 12x - 16",
                "topic": "calculus",
                "difficulty": "medium"
            }
        ]
        
        # Save sample dataset
        with open('sample_jee_dataset.jsonl', 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created sample JEE dataset with {len(sample_data)} questions")
        return sample_data
    
    async def ask_agent(self, question: str) -> Dict[str, Any]:
        """Ask the math agent a question"""
        try:
            response = await self.client.post(
                f"{self.api_base_url}/ask",
                json={"question": question},
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error asking agent: {e}")
            raise
    
    def evaluate_answer(self, expected: str, actual: str) -> tuple[bool, float]:
        """
        Evaluate if the agent's answer is correct
        Returns (is_correct, similarity_score)
        """
        # Simple string matching for now
        # In practice, you might want more sophisticated math answer comparison
        
        # Normalize answers
        expected_norm = expected.lower().strip().replace(" ", "")
        actual_norm = actual.lower().strip().replace(" ", "")
        
        # Exact match
        if expected_norm == actual_norm:
            return True, 1.0
            
        # Check if expected answer is contained in actual answer
        if expected_norm in actual_norm:
            return True, 0.8
            
        # Check for key mathematical terms
        expected_terms = set(expected_norm.replace("=", " ").replace(",", " ").split())
        actual_terms = set(actual_norm.replace("=", " ").replace(",", " ").split())
        
        if expected_terms & actual_terms:  # Intersection
            similarity = len(expected_terms & actual_terms) / len(expected_terms | actual_terms)
            return similarity > 0.5, similarity
        
        return False, 0.0
    
    async def run_single_test(self, test_case: Dict[str, Any]) -> BenchmarkResult:
        """Run a single benchmark test"""
        question_id = test_case.get('id', 'unknown')
        question = test_case['question']
        expected_answer = test_case['answer']
        
        logger.info(f"Testing question {question_id}: {question[:50]}...")
        
        start_time = time.time()
        error = None
        
        try:
            response = await self.ask_agent(question)
            response_time = time.time() - start_time
            
            agent_answer = response.get('answer', '')
            source = response.get('source', 'unknown')
            
            is_correct, similarity_score = self.evaluate_answer(expected_answer, agent_answer)
            
            result = BenchmarkResult(
                question_id=question_id,
                question=question,
                expected_answer=expected_answer,
                agent_answer=agent_answer,
                source=source,
                response_time=response_time,
                is_correct=is_correct,
                similarity_score=similarity_score
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            error = str(e)
            logger.error(f"Error testing question {question_id}: {e}")
            
            result = BenchmarkResult(
                question_id=question_id,
                question=question,
                expected_answer=expected_answer,
                agent_answer="ERROR",
                source="error",
                response_time=response_time,
                is_correct=False,
                similarity_score=0.0,
                error=error
            )
        
        self.results.append(result)
        return result
    
    async def run_benchmark(self, dataset_path: str, max_questions: Optional[int] = None) -> BenchmarkSummary:
        """Run the complete benchmark suite"""
        logger.info("🧪 Starting JEE Benchmark for Math Agent")
        start_time = time.time()
        
        # Load dataset
        dataset = await self.load_jee_dataset(dataset_path)
        
        if max_questions:
            dataset = dataset[:max_questions]
        
        logger.info(f"Running benchmark on {len(dataset)} questions...")
        
        # Run tests
        self.results = []
        for test_case in dataset:
            await self.run_single_test(test_case)
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        total_runtime = time.time() - start_time
        
        # Calculate summary statistics
        total_questions = len(self.results)
        correct_answers = sum(1 for r in self.results if r.is_correct)
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        response_times = [r.response_time for r in self.results if r.error is None]
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        
        kb_usage = sum(1 for r in self.results if r.source == 'kb')
        web_usage = sum(1 for r in self.results if r.source == 'web')
        
        kb_usage_rate = kb_usage / total_questions if total_questions > 0 else 0.0
        web_usage_rate = web_usage / total_questions if total_questions > 0 else 0.0
        
        errors = sum(1 for r in self.results if r.error is not None)
        
        summary = BenchmarkSummary(
            total_questions=total_questions,
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            kb_usage_rate=kb_usage_rate,
            web_usage_rate=web_usage_rate,
            total_runtime=total_runtime,
            errors=errors
        )
        
        return summary
    
    def save_results(self, output_dir: str = "benchmark_results"):
        """Save benchmark results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_path / f"jee_benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        # Save CSV for analysis
        csv_file = output_path / f"jee_benchmark_results_{timestamp}.csv"
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {results_file} and {csv_file}")
    
    def print_summary(self, summary: BenchmarkSummary):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("🧮 JEE BENCHMARK RESULTS - MATH ROUTING AGENT")
        print("="*60)
        print(f"📊 Total Questions:     {summary.total_questions}")
        print(f"✅ Correct Answers:     {summary.correct_answers}")
        print(f"🎯 Accuracy:            {summary.accuracy:.2%}")
        print(f"⏱️  Avg Response Time:   {summary.avg_response_time:.2f}s")
        print(f"📚 KB Usage Rate:       {summary.kb_usage_rate:.2%}")
        print(f"🌐 Web Usage Rate:      {summary.web_usage_rate:.2%}")
        print(f"⏰ Total Runtime:       {summary.total_runtime:.2f}s")
        print(f"❌ Errors:              {summary.errors}")
        print("="*60)
        
        # Print per-topic breakdown if available
        topics = {}
        for result in self.results:
            # Extract topic from question_id or classify based on content
            topic = "general"  # Default topic
            if "calculus" in result.question.lower():
                topic = "calculus"
            elif "algebra" in result.question.lower() or "equation" in result.question.lower():
                topic = "algebra"
            elif "trigonometry" in result.question.lower() or "sin" in result.question.lower() or "cos" in result.question.lower():
                topic = "trigonometry"
            elif "integral" in result.question.lower() or "derivative" in result.question.lower():
                topic = "calculus"
            
            if topic not in topics:
                topics[topic] = {"correct": 0, "total": 0}
            
            topics[topic]["total"] += 1
            if result.is_correct:
                topics[topic]["correct"] += 1
        
        if len(topics) > 1:
            print("\n📈 PERFORMANCE BY TOPIC:")
            print("-" * 40)
            for topic, stats in topics.items():
                accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                print(f"{topic.title():12} {stats['correct']:3}/{stats['total']:3} ({accuracy:.1%})")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()

async def main():
    """Main benchmark execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="JEE Benchmark for Math Agent")
    parser.add_argument("--dataset", default="sample_jee_dataset.jsonl", help="Path to JEE dataset")
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions to test")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Math Agent API URL")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize benchmark suite
    benchmark = JEEBenchmarkSuite(api_base_url=args.api_url)
    
    try:
        # Run benchmark
        summary = await benchmark.run_benchmark(args.dataset, args.max_questions)
        
        # Print results
        benchmark.print_summary(summary)
        
        # Save results
        benchmark.save_results(args.output_dir)
        
        # Print top errors if any
        if summary.errors > 0:
            print(f"\n❌ ERRORS ({summary.errors}):")
            print("-" * 40)
            error_results = [r for r in benchmark.results if r.error]
            for result in error_results[:5]:  # Show first 5 errors
                print(f"Q: {result.question[:50]}...")
                print(f"E: {result.error}")
                print()
        
        return summary.accuracy >= 0.5  # Return True if accuracy >= 50%
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return False
        
    finally:
        await benchmark.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)