import time
import json
import tiktoken
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from math_agent import chat_with_math_agent, available_models, TOOLS
from langsmith import Client
from langsmith.schemas import Run

# Initialize LangSmith client
client = Client()

# Token counter setup
token_encoder = tiktoken.get_encoding("cl100k_base")  # Default encoder for most models

# Cost per 1M tokens (prices in USD, as of 2024-09)
MODEL_COSTS = {
    # Google models (Gemini)
    "google": {
        "model_name": "gemini-pro",
        "input": 0.5,    # $0.50 per 1M tokens
        "output": 1.5,   # $1.50 per 1M tokens
        "context_window": 30720  # 30K tokens
    },
    # Groq models
    "groq": {
        "model_name": "mixtral-8x7b-32768",
        "input": 0.27,   # $0.27 per 1M tokens
        "output": 0.27,  # $0.27 per 1M tokens
        "context_window": 32768  # 32K tokens
    },
    # OpenRouter - NVIDIA Nemotron-Nano-9B
    "openrouter": {
        "model_name": "nvidia/nemotron-nano-9b-v2",
        "input": 0.1,     # $0.10 per 1M input tokens
        "output": 0.3,    # $0.30 per 1M output tokens
        "context_window": 8192  # 8K tokens context window
    },
    # Add more models as needed
    "gpt-4": {
        "model_name": "gpt-4",
        "input": 30.0,   # $30.00 per 1M tokens
        "output": 60.0,  # $60.00 per 1M tokens
        "context_window": 8192  # 8K tokens
    },
    "gpt-3.5-turbo": {
        "model_name": "gpt-3.5-turbo",
        "input": 0.5,    # $0.50 per 1M tokens
        "output": 1.5,   # $1.50 per 1M tokens
        "context_window": 16385  # 16K tokens
    }
}

def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    return len(token_encoder.encode(text))

# 1. Test Cases for Accuracy
test_cases = [
    # Basic Arithmetic
    {"query": "What is 15 + 7?", "expected": "22"},
    {"query": "Subtract 8 from 20", "expected": "12"},
    {"query": "Multiply 6 by 9", "expected": "54"},
    {"query": "Divide 100 by 4", "expected": "25.0"},
    {"query": "What is 2 to the power of 5?", "expected": "32"},
    {"query": "Add -5 and 8", "expected": "3"},
    {"query": "-10 * -3", "expected": "30"},
    {"query": "What is 45 - 60?", "expected": "-15"},
    {"query": "Compute 7^3", "expected": "343"},
    {"query": "Divide -12 by 5", "expected": "-2.4"},

    # Fractions and Decimals
    {"query": "What is 0.5 + 0.25?", "expected": "0.75"},
    {"query": "Calculate 3/4 + 1/2", "expected": "1.25"},
    {"query": "What is 2.5 - 1.75?", "expected": "0.75"},
    {"query": "Multiply 0.2 by 0.4", "expected": "0.08"},
    {"query": "Divide 1 by 8", "expected": "0.125"},

    # Square Roots and Powers
    {"query": "Square root of 144", "expected": "12.0"},
    {"query": "What is 9 squared?", "expected": "81"},
    {"query": "Cube root of 27", "expected": "3.0"},
    {"query": "Square root of 0", "expected": "0.0"},
    {"query": "What is 16 to the power of 0.5?", "expected": "4.0"},

    # Factorials
    {"query": "5 factorial", "expected": "120"},
    {"query": "What is 7! ?", "expected": "5040"},
    {"query": "Compute 0!", "expected": "1"},
    {"query": "Factorial of 1", "expected": "1"},
    {"query": "Factorial of 10", "expected": "3628800"},

    # Linear Equations (ax + b = 0)
    {"query": "Solve for x: 3x + 5 = 20", "expected": "5.0"},
    {"query": "Find x if 4x - 7 = 9", "expected": "4.0"},
    {"query": "Solve ax + b = 0 with a=2, b=-4", "expected": "x = 2.0"},
    {"query": "Solve for x: 5x + 0 = 0", "expected": "0.0"},
    {"query": "Solve ax + b = 0 with a=0, b=7", "expected": "no solution"},

    # Statistics
    {"query": "Mean of 10, 20, 30, 40, 50", "expected": "30.0"},
    {"query": "Median of 7, 3, 5, 1, 9", "expected": "5"},
    {"query": "Standard deviation of 2, 4, 6, 8, 10", "expected": "2.8284271247461903"},
    {"query": "Mean of 1", "expected": "1"},
    {"query": "Stdev of 5, 5", "expected": "0.0"},

    # Trigonometry (degrees)
    {"query": "Sine of 30 degrees", "expected": "0.5"},
    {"query": "Cosine of 60 degrees", "expected": "0.5"},
    {"query": "Tangent of 45 degrees", "expected": "1.0"},
    {"query": "Sine of 0 degrees", "expected": "0.0"},
    {"query": "Cosine of 180 degrees", "expected": "-1.0"},

    # Combined Operations
    {"query": "(5 + 3) * 2 - 4", "expected": "12"},
    {"query": "10 + 2 * 3^2", "expected": "28"},
    {"query": "((2 + 3) ^ 3) - 5", "expected": "120"},
    {"query": "sqrt(16) + 3^2", "expected": "13.0"},
    {"query": "((1/2) + (3/4)) * 8", "expected": "10.0"},

    # Special Cases and Errors
    {"query": "What is 0 divided by 5?", "expected": "0.0"},
    {"query": "5 divided by 0", "expected": "error: division by zero"},
    {"query": "Square root of -1", "expected": "unsupported operation"},
    {"query": "Factorial of -3", "expected": "unsupported operation"},
    {"query": "Unknown op: 3 ⊗ 4", "expected": "unsupported operation"},

    # Larger Numbers
    {"query": "123 * 456", "expected": "56088"},
    {"query": "1000 - 999", "expected": "1"},
    {"query": "2^20", "expected": "1048576"},
    {"query": "Square root of 10,000", "expected": "100.0"},
    {"query": "Mean of 100, 200, 300, 400", "expected": "250"},

    # Word Problems
    {"query": "If a train travels 120 miles in 2 hours, what is its speed?", "expected": "60.0"},
    {"query": "Area of a rectangle with length 8 and width 5", "expected": "40"},
    {"query": "A price increases from 80 to 100. What is the percent increase?", "expected": "25.0"},
    {"query": "Average of quiz scores 70, 85, 90", "expected": "81.66666666666667"},
    {"query": "A right triangle with hypotenuse 10 and one leg 6, find the other leg", "expected": "8.0"}
]


# 2. Model Comparison
# Expected tools for each test case type
EXPECTED_TOOLS = {
    # Basic Arithmetic
    "add": "arithmetic",
    "subtract": "arithmetic",
    "multiply": "arithmetic",
    "divide": "arithmetic",
    "power": "arithmetic",
    "+": "arithmetic",
    "-": "arithmetic",
    "*": "arithmetic",
    "/": "arithmetic",
    "^": "arithmetic",
    "to the power of": "arithmetic",
    "plus": "arithmetic",
    "minus": "arithmetic",
    "times": "arithmetic",
    "divided by": "arithmetic",
    "sum of": "arithmetic",
    "product of": "arithmetic",
    "difference between": "arithmetic",
    
    # Special Math
    "factorial": "special_math",
    "sqrt": "special_math",
    "square root": "special_math",
    "cube root": "special_math",
    "square of": "special_math",
    "cube of": "special_math",
    "!": "special_math",
    "root": "special_math",
    
    # Trigonometry
    "sin": "trigonometry",
    "cos": "trigonometry",
    "tan": "trigonometry",
    "sine": "trigonometry",
    "cosine": "trigonometry",
    "tangent": "trigonometry",
    "degrees": "trigonometry",
    
    # Statistics
    "mean": "statistics_tool",
    "median": "statistics_tool",
    "stdev": "statistics_tool",
    "average": "statistics_tool",
    "standard deviation": "statistics_tool",
    "avg": "statistics_tool",
    "statistics": "statistics_tool",
    "of the numbers": "statistics_tool",
    
    # Linear Equations
    "solve for": "solve_linear",
    "ax + b": "solve_linear",
    "x =": "solve_linear",
    "find x": "solve_linear",
    "equation": "solve_linear",
    "linear": "solve_linear",
    "solve the equation": "solve_linear",
    "what is x": "solve_linear",
    
    # Word Problems
    "speed": "arithmetic",
    "distance": "arithmetic",
    "time": "arithmetic",
    "area": "arithmetic",
    "perimeter": "arithmetic",
    "volume": "arithmetic",
    "percent": "arithmetic",
    "percentage": "arithmetic",
    "increase": "arithmetic",
    "decrease": "arithmetic",
    "ratio": "arithmetic",
    "proportion": "arithmetic",
    "hypotenuse": "special_math",
    "right triangle": "special_math",
    "pythagorean": "special_math"
}

def get_expected_tool(query: str) -> str:
    """Determine the expected tool based on the query."""
    query = query.lower()
    for key, tool in EXPECTED_TOOLS.items():
        if key in query:
            return tool
    return "unknown"

def get_tool_usage_from_langsmith(run_id: str) -> List[Dict[str, Any]]:
    """Fetch tool usage data from LangSmith for a specific run."""
    try:
        run = client.read_run(run_id)
        tool_runs = []
        
        def process_run(run: Run):
            if run.run_type == "tool" and hasattr(run, 'name') and run.name:
                tool_runs.append({
                    'name': run.name,
                    'input': run.inputs,
                    'output': run.outputs if hasattr(run, 'outputs') else None,
                    'start_time': run.start_time,
                    'end_time': run.end_time if hasattr(run, 'end_time') else None,
                    'error': run.error if hasattr(run, 'error') else None
                })
            
            # Process child runs
            for child in run.child_runs or []:
                process_run(child)
        
        process_run(run)
        return tool_runs
    except Exception as e:
        print(f"Error fetching tool usage from LangSmith: {e}")
        return []

def run_model_comparison(models):
    results = {}
    
    for model_name in models:
        # Generate a unique run ID for this evaluation
        run_id = str(uuid.uuid4())
        print(f"\nStarting evaluation for {model_name} (Run ID: {run_id})")
        print(f"\nTesting model: {model_name}")
        model_results = {
            "correct": 0,
            "total": len(test_cases),
            "tool_selection_accuracy": {
                "correct": 0,
                "total": 0,
                "details": []
            },
            "response_times": [],
            "tool_usage": {},
            "token_counts": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tool_call_tokens": 0
            },
            "latency": {
                "total": 0.0,
                "tool_calls": 0,
                "avg_per_tool_call": 0.0
            },
            "test_cases": []
        }
        
        for i, test in enumerate(test_cases):
            test_result = {
                "query": test["query"],
                "expected": test["expected"],
                "start_time": time.time(),
                "tool_calls": 0,
                "tool_call_times": [],
                "token_usage": {"input": 0, "output": 0, "tool_call": 0},
                "response": None,
                "is_correct": False,
                "total_time": 0.0
            }
            
            # Run the test
            try:
                # Track token usage for the prompt
                prompt_tokens = count_tokens(test["query"])
                test_result["token_usage"]["input"] = prompt_tokens
                
                # Add test case info to the result
                test_result["test_case"] = i
                test_result["model"] = model_name
                test_result["timestamp"] = datetime.utcnow().isoformat()
                
                # Get expected tool for this test case
                expected_tool = get_expected_tool(test["query"])
                test_result["expected_tool"] = expected_tool
                
                # Run the model with the unique run ID in metadata
                response = chat_with_math_agent(
                    test["query"],
                    model_name,
                    metadata={"evaluation_run_id": run_id, "test_case": i}
                )
                end_time = time.time()
                
                # Calculate response time
                response_time = end_time - test_result["start_time"]
                
                # Update test result
                test_result["response"] = str(response)
                test_result["is_correct"] = str(test["expected"]).lower() in str(response).lower()
                test_result["total_time"] = response_time
                test_result["run_id"] = run_id
                
                # Initialize tool calls counter based on whether a tool is expected
                test_result["tool_calls"] = 1 if expected_tool != 'unknown' else 0
                
                # Get tool usage from LangSmith
                time.sleep(2)  # Give LangSmith a moment to process the run
                tool_runs = get_tool_usage_from_langsmith(run_id)
                
                if tool_runs:
                    # Update tool calls count with actual number of tool calls
                    test_result["tool_calls"] = len(tool_runs)
                    
                    # Get the first tool used (assuming one tool per query for simplicity)
                    used_tool = tool_runs[0]['name']
                    test_result["used_tool"] = used_tool
                    test_result["tool_selection_correct"] = used_tool == expected_tool
                    test_result["tool_runs"] = tool_runs
                    
                    # Update tool selection accuracy
                    model_results["tool_selection_accuracy"]["total"] += 1
                    if used_tool == expected_tool:
                        model_results["tool_selection_accuracy"]["correct"] += 1
                        
                    # Record tool usage statistics
                    for tool_run in tool_runs:
                        tool_name = tool_run['name']
                        if tool_name not in model_results["tool_usage"]:
                            model_results["tool_usage"][tool_name] = 0
                        model_results["tool_usage"][tool_name] += 1
                    
                    # Record tool usage
                    if used_tool not in model_results["tool_usage"]:
                        model_results["tool_usage"][used_tool] = 0
                    model_results["tool_usage"][used_tool] += 1
                    
                    model_results["tool_selection_accuracy"]["details"].append({
                        "query": test["query"],
                        "expected_tool": expected_tool,
                        "used_tool": used_tool,
                        "correct": used_tool == expected_tool
                    })
                
                # Count output tokens
                output_tokens = count_tokens(str(response))
                test_result["token_usage"]["output"] = output_tokens
                
                # Update model results
                if test_result["is_correct"]:
                    model_results["correct"] += 1
                
                # Update token counts
                model_results["token_counts"]["total_input_tokens"] += test_result["token_usage"]["input"]
                model_results["token_counts"]["total_output_tokens"] += test_result["token_usage"]["output"]
                
                # Update latency
                model_results["latency"]["total"] += response_time
                model_results["latency"]["tool_calls"] += test_result["tool_calls"]
                
                # Calculate cost for this test case
                model_cost = MODEL_COSTS.get(model_name, MODEL_COSTS["google"])
                input_cost = (test_result['token_usage']['input'] / 1_000_000) * model_cost["input"]
                output_cost = (test_result['token_usage']['output'] / 1_000_000) * model_cost["output"]
                total_cost = input_cost + output_cost
                
                # Print test results
                print(f"\nTest {i+1}/{len(test_cases)}")
                print("-" * 50)
                print(f"Q: {test['query']}")
                print(f"A: {response}")
                print(f"Expected: {test['expected']}")
                print(f"Correct: {test_result['is_correct']}")
                print(f"Time: {response_time:.2f}s")
                
                # Print token and cost information
                print(f"Tokens: {test_result['token_usage']['input']} in, {test_result['token_usage']['output']} out")
                print(f"Cost: ${total_cost:.8f} (Input: ${input_cost:.8f}, Output: ${output_cost:.8f})")
                
            except Exception as e:
                end_time = time.time()
                test_result["error"] = str(e)
                test_result["total_time"] = end_time - test_result["start_time"]
                print(f"Error in test case {i+1}: {str(e)}")
            
            # Add test case result
            model_results["test_cases"].append(test_result)
            
            # Add delay between test cases (except after the last one)
            if i < len(test_cases) - 1:
                print("\nWaiting 5 seconds before the next test case...")
                time.sleep(5)
        
        # Calculate final metrics
        model_results["accuracy"] = model_results["correct"] / model_results["total"]
        model_results["avg_response_time"] = model_results["latency"]["total"] / model_results["total"]
        
        # Calculate tool selection accuracy
        if model_results["tool_selection_accuracy"]["total"] > 0:
            model_results["tool_selection_accuracy"]["accuracy"] = (
                model_results["tool_selection_accuracy"]["correct"] / 
                model_results["tool_selection_accuracy"]["total"]
            )
        else:
            model_results["tool_selection_accuracy"]["accuracy"] = 0.0
        
        # Calculate token costs
        model_cost = MODEL_COSTS.get(model_name, MODEL_COSTS["google"])  # Default to google pricing
        input_cost = (model_results["token_counts"]["total_input_tokens"] / 1_000_000) * model_cost["input"]
        output_cost = (model_results["token_counts"]["total_output_tokens"] / 1_000_000) * model_cost["output"]
        
        model_results["cost"] = {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "cost_per_1k_tokens": {
                "input": (model_cost["input"] / 1000),
                "output": (model_cost["output"] / 1000)
            }
        }
        
        # Calculate average tokens per call
        model_results["avg_tokens_per_call"] = {
            "input": model_results["token_counts"]["total_input_tokens"] / model_results["total"],
            "output": model_results["token_counts"]["total_output_tokens"] / model_results["total"]
        }
        
        # Calculate tool call statistics
        total_tool_calls = model_results["latency"]["tool_calls"]
        if total_tool_calls > 0:
            model_results["latency"]["avg_per_tool_call"] = (
                model_results["latency"]["total"] / total_tool_calls
            )
        
        results[model_name] = model_results
    
    return results

# 3. Generate Report
def generate_report(results):
    print("\n" + "="*80)
    print("=== EVALUATION REPORT ===".center(80))
    print("="*80)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Model Comparison Summary
    print("MODEL COMPARISON SUMMARY".center(80))
    print("-" * 80)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'Avg Time':<10} | {'Tokens/Call':<15} | {'Cost':<10} | {'Tool Calls':<10} | Correct/Total")
    print("-" * 80)
    
    for model_name, data in results.items():
        total_tokens = data["token_counts"]["total_input_tokens"] + data["token_counts"]["total_output_tokens"]
        tokens_per_call = f"{data['avg_tokens_per_call']['input']:.0f} in, {data['avg_tokens_per_call']['output']:.0f} out"
        cost = f"${data['cost']['total_cost']:.4f}"
        tool_calls = data["latency"]["tool_calls"]
        
        print(f"{model_name:<15} | {data['accuracy']:<10.2%} | {data['avg_response_time']:<8.2f}s | {tokens_per_call:<15} | {cost:<10} | {tool_calls:<10} | {data['correct']}/{data['total']}")
    
    # Detailed breakdown for each model
    for model_name, data in results.items():
        print("\n" + f" DETAILED RESULTS: {model_name.upper()} ".center(80, "="))
        
        # Tool Selection Accuracy
        if data["tool_selection_accuracy"]["total"] > 0:
            print("\nTOOL SELECTION ACCURACY:")
            print(f"  • Correct: {data['tool_selection_accuracy']['correct']}/{data['tool_selection_accuracy']['total']} "
                  f"({data['tool_selection_accuracy']['accuracy']:.1%})")
            
            # Show tool usage distribution
            if data["tool_usage"]:
                print("\nTOOL USAGE DISTRIBUTION:")
                for tool, count in data["tool_usage"].items():
                    print(f"  • {tool}: {count} times")
        
        # Token Usage
        print("\nTOKEN USAGE:")
        print(f"  • Input tokens:  {data['token_counts']['total_input_tokens']:,} (${data['cost']['input_cost']:.4f})")
        print(f"  • Output tokens: {data['token_counts']['total_output_tokens']:,} (${data['cost']['output_cost']:.4f})")
        print(f"  • Total tokens:  {data['token_counts']['total_input_tokens'] + data['token_counts']['total_output_tokens']:,} (${data['cost']['total_cost']:.4f})")
        print(f"  • Avg per call:  {data['avg_tokens_per_call']['input']:.1f} in, {data['avg_tokens_per_call']['output']:.1f} out")
        
        # Latency
        print("\nLATENCY:")
        print(f"  • Total time:    {data['latency']['total']:.2f}s")
        print(f"  • Avg per call:  {data['avg_response_time']:.2f}s")
        if data["latency"]["tool_calls"] > 0:
            print(f"  • Tool calls:    {data['latency']['tool_calls']} (avg {data['latency']['avg_per_tool_call']:.2f}s per tool call)")
        
        # Accuracy
        print("\nACCURACY:")
        print(f"  • Correct:       {data['correct']}/{data['total']} ({data['accuracy']:.1%})")
        
        # Cost breakdown
        print("\nCOST BREAKDOWN:")
        print(f"  • Input cost:    ${data['cost']['input_cost']:.4f} (${data['cost']['cost_per_1k_tokens']['input']:.4f} per 1K tokens)")
        print(f"  • Output cost:   ${data['cost']['output_cost']:.4f} (${data['cost']['cost_per_1k_tokens']['output']:.4f} per 1K tokens)")
        print(f"  • Total cost:    ${data['cost']['total_cost']:.4f}")
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print(f"Detailed results saved to: {filename}".center(80))
    print("="*80)

def parse_model_selection(available_models, model_args=None):
    """Parse model selection from command line arguments."""
    model_list = list(available_models.keys())
    
    if not model_args or 'all' in model_args:
        return model_list
    
    try:
        selected_indices = []
        for arg in model_args:
            if '-' in arg:
                # Handle ranges like 1-3
                start, end = map(int, arg.split('-'))
                selected_indices.extend(range(start-1, end))
            else:
                selected_indices.append(int(arg)-1)
        
        # Remove duplicates and invalid indices
        selected_indices = list(set(
            idx for idx in selected_indices 
            if 0 <= idx < len(model_list)
        ))
        
        if not selected_indices:
            print("No valid models selected. Using all available models.")
            return model_list
            
        return [model_list[i] for i in sorted(selected_indices)]
        
    except (ValueError, IndexError) as e:
        print(f"Error parsing model selection: {e}")
        print("Available models:")
        for i, model in enumerate(model_list, 1):
            print(f"  {i}. {model}")
        print("\nUsage: python evaluate_math_agent.py [model_numbers]")
        print("Example: python evaluate_math_agent.py 1 2  # Select models 1 and 2")
        print("         python evaluate_math_agent.py 1-3  # Select models 1 through 3")
        print("         python evaluate_math_agent.py all  # Select all models (default)")
        return []

def get_model_cost_estimate(model_name, input_tokens, output_tokens):
    """Calculate cost estimate for a model based on token usage."""
    model_info = MODEL_COSTS.get(model_name, MODEL_COSTS["google"])  # Default to google pricing
    
    input_cost = (input_tokens / 1_000_000) * model_info["input"]
    output_cost = (output_tokens / 1_000_000) * model_info["output"]
    
    return {
        "model_name": model_info["model_name"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "cost_per_1k_input": model_info["input"] / 1000,
        "cost_per_1k_output": model_info["output"] / 1000
    }

if __name__ == "__main__":
    import sys
    from math_agent import available_models
    
    if not available_models:
        print("Error: No models available. Please check your API keys.")
        sys.exit(1)
    
    # If models are specified as command line arguments, use those
    if len(sys.argv) > 1:
        selected_models = parse_model_selection(available_models, sys.argv[1:])
    else:
        # Otherwise, show interactive model selection
        print("Available models:")
        models_list = list(available_models)
        for i, model in enumerate(models_list, 1):
            print(f"{i}. {model}")
        print("A. All models")
        
        # Get user selection
        selection = input("\nSelect model(s) to test (comma-separated numbers or 'A' for all): ").strip().upper()
        
        if selection == 'A':
            selected_models = models_list
        else:
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                selected_models = [models_list[i] for i in selected_indices if 0 <= i < len(models_list)]
                if not selected_models:
                    raise ValueError("No valid models selected")
            except (ValueError, IndexError) as e:
                print("Invalid selection. Running with all models.")
                selected_models = models_list
    
    print(f"\nSelected models: {', '.join(selected_models)}\n")
    
    if not selected_models:
        sys.exit(1)
    
    print(f"\nSelected models for evaluation: {', '.join(selected_models)}")
    
    # Run evaluation
    results = run_model_comparison(selected_models)
    
    # Generate and display report
    generate_report(results)
    
    # Show cost estimate
    print("\n" + "="*50)
    print("COST ESTIMATE SUMMARY".center(50))
    print("="*50)
    for model_name, data in results.items():
        cost_estimate = get_model_cost_estimate(
            model_name,
            data["token_counts"]["total_input_tokens"],
            data["token_counts"]["total_output_tokens"]
        )
        print(f"\nModel: {model_name} ({cost_estimate['model_name']})")
        print(f"  • Input:  {cost_estimate['input_tokens']:,} tokens (${cost_estimate['input_cost']:.4f})")
        print(f"  • Output: {cost_estimate['output_tokens']:,} tokens (${cost_estimate['output_cost']:.4f})")
        print(f"  • Total:  ${cost_estimate['total_cost']:.4f}")
        print(f"  • Rate:   ${cost_estimate['cost_per_1k_input']:.4f}/1K in, ${cost_estimate['cost_per_1k_output']:.4f}/1K out")
