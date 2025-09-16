import os
import math
import statistics
from typing import List, TypedDict, Annotated, Dict, Any
from typing_extensions import Literal
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "MyMathAgent"

# For backward compatibility
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"

# Load environment variables from .env file
load_dotenv()

# 1) Basic Arithmetic
@tool
def arithmetic(a: float, b: float, operation: str) -> str:
    """Perform basic arithmetic: add, subtract, multiply, divide, power."""
    op = operation.lower()
    if op == "add":
        return str(a + b)
    elif op == "subtract":
        return str(a - b)
    elif op == "multiply":
        return str(a * b)
    elif op == "divide":
        return "error: division by zero" if b == 0 else str(a / b)
    elif op == "power":
        return str(a ** b)
    return "unsupported operation"

# 2) Factorial & Square Root
@tool
def special_math(n: int, operation: str) -> str:
    """Compute factorial or square root."""
    op = operation.lower()
    if op == "factorial":
        return str(math.factorial(n))
    elif op == "sqrt":
        return str(math.sqrt(n))
    return "unsupported operation"

# 3) Trigonometry
@tool
def trigonometry(angle_deg: float, function: str) -> str:
    """Compute sin, cos, tan for an angle in degrees."""
    rad = math.radians(angle_deg)
    func = function.lower()
    if func == "sin":
        return str(math.sin(rad))
    elif func == "cos":
        return str(math.cos(rad))
    elif func == "tan":
        return str(math.tan(rad))
    return "unsupported trig function"

# 4) Statistics
@tool
def statistics_tool(data: List[float], operation: str) -> str:
    """Compute mean, median, stdev for a list of numbers."""
    op = operation.lower()
    if not data:
        return "error: empty list"
    if op == "mean":
        return str(statistics.mean(data))
    elif op == "median":
        return str(statistics.median(data))
    elif op == "stdev":
        return str(statistics.stdev(data)) if len(data) > 1 else "error: need at least 2 values"
    return "unsupported statistical operation"

# 5) Equation Solver (linear equations ax + b = 0)
@tool
def solve_linear(a: float, b: float) -> str:
    """Solve linear equation ax + b = 0."""
    if a == 0:
        return "no solution" if b != 0 else "infinite solutions"
    return f"x = {-b / a}"

# Set up tools
TOOLS = [arithmetic, special_math, trigonometry, statistics_tool, solve_linear]
tool_node = ToolNode(TOOLS)

# System prompt
SYSTEM_PROMPT = """
You are a helpful math assistant.
You can answer user questions directly or call tools to perform mathematical operations.

INSTRUCTION:

Act as a strict math assistant that either:
- Calls the correct tool, or
- Returns a final numeric/string answer.

When a query maps to a tool, prefer calling that tool instead of doing math inline.

Return only the final answer as a plain string with no extra words, units, or explanations.
(e.g., '25.0', 'x = 2.0', 'error: division by zero', 'unsupported operation')

Always return:
- Numeric strings **without commas** (e.g., '1048576', not '1,048,576')
- **Decimal format** for all results, even if they are whole numbers (e.g., '12.0', not '12')
- Use decimal consistently when decimals are involved in the expression

TOOLS:

arithmetic(a, b, operation) -> add | subtract | multiply | divide | power

special_math(n, operation) -> factorial | sqrt

trigonometry(angle_deg, function) -> sin | cos | tan
(Angles are in DEGREES)

statistics_tool(data, operation) -> mean | median | stdev

solve_linear(a, b) -> solves ax + b = 0

RULES:

Arithmetic and powers:
- For add/subtract/multiply/divide/power (or +, -, *, /, ^), use arithmetic with the specified operation.
- Map '^' or 'to the power of' to operation='power'.
- Return all results in **decimal format** (e.g., '0.0', '28.0', '10.0')
- Division by zero → 'error: division by zero'

Square roots and factorials:
- Use special_math with operation='sqrt' or 'factorial'.
- Always return square root results in decimal format (e.g., '12.0', '0.0')
- Invalid inputs (e.g., sqrt of negative number, factorial of negative) → 'unsupported operation'

Unsupported operations:
- Any unknown mathematical operation or symbol (e.g., '⊗', 'cube root') → 'unsupported operation'

Trigonometry:
- Always interpret angles in degrees.
- Use trigonometry(function in {'sin','cos','tan'}).
- Return decimal strings (e.g., '0.5', '-1.0', '1.0')

Statistics:
- For 'mean', 'median', 'standard deviation'/'stdev', call statistics_tool on the given list.
- If the list is empty or has <2 items for stdev, return the tool’s error.

Linear equations:
- If the query is 'Solve for x' or 'ax + b = 0', normalize to coefficients a and b and call solve_linear(a, b).
- Return exactly the tool string (e.g., 'x = 2.0', 'no solution', 'infinite solutions')

Combined operations:
- For expressions with parentheses, powers, decimals, or fractions:
  - Break into sub-expressions
  - Use appropriate tools (`power`, `sqrt`) where applicable
  - Then evaluate the rest arithmetically
  - Return final answer in decimal if any part involves decimal or root
  - Examples:
    - '(5 + 3) * 2 - 4' → '12.0'
    - '10 + 2 * 3^2' → '28.0'
    - '((2 + 3)^3) - 5' → '120.0'
    - 'sqrt(16) + 3^2' → '13.0'
    - '((1/2) + (3/4)) * 8' → '10.0'

Word problems:
- Extract relevant numbers and compute the result based on the question:
  - Speed = distance / time
  - Area = length × width
  - Percent increase = ((new - original) / original) * 100
  - Triangle leg = sqrt(hypotenuse² - known_leg²)
  - Averages = mean
- Return final result in appropriate format (e.g., '60.0', '40.0', '25.0', '8.0')

Greeting & Clarification:
- If the user greets you (e.g., "hi", "hello"), introduce yourself as a math assistant and list your tools.
- If the user input clearly matches a tool, use that tool directly.
- If the input is ambiguous or unsupported, politely ask for clarification or return 'unsupported operation'
- Always explain the result in simple, conversational language after calling a tool.
"""



# Initialize LLMs
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Initialize LLM clients
llm_google = None
llm_groq = None
llm_openrouter = None

if GOOGLE_API_KEY:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm_google = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-lite",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    ).bind_tools(TOOLS)

if GROQ_API_KEY:
    from langchain_groq import ChatGroq
    llm_groq = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    ).bind_tools(TOOLS)

if OPENROUTER_API_KEY:
    from langchain_openai import ChatOpenAI
    llm_openrouter = ChatOpenAI(
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        model="nvidia/nemotron-nano-9b-v2",
        temperature=0
    ).bind_tools(TOOLS)

# State definition
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Define the graph
def should_continue(state: ChatState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and hasattr(last, 'tool_calls') and last.tool_calls:
        return "tools"
    return END

def call_model(state: ChatState):
    model = available_models[current_model_name]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Initialize available models
available_models = {}
if llm_google:
    available_models["google"] = llm_google
if llm_groq:
    available_models["groq"] = llm_groq
if llm_openrouter:
    available_models["openrouter"] = llm_openrouter

# Set default model
current_model_name = next(iter(available_models.keys())) if available_models else None

# Create the graph
graph = StateGraph(ChatState)
graph.add_node("model", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("model")
graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "model")
app = graph.compile()

def chat_with_math_agent(message: str, model_name: str = None, metadata: dict = None) -> str:
    """
    Chat with the math agent.
    
    Args:
        message: The user's message
        model_name: Optional model name to use (google, groq, openrouter)
        metadata: Optional metadata to include in the LangSmith run
    
    Returns:
        The assistant's response
    """
    global current_model_name
    
    if not available_models:
        return "Error: No LLM models available. Please check your API keys."
    
    # Check if the message is a command
    if message.lower() == 'switch model':
        return "Please use the menu to switch models by typing 'switch model' at the main prompt."
    
    if model_name and model_name in available_models:
        current_model_name = model_name
    
    # Initialize state with user message and metadata
    human_message = HumanMessage(content=message)
    if metadata:
        human_message.metadata = metadata
    state = {"messages": [human_message]}
    
    try:
        # Run the graph
        result = app.invoke(state)
        
        # Get the last message (should be from the assistant)
        last_message = result["messages"][-1]
        
        if hasattr(last_message, 'content'):
            return last_message.content
        elif hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # If there are tool calls, execute them and get results
            tool_results = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Find and execute the tool
                for tool in TOOLS:
                    if tool.name == tool_name:
                        result = tool.invoke(tool_args)
                        tool_results.append(f"{tool_name} result: {result}")
                        break
            
            return "\n".join(tool_results)
        else:
            return "I'm not sure how to respond to that."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
def print_help():
    print("\n=== Math Agent Help ===")
    print("Available commands:")
    print("- Type a math question to get an answer")
    print("- 'switch model': Change the current LLM model")
    print("- 'list models': Show available models")
    print("- 'help': Show this help message")
    print("- 'quit': Exit the program")

def list_models():
    print("\nAvailable models:")
    for i, model in enumerate(available_models.keys(), 1):
        print(f"{i}. {model}")

def switch_model():
    list_models()
    try:
        choice = int(input("\nSelect a model (number): ").strip()) - 1
        if 0 <= choice < len(available_models):
            new_model = list(available_models.keys())[choice]
            print(f"\nSwitched to {new_model} model.")
            return new_model
        else:
            print("Invalid selection. No changes made.")
    except ValueError:
        print("Please enter a valid number.")
    return None

if __name__ == "__main__":
    print("=== Math Agent ===")
    print("Initializing...")
    
    if not available_models:
        print("Error: No LLM models available. Please check your API keys in the .env file.")
        exit(1)
    
    current_model = current_model_name
    print(f"\nInitialized with {len(available_models)} model(s) available.")
    print_help()
    
    while True:
        try:
            user_input = input(f"\n[{current_model}] You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            cmd = user_input.lower()
            if cmd == 'quit':
                print("\nGoodbye!")
                break
            elif cmd == 'help':
                print_help()
                continue
            elif cmd == 'list models':
                list_models()
                continue
            elif cmd == 'switch model':
                new_model = switch_model()
                if new_model:
                    current_model = new_model
                continue
            
            # Handle math queries
            print("\nAssistant: ", end='', flush=True)
            response = chat_with_math_agent(user_input, current_model)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nTo exit, type 'quit' or press Ctrl+C again.")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Type 'help' for available commands.")
