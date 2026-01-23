"""
Base Agent Infrastructure using aisuite

This module provides the base agent class and aisuite client configuration
for the SEPA trading workflow. Supports switching between OpenAI and Anthropic.

Note: aisuite is optional. Agents can run without LLM reasoning using direct
tool execution when aisuite is not installed.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

# Optional aisuite import
try:
    import aisuite as ai
    AISUITE_AVAILABLE = True
except ImportError:
    ai = None
    AISUITE_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition for a tool that can be called by the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    

@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    data: Any
    message: str
    tool_calls: List[Dict] = field(default_factory=list)
    reasoning: str = ""


class AIClient:
    """
    Unified AI client using aisuite for multi-provider support.
    
    Supports:
    - OpenAI (GPT-4, GPT-4o)
    - Anthropic (Claude 3.5 Sonnet)
    
    Note: Returns None/errors gracefully if aisuite is not installed.
    """
    
    def __init__(self, default_model: str = "openai:gpt-4o"):
        """
        Initialize the AI client.
        
        Args:
            default_model: Default model in format "provider:model"
                          e.g., "openai:gpt-4o" or "anthropic:claude-3-5-sonnet-20241022"
        """
        self.default_model = default_model
        self.client = None
        
        if AISUITE_AVAILABLE and ai is not None:
            try:
                self.client = ai.Client()
                self._verify_credentials()
            except Exception as e:
                logger.warning(f"Could not initialize aisuite client: {e}")
        else:
            logger.info("aisuite not available - agents will use direct tool execution only")
    
    def _verify_credentials(self):
        """Check that required API keys are available."""
        provider = self.default_model.split(":")[0]
        
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not set - OpenAI models will not work")
        elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not set - Anthropic models will not work")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to self.default_model)
            tools: Optional list of tool definitions for function calling
            temperature: Sampling temperature
        
        Returns:
            Response dictionary with 'content' and optionally 'tool_calls'
        """
        if self.client is None:
            return {"content": "", "error": "AI client not available (aisuite not installed)"}
        
        model = model or self.default_model
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            
            if tools:
                kwargs["tools"] = tools
            
            response = self.client.chat.completions.create(**kwargs)
            
            # Extract response
            choice = response.choices[0]
            result = {
                "content": choice.message.content or "",
                "tool_calls": []
            }
            
            # Handle tool calls if present
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"AI client error: {e}")
            return {"content": "", "error": str(e)}


class BaseAgent(ABC):
    """
    Abstract base class for all SEPA agents.
    
    Each agent:
    - Has a specific purpose in the SEPA workflow
    - Uses tools to gather and analyze data
    - Returns structured results for the next agent
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        ai_client: Optional[AIClient] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name (e.g., "TrendTemplateAgent")
            description: What this agent does
            ai_client: Shared AI client instance
            config: Agent-specific configuration
        """
        self.name = name
        self.description = description
        self.ai_client = ai_client or AIClient()
        self.config = config or {}
        self.tools: List[ToolDefinition] = []
        self._setup_tools()
    
    @abstractmethod
    def _setup_tools(self):
        """Set up the tools available to this agent. Override in subclass."""
        pass
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent. Override in subclass."""
        pass
    
    def _format_tools_for_llm(self) -> List[Dict]:
        """Convert tool definitions to LLM-compatible format."""
        formatted = []
        for tool in self.tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return formatted
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Execute a tool by name with given arguments."""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    return tool.function(**arguments)
                except Exception as e:
                    logger.error(f"Tool {tool_name} execution error: {e}")
                    return {"error": str(e)}
        
        return {"error": f"Tool {tool_name} not found"}
    
    def run(self, input_data: Any, max_iterations: int = 5) -> AgentResult:
        """
        Run the agent with the given input.
        
        Args:
            input_data: Input data for the agent (varies by agent type)
            max_iterations: Maximum tool call iterations
        
        Returns:
            AgentResult with the agent's output
        """
        system_prompt = self._get_system_prompt()
        tools = self._format_tools_for_llm() if self.tools else None
        
        # Build initial message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_input(input_data)}
        ]
        
        all_tool_calls = []
        
        for iteration in range(max_iterations):
            response = self.ai_client.chat(
                messages=messages,
                tools=tools,
                temperature=0.3  # Lower temperature for more deterministic analysis
            )
            
            if "error" in response:
                return AgentResult(
                    success=False,
                    data=None,
                    message=f"AI error: {response['error']}",
                    reasoning=""
                )
            
            # Check for tool calls
            if response.get("tool_calls"):
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    
                    logger.info(f"[{self.name}] Calling tool: {tool_name}")
                    
                    result = self._execute_tool(tool_name, tool_args)
                    all_tool_calls.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result
                    })
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, default=str)
                    })
            else:
                # No more tool calls - agent is done
                return AgentResult(
                    success=True,
                    data=self._parse_output(response["content"]),
                    message=response["content"],
                    tool_calls=all_tool_calls,
                    reasoning=response["content"]
                )
        
        # Max iterations reached
        return AgentResult(
            success=False,
            data=None,
            message="Max iterations reached",
            tool_calls=all_tool_calls
        )
    
    def _format_input(self, input_data: Any) -> str:
        """Format input data for the LLM. Override for custom formatting."""
        if isinstance(input_data, str):
            return input_data
        return json.dumps(input_data, default=str)
    
    def _parse_output(self, content: str) -> Any:
        """Parse LLM output. Override for custom parsing."""
        # Try to parse as JSON
        try:
            return json.loads(content)
        except:
            return content


# Shared client instance
_shared_client: Optional[AIClient] = None

def get_ai_client(model: str = "openai:gpt-4o") -> AIClient:
    """Get or create a shared AI client instance."""
    global _shared_client
    if _shared_client is None:
        _shared_client = AIClient(default_model=model)
    return _shared_client


# Example usage
if __name__ == "__main__":
    # Test the AI client
    client = get_ai_client()
    
    response = client.chat(
        messages=[
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ],
        temperature=0.0
    )
    
    print(f"Response: {response}")
