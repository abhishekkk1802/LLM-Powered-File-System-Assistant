"""
LLM File System Assistant
========================

LLM integration module for natural language file operations.
Part B of the LLM Function Calling Assignment (40% of total grade).

This module provides:
- Natural language query processing
- Automatic tool selection and execution
- Multi-step workflow completion
- OpenRouter integration for multiple AI models
- Interactive chat interface

Supported Models:
- OpenAI GPT (gpt-4o, gpt-4o-mini)
- Anthropic Claude (claude-3-sonnet, claude-3-haiku)
- Google Gemini, Meta Llama, and others via OpenRouter

Author: Assignment Submission  
Course: LLM Function Calling and Tool Use
Date: 2024

Usage:
    from llm_file_assistant import LLMFileAssistant
    
    # Initialize assistant
    assistant = LLMFileAssistant(provider="openai", model="openai/gpt-4o-mini")
    
    # Process natural language queries
    result = assistant.process_query("Find resumes mentioning Python")
    
    # Start interactive session
    assistant.chat_session()
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from fs_tools import (
    read_file, list_files, write_file, search_in_file, TOOL_DEFINITIONS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment and LLM client imports
from dotenv import load_dotenv
import openai
import anthropic

# Load environment variables from .env file
load_dotenv()


class LLMFileAssistant:
    """
    LLM-powered file system assistant that can understand natural language
    queries and execute appropriate file operations.
    """
    
    def __init__(self, provider: str = "openai", model: str = None, api_key: str = None, base_url: str = None):
        """
        Initialize the LLM File Assistant.
        
        Args:
            provider (str): LLM provider ("openai" or "anthropic")
            model (str): Model name (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-sonnet")
            api_key (str): API key for the LLM provider
            base_url (str): Base URL for API (for OpenRouter or custom endpoints)
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url or self._get_base_url()
        
        # Set default models (OpenRouter format)
        if not self.model:
            if self.provider == "openai":
                self.model = "openai/gpt-4o-mini"  # OpenRouter format
            elif self.provider == "anthropic":
                self.model = "anthropic/claude-3-sonnet-20240229"  # OpenRouter format
        
        # Initialize client
        self.client = self._initialize_client()
        
        # Available tools mapping
        self.tools_mapping = {
            "read_file": read_file,
            "list_files": list_files,
            "write_file": write_file,
            "search_in_file": search_in_file
        }
        
        logger.info(f"Initialized LLM File Assistant with {self.provider} - {self.model}")
        if self.base_url:
            logger.info(f"Using custom base URL: {self.base_url}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return None
    
    def _get_base_url(self) -> Optional[str]:
        """Get base URL from environment variables."""
        if self.provider == "openai":
            return os.getenv("OPENAI_BASE_URL")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_BASE_URL")
        return None
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            
            # Initialize with base_url if provided (for OpenRouter)
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            return openai.OpenAI(**client_kwargs)
        
        elif self.provider == "anthropic":
            if not self.api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            
            # Note: Anthropic through OpenRouter still uses OpenAI client with different model names
            if self.base_url:
                return openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                return anthropic.Anthropic(api_key=self.api_key)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def process_query(self, user_query: str, context: Dict = None) -> Dict:
        """
        Process a natural language query and execute appropriate file operations.
        
        Args:
            user_query (str): User's natural language query
            context (dict): Optional context information
            
        Returns:
            dict: Response with results and executed actions
        """
        try:
            # Prepare system message
            system_message = self._get_system_message(context)
            
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query}
            ]
            
            # Get LLM response with function calling
            if self.provider == "openai":
                response = self._call_openai(messages)
            elif self.provider == "anthropic":
                response = self._call_anthropic(messages)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing query: {str(e)}",
                "response": "I apologize, but I encountered an error processing your request.",
                "function_calls": []
            }
    
    def _get_system_message(self, context: Dict = None) -> str:
        """Generate system message for the LLM."""
        base_message = """You are an intelligent file system assistant. You MUST complete full workflows in a single response.

AVAILABLE TOOLS:
- list_files(directory, extension): List files in directories with optional filtering
- read_file(filepath): Read and extract text from PDF, TXT, DOCX files
- write_file(filepath, content): Create or overwrite files with content
- search_in_file(filepath, keyword): Search for keywords in file content

CRITICAL: Execute complete workflows, not just single steps!

WORKFLOW EXAMPLES:
1. "Find files mentioning Python":
   → list_files(examples/resumes) 
   → search_in_file for each resume file with keyword "Python"
   → Summarize findings

2. "Read all resumes in resumes folder":
   → list_files(examples/resumes, .txt)
   → read_file for EACH resume found
   → Present all resume contents

3. "Create summary for john_doe_resume.txt":
   → read_file(examples/resumes/john_doe_resume.txt)
   → Analyze content and create summary
   → write_file(summary_john_doe.txt, summary_content)

4. "Find PDF files":
   → list_files(examples, .pdf)
   → list_files(., .pdf) 
   → Show all PDF files found

EXECUTION RULES:
- Complete the ENTIRE task in one turn
- Use multiple function calls to finish workflows
- Don't stop after the first function call
- The examples/resumes directory contains sample resume files"""

        if context:
            base_message += f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"
        
        return base_message
    
    def _call_openai(self, messages: List[Dict]) -> Dict:
        """Call OpenAI API with function calling - supports multi-turn workflows."""
        try:
            all_function_calls = []
            max_iterations = 5  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[{"type": "function", "function": tool} for tool in TOOL_DEFINITIONS],
                    tool_choice="auto",
                    temperature=0.1
                )
                
                message = response.choices[0].message
                
                # If no tool calls, we're done
                if not message.tool_calls:
                    final_content = message.content
                    break
                
                # Execute function calls
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": message.tool_calls
                })
                
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the function
                    if function_name in self.tools_mapping:
                        result = self.tools_mapping[function_name](**function_args)
                        all_function_calls.append({
                            "function": function_name,
                            "arguments": function_args,
                            "result": result
                        })
                        
                        # Add function result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })
                
                # Continue the conversation to see if more function calls are needed
            
            # Get final response if we haven't already
            if iteration >= max_iterations:
                # Force a final response
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages + [{"role": "user", "content": "Please provide your final response based on the function calls executed."}],
                    temperature=0.1
                )
                final_content = final_response.choices[0].message.content
            
            return {
                "success": True,
                "response": final_content,
                "function_calls": all_function_calls,
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _call_anthropic(self, messages: List[Dict]) -> Dict:
        """Call Anthropic API with function calling."""
        # Note: This is a simplified implementation
        # Anthropic's function calling might have different syntax
        try:
            # Convert messages for Anthropic format
            system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            
            response = self.client.messages.create(
                model=self.model,
                system=system_msg,
                messages=user_messages,
                max_tokens=1000
            )
            
            # For now, return without function calling (would need to implement tool use)
            return {
                "success": True,
                "response": response.content[0].text,
                "function_calls": [],
                "model_used": self.model,
                "note": "Function calling not fully implemented for Anthropic"
            }
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    def chat_session(self):
        """Start an interactive chat session."""
        print("🤖 LLM File System Assistant")
        print("=" * 50)
        print("I can help you with file operations! Try commands like:")
        print("- 'Read all PDF files in the documents folder'")
        print("- 'Find files mentioning Python'")
        print("- 'Create a summary of resume_john.pdf'")
        print("- 'List all text files in the current directory'")
        print("\nType 'quit', 'exit', or 'bye' to end the session.")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Goodbye! Thanks for using the File System Assistant!")
                    break
                
                if not user_input:
                    continue
                
                print("\n🔄 Processing your request...")
                
                # Process the query
                result = self.process_query(user_input)
                
                if result["success"]:
                    print(f"\n🤖 Assistant: {result['response']}")
                    
                    # Show function calls if any
                    if result["function_calls"]:
                        print("\n📋 Actions performed:")
                        for i, call in enumerate(result["function_calls"], 1):
                            print(f"  {i}. {call['function']}({', '.join(f'{k}={v}' for k, v in call['arguments'].items())})")
                            
                            # Handle different function result formats
                            if call["function"] == "list_files":
                                files = call["result"]
                                if isinstance(files, list):
                                    print(f"     → Found {len(files)} files")
                                else:
                                    print(f"     → Error in list_files")
                            elif call["function"] == "read_file":
                                if call["result"].get("success"):
                                    size = call["result"]["metadata"]["size_bytes"] if call["result"]["metadata"] else 0
                                    print(f"     → Read file ({size} bytes)")
                                else:
                                    print(f"     → Error: {call['result'].get('error', 'Unknown error')}")
                            elif call["function"] == "search_in_file":
                                if call["result"].get("success"):
                                    matches = call["result"]["total_matches"]
                                    print(f"     → Found {matches} matches")
                                else:
                                    print(f"     → Error: {call['result'].get('error', 'Unknown error')}")
                            elif call["function"] == "write_file":
                                if call["result"].get("success"):
                                    print(f"     → File written successfully")
                                else:
                                    print(f"     → Error: {call['result'].get('error', 'Unknown error')}")
                else:
                    print(f"\n❌ Error: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                logger.error(f"Chat session error: {str(e)}")


def main():
    """Main function to demonstrate the LLM File Assistant."""
    print("🚀 Starting LLM File System Assistant")
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("\n⚠️  No API keys found!")
        print("Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        print("\nFor OpenRouter, set:")
        print("  OPENAI_API_KEY=your_openrouter_key")
        print("  OPENAI_BASE_URL=https://openrouter.ai/api/v1")
        return
    
    # Choose provider and show configuration
    provider = "openai" if openai_key else "anthropic"
    
    if openai_base_url:
        print(f"📡 Using OpenRouter endpoint: {openai_base_url}")
        print("🎯 Available models: openai/gpt-4o-mini, anthropic/claude-3-sonnet, etc.")
    
    try:
        # Initialize assistant
        assistant = LLMFileAssistant(provider=provider)
        
        # Start interactive session
        assistant.chat_session()
        
    except Exception as e:
        print(f"\n❌ Error initializing assistant: {str(e)}")
        logger.error(f"Main error: {str(e)}")


if __name__ == "__main__":
    main()