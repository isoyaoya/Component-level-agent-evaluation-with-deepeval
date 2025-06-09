from typing import List, Dict, Any, Tuple
from deepeval.test_case import ToolCall

def extract_tool_calls_from_strands(agent_messages: List[Dict[str, Any]]) -> Tuple[List[ToolCall], str]:
    """
    Extract tool calls from Strands agent messages and convert them to deepeval ToolCall format.
    Also extracts the final text response.
    """
    tool_calls = []
    tool_use_map = {}  # Map to store tool uses by toolUseId
    
    # First pass: collect all tool uses and their results
    for message in agent_messages:
        for content in message.get("content", []):
            if isinstance(content, dict):
                # Handle tool use
                if "toolUse" in content:
                    tool_use = content["toolUse"]
                    tool_use_id = tool_use["toolUseId"]
                    tool_use_map[tool_use_id] = {
                        "name": tool_use["name"],
                        "input": tool_use["input"],
                        "output": None
                    }
                
                # Handle tool result
                if "toolResult" in content:
                    tool_result = content["toolResult"]  # Fix: access toolResult directly from content
                    tool_use_id = tool_result["toolUseId"]
                    if tool_use_id in tool_use_map:
                        tool_use_map[tool_use_id]["output"] = tool_result["content"]

    # Convert to ToolCall objects
    for tool_data in tool_use_map.values():
        if tool_data["output"] is not None:  # Only include completed tool calls
            # Convert input parameters to a dictionary if it's a string
            input_params = tool_data["input"]
            if isinstance(input_params, str):
                input_params = {"input": input_params}
                
            # Ensure output is always a list of strings
            output = tool_data["output"]
            if isinstance(output, str):
                output = [output]
            elif output is None:
                output = []
            elif isinstance(output, list):
                # Make sure all items in the list are strings
                output = [str(item) if not isinstance(item, str) else item for item in output]

            try:
                # Ensure input_parameters is a dictionary
                if not isinstance(input_params, dict):
                    input_params = {"input": str(input_params)}
                
                # Ensure output is a list of strings
                if not isinstance(output, list):
                    output = [str(output)] if output is not None else []
                else:
                    output = [str(item) for item in output]
                
                # Create a proper ToolCall instance as a pydantic model
                tool_call = ToolCall(
                    name=str(tool_data["name"]),
                    description=f"Tool used in the conversation: {tool_data['name']}",
                    input_parameters=input_params,
                    output=output,
                    reasoning=None  # Optional field from the pydantic model
                )
                # Validate that it's a proper ToolCall instance
                if not isinstance(tool_call, ToolCall):
                    raise TypeError(f"Failed to create proper ToolCall instance: {tool_call}")
                print(f"Created ToolCall: {tool_call.__dict__}")  # Debug print
                tool_calls.append(tool_call)
            except Exception as e:
                print(f"Error creating ToolCall for {tool_data['name']}: {str(e)}")
                continue

    # Extract final text from the results
    final_text = ""
    if agent_messages:
        last_message = agent_messages[-1]
        if "content" in last_message and last_message["content"]:
            last_content = last_message["content"][0]
            if isinstance(last_content, dict) and "response" in last_content:
                final_text = str(last_content["response"])
            else:
                final_text = str(last_content)

    # Debug print to verify ToolCall objects
    print("Created tool calls:")
    for tc in tool_calls:
        print(f"  - {type(tc).__name__}: {tc.name}")

    return tool_calls, final_text
