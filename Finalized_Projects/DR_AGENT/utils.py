import os
import re
import json
import logging
import mimetypes
import base64
from typing import Dict, List, Any

from google.adk.tools import FunctionTool, ToolContext
from google.genai import types  # For creating message Content/Parts

logger = logging.getLogger(__name__)

# ==============================================================================
# --- Interactive & Placeholder Tools ---
# ==============================================================================

def ask_patient_for_clarification(question_text: str, tool_context: ToolContext) -> Dict[str, str]:
    """Simulates asking the patient a clarifying question."""
    logger.info(f"TOOL CALLED: ask_patient_for_clarification with question: '{question_text}'")
    tool_context.state['last_question'] = question_text
    return {"status": "success", "result": f"ACTION_REQUIRED: ASK_PATIENT -> '{question_text}'"}

def transcribe_audio(audio_content: str, tool_context: ToolContext) -> Dict[str, str]:
    """Placeholder for audio-to-text transcription."""
    logger.info("TOOL CALLED: transcribe_audio")
    return {"status": "success", "result": audio_content}

# ==============================================================================
# --- Register Tools for ADK ---
# ==============================================================================

clarification_tool = FunctionTool(func=ask_patient_for_clarification)
audio_tool = FunctionTool(func=transcribe_audio)

# ==============================================================================
# --- Agent Communication Helper ---
# ==============================================================================

async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:  # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            break  # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")

# ==============================================================================
# --- Artifact and Prompt Preparation Utilities ---
# ==============================================================================

def encode_file_for_prompt(filepath: str) -> Dict[str, Any]:
    """
    Reads a file, encodes it to base64, and returns a dictionary
    formatted for a multimodal prompt part.

    Args:
        filepath: The path to the file.

    Returns:
        A dictionary containing the base64 encoded data and its MIME type,
        or None if the file is not found.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}")
        return None

    # Guess the MIME type of the file
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type is None:
        # Default to a generic binary type if MIME type cannot be determined
        mime_type = "application/octet-stream"

    try:
        with open(filepath, "rb") as f:
            file_content = f.read()
            # Encode the binary content to a Base64 string
            encoded_content = base64.b64encode(file_content).decode('utf-8')
        
        return {
            "mime_type": mime_type,
            "data": encoded_content,
            "raw_bytes": file_content  # Also include raw bytes for ADK conversion
        }
    except Exception as e:
        print(f"Error encoding file {filepath}: {e}")
        return None


def create_adk_part_from_file(filepath: str) -> types.Part:
    """
    Reads a file and returns a proper ADK Part object with inline_data.

    Args:
        filepath: The path to the file.

    Returns:
        A types.Part object, or None if the file is not found.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}")
        return None

    # Guess the MIME type of the file
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type is None:
        # Default to a generic binary type if MIME type cannot be determined
        mime_type = "application/octet-stream"

    try:
        with open(filepath, "rb") as f:
            file_content = f.read()
        
        # Create proper ADK Part with inline_data using Blob
        return types.Part(
            inline_data=types.Blob(
                data=file_content,
                mime_type=mime_type
            )
        )
    except Exception as e:
        print(f"Error creating Part from file {filepath}: {e}")
        return None


def build_prompt_parts_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Iterates over all files in a given folder, encodes each one,
    and returns a list of prompt parts as dictionaries.

    Args:
        folder_path: The path to the folder containing patient files.

    Returns:
        A list of dictionaries, where each dictionary represents a file
        and contains base64 encoded data.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at {folder_path}")
        return []

    prompt_parts = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            encoded_file = encode_file_for_prompt(filepath)
            if encoded_file:
                prompt_parts.append(encoded_file)
    
    return prompt_parts


def build_adk_parts_from_folder(folder_path: str) -> List[types.Part]:
    """
    Iterates over all files in a given folder and returns a list of ADK Part objects.

    Args:
        folder_path: The path to the folder containing patient files.

    Returns:
        A list of types.Part objects ready for use with ADK as message parts.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at {folder_path}")
        return []

    parts = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            part = create_adk_part_from_file(filepath)
            if part:
                parts.append(part)
                logger.info(f"Created ADK Part from file: {filename} (MIME: {part.inline_data.mime_type})")
    
    return parts


def convert_dict_to_adk_parts(dict_artifacts: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Converts old-style dictionary artifacts to proper ADK Part objects.
    
    Args:
        dict_artifacts: List of dictionaries with 'data' (base64) and 'mime_type' keys
        
    Returns:
        List of types.Part objects
    """
    adk_parts = []
    
    for dict_artifact in dict_artifacts:
        if not dict_artifact or "data" not in dict_artifact or "mime_type" not in dict_artifact:
            continue
            
        try:
            # If raw_bytes is available, use it; otherwise decode base64
            if "raw_bytes" in dict_artifact:
                raw_bytes = dict_artifact["raw_bytes"]
            else:
                raw_bytes = base64.b64decode(dict_artifact["data"])
            
            # Create proper ADK Part
            part = types.Part(
                inline_data=types.Blob(
                    data=raw_bytes,
                    mime_type=dict_artifact["mime_type"]
                )
            )
            adk_parts.append(part)
            
        except Exception as e:
            logger.error(f"Failed to convert dictionary artifact to ADK Part: {e}")
            continue
    
    return adk_parts


# Backward compatibility aliases
def build_adk_artifacts_from_folder(folder_path: str) -> List[types.Part]:
    """
    Alias for build_adk_parts_from_folder for backward compatibility.
    Returns ADK Part objects (not Artifact objects as the name might suggest).
    """
    return build_adk_parts_from_folder(folder_path)