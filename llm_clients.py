# llm_clients.py
from __future__ import annotations

from typing import List, Dict
import json

import secrets  # your own secrets.py (NOT committed to git)


def openai_chat_fn(messages: List[Dict[str, str]]) -> str:
    
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    try:
        client = OpenAI(
            api_key=secrets.OPENAI_API_KEY,
            timeout=30.0,  # overall client timeout
        )

        resp = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=100,   # new-style param; NOT max_tokens
            # IMPORTANT: per-request timeout so it can't hang forever
            timeout=10.0,                # seconds
        )

        text = resp.choices[0].message.content
        return text if text else '{"action_index": 0}'
    except Exception as e:
        print(f"⚠️ OpenAI API error: {e}")
        # Safe fallback so the game keeps going even if the API dies
        return '{"action_index": 0}'

def claude_chat_fn(messages: List[Dict[str, str]]) -> str:
    """
    Claude -> Claude Haiku 4.5 (CHEAPEST Claude model - great for testing!)
    
    Cost: $0.80 per 1M input tokens, $4.00 per 1M output tokens
    (4x cheaper than Sonnet 4 for input!)
    """
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("pip install anthropic") from e

    client = anthropic.Anthropic(api_key=secrets.ANTHROPIC_API_KEY)

    # Separate system prompt and user/assistant messages
    system_parts: List[str] = []
    converted_messages: List[Dict[str, str]] = []

    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            converted_messages.append({"role": role, "content": content})

    system_prompt = "\n\n".join(system_parts) if system_parts else None

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Haiku 4.5 - cheapest!
            max_tokens=100,
            temperature=0.7,
            system=system_prompt,
            messages=converted_messages,
            timeout=30.0,
        )

        # Anthropic Python SDK: resp.content is a list of blocks
        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "".join(text_parts).strip()
        
        return text if text else '{"action_index": 0}'
    except Exception as e:
        print(f"⚠️ Claude API error: {e}")
        return '{"action_index": 0}'


def gemini_chat_fn(messages: List[Dict[str, str]]) -> str:
    """
    Gemini -> Gemini 2.5 
    Returns a JSON string like {"action_index": 3}.
    
    NOTE: This model can timeout. We handle it gracefully.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("pip install google-generativeai") from e

    try:
        genai.configure(api_key=secrets.GEMINI_API_KEY)

        # Use experimental model with better JSON support
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.3,
                "max_output_tokens": 100,  # Keep it short
            }
        )

        # Combine all messages into a single prompt
        combined = []
        for m in messages:
            role = m.get("role", "").upper()
            content = m.get("content", "")
            if role == "SYSTEM":
                combined.append(f"INSTRUCTIONS: {content}\n")
            else:
                combined.append(f"{content}\n")
        prompt = "".join(combined)

        # Add explicit JSON formatting instruction
        prompt += "\n\nYou MUST respond with ONLY a JSON object in this exact format: {\"action_index\": <number>}"

        resp = model.generate_content(
            prompt,
            request_options={"timeout": 10},  # Short timeout to fail fast
        )

        # Robust text extraction
        text = None
        
        # Try multiple extraction methods
        try:
            # Method 1: Direct text attribute
            if hasattr(resp, 'text') and resp.text:
                text = resp.text.strip()
        except Exception:
            pass
        
        if not text:
            try:
                # Method 2: Via candidates
                if hasattr(resp, 'candidates') and resp.candidates:
                    cand = resp.candidates[0]
                    if hasattr(cand, 'content') and cand.content.parts:
                        parts_text = []
                        for part in cand.content.parts:
                            if hasattr(part, 'text') and part.text:
                                parts_text.append(part.text)
                        if parts_text:
                            text = "".join(parts_text).strip()
            except Exception:
                pass
        
        if not text:
            print("⚠️ Gemini returned empty response; using fallback")
            return '{"action_index": 0}'

        # Validate it's actually JSON
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            match = re.search(r'\{[^}]*"action_index"[^}]*\}', text)
            if match:
                return match.group(0)
            else:
                print(f"⚠️ Gemini returned non-JSON: {text[:100]}")
                return '{"action_index": 0}'

    except Exception as e:
        # Log the specific error type
        error_msg = str(e)
        if "504" in error_msg or "timeout" in error_msg.lower():
            print(f"⚠️ Gemini TIMEOUT: {error_msg[:100]}")
        else:
            print(f"⚠️ Gemini API error: {error_msg[:100]}")
        return '{"action_index": 0}'


def grok_chat_fn(messages: List[Dict[str, str]]) -> str:
    """
    Grok -> grok-4-fast-reasoning via xAI SDK.
    """
    try:
        from xai_sdk import Client
        from xai_sdk.chat import user as x_user, system as x_system
    except ImportError as e:
        raise RuntimeError("pip install xai-sdk") from e

    try:
        # Much shorter timeout: 30 seconds max
        client = Client(api_key=secrets.GROK_API_KEY, timeout=30)

        chat = client.chat.create(model="grok-4-fast-reasoning")

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                chat.append(x_system(content))
            else:
                chat.append(x_user(content))

        response = chat.sample()
        text = str(response.content).strip()
        return text if text else '{"action_index": 0}'
    except Exception as e:
        print(f"⚠️ Grok API error (timeout or network issue): {e}")
        return '{"action_index": 0}'