# llm_clients.py
from __future__ import annotations

from typing import List, Dict
import json

import secrets  # your own secrets.py (NOT committed to git)


def local_stub_chat_fn(messages: List[Dict[str, str]]) -> str:
    """
    Fast, offline-safe chat function that always returns the first action index.

    This keeps the game loop moving without making any external API calls and
    avoids hanging when credentials or network access are unavailable.
    """
    return json.dumps({"action_index": 0})


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
        print(f"❌ OpenAI API error: {e}")
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
        print(f"❌ Claude API error: {e}")
        return '{"action_index": 0}'


def gemini_chat_fn(messages: List[Dict[str, str]]) -> str:
    """
    Gemini -> Gemini 2.5 Flash (cheapest Gemini model)
    Returns a JSON string like {"action_index": 3}.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("pip install google-generativeai") from e

    try:
        genai.configure(api_key=secrets.GEMINI_API_KEY)

        model = genai.GenerativeModel("gemini-2.5-flash")

        combined = []
        for m in messages:
            role = m.get("role", "").upper()
            content = m.get("content", "")
            combined.append(f"{role}: {content}\n")
        prompt = "".join(combined)

        resp = model.generate_content(
            prompt,
            generation_config={
                # Force JSON
                "response_mime_type": "application/json",
                "max_output_tokens": 64,
                "temperature": 0.3,
            },
            request_options={"timeout": 20},
        )

        # --- robust extraction ---
        text = None

        # Prefer structured parts
        if getattr(resp, "candidates", None):
            cand = resp.candidates[0]
            if getattr(cand, "content", None) and cand.content.parts:
                # Parts can be text or JSON; both support `.text` here
                parts = cand.content.parts
                buf = []
                for p in parts:
                    # some SDK versions use .text, some .function_call, etc
                    if hasattr(p, "text") and p.text:
                        buf.append(p.text)
                text = "".join(buf).strip() if buf else None

        # If we still have nothing, try `resp.text` as a best-effort
        if not text:
            try:
                text = resp.text.strip()
            except Exception:
                text = None

        if not text:
            print("❌ Gemini returned no usable text; falling back to default action.")
            return '{"action_index": 0}'

        # At this point `text` *should* be JSON, but we still guard in the agent.
        print(f"✅ Gemini raw text: {text[:80]}...")
        return text

    except Exception as e:
        print(f"❌ Gemini API error: {e}")
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
        client = Client(api_key=secrets.GROK_API_KEY, timeout=3600)

        # Use the model you requested
        chat = client.chat.create(model="grok-4-fast-reasoning")

        # Convert OpenAI-style messages to xAI-style chat calls
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                chat.append(x_system(content))
            else:
                chat.append(x_user(content))

        response = chat.sample()

        text = str(response.content).strip()
        # Ensure we always return something JSON-like
        return text if text else '{"action_index": 0}'
    except Exception as e:
        print(f"Grok API error: {e}")
        return '{"action_index": 0}'