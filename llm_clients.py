# llm_clients.py - Simplified and fixed
from __future__ import annotations

from typing import List, Dict
import json
import time

import secrets


def openai_chat_fn(messages: List[Dict[str, str]]) -> str:
    """OpenAI - simplified, no fancy params."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    try:
        client = OpenAI(api_key=secrets.OPENAI_API_KEY)
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # Changed to a model that definitely works
            messages=messages,
            max_tokens=100,
        )
        
        text = resp.choices[0].message.content
        if not text:
            return '{"action_index": 0}'
        
        # Try to extract JSON if it's wrapped in text
        try:
            json.loads(text)
            return text
        except:
            import re
            match = re.search(r'\{[^}]*"action_index"[^}]*\}', text)
            if match:
                return match.group(0)
            return '{"action_index": 0}'
            
    except Exception as e:
        print(f"⚠️ OpenAI error: {e}")
        return '{"action_index": 0}'


def claude_chat_fn(messages: List[Dict[str, str]]) -> str:
    """Claude - simplified."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("pip install anthropic") from e

    try:
        client = anthropic.Anthropic(api_key=secrets.ANTHROPIC_API_KEY)
        
        # Separate system from other messages
        system_content = None
        user_messages = []
        
        for m in messages:
            if m.get("role") == "system":
                system_content = m.get("content", "")
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})
        
        # Make API call
        kwargs = {
            "model": "claude-3-5-haiku-20241022",  # Using Haiku 3.5 which is more reliable
            "max_tokens": 100,
            "messages": user_messages,
        }
        
        if system_content:
            kwargs["system"] = system_content
        
        resp = client.messages.create(**kwargs)
        
        # Extract text
        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text
        
        if not text:
            return '{"action_index": 0}'
        
        # Try to parse or extract JSON
        try:
            json.loads(text)
            return text
        except:
            import re
            match = re.search(r'\{[^}]*"action_index"[^}]*\}', text)
            if match:
                return match.group(0)
            return '{"action_index": 0}'
            
    except Exception as e:
        print(f"⚠️ Claude error: {e}")
        return '{"action_index": 0}'


def gemini_chat_fn(messages: List[Dict[str, str]]) -> str:
    """Gemini - simplified."""
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("pip install google-generativeai") from e

    try:
        genai.configure(api_key=secrets.GEMINI_API_KEY)
        
        model = genai.GenerativeModel("gemini-2.5-flash")  # Using 1.5 which is stable
        
        # Combine messages into one prompt
        prompt = ""
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                prompt += f"{content}\n\n"
            elif role == "user":
                prompt += f"{content}\n"
        
        resp = model.generate_content(prompt)
        
        text = resp.text if hasattr(resp, 'text') else ""
        
        if not text:
            return '{"action_index": 0}'
        
        # Try to parse or extract JSON
        try:
            json.loads(text)
            return text
        except:
            import re
            match = re.search(r'\{[^}]*"action_index"[^}]*\}', text)
            if match:
                return match.group(0)
            # Sometimes Gemini just returns a number
            match = re.search(r'"action_index"\s*:\s*(\d+)', text)
            if match:
                return f'{{"action_index": {match.group(1)}}}'
            return '{"action_index": 0}'
            
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        return '{"action_index": 0}'