# test_apis.py - Simple API test
from llm_clients import openai_chat_fn, claude_chat_fn, gemini_chat_fn
import json


def test_api(name: str, chat_fn):
    """Test a single API."""
    print(f"\nTesting {name}...")
    
    messages = [
        {"role": "system", "content": "Respond with JSON only."},
        {"role": "user", "content": 'Pick a number 0-5. Respond: {"action_index": <number>}'}
    ]
    
    try:
        response = chat_fn(messages)
        print(f"  Response: {response[:100]}")
        
        parsed = json.loads(response)
        
        if "action_index" in parsed:
            idx = parsed["action_index"]
            if isinstance(idx, int) and 0 <= idx <= 5:
                print(f"  ✅ WORKING - Got valid index: {idx}")
                return True
            else:
                print(f"  ⚠️  Got index but invalid: {idx}")
                return False
        else:
            print(f"  ❌ FAILED - No action_index in response")
            return False
            
    except Exception as e:
        print(f"  ❌ FAILED - {str(e)[:100]}")
        return False


def main():
    print("\n" + "="*60)
    print("API TEST")
    print("="*60)
    
    results = {
        "OpenAI gpt-4o-mini": test_api("OpenAI", openai_chat_fn),
        "Claude Haiku 3.5": test_api("Claude", claude_chat_fn),
        "Gemini 1.5 Flash": test_api("Gemini", gemini_chat_fn),
    }
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    if all(results.values()):
        print("\n✅ All working! Run: python main.py")
    else:
        print("\n⚠️  Some APIs failed. Check your secrets.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()