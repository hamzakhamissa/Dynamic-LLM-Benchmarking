# main_single_model.py - Simplified test with just one LLM
"""
Use this to test with just ONE working API model + random agents.
This helps debug the game logic without API issues.

Usage:
    python main_single_model.py openai
    python main_single_model.py claude  
    python main_single_model.py gemini
"""

import sys
from env import PyCatanEngine
from agents import LLMJsonAgent, RandomAgent
from orchestrator import GameOrchestrator
from metrics import compute_win_rates, hallucination_stats, average_turns
from llm_clients import openai_chat_fn, claude_chat_fn, gemini_chat_fn


def main():
    # Choose which model to test
    if len(sys.argv) > 1:
        model_choice = sys.argv[1].lower()
    else:
        print("Usage: python main_single_model.py [openai|claude|gemini]")
        print("\nDefaulting to OpenAI...")
        model_choice = "openai"
    
    # Select the chat function
    if model_choice == "openai":
        llm_name = "OpenAI_gpt-5-nano"
        chat_fn = openai_chat_fn
    elif model_choice == "claude":
        llm_name = "Claude_Haiku_4.5"
        chat_fn = claude_chat_fn
    elif model_choice == "gemini":
        llm_name = "Gemini_2.5_Flash"
        chat_fn = gemini_chat_fn
    else:
        print(f"Unknown model: {model_choice}")
        print("Choose: openai, claude, or gemini")
        return
    
    # Configuration
    N_GAMES = 3
    TARGET_VP = 8
    MAX_TURNS = 150
    
    print("=" * 70)
    print("🎮 SIMPLIFIED CATAN TEST")
    print("=" * 70)
    print(f"\n🤖 Testing: {llm_name}")
    print(f"📊 Games: {N_GAMES}")
    print(f"🎯 Target VP: {TARGET_VP}")
    print(f"⏱️  Max turns: {MAX_TURNS}")
    print("\n👥 Players:")
    print(f"  Player 0: {llm_name}")
    print(f"  Player 1: Random Agent")
    print(f"  Player 2: Random Agent")
    print(f"  Player 3: Random Agent")
    print("=" * 70 + "\n")
    
    # Build agents: 1 LLM + 3 random
    agents = [
        LLMJsonAgent(llm_name, chat_fn),
        RandomAgent("Random_1", seed=42),
        RandomAgent("Random_2", seed=43),
        RandomAgent("Random_3", seed=44),
    ]
    
    # Setup engine
    engine = PyCatanEngine(
        num_players=4,
        target_vp=TARGET_VP,
        max_turns=MAX_TURNS,
        seed=42,
    )
    
    orchestrator = GameOrchestrator(engine, agents)
    
    # Run games
    print("🎲 Starting games...\n")
    results = orchestrator.play_many_games(n_games=N_GAMES)
    
    # Basic metrics
    avg_t = average_turns(results)
    win_rates = compute_win_rates(results)
    hall = hallucination_stats(results)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n📈 Average turns: {avg_t:.1f}")
    
    print("\n🏆 Win Rates:")
    for i, rate in win_rates.items():
        agent_name = agents[i].name
        print(f"  {agent_name:<25} {rate:>6.1%}")
    
    print(f"\n🧠 {llm_name} Hallucinations:")
    llm_stats = hall[0]  # Player 0 is the LLM
    print(f"  Total decisions:      {llm_stats['decisions']:>6.0f}")
    print(f"  Index errors:         {llm_stats['index_hallucinations']:>6.0f}")
    print(f"  Action failures:      {llm_stats['action_failures']:>6.0f}")
    print(f"  API errors:           {llm_stats['api_errors']:>6.0f}")
    print(f"  Total hallucinations: {llm_stats['total_hallucinations']:>6.0f}")
    print(f"  Hallucination rate:   {llm_stats['hallucination_rate']:>6.1%}")
    
    print("\n" + "=" * 70)
    
    if llm_stats['api_errors'] > llm_stats['decisions'] * 0.5:
        print("⚠️  WARNING: >50% API errors detected!")
        print("   Check your API key and rate limits.")
    elif llm_stats['hallucination_rate'] < 0.1:
        print("✅ SUCCESS: Hallucination rate < 10%")
        print("   This model is working well!")
    else:
        print("⚠️  Hallucination rate is high")
        print("   Consider adjusting prompts or model settings")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()