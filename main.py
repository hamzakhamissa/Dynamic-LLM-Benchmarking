# main.py - Safe Testing Version with Claude Haiku + GPT-4o-mini
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Dict, Any

from env import PyCatanEngine
from agents import LLMJsonAgent, RandomAgent
from orchestrator import GameOrchestrator
from metrics import (
    compute_win_rates,
    hallucination_stats,
    trade_behavior,
    overall_scores,
    average_turns,
)
from llm_clients import (
    openai_chat_fn,
    claude_chat_fn,
    # gemini_chat_fn,
    grok_chat_fn
)


def build_agents():
    """
    3 LLM agents (OpenAI, Claude, Grok) + 1 random baseline.

    Player 0: OpenAI (gpt-5-nano)
    Player 1: Claude (Haiku 4.5)
    Player 2: Grok (4-fast-reasoning)
    Player 3: Random baseline
    """
    return [
        LLMJsonAgent("OpenAI_gpt-5-nano", openai_chat_fn),    
        LLMJsonAgent("Claude_Haiku_4.5", claude_chat_fn),            
        LLMJsonAgent("Grok_4_fast_reasoning", grok_chat_fn),         
        RandomAgent("Random_baseline"),
    ]


def save_results_to_json(results, metrics, filename=None):
    """
    Save game results and metrics to a JSON file for later analysis.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_results_{timestamp}.json"
    
    # Convert results to serializable format
    serializable_results = []
    for game_idx, game in enumerate(results):
        game_data = {
            "game_index": game_idx,
            "winner_index": game.winner_index,
            "final_turn": game.final_state.turn,
            "final_victory_points": game.final_state.victory_points,
            "final_resources": game.final_state.resources,
            "steps": []
        }
        
        for step in game.steps:
            step_data = {
                "turn": step.state_before.turn,
                "acting_player": step.acting_player_index,
                "action_type": step.action.type.name,
                "action_payload": str(step.action.payload),  # Convert to string for serialization
                "legal_actions_count": step.legal_actions_count,
                "victory_points_before": step.state_before.victory_points,
                "victory_points_after": step.state_after.victory_points,
                "resources_before": step.state_before.resources,
                "resources_after": step.state_after.resources,
                "info": {
                    k: v for k, v in step.info.items() 
                    if k not in ['raw_llm_response']  # Exclude raw responses to keep file smaller
                }
            }
            
            # Add LLM-specific info if present
            if 'llm_valid_index' in step.info:
                step_data['llm_valid_index'] = step.info['llm_valid_index']
                step_data['llm_used_fallback'] = step.info['llm_used_fallback']
            
            game_data["steps"].append(step_data)
        
        serializable_results.append(game_data)
    
    # Combine with metrics
    output = {
        "timestamp": datetime.now().isoformat(),
        "num_games": len(results),
        "metrics": metrics,
        "games": serializable_results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {filename}")
    return filename


def main():
    print("=" * 60)
    print("4 LLM AGENTS: OpenAI / Claude / Gemini / Grok")
    print("=" * 60)
    print("Models:")
    print("  - OpenAI : gpt-5-nano")
    print("  - Claude : claude-haiku-4-5-20251001")
    print("  - Gemini : gemini-2.5-flash")
    print("  - Grok   : grok-4-fast-reasoning")
    print(f"Games: 1")
    print(f"Max turns: 50")
    print("=" * 60 + "\n")
    
    engine = PyCatanEngine(
        num_players=4,
        target_vp=2,
        max_turns=50,  # Reduce to 50 for faster testing
        seed=42,
    )

    agents = build_agents()
    orchestrator = GameOrchestrator(engine, agents)

    # Just 1 game for initial testing
    n_games = 2
    
    print("Starting game...\n")
    results = orchestrator.play_many_games(n_games=n_games)
    
    print(f"\n✅ Played {n_games} game(s).")
    print(f"Average turns: {average_turns(results):.1f}")

    # Calculate metrics
    win_rates = compute_win_rates(results)
    hall = hallucination_stats(results)
    trades = trade_behavior(results)
    scores = overall_scores(results)

    # Prepare metrics dict for JSON
    metrics = {
        "average_turns": average_turns(results),
        "win_rates": {f"player_{i}": rate for i, rate in win_rates.items()},
        "hallucination_stats": {f"player_{i}": st for i, st in hall.items()},
        "trade_behavior": {f"player_{i}": st for i, st in trades.items()},
        "overall_scores": {f"player_{i}": st for i, st in scores.items()},
    }

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\n=== Win rates ===")
    agent_names = ["GPT-4o-mini", "Random_1", "Random_2", "Random_3"]
    for i, rate in win_rates.items():
        print(f"{agent_names[i]:<20} (Player {i}): {rate:.3f}")

    print("\n=== Hallucination stats ===")
    for i, st in hall.items():
        print(f"{agent_names[i]:<20} (Player {i}): {st}")

    print("\n=== Trade behaviour ===")
    for i, st in trades.items():
        print(f"{agent_names[i]:<20} (Player {i}):")
        print(f"  - Trades: {st['num_trades']:.0f}")
        print(f"  - Helping leader: {st['trades_helping_leader_ratio']:.2%}")
        print(f"  - Helping last: {st['trades_helping_last_ratio']:.2%}")
        print(f"  - Selfish ratio: {st['selfish_trade_ratio']:.2%}")

    print("\n=== Overall scores ===")
    for i, st in scores.items():
        print(f"{agent_names[i]:<20} (Player {i}):")
        print(f"  - Win rate: {st['win_rate']:.3f}")
        print(f"  - Overall score: {st['overall_score']:.3f}")
        print(f"  - Game sense: {st['game_sense']:.3f}")

    # Save to JSON
    print("\n" + "="*60)
    save_results_to_json(results, metrics)
    print("="*60)
    
    print("\n💡 NEXT STEPS:")
    print("   1. ✅ Test worked? Increase n_games to 3-5")
    print("   2. 🤖 Add more LLM agents? Try 2 GPT agents")
    print("   3. 📊 Different models? Add Claude/Gemini later")
    print("   4. ⏱️  Games too short? Increase max_turns to 100")
    print("   5. 📁 Analyze results in the generated JSON file")


if __name__ == "__main__":
    main()