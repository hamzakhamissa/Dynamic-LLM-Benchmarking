# main.py - Safe Testing Version with Claude Haiku + GPT-4o-mini
from __future__ import annotations

import json
import os
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
    grok_chat_fn,
    local_stub_chat_fn,
)


def build_agents(use_remote_llms: bool = False):
    """
    Build either real LLM agents (requires credentials/network) or fast offline agents.

    When `use_remote_llms` is False (default), we avoid any external API calls to
    keep local runs quick and reliable. The first player uses a deterministic stub
    chat function to exercise the LLM parsing logic without leaving the machine;
    the remaining three players are random baselines.
    """
    if use_remote_llms:
        return [
            LLMJsonAgent("OpenAI_gpt-5-nano", openai_chat_fn),
            LLMJsonAgent("Claude_Haiku_4.5", claude_chat_fn),
            LLMJsonAgent("Grok_4_fast_reasoning", grok_chat_fn),
            RandomAgent("Random_baseline"),
        ]

    return [
        LLMJsonAgent("Stub_LLM_Player", local_stub_chat_fn),
        RandomAgent("Random_1"),
        RandomAgent("Random_2"),
        RandomAgent("Random_3"),
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
    use_remote_llms = os.getenv("USE_REMOTE_LLMS", "0") == "1"
    n_games = int(os.getenv("N_GAMES", "1"))

    print("=" * 60)
    if use_remote_llms:
        print("4 LLM AGENTS: OpenAI / Claude / Gemini / Grok")
    else:
        print("4 OFFLINE AGENTS: Stub LLM + 3 Random baselines")
    print("=" * 60)
    print("Models:")
    if use_remote_llms:
        print("  - OpenAI : gpt-5-nano")
        print("  - Claude : claude-haiku-4-5-20251001")
        print("  - Gemini : gemini-2.5-flash")
        print("  - Grok   : grok-4-fast-reasoning")
    else:
        print("  - Stub   : local JSON stub (always index 0)")
        print("  - Random : three random baselines")
    print(f"Games: {n_games}")
    print(f"Max turns: 50")
    print("=" * 60 + "\n")
    
    engine = PyCatanEngine(
        num_players=4,
        target_vp=2,
        max_turns=50,  # Reduce to 50 for faster testing
        seed=42,
    )

    agents = build_agents(use_remote_llms)
    orchestrator = GameOrchestrator(engine, agents)
    agent_names = [a.name for a in agents]
    # Just 1 game for initial testing (override via N_GAMES)
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
