# main.py - Enhanced benchmark with better tracking
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
    resource_efficiency,
    overall_scores,
    average_turns,
)
from llm_clients import (
    openai_chat_fn,
    claude_chat_fn,
    gemini_chat_fn,
    # grok_chat_fn
)


def build_agents():
    """
    3 LLM agents (OpenAI, Claude, Gemini) + 1 random baseline.

    Player 0: OpenAI (gpt-5-nano)
    Player 1: Claude (Haiku 4.5)
    Player 2: Gemini (2.5 Flash)
    Player 3: Random baseline
    
    NOTE: Grok removed due to API instability/timeouts
    """
    return [
        LLMJsonAgent("OpenAI_gpt-5-nano", openai_chat_fn),
        LLMJsonAgent("Claude_Haiku_4.5", claude_chat_fn),
        LLMJsonAgent("Gemini_2.5_Flash", gemini_chat_fn),
        RandomAgent("Random_baseline", seed=42),
        # LLMJsonAgent("Grok-4-fast-reasoning", grok_chat_fn)
    ]


def save_results_to_json(results, metrics, filename=None):
    """
    Save game results and metrics to a JSON file for later analysis.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_results_{timestamp}.json"

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
                "action_payload": str(step.action.payload),
                "legal_actions_count": step.legal_actions_count,
                "victory_points_before": step.state_before.victory_points,
                "victory_points_after": step.state_after.victory_points,
                "resources_before": step.state_before.resources,
                "resources_after": step.state_after.resources,
                "info": {
                    k: v for k, v in step.info.items()
                    if k not in ["raw_llm_response"]
                },
            }

            if "llm_valid_index" in step.info:
                step_data["llm_valid_index"] = step.info["llm_valid_index"]
                step_data["llm_used_fallback"] = step.info["llm_used_fallback"]
            
            if "action_failed" in step.info:
                step_data["action_failed"] = step.info["action_failed"]

            game_data["steps"].append(step_data)

        serializable_results.append(game_data)

    output = {
        "timestamp": datetime.now().isoformat(),
        "num_games": len(results),
        "metrics": metrics,
        "games": serializable_results,
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\ Results saved to: {filename}")
    return filename


def main():
    # Config
    n_games = 1  # Start with 5 for testing
    target_vp = 8  # Lower from 10 to ensure wins (games finishing in 50-70 turns)
    max_turns = 150  # Increased safety net

    print("=" * 60)
    print("DYNAMIC CATAN BENCHMARK v2.0")
    print("=" * 60)
    print("Models:")
    print("  - OpenAI : gpt-5-nano")
    print("  - Claude : claude-haiku-4-5-20251001")
    print("  - Gemini : gemini-2.5-flash")
    print("  - Baseline: Random_agent")
    print(f"\nGame Settings:")
    print(f"  - Games: {n_games}")
    print(f"  - Victory Points: {target_vp}")
    print(f"  - Max turns: {max_turns}")
    print("=" * 60 + "\n")

    engine = PyCatanEngine(
        num_players=4,
        target_vp=target_vp,
        max_turns=max_turns,
        seed=42,
    )

    agents = build_agents()
    orchestrator = GameOrchestrator(engine, agents)

    print("🎮 Starting games...\n")
    results = orchestrator.play_many_games(n_games=n_games)

    avg_t = average_turns(results)
    print(f"\n{'='*60}")
    print(f"Completed {n_games} game(s)!")
    print(f"Average turns per game: {avg_t:.1f}")
    print(f"{'='*60}\n")

    # Compute metrics
    win_rates = compute_win_rates(results)
    hall = hallucination_stats(results)
    trades = trade_behavior(results)
    efficiency = resource_efficiency(results)
    scores = overall_scores(results)

    metrics = {
        "average_turns": avg_t,
        "win_rates": {f"player_{i}": rate for i, rate in win_rates.items()},
        "hallucination_stats": {f"player_{i}": st for i, st in hall.items()},
        "trade_behavior": {f"player_{i}": st for i, st in trades.items()},
        "resource_efficiency": {f"player_{i}": st for i, st in efficiency.items()},
        "overall_scores": {f"player_{i}": st for i, st in scores.items()},
    }

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    agent_names = [a.name for a in agents]

    print("\n===  Win Rates ===")
    for i, rate in win_rates.items():
        print(f"{agent_names[i]:<24} (Player {i}): {rate:.1%}")

    print("\n===  Hallucination Stats ===")
    for i, st in hall.items():
        print(f"{agent_names[i]:<24} (Player {i}):")
        print(f"  Total decisions: {st['decisions']:.0f}")
        print(f"  Index errors: {st['index_hallucinations']:.0f}")
        print(f"  Action failures: {st['action_failures']:.0f}")
        print(f"  Hallucination rate: {st['hallucination_rate']:.1%}")
        print(f"  Penalty score: {st['hallucination_penalty_score']:.3f}")

    print("\n===  Trade Behavior ===")
    for i, st in trades.items():
        print(f"{agent_names[i]:<24} (Player {i}):")
        print(f"  Player trades: {st['num_player_trades']:.0f}")
        print(f"  Bank trades: {st['num_bank_trades']:.0f}")
        print(f"  Total trades: {st['total_trades']:.0f}")
        print(f"  Helping leader: {st['trades_helping_leader_ratio']:.1%}")
        print(f"  Helping last: {st['trades_helping_last_ratio']:.1%}")

    print("\n===  Resource Efficiency ===")
    for i, st in efficiency.items():
        print(f"{agent_names[i]:<24} (Player {i}):")
        print(f"  Build rate: {st['build_rate']:.3f} builds/turn")
        print(f"  Avg final resources: {st['avg_final_resources']:.1f}")
        print(f"  Efficiency score: {st['efficiency_score']:.3f}")

    print("\n=== Overall Scores ===")
    for i, st in scores.items():
        print(f"{agent_names[i]:<24} (Player {i}):")
        print(f"  Win rate: {st['win_rate']:.1%}")
        print(f"  Overall score: {st['overall_score']:.3f}")
        print(f"  Game sense: {st['game_sense']:.3f}")
        print(f"  Trade activity: {st['trade_activity']:.3f}")

    print("\n" + "=" * 60)
    save_results_to_json(results, metrics)
    print("=" * 60)


if __name__ == "__main__":
    main()