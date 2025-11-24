# main.py - Enhanced benchmark with better configuration
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
)


def build_agents():
    """
    3 LLM agents + 1 random baseline.
    
    Player 0: OpenAI (gpt-5-nano)
    Player 1: Claude (Haiku 4.5) - Cheapest Claude model
    Player 2: Gemini (2.5 Flash)
    Player 3: Random baseline
    """
    return [
        LLMJsonAgent("OpenAI_gpt-5-nano", openai_chat_fn),
        LLMJsonAgent("Claude_Haiku_4.5", claude_chat_fn),
        LLMJsonAgent("Gemini_2.5_Flash", gemini_chat_fn),
        RandomAgent("Random_baseline", seed=42),
    ]


def save_results_to_json(results, metrics, filename=None):
    """Save results to JSON for analysis."""
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
                "info": {
                    k: v for k, v in step.info.items()
                    if k not in ["raw_llm_response"]  # Exclude large text
                },
            }

            # Add LLM-specific tracking
            if "llm_valid_index" in step.info:
                step_data["llm_valid_index"] = step.info["llm_valid_index"]
                step_data["llm_used_fallback"] = step.info["llm_used_fallback"]
                step_data["llm_api_error"] = step.info.get("llm_api_error", False)
            
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

    print(f"\n💾 Results saved to: {filename}")
    return filename


def print_results(results, agents):
    """Print formatted results."""
    avg_t = average_turns(results)
    win_rates = compute_win_rates(results)
    hall = hallucination_stats(results)
    trades = trade_behavior(results)
    efficiency = resource_efficiency(results)
    scores = overall_scores(results)

    agent_names = [a.name for a in agents]

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n📊 Games Completed: {len(results)}")
    print(f"📈 Average Turns: {avg_t:.1f}")

    print("\n🏆 WIN RATES")
    print("-" * 70)
    for i, rate in sorted(win_rates.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(rate * 50)
        print(f"{agent_names[i]:<25} {rate:>6.1%} {bar}")

    print("\n🧠 HALLUCINATION ANALYSIS")
    print("-" * 70)
    for i, st in hall.items():
        print(f"\n{agent_names[i]} (Player {i}):")
        print(f"  Total decisions:     {st['decisions']:>6.0f}")
        print(f"  Index errors:        {st['index_hallucinations']:>6.0f}")
        print(f"  Action failures:     {st['action_failures']:>6.0f}")
        print(f"  Total hallucinations: {st['total_hallucinations']:>6.0f}")
        print(f"  Hallucination rate:  {st['hallucination_rate']:>6.1%}")
        print(f"  Penalty score:       {st['hallucination_penalty_score']:>6.3f}")

    print("\n💰 TRADE BEHAVIOR")
    print("-" * 70)
    for i, st in trades.items():
        print(f"\n{agent_names[i]} (Player {i}):")
        print(f"  Player trades:       {st['num_player_trades']:>6.0f}")
        print(f"  Bank trades:         {st['num_bank_trades']:>6.0f}")
        print(f"  Total trades:        {st['total_trades']:>6.0f}")
        if st['total_trades'] > 0:
            print(f"  Helping leader:      {st['trades_helping_leader_ratio']:>6.1%}")
            print(f"  Helping last place:  {st['trades_helping_last_ratio']:>6.1%}")

    print("\n⚡ RESOURCE EFFICIENCY")
    print("-" * 70)
    for i, st in efficiency.items():
        print(f"\n{agent_names[i]} (Player {i}):")
        print(f"  Build rate:          {st['build_rate']:>6.3f} builds/turn")
        print(f"  Avg final resources: {st['avg_final_resources']:>6.1f} cards")
        print(f"  Efficiency score:    {st['efficiency_score']:>6.3f}")

    print("\n🎯 OVERALL SCORES")
    print("-" * 70)
    print("Formula: win_rate × hallucination_penalty × (0.5 + 0.3×trade_activity + 0.2×game_sense)")
    print("  • Base: 50% credit for winning")
    print("  • Trade activity: +30% for active trading (max 15 trades)")
    print("  • Game sense: +20% for strategic play (help weak, avoid helping leader)")
    print()
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    for i, st in sorted_scores:
        print(f"\n{agent_names[i]} (Player {i}):")
        print(f"  Win rate:            {st['win_rate']:>6.1%}")
        print(f"  Overall score:       {st['overall_score']:>6.3f} ⭐")
        print(f"  Game sense:          {st['game_sense']:>6.3f}")
        print(f"  Trade activity:      {st['trade_activity']:>6.3f}")

    print("\n" + "=" * 70)

    # Return metrics for saving
    return {
        "average_turns": avg_t,
        "win_rates": {f"player_{i}": rate for i, rate in win_rates.items()},
        "hallucination_stats": {f"player_{i}": st for i, st in hall.items()},
        "trade_behavior": {f"player_{i}": st for i, st in trades.items()},
        "resource_efficiency": {f"player_{i}": st for i, st in efficiency.items()},
        "overall_scores": {f"player_{i}": st for i, st in scores.items()},
    }


def main():
    # Configuration
    N_GAMES = 5  # Start with 5 games for testing
    TARGET_VP = 8  # Lower target for faster games
    MAX_TURNS = 150  # Safety net

    print("=" * 70)
    print("🎮 CATAN LLM BENCHMARK v2.1")
    print("=" * 70)
    print("\n📋 Configuration:")
    print(f"  Games:         {N_GAMES}")
    print(f"  Victory Points: {TARGET_VP}")
    print(f"  Max Turns:     {MAX_TURNS}")
    print("\n🤖 Models:")
    print("  • OpenAI gpt-5-nano")
    print("  • Claude Haiku 4.5 (cheapest)")
    print("  • Gemini 2.5 Flash")
    print("  • Random Baseline")
    print("\n✨ Features:")
    print("  ✓ Retry logic with exponential backoff")
    print("  ✓ Enhanced JSON parsing")
    print("  ✓ API error tracking")
    print("  ✓ Bank trades (4:1)")
    print("  ✓ Robber mechanic")
    print("  ✓ Discard phase")
    print("\n🔜 TODO (from README):")
    print("  • Port trades (3:1, 2:1)")
    print("  • Proper hex board")
    print("  • Dynamic scenarios")
    print("  • Strategy pivot detection")
    print("=" * 70)

    # Setup
    engine = PyCatanEngine(
        num_players=4,
        target_vp=TARGET_VP,
        max_turns=MAX_TURNS,
        seed=42,
    )

    agents = build_agents()
    orchestrator = GameOrchestrator(engine, agents)

    # Run games
    print("\n🎲 Starting games...\n")
    results = orchestrator.play_many_games(n_games=N_GAMES)

    # Display results
    metrics = print_results(results, agents)

    # Save to JSON
    save_results_to_json(results, metrics)

    print("\n✅ Benchmark complete!")
    print("\n💡 NEXT STEPS:")
    print("  1. Review game_results_*.json for detailed analysis")
    print("  2. Check hallucination rates - should be <10%")
    print("  3. If API errors persist, increase retry delays")
    print("  4. Implement port trades for better gameplay")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()