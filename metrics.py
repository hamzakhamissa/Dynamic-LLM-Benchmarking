# metrics.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Any

from orchestrator import GameResult, StepRecord
from env import ActionType


def compute_win_rates(results: List[GameResult]) -> Dict[int, float]:
    wins = defaultdict(int)
    for g in results:
        if g.winner_index is not None:
            wins[g.winner_index] += 1
    total = len(results)
    return {i: wins[i] / total for i in range(4)}


def average_turns(results: List[GameResult]) -> float:
    if not results:
        return 0.0
    return sum(g.final_state.turn for g in results) / len(results)


def hallucination_stats(results: List[GameResult]) -> Dict[int, Dict[str, float]]:
    """
    Counts how often each agent:
    1. Tried to pick an invalid action_index (llm_used_fallback from agents.py)
    2. Tried an illegal action that failed (action_failed from env.py)
    """
    stats = {
        i: {"total_decisions": 0, "fallbacks": 0, "action_failures": 0}
        for i in range(4)
    }

    for g in results:
        for step in g.steps:
            i = step.acting_player_index
            stats[i]["total_decisions"] += 1
            
            # Track index parsing failures
            if step.info.get("llm_used_fallback", False):
                stats[i]["fallbacks"] += 1
            
            # Track action execution failures
            if step.info.get("action_failed", False):
                stats[i]["action_failures"] += 1

    out: Dict[int, Dict[str, float]] = {}
    for i in range(4):
        total = stats[i]["total_decisions"] or 1
        fallbacks = stats[i]["fallbacks"]
        failures = stats[i]["action_failures"]
        total_hallucinations = fallbacks + failures
        rate = total_hallucinations / total
        
        # Heavy penalty: after 20% hallucination, score goes to 0
        penalty_score = max(0.0, 1.0 - 5.0 * rate)
        
        out[i] = {
            "decisions": float(total),
            "index_hallucinations": float(fallbacks),
            "action_failures": float(failures),
            "total_hallucinations": float(total_hallucinations),
            "hallucination_rate": rate,
            "hallucination_penalty_score": penalty_score,
        }
    return out


def _sum_resources(res_dict: Dict[str, int]) -> int:
    return sum(res_dict.values())


def trade_behavior(results: List[GameResult]) -> Dict[int, Dict[str, float]]:
    """
    Measure trade activity including both player trades and bank trades.
    """
    trade_counts = {
        i: {
            "num_player_trades": 0,
            "num_bank_trades": 0,
            "num_trades_with_leader": 0,
            "num_trades_helping_leader": 0,
            "num_trades_helping_last": 0,
            "selfish_trades": 0,
            "altruistic_trades": 0,
        }
        for i in range(4)
    }

    for g in results:
        for step in g.steps:
            from_idx = step.acting_player_index
            
            # Track bank trades
            if step.action.type == ActionType.BANK_TRADE:
                trade_counts[from_idx]["num_bank_trades"] += 1
                continue
            
            # Track player-to-player trades
            if step.action.type != ActionType.TRADE:
                continue

            trade_counts[from_idx]["num_player_trades"] += 1
            
            payload = step.action.payload
            to_idx = int(payload["to_player"])

            # Determine leader / last place BEFORE trade
            vp = step.state_before.victory_points
            max_vp = max(vp)
            min_vp = min(vp)
            leaders = [i for i, v in enumerate(vp) if v == max_vp]
            lasts = [i for i, v in enumerate(vp) if v == min_vp]
            leader = leaders[0]
            last = lasts[0]

            # Resource values before / after
            before_res = step.state_before.resources
            after_res = step.state_after.resources

            from_before = _sum_resources(before_res[from_idx])
            from_after = _sum_resources(after_res[from_idx])
            to_before = _sum_resources(before_res[to_idx])
            to_after = _sum_resources(after_res[to_idx])

            from_delta = from_after - from_before
            to_delta = to_after - to_before

            stats = trade_counts[from_idx]

            if to_idx == leader or from_idx == leader:
                stats["num_trades_with_leader"] += 1

            if to_idx == leader and to_delta > 0:
                stats["num_trades_helping_leader"] += 1

            if to_idx == last and to_delta > 0:
                stats["num_trades_helping_last"] += 1

            if from_delta > to_delta:
                stats["selfish_trades"] += 1
            elif from_delta < to_delta:
                stats["altruistic_trades"] += 1

    # Normalize
    out: Dict[int, Dict[str, float]] = {}
    for i in range(4):
        s = trade_counts[i]
        total_player = s["num_player_trades"] or 1
        total_trades = s["num_player_trades"] + s["num_bank_trades"]
        selfish_plus_altru = s["selfish_trades"] + s["altruistic_trades"] or 1

        out[i] = {
            "num_player_trades": float(s["num_player_trades"]),
            "num_bank_trades": float(s["num_bank_trades"]),
            "total_trades": float(total_trades),
            "trades_with_leader_ratio": s["num_trades_with_leader"] / total_player,
            "trades_helping_leader_ratio": s["num_trades_helping_leader"] / total_player,
            "trades_helping_last_ratio": s["num_trades_helping_last"] / total_player,
            "selfish_trade_ratio": s["selfish_trades"] / selfish_plus_altru,
            "altruistic_trade_ratio": s["altruistic_trades"] / selfish_plus_altru,
        }

    return out


def resource_efficiency(results: List[GameResult]) -> Dict[int, Dict[str, float]]:
    """
    Measure how efficiently each player uses resources.
    - Resource waste: cards held at end of game
    - Build rate: builds per turn
    """
    stats = {
        i: {
            "total_builds": 0,
            "total_turns": 0,
            "final_resources": 0,
        }
        for i in range(4)
    }
    
    for g in results:
        for step in g.steps:
            i = step.acting_player_index
            stats[i]["total_turns"] += 1
            
            if step.action.type in (
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_CITY,
                ActionType.BUILD_ROAD,
            ):
                stats[i]["total_builds"] += 1
        
        # Final resources
        for i in range(4):
            stats[i]["final_resources"] += _sum_resources(g.final_state.resources[i])
    
    out: Dict[int, Dict[str, float]] = {}
    for i in range(4):
        total_turns = stats[i]["total_turns"] or 1
        build_rate = stats[i]["total_builds"] / total_turns
        avg_final_resources = stats[i]["final_resources"] / len(results)
        
        out[i] = {
            "build_rate": build_rate,
            "avg_final_resources": avg_final_resources,
            "efficiency_score": build_rate / (1.0 + 0.1 * avg_final_resources),
        }
    
    return out


def overall_scores(results: List[GameResult]) -> Dict[int, Dict[str, float]]:
    """
    Combine:
      - win rate
      - hallucination penalty
      - trade activity
      - resource efficiency
    into a single score per agent.
    """
    win_rates = compute_win_rates(results)
    hall = hallucination_stats(results)
    trades = trade_behavior(results)
    efficiency = resource_efficiency(results)

    scores: Dict[int, Dict[str, float]] = {}
    for i in range(4):
        win = win_rates.get(i, 0.0)
        hall_penalty = hall[i]["hallucination_penalty_score"]
        trade_stats = trades[i]
        eff_stats = efficiency[i]

        # Reward trading activity (both types)
        trade_activity = min(1.0, trade_stats["total_trades"] / 15.0)
        
        # Reward strategic trading
        leader_feed = trade_stats["trades_helping_leader_ratio"]
        last_help = trade_stats["trades_helping_last_ratio"]
        
        # Game sense: help last, avoid helping leader
        game_sense = (
            0.4 * last_help
            - 0.6 * leader_feed
            + 0.2 * eff_stats["efficiency_score"]
        )

        # Final composite score
        overall = max(0.0, win * hall_penalty * (
            0.4 +  # Base score
            0.3 * trade_activity +
            0.3 * game_sense
        ))

        scores[i] = {
            "win_rate": win,
            "hallucination_penalty": hall_penalty,
            "trade_activity": trade_activity,
            "game_sense": game_sense,
            "efficiency": eff_stats["efficiency_score"],
            "overall_score": overall,
        }

    return scores