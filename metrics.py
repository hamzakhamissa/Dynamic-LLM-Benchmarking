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
    Counts how often each agent tried to pick an invalid action_index
    (i.e. we had to use fallback).
    """
    stats = {
        i: {"total_decisions": 0, "fallbacks": 0}
        for i in range(4)
    }

    for g in results:
        for step in g.steps:
            i = step.acting_player_index
            stats[i]["total_decisions"] += 1
            if step.info.get("llm_used_fallback", False):
                stats[i]["fallbacks"] += 1

    out: Dict[int, Dict[str, float]] = {}
    for i in range(4):
        total = stats[i]["total_decisions"] or 1
        fallbacks = stats[i]["fallbacks"]
        rate = fallbacks / total
        # heavy penalty: after 20% hallucination, score goes to 0
        penalty_score = max(0.0, 1.0 - 5.0 * rate)
        out[i] = {
            "decisions": float(total),
            "hallucinations": float(fallbacks),
            "hallucination_rate": rate,
            "hallucination_penalty_score": penalty_score,
        }
    return out


def _sum_resources(res_dict: Dict[str, int]) -> int:
    return sum(res_dict.values())


def trade_behavior(results: List[GameResult]) -> Dict[int, Dict[str, float]]:
    """
    Measure greed vs cooperation via gift trades.

    For each TRADE step:
      - identify from / to players
      - compute net resource change for each
      - check leader & last-place roles at that step
    """
    trade_counts = {
        i: {
            "num_trades": 0,
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
            if step.action.type != ActionType.TRADE:
                continue

            from_idx = step.acting_player_index
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
            stats["num_trades"] += 1

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
        total = s["num_trades"] or 1
        selfish_plus_altru = s["selfish_trades"] + s["altruistic_trades"] or 1

        out[i] = {
            "num_trades": float(s["num_trades"]),
            "trades_with_leader_ratio": s["num_trades_with_leader"] / total,
            "trades_helping_leader_ratio": s["num_trades_helping_leader"] / total,
            "trades_helping_last_ratio": s["num_trades_helping_last"] / total,
            "selfish_trade_ratio": s["selfish_trades"] / selfish_plus_altru,
            "altruistic_trade_ratio": s["altruistic_trades"] / selfish_plus_altru,
        }

    return out


def overall_scores(results: List[GameResult]) -> Dict[int, Dict[str, float]]:
    """
    Combine:
      - win rate
      - hallucination penalty
      - cooperative-but-not-dumb trading
    into a single score per agent.
    """
    win_rates = compute_win_rates(results)
    hall = hallucination_stats(results)
    trades = trade_behavior(results)

    scores: Dict[int, Dict[str, float]] = {}
    for i in range(4):
        win = win_rates.get(i, 0.0)
        hall_penalty = hall[i]["hallucination_penalty_score"]
        trade_stats = trades[i]

        # Encourage *some* trading, discourage feeding the leader.
        trade_activity = min(1.0, trade_stats["num_trades"] / 10.0)
        leader_feed = trade_stats["trades_helping_leader_ratio"]
        last_help = trade_stats["trades_helping_last_ratio"]
        selfish = trade_stats["selfish_trade_ratio"]

        # "Game sense" score:
        # - reward helping last place
        # - punish helping leader
        # - slight reward for moderate selfishness (playing to win)
        game_sense = (
            0.4 * last_help
            - 0.6 * leader_feed
            + 0.2 * (0.5 - abs(selfish - 0.6))  # prefer mildly selfish ~0.6
        )

        # Final composite score (tunable)
        overall = max(0.0, win * hall_penalty * (0.5 + 0.5 * trade_activity + game_sense))

        scores[i] = {
            "win_rate": win,
            "hallucination_penalty": hall_penalty,
            "trade_activity": trade_activity,
            "game_sense": game_sense,
            "overall_score": overall,
        }

    return scores
