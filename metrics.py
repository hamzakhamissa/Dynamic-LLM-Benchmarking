# metrics.py - Fixed scoring and added formula documentation
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from orchestrator import GameResult
from env import ActionType


def compute_win_rates(results: List[GameResult]) -> Dict[int, float]:
    """
    Calculate win rate for each player.
    
    Formula: wins / total_games
    """
    wins = defaultdict(int)
    for g in results:
        if g.winner_index is not None:
            wins[g.winner_index] += 1
    total = len(results) or 1
    return {i: wins[i] / total for i in range(4)}


def average_turns(results: List[GameResult]) -> float:
    """Calculate average game length in turns."""
    if not results:
        return 0.0
    return sum(g.final_state.turn for g in results) / len(results)


def hallucination_stats(results: List[GameResult]) -> Dict[int, Dict[str, float]]:
    """
    Track decision quality:
    - Index parsing failures (invalid action_index)
    - Action execution failures (illegal moves)
    - API errors (timeout/failure)
    
    Penalty Formula:
      penalty_score = max(0, 1 - 5 * hallucination_rate)
      - 0% hallucinations → 1.0 penalty score (no penalty)
      - 10% hallucinations → 0.5 penalty score (50% penalty)
      - 20% hallucinations → 0.0 penalty score (maximum penalty)
    """
    stats = {
        i: {
            "total_decisions": 0,
            "fallbacks": 0,
            "action_failures": 0,
            "api_errors": 0,
        }
        for i in range(4)
    }

    for g in results:
        for step in g.steps:
            i = step.acting_player_index
            stats[i]["total_decisions"] += 1
            
            # Track parsing failures
            if step.info.get("llm_used_fallback", False):
                stats[i]["fallbacks"] += 1
            
            # Track action failures
            if step.info.get("action_failed", False):
                stats[i]["action_failures"] += 1
            
            # Track API errors
            if step.info.get("llm_api_error", False):
                stats[i]["api_errors"] += 1

    out: Dict[int, Dict[str, float]] = {}
    for i in range(4):
        total = stats[i]["total_decisions"] or 1
        fallbacks = stats[i]["fallbacks"]
        failures = stats[i]["action_failures"]
        api_errors = stats[i]["api_errors"]
        
        total_hallucinations = fallbacks + failures
        rate = total_hallucinations / total
        
        # Penalty: after 20% hallucination, score drops to 0
        penalty_score = max(0.0, 1.0 - 5.0 * rate)
        
        out[i] = {
            "decisions": float(total),
            "index_hallucinations": float(fallbacks),
            "action_failures": float(failures),
            "api_errors": float(api_errors),
            "total_hallucinations": float(total_hallucinations),
            "hallucination_rate": rate,
            "hallucination_penalty_score": penalty_score,
        }
    return out


def _sum_resources(res_dict: Dict[str, int]) -> int:
    """Sum all resources in a dict."""
    return sum(res_dict.values())


def trade_behavior(results: List[GameResult]) -> Dict[int, Dict[str, float]]:
    """
    Analyze trading patterns.
    
    Metrics:
    - num_player_trades: Direct trades with other players
    - num_bank_trades: 4:1 trades with the bank
    - trades_helping_leader_ratio: % of trades that help the leader (BAD strategy)
    - trades_helping_last_ratio: % of trades that help last place (GOOD strategy)
    - selfish_trade_ratio: % where you gain more than partner
    - altruistic_trade_ratio: % where partner gains more than you
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
            
            # Bank trades
            if step.action.type == ActionType.BANK_TRADE:
                trade_counts[from_idx]["num_bank_trades"] += 1
                continue
            
            # Player trades
            if step.action.type != ActionType.TRADE:
                continue

            trade_counts[from_idx]["num_player_trades"] += 1
            
            payload = step.action.payload
            to_idx = int(payload["to_player"])

            # Determine leader/last place before trade
            vp = step.state_before.victory_points
            max_vp = max(vp)
            min_vp = min(vp)
            leaders = [i for i, v in enumerate(vp) if v == max_vp]
            lasts = [i for i, v in enumerate(vp) if v == min_vp]
            leader = leaders[0]
            last = lasts[0]

            # Resource deltas
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
    Measure resource usage efficiency.
    
    Metrics:
    - build_rate: buildings per turn (higher = better)
    - avg_final_resources: cards left at end (lower = better, means you spent them)
    
    Formula:
      efficiency_score = build_rate / (1 + 0.1 * avg_final_resources)
      
    Interpretation:
    - High build_rate + low leftover resources = high efficiency
    - 0.3 builds/turn with 5 cards left → 0.3 / 1.5 = 0.20 efficiency
    - 0.1 builds/turn with 20 cards left → 0.1 / 3.0 = 0.03 efficiency
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
    num_games = len(results) or 1
    
    for i in range(4):
        total_turns = stats[i]["total_turns"] or 1
        build_rate = stats[i]["total_builds"] / total_turns
        avg_final_resources = stats[i]["final_resources"] / num_games
        
        # Efficiency: higher build rate, lower leftover resources
        out[i] = {
            "build_rate": build_rate,
            "avg_final_resources": avg_final_resources,
            "efficiency_score": build_rate / (1.0 + 0.1 * avg_final_resources),
        }
    
    return out


def overall_scores(results: List[GameResult]) -> Dict[int, Dict[str, float]]:
    """
    Composite score combining multiple factors.
    
    FORMULA:
      overall_score = win_rate * hallucination_penalty * (
          0.5 +                              # Base score (50%)
          0.3 * min(trade_activity, 1.0) +  # Reward active trading (30%)
          0.2 * max(game_sense, 0)          # Reward strategic play (20%)
      )
    
    Where:
      - win_rate: 0.0 to 1.0 (% of games won)
      - hallucination_penalty: 1.0 if no errors, 0.0 if >20% error rate
      - trade_activity: total_trades / 15 (capped at 1.0)
      - game_sense: 0.4 * helping_last - 0.6 * helping_leader + 0.2 * efficiency
    
    Components:
    - Win rate: Direct measure of success
    - Hallucination penalty: Punishes unreliable models
    - Trade activity: Rewards engaging with game mechanics
    - Game sense: Rewards helping underdogs, avoiding helping leaders
    
    Score Interpretation:
    - 0.6+  : Excellent (60%+ win rate, no errors, strategic play)
    - 0.4-0.6: Good (winning some, mostly reliable)
    - 0.2-0.4: Fair (occasional wins or high error rate)
    - <0.2  : Poor (rarely winning or very unreliable)
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

        # Trade activity score (normalize to 0-1, cap at 15 trades)
        trade_activity = min(1.0, trade_stats["total_trades"] / 15.0)
        
        # Strategic trading
        leader_feed = trade_stats["trades_helping_leader_ratio"]
        last_help = trade_stats["trades_helping_last_ratio"]
        
        # Game sense: help weak, avoid helping strong
        game_sense = (
            0.4 * last_help
            - 0.6 * leader_feed
            + 0.2 * eff_stats["efficiency_score"]
        )

        # FIXED: More generous scoring
        # Base of 0.5 means even with 0 trades/sense, you get 50% credit for winning
        overall = win * hall_penalty * (
            0.5 +  # Base score (was 0.4)
            0.3 * trade_activity +
            0.2 * max(game_sense, 0)  # Only positive game sense counts
        )

        scores[i] = {
            "win_rate": win,
            "hallucination_penalty": hall_penalty,
            "trade_activity": trade_activity,
            "game_sense": game_sense,
            "efficiency": eff_stats["efficiency_score"],
            "overall_score": overall,
        }

    return scores