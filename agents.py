# agents.py - Enhanced with better parsing and clearer prompts
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import json
import random
import re

from env import Action, ActionType, GameState

ChatFn = Callable[[List[Dict[str, str]]], str]


@dataclass
class StepDecisionInfo:
    raw_response: str
    valid_index: bool
    used_fallback: bool
    api_error: bool  # NEW: track if API failed


class RandomAgent:
    """Simple random policy baseline."""

    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        self.name = name
        self.rng = random.Random(seed)
        self.last_decision_info: Optional[StepDecisionInfo] = None

    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        action = self.rng.choice(legal_actions)
        self.last_decision_info = StepDecisionInfo(
            raw_response="(random)",
            valid_index=True,
            used_fallback=False,
            api_error=False,
        )
        return action


class LLMJsonAgent:
    """LLM agent with improved JSON parsing and clearer prompts."""

    def __init__(self, name: str, chat_fn: ChatFn) -> None:
        self.name = name
        self.chat_fn = chat_fn
        self.last_decision_info: Optional[StepDecisionInfo] = None

    def _serialize_state(self, state: GameState) -> str:
        """Convert GameState into JSON."""
        return json.dumps(
            {
                "turn": state.turn,
                "current_player_index": state.current_player_index,
                "victory_points": list(state.victory_points),
                "resources": state.resources,
                "longest_road_owner": state.longest_road_owner,
                "largest_army_owner": state.largest_army_owner,
                "robber_position": state.robber_position,
                "pending_discard": state.pending_discard,
            },
            indent=2,
        )

    def _describe_action(self, idx: int, action: Action) -> str:
        """Convert action to clear, readable description."""
        t = action.type
        p = action.payload

        if t == ActionType.BUILD_SETTLEMENT:
            coords = p.get("coords")
            return f"{idx}. BUILD_SETTLEMENT at location {coords} [Costs: 1 brick, 1 lumber, 1 wool, 1 grain] [Gain: +1 VP]"

        if t == ActionType.BUILD_CITY:
            coords = p.get("coords")
            return f"{idx}. BUILD_CITY at location {coords} (upgrade settlement) [Costs: 3 ore, 2 grain] [Gain: +1 VP]"

        if t == ActionType.BUILD_ROAD:
            path = p.get("path")
            return f"{idx}. BUILD_ROAD on path {path} [Costs: 1 brick, 1 lumber]"

        if t == ActionType.TRADE:
            to_player = p.get("to_player")
            resource = p.get("resource")
            return f"{idx}. TRADE: Give 1 {resource} to Player {to_player}"

        if t == ActionType.BANK_TRADE:
            give = p.get("give")
            receive = p.get("receive")
            return f"{idx}. BANK_TRADE: Give 4 {give} → Get 1 {receive}"

        if t == ActionType.DISCARD:
            resource = p.get("resource")
            count = p.get("count")
            return f"{idx}. DISCARD {count} {resource} (required: robber rolled 7)"

        if t == ActionType.END_TURN:
            return f"{idx}. END_TURN (finish your turn)"

        return f"{idx}. UNKNOWN_ACTION"

    def _extract_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Aggressively extract JSON from LLM response.
        Handles: pure JSON, markdown fences, explanations, etc.
        """
        # 1) Direct parse
        try:
            return json.loads(raw.strip())
        except Exception:
            pass

        # 2) Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE)
        try:
            return json.loads(cleaned.strip())
        except Exception:
            pass

        # 3) Find first {...} block
        match = re.search(r"\{[^}]*\"action_index\"[^}]*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        # 4) Try finding just the number after "action_index"
        match = re.search(r"['\"]?action_index['\"]?\s*:\s*(\d+)", raw, flags=re.IGNORECASE)
        if match:
            try:
                return {"action_index": int(match.group(1))}
            except Exception:
                pass

        return None

    def choose_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        """
        Main decision logic with improved error handling.
        """
        state_json = self._serialize_state(state)
        action_lines = [self._describe_action(i, a) for i, a in enumerate(legal_actions)]
        actions_text = "\n".join(action_lines)

        vps = list(state.victory_points)
        my_idx = state.current_player_index
        my_vp = vps[my_idx]
        leader_vp = max(vps)
        my_resources = state.resources[my_idx]
        total_resources = sum(my_resources.values())

        # Get opponent info for trading context
        opponent_vps = [(idx, vps[idx]) for idx in range(len(vps)) if idx != my_idx]
        opponent_vps.sort(key=lambda x: x[1], reverse=True)  # Sort by VP
        leader_idx, leader_vp = opponent_vps[0] if opponent_vps else (None, 0)
        weakest_idx, weakest_vp = opponent_vps[-1] if opponent_vps else (None, 0)
        
        trading_hint = ""
        if leader_idx is not None and my_vp < leader_vp:
            trading_hint = f"\n💡 Trading tip: Player {leader_idx} is leading with {leader_vp} VP. Avoid giving them resources!"
            if weakest_idx is not None and weakest_idx != leader_idx:
                trading_hint += f"\n   Consider trading with Player {weakest_idx} ({weakest_vp} VP) to build an alliance."
        context_hint = ""
        if state.pending_discard:
            discard_count = total_resources // 2
            context_hint = f"""
⚠️ DISCARD PHASE ACTIVE ⚠️
You rolled a 7 and have {total_resources} cards (>7 limit).
You MUST discard {discard_count} cards immediately.
Look for DISCARD actions in the list below and choose one.
DO NOT choose BUILD or TRADE actions during discard phase.
"""

        system_prompt = """You are an expert Settlers of Catan AI player.

Your task: Choose the BEST action from the numbered list provided.

CRITICAL RULES:
1. You must output ONLY valid JSON: {"action_index": <number>}
2. The number must be from the provided action list (0 to N-1)
3. No explanations, no extra text, just the JSON object
4. During DISCARD phase, you MUST choose a DISCARD action

Strategy priorities (in order):
1. BUILD_SETTLEMENT and BUILD_CITY are TOP priority (they give victory points!)
2. TRADE with other players strategically:
   - Give resources to players who are BEHIND (not to the leader!)
   - This is better than hoarding - it builds alliances
   - Even "bad" trades can help you win by slowing the leader
3. Use BANK_TRADE when you have 4+ of one resource
4. BUILD_ROAD to expand and get Longest Road bonus
5. Only END_TURN when no productive actions are available

Trading strategy:
- Look at victory points before trading
- NEVER help the player with the most victory points
- DO help players who are behind - they might help you later
- DISCARD actions are mandatory when the robber is rolled"""

        user_prompt = f"""GAME STATE:
{state_json}

YOUR STATUS:
- You are Player {my_idx}
- Your VP: {my_vp} / 10 needed to win
- Leader VP: {leader_vp}
- Your resources: {total_resources} cards total
- Your hand: {my_resources}
{trading_hint}
{context_hint}

AVAILABLE ACTIONS (choose ONE by index):
{actions_text}

Remember: Output ONLY this format: {{"action_index": <number>}}
Choose the action index that best helps you win the game.
Consider: Building > Trading with weak players > Bank trades > END_TURN"""

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call LLM
        raw = self.chat_fn(messages)

        # Check for API failure marker
        api_failed = False
        try:
            temp_parse = json.loads(raw)
            if temp_parse.get("error") == "api_failed":
                api_failed = True
        except:
            pass

        # Parse response
        used_fallback = False
        valid_index = False
        chosen_idx = 0

        parsed = self._extract_json(raw)
        if isinstance(parsed, dict) and "action_index" in parsed:
            idx = parsed["action_index"]
            # Handle string indices
            if isinstance(idx, str) and idx.isdigit():
                idx = int(idx)
            if isinstance(idx, int) and 0 <= idx < len(legal_actions):
                chosen_idx = idx
                valid_index = True
                print(f"✓ {self.name} chose action {chosen_idx}: {legal_actions[chosen_idx].type.name}")
            else:
                print(f"⚠️ {self.name} returned out-of-range index: {idx} (max: {len(legal_actions)-1})")

        # Fallback if parsing failed or API failed
        if not valid_index or api_failed:
            used_fallback = True
            print(f"⚠️ {self.name} using fallback (API failed: {api_failed})")

            # Smart fallback priority
            # 1) Discard if required
            if state.pending_discard:
                discard_indices = [
                    i for i, a in enumerate(legal_actions)
                    if a.type == ActionType.DISCARD
                ]
                if discard_indices:
                    chosen_idx = discard_indices[0]
                    print(f"   → Fallback chose DISCARD action {chosen_idx}")
            
            # 2) Build actions (highest priority)
            if not state.pending_discard:
                build_indices = [
                    i for i, a in enumerate(legal_actions)
                    if a.type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.BUILD_ROAD)
                ]
                if build_indices:
                    chosen_idx = build_indices[0]
                    print(f"   → Fallback chose BUILD action {chosen_idx}")
                else:
                    # 3) Bank trades
                    bank_indices = [
                        i for i, a in enumerate(legal_actions)
                        if a.type == ActionType.BANK_TRADE
                    ]
                    if bank_indices:
                        chosen_idx = bank_indices[0]
                        print(f"   → Fallback chose BANK_TRADE {chosen_idx}")
                    else:
                        # 4) END_TURN
                        end_indices = [
                            i for i, a in enumerate(legal_actions)
                            if a.type == ActionType.END_TURN
                        ]
                        if end_indices:
                            chosen_idx = end_indices[0]
                            print(f"   → Fallback chose END_TURN {chosen_idx}")

        # Safety check
        if legal_actions and 0 <= chosen_idx < len(legal_actions):
            self.last_decision_info = StepDecisionInfo(
                raw_response=raw,
                valid_index=valid_index and not api_failed,
                used_fallback=used_fallback or api_failed,
                api_error=api_failed,
            )
            return legal_actions[chosen_idx]
        else:
            # Emergency fallback
            print(f"⚠️ {self.name} EMERGENCY FALLBACK")
            self.last_decision_info = StepDecisionInfo(
                raw_response=raw,
                valid_index=False,
                used_fallback=True,
                api_error=api_failed,
            )
            return Action(ActionType.END_TURN, payload={})