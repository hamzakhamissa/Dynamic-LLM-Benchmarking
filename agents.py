# agents.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import json
import random
import re

from env import Action, ActionType, GameState

# A chat function takes OpenAI-style messages and returns a *string* response
ChatFn = Callable[[List[Dict[str, str]]], str]
"""
Expected behavior:
- Input:  messages = [{"role": "system"|"user"|"assistant", "content": "..."}]
- Output: string that *should* contain JSON like: {"action_index": 3}
"""


@dataclass
class StepDecisionInfo:
    raw_response: str
    valid_index: bool
    used_fallback: bool


class RandomAgent:
    """
    Simple random policy, used as a baseline or filler when not using an LLM.
    """

    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        self.name = name
        self.rng = random.Random(seed)
        self.last_decision_info: Optional[StepDecisionInfo] = None

    def choose_action(
        self, state: GameState, legal_actions: List[Action]
    ) -> Action:
        # Random baseline always has "valid" index and never uses fallback logic.
        action = self.rng.choice(legal_actions)
        self.last_decision_info = StepDecisionInfo(
            raw_response="(random)",
            valid_index=True,
            used_fallback=False,
        )
        return action


class LLMJsonAgent:
    """
    Wraps an LLM that returns JSON and chooses an action index.

    The model sees:
      - a compact JSON summary of the game state
      - a numbered list of candidate actions as natural language

    It must output:
      { "action_index": <integer> }
    """

    def __init__(self, name: str, chat_fn: ChatFn) -> None:
        self.name = name
        self.chat_fn = chat_fn
        self.last_decision_info: Optional[StepDecisionInfo] = None

    # --------- Helpers for prompt construction ---------

    def _serialize_state(self, state: GameState) -> str:
        """
        Convert GameState into JSON-serializable dict.
        """
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
        """
        Turn an Action into a human-readable line that the LLM can reason about.
        """
        t = action.type
        p = action.payload

        if t == ActionType.BUILD_SETTLEMENT:
            coords = p.get("coords")
            return f"{idx}: BUILD_SETTLEMENT at coords={coords}"

        if t == ActionType.BUILD_CITY:
            coords = p.get("coords")
            return f"{idx}: BUILD_CITY at coords={coords} (upgrade settlement to city)"

        if t == ActionType.BUILD_ROAD:
            path = p.get("path")
            return f"{idx}: BUILD_ROAD along path={path}"

        if t == ActionType.TRADE:
            to_player = p.get("to_player")
            resource = p.get("resource")
            return f"{idx}: TRADE give 1 {resource} to player {to_player}"

        if t == ActionType.BANK_TRADE:
            give = p.get("give")
            receive = p.get("receive")
            return f"{idx}: BANK_TRADE give 4 {give} for 1 {receive}"

        if t == ActionType.DISCARD:
            resource = p.get("resource")
            count = p.get("count")
            return f"{idx}: DISCARD {count} {resource} (required after rolling 7)"

        if t == ActionType.END_TURN:
            return f"{idx}: END_TURN"

        return f"{idx}: UNKNOWN_ACTION payload={p}"

    def _extract_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Try hard to recover a JSON object from the LLM's response.
        Handles:
          - pure JSON:        {"action_index": 3}
          - fenced code:      ```json { "action_index": 3 } ```
          - explanations:     "I choose... {\"action_index\": 3} because..."
        """
        # 1) Direct parse
        try:
            return json.loads(raw.strip())
        except Exception:
            pass

        # 2) Strip common markdown fences
        cleaned = re.sub(r"```json\s*|\s*```", "", raw, flags=re.IGNORECASE)
        try:
            return json.loads(cleaned.strip())
        except Exception:
            pass

        # 3) Grep the first {...} block
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                pass

        return None

    # --------- Main policy ---------

    def choose_action(
        self, state: GameState, legal_actions: List[Action]
    ) -> Action:
        """
        1. Build a prompt from GameState + legal actions.
        2. Call LLM via chat_fn.
        3. Parse JSON -> action_index.
        4. If anything goes wrong, fall back to a safe heuristic action.
        """
        state_json = self._serialize_state(state)
        action_lines = [
            self._describe_action(i, a) for i, a in enumerate(legal_actions)
        ]
        actions_text = "\n".join(action_lines)

        vps = list(state.victory_points)
        my_idx = state.current_player_index
        my_vp = vps[my_idx]
        leader_vp = max(vps)
        my_resources = state.resources[my_idx]
        total_resources = sum(my_resources.values())

        # Special context for discard phase
        context_hint = ""
        if state.pending_discard:
            context_hint = "\n⚠️ DISCARD PHASE: You rolled a 7 and have >7 cards. You MUST discard half your cards."

        system_prompt = (
            "You are a strong Settlers of Catan bot. "
            "Always follow the rules. "
            "Prioritize building settlements and cities to gain victory points. "
            "Use bank trades (4:1) when you have excess resources. "
            "You must output ONLY valid JSON of the form "
            '{"action_index": <integer>} and nothing else.'
        )

        user_prompt = f"""
You are playing a 4-player game of Settlers of Catan.

Game state (JSON):
{state_json}

Interpreting the state:
- You are player index {my_idx}.
- Your victory points: {my_vp} / {10} needed to win
- Highest victory points at table: {leader_vp}
- Your total resources: {total_resources} cards
- Robber position: {state.robber_position}
{context_hint}

Available discrete actions (choose exactly ONE by index):
{actions_text}

Strategy tips:
- BUILD_SETTLEMENT and BUILD_CITY give victory points (highest priority)
- BANK_TRADE (4:1) helps when you have 4+ of one resource
- Only END_TURN when you can't do anything productive
- DISCARD actions are mandatory when prompted

Return ONLY a JSON object, no explanation, no extra keys.
The object MUST look like:
{{"action_index": <index_of_best_action>}}
"""

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = self.chat_fn(messages)

        # Check if API returned a fallback/error response
        api_failed = (raw == '{"action_index": 0}')
        
        used_fallback = False
        valid_index = False
        chosen_idx = 0  # default

        parsed = self._extract_json(raw)
        if isinstance(parsed, dict) and "action_index" in parsed:
            idx = parsed["action_index"]
            # tolerate string indices too
            if isinstance(idx, str) and idx.isdigit():
                idx = int(idx)
            if isinstance(idx, int) and 0 <= idx < len(legal_actions):
                chosen_idx = idx
                valid_index = True

        if not valid_index:
            # Fallback heuristic:
            #   1) Prefer any build action (settlement/city/road)
            #   2) Prefer bank trades if we have 4+ of something
            #   3) Else prefer END_TURN
            #   4) Else index 0
            used_fallback = True

            # Priority 1: Build actions
            build_indices = [
                i
                for i, a in enumerate(legal_actions)
                if a.type in (
                    ActionType.BUILD_SETTLEMENT,
                    ActionType.BUILD_CITY,
                    ActionType.BUILD_ROAD,
                )
            ]
            if build_indices:
                chosen_idx = build_indices[0]
            else:
                # Priority 2: Bank trades
                bank_indices = [
                    i for i, a in enumerate(legal_actions)
                    if a.type == ActionType.BANK_TRADE
                ]
                if bank_indices:
                    chosen_idx = bank_indices[0]
                else:
                    # Priority 3: Discard actions (during robber phase)
                    discard_indices = [
                        i for i, a in enumerate(legal_actions)
                        if a.type == ActionType.DISCARD
                    ]
                    if discard_indices:
                        chosen_idx = discard_indices[0]
                    else:
                        # Priority 4: END_TURN
                        end_indices = [
                            i
                            for i, a in enumerate(legal_actions)
                            if a.type == ActionType.END_TURN
                        ]
                        if end_indices:
                            chosen_idx = end_indices[0]
                        else:
                            # Last resort: first action (if list not empty)
                            chosen_idx = 0 if legal_actions else None

        # Safety check
        if legal_actions and 0 <= chosen_idx < len(legal_actions):
            self.last_decision_info = StepDecisionInfo(
                raw_response=raw,
                valid_index=valid_index and not api_failed,  # Mark API failures
                used_fallback=used_fallback or api_failed,
            )
            return legal_actions[chosen_idx]
        else:
            # Emergency fallback: create a safe END_TURN action
            print(f"⚠️ No valid actions available! Creating emergency END_TURN")
            self.last_decision_info = StepDecisionInfo(
                raw_response=raw,
                valid_index=False,
                used_fallback=True,
            )
            return Action(ActionType.END_TURN, payload={})