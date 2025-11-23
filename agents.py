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
            return f"{idx}: BUILD_CITY at coords={coords}"

        if t == ActionType.BUILD_ROAD:
            path = p.get("path")
            return f"{idx}: BUILD_ROAD along path={path}"

        if t == ActionType.TRADE:
            # current env.py: gift-style trade (we can refine rules later)
            to_player = p.get("to_player")
            resource = p.get("resource")
            return f"{idx}: TRADE give 1 {resource} to player {to_player}"

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

        system_prompt = (
            "You are a strong Settlers of Catan bot. "
            "Always follow the rules. "
            "You must output ONLY valid JSON of the form "
            '{"action_index": <integer>} and nothing else.'
        )

        user_prompt = f"""
You are playing a 4-player game of Settlers of Catan.

Game state (JSON):
{state_json}

Interpreting the state:
- You are player index {my_idx}.
- Your victory points: {my_vp}
- Highest victory points at table: {leader_vp}

Available discrete actions (choose exactly ONE by index):
{actions_text}

Return ONLY a JSON object, no explanation, no extra keys.
The object MUST look like:
{{"action_index": <index_of_best_action>}}
"""

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = self.chat_fn(messages)

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
            #   2) Else prefer END_TURN
            #   3) Else index 0
            used_fallback = True

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
                end_indices = [
                    i
                    for i, a in enumerate(legal_actions)
                    if a.type == ActionType.END_TURN
                ]
                if end_indices:
                    chosen_idx = end_indices[0]
                else:
                    chosen_idx = 0

        self.last_decision_info = StepDecisionInfo(
            raw_response=raw,
            valid_index=valid_index,
            used_fallback=used_fallback,
        )

        return legal_actions[chosen_idx]
