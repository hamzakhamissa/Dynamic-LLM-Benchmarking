# orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from env import CatanEngine, GameState, Action
from agents import RandomAgent, LLMJsonAgent


@dataclass
class StepRecord:
    state_before: GameState
    state_after: GameState
    acting_player_index: int
    action: Action
    legal_actions_count: int
    info: Dict[str, Any]  # extra info (roll, winner, hallucination flags, etc.)


@dataclass
class GameResult:
    steps: List[StepRecord]
    winner_index: Optional[int]
    final_state: GameState


class GameOrchestrator:
    """
    Glue:
      - uses a CatanEngine
      - one agent per player
      - logs all steps for metric computation
    """

    def __init__(
        self,
        engine: CatanEngine,
        agents: List[object],  # RandomAgent or LLMJsonAgent
    ) -> None:
        assert len(agents) == 4, "Expected exactly 4 agents (4-player Catan)."
        self.engine = engine
        self.agents = agents

    def play_single_game(self) -> GameResult:
        state = self.engine.start_game()
        done = False
        steps: List[StepRecord] = []
        winner_index: Optional[int] = None

        while not done:
            player_idx = state.current_player_index
            agent = self.agents[player_idx]

            legal_actions = self.engine.get_legal_actions()
            action = agent.choose_action(state, legal_actions)

            # Collect hallucination metadata if available
            decision_info = getattr(agent, "last_decision_info", None)
            step_meta: Dict[str, Any] = {}
            if decision_info is not None:
                step_meta["llm_valid_index"] = getattr(
                    decision_info, "valid_index", True
                )
                step_meta["llm_used_fallback"] = getattr(
                    decision_info, "used_fallback", False
                )
                step_meta["raw_llm_response"] = getattr(
                    decision_info, "raw_response", ""
                )

            new_state, done, env_info = self.engine.step(action)
            step_meta.update(env_info)

            steps.append(
                StepRecord(
                    state_before=state,
                    state_after=new_state,
                    acting_player_index=player_idx,
                    action=action,
                    legal_actions_count=len(legal_actions),
                    info=step_meta,
                )
            )

            state = new_state
            if done:
                winner_index = env_info.get("winner_index")

        return GameResult(
            steps=steps,
            winner_index=winner_index,
            final_state=state,
        )

    def play_many_games(self, n_games: int) -> List[GameResult]:
        results: List[GameResult] = []
        for _ in range(n_games):
            results.append(self.play_single_game())
        return results
