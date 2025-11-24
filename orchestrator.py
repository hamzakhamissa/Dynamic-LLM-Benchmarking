# orchestrator.py - Enhanced with better tracking
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
    info: Dict[str, Any]


@dataclass
class GameResult:
    steps: List[StepRecord]
    winner_index: Optional[int]
    final_state: GameState


class GameOrchestrator:
    """
    Enhanced orchestrator with better error tracking.
    """

    def __init__(
        self,
        engine: CatanEngine,
        agents: List[object],
    ) -> None:
        assert len(agents) == 4, "Expected exactly 4 agents (4-player Catan)."
        self.engine = engine
        self.agents = agents

    def play_single_game(self) -> GameResult:
        state = self.engine.start_game()
        done = False
        steps: List[StepRecord] = []
        winner_index: Optional[int] = None

        turn_count = 0
        while not done:
            turn_count += 1
            player_idx = state.current_player_index
            agent = self.agents[player_idx]

            legal_actions = self.engine.get_legal_actions()
            
            if not legal_actions:
                print(f"⚠️ Turn {turn_count}: No legal actions for Player {player_idx}!")
                break

            # Get action from agent
            action = agent.choose_action(state, legal_actions)

            # Collect decision metadata
            decision_info = getattr(agent, "last_decision_info", None)
            step_meta: Dict[str, Any] = {}
            
            if decision_info is not None:
                step_meta["llm_valid_index"] = getattr(decision_info, "valid_index", True)
                step_meta["llm_used_fallback"] = getattr(decision_info, "used_fallback", False)
                step_meta["llm_api_error"] = getattr(decision_info, "api_error", False)
                step_meta["raw_llm_response"] = getattr(decision_info, "raw_response", "")

            # Execute action in environment
            new_state, done, env_info = self.engine.step(action)
            step_meta.update(env_info)

            # Record step
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
        for game_num in range(n_games):
            print(f"\n{'='*60}")
            print(f"Starting Game {game_num + 1}/{n_games}")
            print(f"{'='*60}")
            result = self.play_single_game()
            results.append(result)
            
            # Print game summary
            if result.winner_index is not None:
                winner_name = self.agents[result.winner_index].name
                print(f"\n🏆 Winner: {winner_name} (Player {result.winner_index})")
            else:
                print(f"\n⚠️ Game ended without winner (turn limit reached)")
            print(f"Final VP: {result.final_state.victory_points}")
            print(f"Total turns: {result.final_state.turn}")
            
        return results