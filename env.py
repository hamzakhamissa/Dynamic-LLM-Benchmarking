# env.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import random


class Resource(Enum):
    BRICK = auto()
    LUMBER = auto()
    WOOL = auto()
    GRAIN = auto()
    ORE = auto()


class ActionType(Enum):
    BUILD_SETTLEMENT = auto()
    BUILD_ROAD = auto()
    BUILD_CITY = auto()
    TRADE = auto()          # gift-style trade: give 1 resource to another player
    END_TURN = auto()


class BuildingType(Enum):
    SETTLEMENT = auto()
    CITY = auto()


@dataclass
class Building:
    owner: "Player"
    building_type: BuildingType


class Player:
    def __init__(self) -> None:
        self.resources: Dict[Resource, int] = {}
        self.roads: set[Tuple[int, int]] = set()

    def has_resources(self, cost: Dict[Resource, int]) -> bool:
        return all(self.resources.get(res, 0) >= amt for res, amt in cost.items())

    def pay_resources(self, cost: Dict[Resource, int]) -> None:
        for res, amt in cost.items():
            self.resources[res] = self.resources.get(res, 0) - amt


class StubBoard:
    """Minimal board to keep tests running without the real pycatan dependency."""

    def __init__(self, rng: random.Random) -> None:
        # 12 intersections arranged linearly; 11 connecting paths
        self.intersections: Dict[int, Optional[Building]] = {i: None for i in range(12)}
        self.paths: Dict[Tuple[int, int], Optional[Player]] = {
            (i, i + 1): None for i in range(11)
        }
        self.rng = rng

    def get_valid_settlement_coords(self, player: Player, ensure_connected: bool) -> List[int]:
        # Allow building on any empty intersection
        return [c for c, b in self.intersections.items() if b is None]

    def assert_valid_road_coords(
        self, player: Player, path_coords: Tuple[int, int], ensure_connected: bool
    ) -> None:
        if path_coords not in self.paths:
            raise ValueError("Invalid path")
        if self.paths[path_coords] is not None:
            raise ValueError("Path already taken")


class StubGame:
    def __init__(self, board: StubBoard, num_players: int, rng: random.Random) -> None:
        self.board = board
        self.players = [Player() for _ in range(num_players)]
        self.longest_road_owner: Optional[Player] = None
        self.largest_army_owner: Optional[Player] = None
        self.rng = rng

    def build_settlement(
        self,
        player: Player,
        coords: int,
        cost_resources: bool,
        ensure_connected: bool,
    ) -> None:
        if coords not in self.board.intersections or self.board.intersections[coords] is not None:
            raise ValueError("Invalid or occupied settlement location")
        if cost_resources and not player.has_resources(SETTLEMENT_COST):
            raise ValueError("Insufficient resources for settlement")
        if cost_resources:
            player.pay_resources(SETTLEMENT_COST)
        self.board.intersections[coords] = Building(owner=player, building_type=BuildingType.SETTLEMENT)

    def build_road(
        self,
        player: Player,
        path_coords: Tuple[int, int],
        cost_resources: bool,
        ensure_connected: bool,
    ) -> None:
        if path_coords not in self.board.paths:
            raise ValueError("Invalid road path")
        if self.board.paths[path_coords] is not None:
            raise ValueError("Road already exists")
        if cost_resources and not player.has_resources(ROAD_COST):
            raise ValueError("Insufficient resources for road")
        if cost_resources:
            player.pay_resources(ROAD_COST)
        self.board.paths[path_coords] = player
        player.roads.add(path_coords)

    def upgrade_settlement_to_city(
        self, player: Player, coords: int, cost_resources: bool
    ) -> None:
        building = self.board.intersections.get(coords)
        if building is None or building.owner is not player:
            raise ValueError("No settlement to upgrade")
        if building.building_type != BuildingType.SETTLEMENT:
            raise ValueError("Already a city here")
        if cost_resources and not player.has_resources(CITY_COST):
            raise ValueError("Insufficient resources for city")
        if cost_resources:
            player.pay_resources(CITY_COST)
        self.board.intersections[coords] = Building(owner=player, building_type=BuildingType.CITY)

    def add_yield_for_roll(self, roll: int) -> None:
        # Give each player one random resource to keep the game progressing.
        for player in self.players:
            res = self.rng.choice(RESOURCE_LIST)
            player.resources[res] = player.resources.get(res, 0) + 1

    def get_victory_points(self, player: Player) -> int:
        vp = 0
        for building in self.board.intersections.values():
            if building is None or building.owner is not player:
                continue
            vp += 2 if building.building_type == BuildingType.CITY else 1
        return vp


PyCatanGame = StubGame
Coords = int


@dataclass(frozen=True)
class Action:
    """Discrete game action that the LLM chooses from."""
    type: ActionType
    payload: Dict[str, Any]


@dataclass
class GameState:
    """Compact snapshot we pass to agents & metrics."""
    turn: int
    current_player_index: int
    victory_points: List[int]
    resources: List[Dict[str, int]]   # per-player {resource_name: count}
    longest_road_owner: Optional[int]
    largest_army_owner: Optional[int]


class CatanEngine:
    """Abstract interface so you can swap implementations if needed."""

    def start_game(self) -> GameState:
        raise NotImplementedError

    def get_legal_actions(self) -> List[Action]:
        raise NotImplementedError

    def step(self, action: Action) -> Tuple[GameState, bool, Dict[str, Any]]:
        """
        Apply action.

        Returns:
          - new_state
          - done (True if game over)
          - info: arbitrary dict (e.g., winner, rollout info)
        """
        raise NotImplementedError


# ---- Concrete implementation using pycatan ---------------------------------


RESOURCE_LIST = [
    Resource.LUMBER,
    Resource.BRICK,
    Resource.WOOL,
    Resource.GRAIN,
    Resource.ORE,
]

# Standard Catan costs
SETTLEMENT_COST = {
    Resource.BRICK: 1,
    Resource.LUMBER: 1,
    Resource.WOOL: 1,
    Resource.GRAIN: 1,
}
ROAD_COST = {
    Resource.BRICK: 1,
    Resource.LUMBER: 1,
}
CITY_COST = {
    Resource.ORE: 3,
    Resource.GRAIN: 2,
}


class PyCatanEngine(CatanEngine):
    """
    Lightweight, dependency-free Catan-style engine.

    Simplifications:
    - Minimal board (12 intersections in a line, 11 paths).
    - One free starting settlement for each player.
    - Per-turn resource drip so actions remain available.
    - Trade = one-way "gift" of 1 resource from current player to another.
    """

    def __init__(
        self,
        num_players: int = 4,
        target_vp: int = 10,
        max_turns: int = 500,
        seed: Optional[int] = None,
    ) -> None:
        assert num_players == 4, "This benchmark assumes exactly 4 players."
        self.num_players = num_players
        self.target_vp = target_vp
        self.max_turns = max_turns
        self.rng = random.Random(seed)
        self.game: Optional[PyCatanGame] = None
        self.current_player_index: int = 0
        self.turn: int = 0

    # ------------- Public API -------------

    def start_game(self) -> GameState:
        board = StubBoard(self.rng)
        self.game = PyCatanGame(board, num_players=self.num_players, rng=self.rng)
        self.turn = 0
        self.current_player_index = 0

        # Seed each player with a few starting resources so they can act.
        for player in self.game.players:
            for res in RESOURCE_LIST:
                player.resources[res] = 2

        # One free starting settlement for each player
        for idx, player in enumerate(self.game.players):
            settlement_coords = list(
                self.game.board.get_valid_settlement_coords(
                    player=player, ensure_connected=False
                )
            )
            if not settlement_coords:
                continue
            coords = self.rng.choice(settlement_coords)
            self.game.build_settlement(
                player=player,
                coords=coords,
                cost_resources=False,
                ensure_connected=False,
            )

        return self._export_state()

    def get_legal_actions(self) -> List[Action]:
        assert self.game is not None
        player = self._current_player()
        actions: List[Action] = []

        # Build actions, conditioned on having enough resources
        actions.extend(self._settlement_actions(player))
        actions.extend(self._road_actions(player))
        actions.extend(self._city_actions(player))

        # Trade actions (gift 1 resource to someone else)
        actions.extend(self._trade_actions(player))

        # Always allow end turn
        actions.append(Action(ActionType.END_TURN, payload={}))

        return actions

    def step(self, action: Action) -> Tuple[GameState, bool, Dict[str, Any]]:
        assert self.game is not None
        info: Dict[str, Any] = {}
        player = self._current_player()

        # Roll dice & distribute resources at the start of each turn
        roll = self.rng.randint(1, 6) + self.rng.randint(1, 6)
        self.game.add_yield_for_roll(roll)
        info["roll"] = roll
        
        # Debug: print progress every 10 turns
        if self.turn % 10 == 0:
            vp = [self.game.get_victory_points(p) for p in self.game.players]
            print(f"Turn {self.turn}: VP = {vp}")

        # Apply action
        try:
            if action.type == ActionType.BUILD_SETTLEMENT:
                coords = action.payload["coords"]
                self.game.build_settlement(
                    player=player,
                    coords=coords,
                    cost_resources=True,
                    ensure_connected=True,
                )
            elif action.type == ActionType.BUILD_ROAD:
                path_coords = action.payload["path"]
                self.game.build_road(
                    player=player,
                    path_coords=path_coords,
                    cost_resources=True,
                    ensure_connected=True,
                )
            elif action.type == ActionType.BUILD_CITY:
                coords = action.payload["coords"]
                self.game.upgrade_settlement_to_city(
                    player=player,
                    coords=coords,
                    cost_resources=True,
                )
            elif action.type == ActionType.TRADE:
                # Gift style: give 1 unit of some resource to another player
                to_idx = int(action.payload["to_player"])
                res_name = action.payload["resource"]
                self._apply_trade(
                    from_idx=self.current_player_index,
                    to_idx=to_idx,
                    res_name=res_name,
                )
                info["is_trade"] = True
                info["trade_payload"] = action.payload
            elif action.type == ActionType.END_TURN:
                pass
            else:
                raise ValueError(f"Unknown action type: {action.type}")
        except Exception as e:
            # If action fails (shouldn't happen with legal actions), log and continue
            info["action_error"] = str(e)
            print(f"Warning: Action failed: {e}")

        # Advance turn
        self.turn += 1
        self.current_player_index = (self.current_player_index + 1) % self.num_players

        state = self._export_state()
        winner_index = self._get_winner_index(state)

        done = winner_index is not None or self.turn >= self.max_turns
        if done and winner_index is not None:
            info["winner_index"] = winner_index

        return state, done, info

    # ------------- Helpers -------------

    def _current_player(self):
        assert self.game is not None
        return self.game.players[self.current_player_index]

    def _player_resources(self, player_idx: int) -> Dict[str, int]:
        assert self.game is not None
        player = self.game.players[player_idx]
        out: Dict[str, int] = {}
        for res in RESOURCE_LIST:
            out[res.name.lower()] = player.resources.get(res, 0)
        return out

    def _has_resources(self, player, cost: Dict[Resource, int]) -> bool:
        return player.has_resources(cost)

    def _settlement_actions(self, player) -> List[Action]:
        assert self.game is not None
        actions: List[Action] = []
        if not self._has_resources(player, SETTLEMENT_COST):
            return actions

        for coords in self.game.board.get_valid_settlement_coords(
            player=player, ensure_connected=True
        ):
            actions.append(
                Action(
                    ActionType.BUILD_SETTLEMENT,
                    payload={"coords": coords},
                )
            )
        return actions

    def _road_actions(self, player) -> List[Action]:
        assert self.game is not None
        actions: List[Action] = []
        if not self._has_resources(player, ROAD_COST):
            return actions

        for path_coords in self.game.board.paths.keys():
            try:
                self.game.board.assert_valid_road_coords(
                    player=player,
                    path_coords=path_coords,
                    ensure_connected=True,
                )
            except Exception:
                continue

            actions.append(
                Action(
                    ActionType.BUILD_ROAD,
                    payload={"path": path_coords},
                )
            )

        return actions

    def _city_actions(self, player) -> List[Action]:
        assert self.game is not None
        actions: List[Action] = []
        if not self._has_resources(player, CITY_COST):
            return actions

        # Upgrade any existing settlement belonging to this player
        for coords, building in self.game.board.intersections.items():
            if building is None:
                continue

            building_owner = getattr(building, "owner", None)
            if building_owner is not player:
                continue

            building_type = getattr(building, "building_type", None)
            if isinstance(building_type, Enum):
                building_type = building_type.name

            if building_type != "SETTLEMENT":
                continue

            actions.append(
                Action(
                    ActionType.BUILD_CITY,
                    payload={"coords": coords},
                )
            )

        return actions

    def _trade_actions(self, player: Player) -> List[Action]:

        actions: List[Action] = []

        # Current player's resources
        player_res = self._player_resources(self.current_player_index)
        # Resource types present in this game
        res_types = [r.name.lower() for r in RESOURCE_LIST]

        # If player has no resources, they can't offer anything
        if sum(player_res.values()) == 0:
            return actions

        for give_res in res_types:
            if player_res.get(give_res, 0) <= 0:
                continue  # can't offer what you don't have

            for other_idx in range(self.num_players):
                if other_idx == self.current_player_index:
                    continue

                actions.append(
                    Action(
                        type=ActionType.TRADE,
                        payload={
                            "to_player": other_idx,
                            "resource": give_res,
                        },
                    )
                )

        return actions



    def _apply_trade(self, from_idx: int, to_idx: int, res_name: str) -> None:
        """
        Apply a gift trade: transfer one unit of `res_name` from the current player
        to another player.
        """
        assert self.game is not None

        res_enum = Resource[res_name.upper()]
        from_player = self.game.players[from_idx]
        to_player = self.game.players[to_idx]

        # Sanity check (should already be enforced by _trade_actions)
        if from_player.resources.get(res_enum, 0) <= 0:
            return

        from_player.resources[res_enum] -= 1
        to_player.resources[res_enum] = to_player.resources.get(res_enum, 0) + 1


    def _export_state(self) -> GameState:
        assert self.game is not None
        vp_list = [
            self.game.get_victory_points(p) for p in self.game.players
        ]
        res_list = [self._player_resources(i) for i in range(self.num_players)]

        longest_owner = None
        if self.game.longest_road_owner is not None:
            longest_owner = self.game.players.index(self.game.longest_road_owner)

        largest_owner = None
        if self.game.largest_army_owner is not None:
            largest_owner = self.game.players.index(self.game.largest_army_owner)

        return GameState(
            turn=self.turn,
            current_player_index=self.current_player_index,
            victory_points=vp_list,
            resources=res_list,
            longest_road_owner=longest_owner,
            largest_army_owner=largest_owner,
        )

    def _get_winner_index(self, state: GameState) -> Optional[int]:
        # Winner = any player hitting target_vp; tie-break: highest VP
        best_vp = max(state.victory_points)
        if best_vp < self.target_vp:
            return None
        winners = [
            i for i, vp in enumerate(state.victory_points) if vp == best_vp
        ]
        # If tie, just pick first
        return winners[0]
