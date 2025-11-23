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
    BANK_TRADE = auto()     # 4:1 trade with bank
    PORT_TRADE = auto()     # 3:1 or 2:1 trade at port
    MOVE_ROBBER = auto()    # Move robber and steal
    DISCARD = auto()        # Discard cards when robber rolled
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
        self.knights_played: int = 0

    def has_resources(self, cost: Dict[Resource, int]) -> bool:
        return all(self.resources.get(res, 0) >= amt for res, amt in cost.items())

    def pay_resources(self, cost: Dict[Resource, int]) -> None:
        for res, amt in cost.items():
            self.resources[res] = self.resources.get(res, 0) - amt
    
    def total_cards(self) -> int:
        return sum(self.resources.values())


class StubBoard:
    """Minimal board to keep tests running without the real pycatan dependency."""

    def __init__(self, rng: random.Random) -> None:
        # 12 intersections arranged linearly; 11 connecting paths
        self.intersections: Dict[int, Optional[Building]] = {i: None for i in range(12)}
        self.paths: Dict[Tuple[int, int], Optional[Player]] = {
            (i, i + 1): None for i in range(11)
        }
        self.rng = rng
        self.robber_position: int = 6  # Start in middle

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
        
        # Longest road bonus (2 VP if you have 5+ roads)
        if self.longest_road_owner == player:
            vp += 2
        
        # Largest army bonus (2 VP if you played 3+ knights)
        if self.largest_army_owner == player:
            vp += 2
            
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
    robber_position: int
    pending_discard: bool  # True if waiting for players to discard


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

    New features:
    - Bank trades (4:1)
    - Robber mechanic (7 rolls)
    - Discard phase
    - Better hallucination tracking
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
        self.pending_discard: bool = False
        self.players_needing_discard: List[int] = []

    # ------------- Public API -------------

    def start_game(self) -> GameState:
        board = StubBoard(self.rng)
        self.game = PyCatanGame(board, num_players=self.num_players, rng=self.rng)
        self.turn = 0
        self.current_player_index = 0
        self.pending_discard = False
        self.players_needing_discard = []

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

        # If we're in discard phase, only allow discards
        if self.pending_discard and self.current_player_index in self.players_needing_discard:
            return self._discard_actions(player)

        # Build actions, conditioned on having enough resources
        actions.extend(self._settlement_actions(player))
        actions.extend(self._road_actions(player))
        actions.extend(self._city_actions(player))

        # Trade actions
        actions.extend(self._trade_actions(player))
        actions.extend(self._bank_trade_actions(player))

        # Always allow end turn
        actions.append(Action(ActionType.END_TURN, payload={}))

        return actions

    def step(self, action: Action) -> Tuple[GameState, bool, Dict[str, Any]]:
        assert self.game is not None
        info: Dict[str, Any] = {}
        player = self._current_player()
        action_succeeded = True

        # Handle discard phase separately
        if self.pending_discard and self.current_player_index in self.players_needing_discard:
            try:
                self._apply_discard(player, action)
                self.players_needing_discard.remove(self.current_player_index)
                
                # Move to next player needing discard
                if self.players_needing_discard:
                    self.current_player_index = self.players_needing_discard[0]
                else:
                    # All discards done, continue to robber move
                    self.pending_discard = False
                    
                state = self._export_state()
                return state, False, info
            except Exception as e:
                action_succeeded = False
                info["action_error"] = str(e)
                info["action_failed"] = True
                print(f"⚠️ Discard failed: {e}")

        # Roll dice & distribute resources at the start of each turn
        roll = self.rng.randint(1, 6) + self.rng.randint(1, 6)
        info["roll"] = roll
        
        # Handle robber (7)
        if roll == 7:
            info["robber_rolled"] = True
            # Find players who need to discard
            self.players_needing_discard = [
                i for i, p in enumerate(self.game.players)
                if p.total_cards() > 7
            ]
            if self.players_needing_discard:
                self.pending_discard = True
                self.current_player_index = self.players_needing_discard[0]
                state = self._export_state()
                return state, False, info
        else:
            self.game.add_yield_for_roll(roll)
        
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
            elif action.type == ActionType.BANK_TRADE:
                give_res = action.payload["give"]
                get_res = action.payload["receive"]
                self._apply_bank_trade(player, give_res, get_res)
                info["is_bank_trade"] = True
                info["bank_trade_payload"] = action.payload
            elif action.type == ActionType.END_TURN:
                pass
            else:
                raise ValueError(f"Unknown action type: {action.type}")
        except Exception as e:
            # Track action failures as hallucinations
            action_succeeded = False
            info["action_error"] = str(e)
            info["action_failed"] = True
            print(f"Action failed: {e}")

        # Record whether action succeeded
        info["action_succeeded"] = action_succeeded

        # Advance turn
        self.turn += 1
        self.current_player_index = (self.current_player_index + 1) % self.num_players

        # Update longest road and largest army
        self._update_special_achievements()

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

    def _bank_trade_actions(self, player: Player) -> List[Action]:
        """4:1 bank trades - trade 4 of one resource for 1 of another."""
        actions: List[Action] = []
        player_res = self._player_resources(self.current_player_index)
        
        for give_res in RESOURCE_LIST:
            give_name = give_res.name.lower()
            if player_res.get(give_name, 0) >= 4:
                for get_res in RESOURCE_LIST:
                    if give_res != get_res:
                        get_name = get_res.name.lower()
                        actions.append(
                            Action(
                                type=ActionType.BANK_TRADE,
                                payload={
                                    "give": give_name,
                                    "receive": get_name,
                                },
                            )
                        )
        
        return actions

    def _discard_actions(self, player: Player) -> List[Action]:
        """Generate discard actions when player has >7 cards after 7 roll."""
        actions: List[Action] = []
        total = player.total_cards()
        discard_count = total // 2
        
        if discard_count == 0:
            # No discard needed, but we need at least one action
            return [Action(ActionType.END_TURN, payload={})]
        
        # Generate all possible single-resource discards
        for res in RESOURCE_LIST:
            res_count = player.resources.get(res, 0)
            if res_count >= discard_count and discard_count > 0:
                actions.append(
                    Action(
                        type=ActionType.DISCARD,
                        payload={
                            "resource": res.name.lower(),
                            "count": discard_count,
                        },
                    )
                )
        
        # If no single resource has enough, offer to discard whatever we have
        if not actions and discard_count > 0:
            # Find the resource with the most cards
            max_res = max(RESOURCE_LIST, key=lambda r: player.resources.get(r, 0))
            max_count = player.resources.get(max_res, 0)
            if max_count > 0:
                actions.append(
                    Action(
                        type=ActionType.DISCARD,
                        payload={
                            "resource": max_res.name.lower(),
                            "count": min(discard_count, max_count),
                        },
                    )
                )
        
        # Fallback: if still no actions, allow END_TURN
        if not actions:
            actions.append(Action(ActionType.END_TURN, payload={}))
        
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
            raise ValueError(f"Player {from_idx} doesn't have {res_name} to trade")

        from_player.resources[res_enum] -= 1
        to_player.resources[res_enum] = to_player.resources.get(res_enum, 0) + 1

    def _apply_bank_trade(self, player: Player, give_res: str, get_res: str) -> None:
        """4:1 bank trade."""
        give_enum = Resource[give_res.upper()]
        get_enum = Resource[get_res.upper()]
        
        if player.resources.get(give_enum, 0) < 4:
            raise ValueError(f"Need 4 {give_res} for bank trade")
        
        player.resources[give_enum] -= 4
        player.resources[get_enum] = player.resources.get(get_enum, 0) + 1

    def _apply_discard(self, player: Player, action: Action) -> None:
        """Apply discard action."""
        if action.type == ActionType.END_TURN:
            # No discard needed (player had ≤7 cards)
            return
            
        if action.type != ActionType.DISCARD:
            raise ValueError("Expected DISCARD action during discard phase")
        
        res_name = action.payload["resource"]
        count = action.payload["count"]
        res_enum = Resource[res_name.upper()]
        
        available = player.resources.get(res_enum, 0)
        if available < count:
            # Discard what we can
            actual_discard = available
            print(f"Player can only discard {actual_discard} {res_name}, not {count}")
        else:
            actual_discard = count
        
        if actual_discard > 0:
            player.resources[res_enum] -= actual_discard

    def _update_special_achievements(self) -> None:
        """Update longest road and largest army owners."""
        assert self.game is not None
        
        # Longest road: 5+ roads
        for player in self.game.players:
            if len(player.roads) >= 5:
                if self.game.longest_road_owner is None or len(player.roads) > len(self.game.longest_road_owner.roads):
                    self.game.longest_road_owner = player
        
        # Largest army: 3+ knights
        for player in self.game.players:
            if player.knights_played >= 3:
                if self.game.largest_army_owner is None or player.knights_played > self.game.largest_army_owner.knights_played:
                    self.game.largest_army_owner = player

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
            robber_position=self.game.board.robber_position,
            pending_discard=self.pending_discard,
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