from enum import Enum
import numpy as np

class Move(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Maze:

    def __init__(self, rows, cols, player_locs):
        self.rows = rows
        self.cols = cols
        self.cells = rows * cols

        # Make sure we have enough numbers in UINT8_MAX to cover the whole grid
        # Each player taking turns sequentially is not currently enforced
        self.num_players = len(player_locs)
        if self.cells > 254 * self.num_players:
            raise ValueError("Grid is too large for this array datatype")
        
        self.grids = np.zeros((self.num_players, rows, cols), dtype=np.uint8)

        for n, (x, y) in enumerate(player_locs):
            self.grids[n, x, y] = 255

        self.moved_players = set()
        self.game_over = False

        self.update_score()

    def step(self):
        if len(self.moved_players) < self.num_players:
            raise Exception("Not all players have moved")

        self.update_score()
        self.moved_players.clear()
        # maze.display()

    def update_score(self):
        flat = self.grids.any(axis=0)
        self.score = flat.sum()

    def get_player_loc(self, player):
        index = self.grids[player].argmax()
        row, col = np.unravel_index(index, self.grids[player].shape)
        return row, col

    def move_player(self, player, direction):
        if player in self.moved_players:
            raise ValueError("This player has already moved this turn")

        if type(direction) != Move:
            raise TypeError("Given direction not a valid Move")

        self.moved_players.add(player)

        if direction in self.get_legal_moves(player):
            grid = self.grids[player]
            row, col = self.get_player_loc(player)

            np.clip(grid, 1, 255, out=grid)
            grid -= 1

            if direction == Move.UP:
                row -= 1
            elif direction == Move.DOWN:
                row += 1
            elif direction == Move.LEFT:
                col -= 1
            elif direction == Move.RIGHT:
                col += 1

            grid[row, col] = 255

        else:
            self.game_over = True

    def get_legal_moves(self, player):
        moves = []
        row, col = self.get_player_loc(player)
        flat = self.grids.any(axis=0)

        # UP
        if row - 1 > 0:
            if not flat[row - 1, col]:
                moves.append(Move.UP)
        # DOWN
        if row + 1 < self.rows:
            if not flat[row + 1, col]:
                moves.append(Move.DOWN)
        # LEFT
        if col - 1 > 0:
            if not flat[row, col - 1]:
                moves.append(Move.LEFT)
        # RIGHT
        if col + 1 < self.cols:
            if not flat[row, col + 1]:
                moves.append(Move.RIGHT)

        return moves

    def display(self):
        if self.game_over:
            print("\nGame over")
        else:
            print("\nOngoing game")
        print(self.grids)
        print(f"Current score: {self.score}/{self.cells}")

        for player in range(self.num_players):
            moves = self.get_legal_moves(player)
            movelist = [f"{x.value}: {x.name}" for x in moves]
            print(
                f"Player {player}'s Legal Moves: {movelist}"
            )
