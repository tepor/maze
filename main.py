import numpy as np

class Move:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Maze:

    def __init__(self, rows, cols, player_locs):
        self.rows = rows
        self.cols = cols
        self.cells = rows * cols
        self.num_players = len(player_locs)

        self.grids = np.zeros((self.num_players, rows, cols), dtype=np.uint8)

        for n, (x, y) in enumerate(player_locs):
            self.grids[n, x, y] = 255

        self.game_over = False


        self.update_score()

    def update_score(self):
        flat = self.grids.any(axis=0)
        self.score = flat.sum()

    def get_player_loc(self, player):
        row = self.grids[player].argmax(axis=0).max()
        col = self.grids[player].argmax(axis=1).max()
        return row, col

    def move_player(self, player, direction):
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
            self.update_score()

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
            print(
                f"Player {player}'s Legal Moves: {self.get_legal_moves(player)}"
            )


if __name__ == "__main__":
    maze = Maze(4, 5, [(0, 0), (3, 4)])
    maze.display()

    maze.move_player(0, "DOWN")
    maze.move_player(0, "DOWN")
    maze.move_player(0, "RIGHT")
    maze.move_player(0, "RIGHT")
    maze.move_player(0, "DOWN")
    maze.move_player(0, "UP")

    maze.display()