import maze



if __name__ == "__main__":
    maze = Maze(4, 5, [(0, 0), (3, 4)])
    maze.display()

    maze.move_player(0, Move.DOWN)
    maze.move_player(1, Move.UP)
    maze.step()
    maze.move_player(0, Move.RIGHT)
    maze.move_player(1, Move.LEFT)
    maze.step()
    maze.move_player(0, Move.DOWN)
    maze.move_player(1, Move.UP)
    maze.step()
    maze.move_player(0, Move.DOWN)
    maze.move_player(1, Move.UP)
    maze.step()
    maze.move_player(0, Move.RIGHT)
    maze.move_player(1, Move.RIGHT)
    maze.step()
    maze.move_player(0, Move.DOWN)
    maze.move_player(1, Move.DOWN)
    maze.step()

    maze.display()

    