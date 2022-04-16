import torch
import torch.nn as nn
import torch.nn.functional as F

import git
import numpy as np

import os
import time
import random as r
import csv

from maze import Maze, Move


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels=1,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    bias=False)

        self.m1 = nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1),
                                     bias=False)
        
        self.m2 = nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1),
                                     bias=False)

        self.m3 = nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1),
                                     bias=False)

        self.m4 = nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1),
                                     bias=False)

        self.m5 = nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1),
                                     bias=False)

        self.output_linear = nn.Linear(3136, 4)

        self.bnorm64 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.bnorm64(x)
        x = F.relu(x)

        x = self.m1(x)
        x = self.bnorm64(x)
        x = F.relu(x)

        x = self.m2(x)
        x = self.bnorm64(x)
        x = F.relu(x)

        x = self.m3(x)
        x = self.bnorm64(x)
        x = F.relu(x)

        x = self.m4(x)
        x = self.bnorm64(x)
        x = F.relu(x)

        x = self.m5(x)
        x = self.bnorm64(x)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.output_linear(x)
        x = F.softmax(x, dim=-1)

        return x


def train():
    learning_rate = 1e-7
    momentum = 0.7
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Prepare for saving the model
    # Create model save path if it does not exist
    if not os.path.exists('models'):
        os.makedirs('models')
    # Get time
    timestring = time.strftime("%y%m%d_%H%M%S")
    # Get the git branch hash if it exists
    try:
        repo = git.Repo(search_parent_directories=True)
        gitstring = repo.head.object.hexsha[:7]
    except:
        gitstring = "nogit"

    model_fn = f"models/global_model_{timestring}_{gitstring}.pt"

    # Set up network
    model = CNN().to(device)
    optimiser = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum)

    # Training
    losses = []
    start_time = time.time()
    games = range(100000)
    for game in games:
        loss = None

        # Generate a game
        rows = 7
        cols = 7
        # start_position = (r.choice(range(rows)), r.choice(range(cols)))
        start_position = [(0,0)]
        maze = Maze(rows, cols, start_position)
        running_reward = 0

        # Play the game
        while not maze.game_over:
            # Convert the game info into Tensor format
            grids = torch.tensor(maze.grids.astype(np.float32)).to(device)
            grids = grids.unsqueeze(0)

            # Run the model
            output = model(grids).squeeze()

            # Make a reward function based on the legal moves
            legal = [1.0 if Move(x) in maze.get_legal_moves(0) else 0.0 for x in range(4)]
            legal = torch.tensor(legal).to(device)
            reward = output * legal
            running_reward += reward.sum()

            # Select a move and run it
            move = torch.multinomial(output, 1)
            maze.move_player(0, Move(int(move)))
            maze.step()
            # maze.display()
            # Score the game        
            # loss = torch.tensor(float(-maze.score), requires_grad=True)

        loss = -running_reward

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
        # Save the result
        losses.append(loss.item())

        if game % 100 == 0:
            print(f"Game: {game+1}/{len(games)}")
            maze.display()
            print(f"Last move probability: {output.data}")
            print(f"Last chosen move: {Move(int(move))}")
            elapsed = time.time() - start_time
            print(
            f"Loss: {loss.item():.7f}, "
            f"Elapsed: {elapsed:.2f}s")

    print("Training complete")
    maze.display()

    elapsed = time.time() - start_time
    print(
    f"Loss: {loss.item():.7f}, "
    f"Elapsed: {elapsed:.2f}s")

    loss_fn = model_fn.replace(".pt", ".csv")
    with open(loss_fn, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\n")
        writer.writerow(["losses"])
        writer.writerow(losses)
    print("Saved loss data to:", loss_fn)


if __name__ == "__main__":
    train()
