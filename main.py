import torch
import torch.nn as nn
import numpy as np
import ptan
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box 

class DummyNet(nn.Module): 
  def __init__(self,obs_size, n_actions): 
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(obs_size, 16), 
        nn.ReLU(), 
        nn.Linear(16, n_actions)
    )
  
  def forward(self,x): 
    return self.net(x)


if __name__ == "__main__":
    console = Console()

    obs_size = 4
    n_actions = 2

    net = DummyNet(obs_size, n_actions)

    # Fake observation (like CartPole)
    obs = np.array([0.1, 0.0, -0.2, 0.05], dtype=np.float32)

    obs_v = torch.tensor([obs])   # shape: [1, obs_size]
    q_vals = net(obs_v)
    q_vals_np = q_vals.detach().numpy()[0]
    
    selector = ptan.actions.ArgmaxActionSelector()
    action = selector(q_vals.detach().numpy())

    print("Chosen action (greedy):", action)

    # Create a rich table for observations
    obs_table = Table(title="Input Observation", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    obs_table.add_column("Index", style="dim", justify="center")
    obs_table.add_column("Value", justify="right", style="yellow")

    for idx, val in enumerate(obs):
        obs_table.add_row(str(idx), f"{val:.4f}")

    # Create a rich table for Q-values
    q_table = Table(title="Q-Values", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    q_table.add_column("Action", style="dim", justify="center")
    q_table.add_column("Q-Value", justify="right", style="green")

    for action_idx, q_val in enumerate(q_vals_np):
        q_table.add_row(f"Action {action_idx}", f"{q_val:.6f}")

    # Best action
    best_action = np.argmax(q_vals_np)

    console.print("\n")
    console.print(Panel.fit(
        "[bold blue]DQN Network Output Demo[/bold blue]",
        border_style="blue"
    ))
    console.print(obs_table)
    console.print(q_table)
    console.print(Panel(
        f"[bold green]Best Action: {best_action}[/bold green] (Q-value: {q_vals_np[best_action]:.6f})",
        border_style="green",
        title="Recommendation"
    ))
    console.print("\n")
