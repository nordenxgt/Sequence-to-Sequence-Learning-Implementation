import os
import matplotlib.pyplot as plt
from typing import List

def plot_loss(train_losses: List[int], val_losses: List[int], save: bool = False, show: bool = False) -> None:
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.title("Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    
    if save:
        if not os.path.exists("results"): os.makedirs("results")
        plt.savefig(f"results/loss.png")
    
    if show:
        plt.show()