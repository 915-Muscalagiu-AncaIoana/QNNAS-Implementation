from matplotlib import pyplot as plt


def plot_rewards(rewards, title='Rewards per Episode', xlabel='Episode', ylabel='Reward'):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Rewards per Episode', color='blue', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()