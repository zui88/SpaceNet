import matplotlib.pyplot as plt


def save_print_history(history):
    fig, ax = plt.subplots()

    train = history.history["loss"]
    val = history.history["val_loss"]

    plt.scatter(len(train) - 1, train[-1])
    plt.scatter(len(val) - 1, val[-1])

    # plt.text(len(train) - 1, train[-1], f"{train[-1]:.4f}")
    plt.text(len(val) - 1, val[-1], f"{val[-1]:.4f}")

    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    # plt.yscale("log")
    ax.legend()
    ax.grid(True)

    fig.savefig("training_history.png", dpi=300)
    plt.show()
