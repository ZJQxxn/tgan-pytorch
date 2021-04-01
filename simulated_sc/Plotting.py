import numpy as np
import matplotlib.pyplot as plt

params = {
    "legend.fontsize": 14,
    "legend.frameon": False,
    "ytick.labelsize": 14,
    "xtick.labelsize": 14,
    # "figure.dpi": 600,
    "axes.prop_cycle": plt.cycler("color", plt.cm.Accent(np.linspace(0, 1, 5))),
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "pdf.fonttype": 42,
    "font.sans-serif": "CMU Serif",
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
}
plt.rcParams.update(params)



def plotTrainingHistory(filename):
    # reading data
    training_history = np.load(filename, allow_pickle=True).item()
    dis_loss = training_history["dis_loss_history"]
    gen_loss = training_history["gen_loss_history"]
    real_loss = training_history["real_loss_history"]
    MSE = training_history["MSE_history"]
    correlation = training_history["correlation_history"]
    correlation = [i[0] for i in correlation]
    epochs = len(dis_loss)
    # ploting
    plt.figure(figsize=(8,8))
    plt.title("Wasserstein Divergence vs. Epoch", fontsize = 20)
    plt.plot(np.arange(epochs), dis_loss, lw = 4)
    plt.xticks(fontsize = 15)
    plt.xlabel("Epoch", fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.ylabel("Wasserstein divergence", fontsize = 20)
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(8,8))
    plt.title("Generated Data Loss", fontsize=20)
    plt.plot(np.arange(epochs), -np.array(gen_loss), lw=4)
    plt.xticks(fontsize=15)
    plt.xlabel("Epoch", fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylabel("loss", fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(8,8))
    plt.title("Real Data Loss", fontsize=20)
    plt.plot(np.arange(epochs), real_loss, lw=4)
    plt.xticks(fontsize=15)
    plt.xlabel("Epoch", fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylabel("loss", fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10,5))
    plt.title("Mean Squared Error", fontsize=20)
    plt.plot(np.arange(epochs), MSE, lw=4)
    plt.xticks(fontsize=15)
    plt.xlabel("Epoch", fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylabel("mean squared error", fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10,5))
    plt.title("Pearson Correlation", fontsize=20)
    plt.plot(np.arange(epochs), correlation, lw=4)
    plt.xticks(fontsize=15)
    plt.xlabel("Epoch", fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylabel("correlation", fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.close()



if __name__ == '__main__':
    filename = "./mose_cell_training_history.npy"
    plotTrainingHistory(filename)