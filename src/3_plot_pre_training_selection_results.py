from init import *

def properties(data, mode = 0):
    """Extracts the different properties from the [data] dictionnary of performances. Either outputs the raw arrays of performances (mode = 0)
       or outputs the performances at the optimum (mode = 1).
       pre: data (dict {string:np.ndarray}): Contains the evolution of the validation set performances through time and through the different experiments.
                                             The four keys are : "loss", "val_loss", "val_acc", "val_AUC".
            mode (int): Changes the output to either be the raw arrays of performances (at mode = 0), or the best performances across trials (mode = 1)
       post: data_loss (np.ndarray): the training loss of the model through the epochs (axis 0) and through the experiments (axis 1)
             data_val_loss (np.ndarray): the validation loss of the model through the epochs (axis 0) and through the experiments (axis 1)
             data_val_acc (np.ndarray): the validation accuracy of the model through the epochs (axis 0) and through the experiments (axis 1)
             data_val_AUC (np.ndarray): the validation AUC of the model through the epochs (axis 0) and through the experiments (axis 1)
             
             min_data_val_loss (np.1darray): the minimum validation loss through the experiments
             epoch_min_data_val_loss (np.1darray): epoch at which the minimum validation loss occurs through the experiments
             acc_at_data_val_loss (np.1darray): accuracy at the minimum validation loss through all the experiments
             AUC_at_data_val_loss (np.1darray): AUC at the minimum validation loss through all the experiments

    """

    data_loss = data["loss"]
    data_val_loss = data["val_loss"]
    data_val_acc = data["val_acc"]
    data_val_AUC = data["val_AUC"]

    min_data_val_loss = np.min(data_val_loss, axis=1)
    epoch_min_data_val_loss = np.argmin(data_val_loss, axis=1)
    acc_at_min_val_loss = np.array([data_val_acc[i,j] for i,j in zip(np.arange(data_val_acc.shape[0]),epoch_min_data_val_loss)])
    AUC_at_min_val_loss = np.array([data_val_AUC[i,j] for i,j in zip(np.arange(data_val_AUC.shape[0]),epoch_min_data_val_loss)])

    if mode==0:
        return data_loss, data_val_loss, data_val_acc, data_val_AUC
    if mode==1:
        return min_data_val_loss, epoch_min_data_val_loss + 1, acc_at_min_val_loss, AUC_at_min_val_loss

def stats_at_min_val_loss(properties):
    """Extracts the mean and std from the best performances through the different experiments.
       pre : properties (np.ndarray): Contains the best performances metrics (i.e. validation loss, epoch of convergence, accuracy, AUC) through
                                      the different experiments.
       post: statistics (np.ndarray): The mean and std of the best performance metrics over the experiments.
    """
    statistics = np.empty((len(properties),2))
    i = 0
    for a_property in properties:
        statistics[i,0] = np.mean(a_property)
        statistics[i,1] = np.std(a_property)
        i+=1
    return statistics

def multiple_ttests(all_data):
    """Performs statistical significance testing of all the arrays present in [all_data] under the form of a half double entry table. Test used
       is a Welch's t.
       pre : all_data (list of np.1darray): Contains the data to be compared with one another using a Welch's t test. Each np.array the values of
                                             a performance metrics through the experiments.
       post: p_vals (2d array): table filled with -1 on the lower triangle and diagonal. Upper diagonal contains the p_values of the comparison of
                                the different sets by pairs.

       """
    p_vals = - np.ones((len(all_data),len(all_data)))
    for i in range(len(all_data)):
        j = i+1
        while j < len(all_data):
            _, p_vals[i,j] = stats.ttest_ind(all_data[i], all_data[j], equal_var = False)
            j+=1
    return np.round(p_vals,5)

# Load the performances saved earlier
with open('../results/pre_trainings_performances.pickle', 'rb') as file:
    raw_results = pickle.load(file)

epochs = np.arange(40)+1


# Extract the performances values (raw and minimum) from the different pre-trainings
no_pt_loss, no_pt_val_loss, no_pt_val_acc, no_pt_val_AUC = properties(raw_results["no_pre"], mode=0)
swap_pt_loss, swap_pt_val_loss, swap_pt_val_acc, swap_pt_val_AUC = properties(raw_results["swap"], mode=0)

min_no_pt_val_loss, epoch_min_no_pt_val_loss, _, _ = properties(raw_results["no_pre"], mode=1)
min_swap_pt_val_loss, epoch_min_swap_pt_val_loss, _, _ = properties(raw_results["swap"], mode=1)

gauss_pt_loss, gauss_pt_val_loss, gauss_pt_val_acc, gauss_pt_val_AUC = properties(raw_results["gauss"], mode=0)
shuffle_pt_loss, shuffle_pt_val_loss, shuffle_pt_val_acc, shuffle_pt_val_AUC = properties(raw_results["shuffle"], mode=0)

min_gauss_pt_val_loss, epoch_min_gauss_pt_val_loss, _, _ = properties(raw_results["gauss"], mode=1)
min_shuffle_pt_val_loss, epoch_min_shuffle_pt_val_loss, _, _ = properties(raw_results["shuffle"], mode=1)

hybrid_pt_loss, hybrid_pt_val_loss, hybrid_pt_val_acc, hybrid_pt_val_AUC = properties(raw_results["hybrid"], mode=0)
min_hybrid_pt_val_loss, epoch_min_hybrid_pt_val_loss, _, _ = properties(raw_results["hybrid"], mode=1)

print("# samples: \n", hybrid_pt_loss.shape[0])

# Show the mean and std of each performance metrics for each pre-training
print("Stats no pre-training:\n", stats_at_min_val_loss(properties(raw_results["no_pre"],1)))
print("\nStats swap pre-training:\n", stats_at_min_val_loss(properties(raw_results["swap"],1)))
print("\nStats noise pre-training:\n", stats_at_min_val_loss(properties(raw_results["gauss"],1)))
print("\nStats shuffle pre-training:\n", stats_at_min_val_loss(properties(raw_results["shuffle"],1)))
print("\nStats hybrid pre-training:\n", stats_at_min_val_loss(properties(raw_results["hybrid"],1)))

names = ["min_data_val_loss", "epoch_min_data_val_loss", "acc_at_min_val_loss", "AUC_at_min_val_loss"]
data_swap = properties(raw_results["swap"],1)
data_no = properties(raw_results["no_pre"],1)
data_gauss = properties(raw_results["gauss"],1)
data_shuffle = properties(raw_results["shuffle"],1)
data_hybrid = properties(raw_results["hybrid"],1)

# Show significance cross testing per metric for all pre-trainings
print("\n\n\n")
for i in range(len(names)):
    print(names[i])
    print(multiple_ttests([data_no[i],data_swap[i],data_gauss[i], data_shuffle[i], data_hybrid[i]]),"\n")

# Show significance cross testing per metric for pooled pre-trainings vs no pre-training
all_data = np.concatenate((data_swap,data_shuffle,data_hybrid,data_gauss),axis=1)
print("\n\n\nComparison between pre-training and no pre-training:")
print(stats_at_min_val_loss(all_data))
for i in range(len(names)):
    print(names[i])
    print(multiple_ttests([data_no[i],all_data[i]]),"\n")

# Check if there is a linear relation between epoch and val_loss
reg_shuffle = LinearRegression().fit(epoch_min_shuffle_pt_val_loss.reshape(-1, 1), min_shuffle_pt_val_loss)
reg_hybrid = LinearRegression().fit(epoch_min_hybrid_pt_val_loss.reshape(-1, 1), min_hybrid_pt_val_loss)
reg_gauss = LinearRegression().fit(epoch_min_gauss_pt_val_loss.reshape(-1, 1), min_gauss_pt_val_loss)
reg_swap = LinearRegression().fit(epoch_min_swap_pt_val_loss.reshape(-1, 1), min_swap_pt_val_loss)
reg_no = LinearRegression().fit(epoch_min_no_pt_val_loss.reshape(-1, 1), min_no_pt_val_loss)

R2_shuffle = reg_shuffle.score(epoch_min_shuffle_pt_val_loss.reshape(-1, 1), min_shuffle_pt_val_loss)
R2_hybrid = reg_hybrid.score(epoch_min_hybrid_pt_val_loss.reshape(-1, 1), min_hybrid_pt_val_loss)
R2_gauss = reg_gauss.score(epoch_min_gauss_pt_val_loss.reshape(-1, 1), min_gauss_pt_val_loss)
R2_swap = reg_swap.score(epoch_min_swap_pt_val_loss.reshape(-1, 1), min_swap_pt_val_loss)
R2_no = reg_no.score(epoch_min_no_pt_val_loss.reshape(-1, 1), min_no_pt_val_loss)

print("R2 score of shuffle regression:", R2_shuffle)
print("Shuffle regression slope:", reg_shuffle.coef_[0],'\n')
print("R2 score of hybrid regression:",R2_hybrid)
print("Hybrid regression slope:", reg_hybrid.coef_[0],'\n')
print("R2 score of white noise regression:", R2_gauss)
print("White noise regression slope:", reg_gauss.coef_[0],'\n')
print("R2 score of mixing regression:",R2_swap)
print("Mixing regression slope:", reg_swap.coef_[0],'\n')
print("R2 score of no pre-training regression:",R2_no)
print("No pre-training regression slope:", reg_no.coef_[0])

def ci_95(data):
    """Computes the 95% confidence interval of a given [data] through the experiment axis.
       pre : data (2d array): An array of data points (axis 0: epochs, axis 1: experiments).
       post: The 95% CI of the input datathrough the experiment axis.
    """
    return 1.96 * np.std(data, axis = 0)/np.sqrt(data.shape[0])

def ci_99(data):
    """Computes the 99% confidence interval of a given [data] through the experiment axis.
       pre : data (2d array): An array of data points (axis 0: epochs, axis 1: experiments).
       post: The 99% CI of the input datathrough the experiment axis.
    """
    return 2.81 * np.std(data, axis = 0)/np.sqrt(data.shape[0])

def scatter_hist(x, y, label, ax, ax_histx, ax_histy):
#https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    n_plot = len(x)
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    for i in range(n_plot):
        ax.scatter(x[i], y[i], label = label[i], marker='.')
        mu = np.mean(x[i])
        sigma = np.std(x[i])
        base = np.linspace(max(1,mu - 3*sigma), min(45,mu + 3*sigma), 100)
        ax_histx.plot(base, stats.norm.pdf(base, mu, sigma))
        mu = np.mean(y[i])
        sigma = np.std(y[i])
        base = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax_histy.plot(stats.norm.pdf(base, mu, sigma)/100,base)


#https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html (lines 172--189)
fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histx.set_ylabel("Density [-]")
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
ax_histy.set_xlabel("Density [-]")
# Draw the scatter plot and marginals.
scatter_hist([epoch_min_shuffle_pt_val_loss, epoch_min_swap_pt_val_loss, epoch_min_gauss_pt_val_loss, epoch_min_hybrid_pt_val_loss, epoch_min_no_pt_val_loss],
    [min_shuffle_pt_val_loss, min_swap_pt_val_loss, min_gauss_pt_val_loss, min_hybrid_pt_val_loss,min_no_pt_val_loss],
    ["Shuffling", "Mixing", "White noise", "Hybrid", "No pre-training"],ax, ax_histx, ax_histy)
ax.legend()
ax.set_xlabel("Epoch of convergence [-]")
ax.set_ylabel("Minimum validation loss [-]")
plt.show()


""" Supplementary graphs

plt.scatter(epoch_min_shuffle_pt_val_loss, min_shuffle_pt_val_loss,marker='.')
plt.plot(np.arange(0,42), reg_shuffle.predict(np.arange(0,42).reshape(-1, 1)), label="Shuffling (R2 = {:.4f})".format(R2_shuffle))
plt.scatter(epoch_min_swap_pt_val_loss, min_swap_pt_val_loss,marker='.')
plt.plot(np.arange(0,42), reg_swap.predict(np.arange(0,42).reshape(-1, 1)), label="Mixing (R2 = {:.4f})".format(R2_swap))

plt.scatter(epoch_min_gauss_pt_val_loss, min_gauss_pt_val_loss,marker='.')
plt.plot(np.arange(0,42), reg_gauss.predict(np.arange(0,42).reshape(-1, 1)), label="White noise (R2 = {:.4f})".format(R2_gauss))

plt.scatter(epoch_min_hybrid_pt_val_loss, min_hybrid_pt_val_loss, marker='.')
plt.plot(np.arange(0,42), reg_hybrid.predict(np.arange(0,42).reshape(-1, 1)), label="Hybrid (R2 = {:.4f})".format(R2_hybrid))

plt.scatter(epoch_min_no_pt_val_loss, min_no_pt_val_loss, marker='.')
plt.plot(np.arange(0,42), reg_no.predict(np.arange(0,42).reshape(-1, 1)), label="No pre-training (R2 = {:.4f})".format(R2_no))

plt.xlabel("Epoch of convergence [-]")
plt.ylabel("Minimum validation loss [-]")
plt.legend()
plt.show()


plt.subplot(211)
plt.title("Training set loss function")
plt.plot(epochs,np.mean(no_pt_loss,axis=0), label = "No pre-training")
plt.fill_between(epochs, (np.mean(no_pt_loss,axis=0)-ci_95(no_pt_loss)), (np.mean(no_pt_loss,axis=0)+ci_95(no_pt_loss)), alpha=.1)
plt.plot(epochs,np.mean(swap_pt_loss, axis=0), label = "Swapping")
plt.fill_between(epochs, (np.mean(swap_pt_loss, axis=0)-ci_95(swap_pt_loss)), (np.mean(swap_pt_loss, axis=0)+ci_95(swap_pt_loss)), alpha=.1)
plt.plot(epochs,np.mean(gauss_pt_loss,axis=0), label = "Noise")
plt.fill_between(epochs, (np.mean(gauss_pt_loss,axis=0)-ci_95(gauss_pt_loss)), (np.mean(gauss_pt_loss,axis=0)+ci_95(gauss_pt_loss)), alpha=.1)
plt.plot(epochs,np.mean(shuffle_pt_loss, axis=0), label = "Shuffle")
plt.fill_between(epochs, (np.mean(shuffle_pt_loss, axis=0)-ci_95(shuffle_pt_loss)), (np.mean(shuffle_pt_loss, axis=0)+ci_95(shuffle_pt_loss)), alpha=.1)
plt.plot(epochs,np.mean(hybrid_pt_loss, axis=0), label = "Hybrid")
plt.fill_between(epochs, (np.mean(hybrid_pt_loss, axis=0)-ci_95(hybrid_pt_loss)), (np.mean(hybrid_pt_loss, axis=0)+ci_95(hybrid_pt_loss)), alpha=.1)
plt.ylabel("Loss [-]")
plt.legend()

plt.subplot(212)
plt.title("Validation set loss function")
plt.plot(epochs,np.mean(no_pt_val_loss,axis=0), label = "No pre-training")
plt.fill_between(epochs, (np.mean(no_pt_val_loss,axis=0)-ci_95(no_pt_val_loss)), (np.mean(no_pt_val_loss,axis=0)+ci_95(no_pt_val_loss)), alpha=.1)
plt.plot(epochs,np.mean(swap_pt_val_loss, axis=0), label = "Swapping")
plt.fill_between(epochs, (np.mean(swap_pt_val_loss, axis=0)-ci_95(swap_pt_val_loss)), (np.mean(swap_pt_val_loss, axis=0)+ci_95(swap_pt_val_loss)), alpha=.1)
plt.plot(epochs,np.mean(gauss_pt_val_loss,axis=0), label = "Noise")
plt.fill_between(epochs, (np.mean(gauss_pt_val_loss,axis=0)-ci_95(gauss_pt_val_loss)), (np.mean(gauss_pt_val_loss,axis=0)+ci_95(gauss_pt_val_loss)), alpha=.1)
plt.plot(epochs,np.mean(shuffle_pt_val_loss, axis=0), label = "Shuffle")
plt.fill_between(epochs, (np.mean(shuffle_pt_val_loss, axis=0)-ci_95(shuffle_pt_val_loss)), (np.mean(shuffle_pt_val_loss, axis=0)+ci_95(shuffle_pt_val_loss)), alpha=.1)
plt.plot(epochs,np.mean(hybrid_pt_val_loss, axis=0), label = "Hybrid")
plt.fill_between(epochs, (np.mean(hybrid_pt_val_loss, axis=0)-ci_95(hybrid_pt_val_loss)), (np.mean(hybrid_pt_val_loss, axis=0)+ci_95(hybrid_pt_val_loss)), alpha=.1)
plt.ylabel("Loss [-]")
plt.xlabel("Epoch [-]")
plt.legend()

plt.show()


plt.subplot(211)
plt.title("Validation set accuracy")
plt.plot(epochs,np.mean(no_pt_val_acc,axis=0), label = "No pre-training")
plt.fill_between(epochs, (np.mean(no_pt_val_acc,axis=0)-ci_95(no_pt_val_acc)), (np.mean(no_pt_val_acc,axis=0)+ci_95(no_pt_val_acc)), alpha=.1)
plt.plot(epochs,np.mean(swap_pt_val_acc, axis=0), label = "Swapping")
plt.fill_between(epochs, (np.mean(swap_pt_val_acc, axis=0)-ci_95(swap_pt_val_acc)), (np.mean(swap_pt_val_acc, axis=0)+ci_95(swap_pt_val_acc)), alpha=.1)
plt.plot(epochs,np.mean(gauss_pt_val_acc,axis=0), label = "Noise")
plt.fill_between(epochs, (np.mean(gauss_pt_val_acc,axis=0)-ci_95(gauss_pt_val_acc)), (np.mean(gauss_pt_val_acc,axis=0)+ci_95(gauss_pt_val_acc)), alpha=.1)
plt.plot(epochs,np.mean(shuffle_pt_val_acc, axis=0), label = "Shuffle")
plt.fill_between(epochs, (np.mean(shuffle_pt_val_acc, axis=0)-ci_95(shuffle_pt_val_acc)), (np.mean(shuffle_pt_val_acc, axis=0)+ci_95(shuffle_pt_val_acc)), alpha=.1)
plt.plot(epochs,np.mean(hybrid_pt_val_acc, axis=0), label = "Hybrid")
plt.fill_between(epochs, (np.mean(hybrid_pt_val_acc, axis=0)-ci_95(hybrid_pt_val_acc)), (np.mean(hybrid_pt_val_acc, axis=0)+ci_95(hybrid_pt_val_acc)), alpha=.1)
plt.ylabel("Accuracy [%]")
plt.legend()

plt.subplot(212)
plt.title("Validation set AUC")
plt.plot(epochs,np.mean(no_pt_val_AUC,axis=0), label = "No pre-training")
plt.fill_between(epochs, (np.mean(no_pt_val_AUC,axis=0)-ci_95(no_pt_val_AUC)), (np.mean(no_pt_val_AUC,axis=0)+ci_95(no_pt_val_AUC)), alpha=.1)
plt.plot(epochs,np.mean(swap_pt_val_AUC, axis=0), label = "Swapping")
plt.fill_between(epochs, (np.mean(swap_pt_val_AUC, axis=0)-ci_95(swap_pt_val_AUC)), (np.mean(swap_pt_val_AUC, axis=0)+ci_95(swap_pt_val_AUC)), alpha=.1)
plt.plot(epochs,np.mean(gauss_pt_val_AUC,axis=0), label = "Noise")
plt.fill_between(epochs, (np.mean(gauss_pt_val_AUC,axis=0)-ci_95(gauss_pt_val_AUC)), (np.mean(gauss_pt_val_AUC,axis=0)+ci_95(gauss_pt_val_AUC)), alpha=.1)
plt.plot(epochs,np.mean(shuffle_pt_val_AUC, axis=0), label = "Shuffle")
plt.fill_between(epochs, (np.mean(shuffle_pt_val_AUC, axis=0)-ci_95(shuffle_pt_val_AUC)), (np.mean(shuffle_pt_val_AUC, axis=0)+ci_95(shuffle_pt_val_AUC)), alpha=.1)
plt.plot(epochs,np.mean(hybrid_pt_val_AUC, axis=0), label = "Hybrid")
plt.fill_between(epochs, (np.mean(hybrid_pt_val_AUC, axis=0)-ci_95(hybrid_pt_val_AUC)), (np.mean(hybrid_pt_val_AUC, axis=0)+ci_95(hybrid_pt_val_AUC)), alpha=.1)
plt.ylabel("AUC [-]")
plt.xlabel("Epoch [-]")
plt.legend()

plt.show()
"""