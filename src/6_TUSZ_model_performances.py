from init import *

# Move to where the TUSZ dataset is located.
#chdir(PATH)

# Load the performances for the pre-training, the fine tuning, and the model without pre-training.
with open('../results/raw_performances_ft_shuffle_big_lr.pickle', 'rb') as file:
    shuffle_ft = pickle.load(file)

with open('../results/raw_performances_ft_no_big_lr.pickle', 'rb') as file:
    no_ft = pickle.load(file)

with open('../results/raw_performances_shuffle_pt.pickle', 'rb') as file:
    pre_training = pickle.load(file)

# Plot of the evolution of the performance metrics through the pre-training.
pt_weights = np.argmin(pre_training["val_loss"])
pt_epochs = np.arange(len(pre_training["loss"]))+1
plt.subplot(211)
plt.plot(pre_training["loss"],label="Training loss", color="skyblue")
plt.plot(pre_training["val_loss"], label = "Validation loss", color="skyblue", linestyle="--")
plt.plot([pt_weights,pt_weights],[np.min(pre_training["loss"]),np.max(pre_training["loss"])],linestyle="--", color="red", label="EOC")
plt.ylabel("Loss [-]")
plt.xticks(pt_epochs-1,pt_epochs)
plt.legend()

plt.subplot(223)
plt.plot(pre_training["val_accuracy"], label="Validation accuracy", color="skyblue")
plt.plot([pt_weights,pt_weights],[np.min(pre_training["val_accuracy"]),np.max(pre_training["val_accuracy"])],linestyle="--", color="red")
plt.xlabel("Epoch [-]")
plt.ylabel("Validation accuracy [-]")
plt.xticks(pt_epochs-1,pt_epochs)

plt.subplot(224)
plt.plot(pre_training["val_AUC"], label="Validation AUC",color="skyblue")
plt.plot([pt_weights,pt_weights],[np.min(pre_training["val_AUC"]),np.max(pre_training["val_AUC"])],linestyle="--", color="red")
plt.xticks(pt_epochs-1,pt_epochs)
plt.xlabel("Epoch [-]")
plt.ylabel("Validation AUC [-]")
plt.show()


# Comparison plot of the evolution of the performance metrics through the fine tuning of a model with vs without pre-training.
shuffle_weights = np.argmin(shuffle_ft["val_loss"])
shuffle_epochs = np.arange(len(shuffle_ft["loss"]))+1
no_weights = np.argmin(no_ft["val_loss"])
no_epochs = np.arange(len(no_ft["loss"]))+1
plt.subplot(211)
plt.plot(shuffle_ft["val_loss"], label = "Validation loss (pre-trained)", color="C0", linestyle="--")
plt.plot(shuffle_ft["loss"], label="Training loss (pre-trained)", color="C0")
plt.plot(no_ft["val_loss"], label = "Validation loss (no pre-training)", color="C4", linestyle="--")
plt.plot(no_ft["loss"], label="Training loss (no pre-training)", color="C4")
plt.plot([shuffle_weights,shuffle_weights],[np.min(shuffle_ft["loss"]),np.max(no_ft["loss"])],linestyle="--", color="red", label="EOC (pre-trained)")
plt.plot([no_weights,no_weights],[np.min(shuffle_ft["loss"]),np.max(no_ft["loss"])],linestyle=":", color="red", label="EOC (no pre-training)")
plt.ylabel("Loss [-]")
plt.xticks(no_epochs-1,no_epochs)
plt.legend()

plt.subplot(223)
plt.plot(shuffle_ft["val_accuracy"], label="Pre-trained",color = "C0")
plt.plot([shuffle_weights,shuffle_weights],[np.min(no_ft["val_accuracy"]),np.max(shuffle_ft["val_accuracy"])],linestyle="--", color="red")
plt.xticks(shuffle_epochs-1,shuffle_epochs)
plt.plot(no_ft["val_accuracy"], label="No pre-training",color = "C4")
plt.plot([no_weights,no_weights],[np.min(no_ft["val_accuracy"]),np.max(shuffle_ft["val_accuracy"])],linestyle=":", color="red")
plt.xticks(no_epochs-1,no_epochs)
plt.xlabel("Epoch [-]")
plt.ylabel("Validation accuracy [-]")
plt.legend()

plt.subplot(224)
plt.plot(shuffle_ft["val_AUC"], label="Pre-trained", color= "C0")
plt.plot(no_ft["val_AUC"], label="No pre-training", color= "C4")
plt.plot([shuffle_weights,shuffle_weights],[np.min(no_ft["val_AUC"]),np.max(shuffle_ft["val_AUC"])],linestyle="--", color="red")
plt.plot([no_weights,no_weights],[np.min(no_ft["val_AUC"]),np.max(shuffle_ft["val_AUC"])],linestyle=":", color="red")
plt.xlabel("Epoch [-]")
plt.xticks(no_epochs-1,no_epochs)
plt.ylabel("Validation AUC [-]")
plt.legend()
plt.show()
