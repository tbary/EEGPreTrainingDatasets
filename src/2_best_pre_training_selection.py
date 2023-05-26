from init import *

def run_experiment(model, checkpoint_filepath, x_train, x_test, y_train, y_test, pretrained_weights=None):
#from https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_with_vision_transformer.ipynb
    """Trains [model] on [x_train] and [y_train] using an AdamW optimizer and a binary cross entropy loss.
       Saves best weights to [checkpoint_filepath]. Tests the model with best weights on [x_test] and reports 
       the accuracy and AUC.
       pre : model (tf.keras.Model): the model to train.
             checkpoint_filepath (string): the path to the directry where the weights must be saved.
             x_train (np.ndarray): training data.
             y_train (np.array): training target. Same length as x_train.
             x_test (np.ndarray): test data.
             y_test (np.array): test target. Same length as x_test.
             pretrained_weights (string or None): path to the weights resulting from model pre-training. No
                                                  weights are loaded if None.
       post: history (tf.keras.History): object that stores the events occuring during the fitting process."""

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Metrics to be monitored: accuracy and AUC.
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(num_thresholds= 10, name = "AUC", from_logits=True)
        ],
    )

    # Best weights are determined as yielding the minimal validation loss.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    # [batch_size] and [num_epochs] are global variables.
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.05,
        callbacks=[checkpoint_callback],
        verbose=2
    )

    # Test the model
    model.load_weights(checkpoint_filepath)
    _,accuracy, AUC = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test AUC: {AUC}")

    return history

# Experiment variables
learning_rate = 0.0001
weight_decay = 0.0001
batch_size = 8
num_epochs = 50
n_exp = 30

# Model parameters
num_heads = 2
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer MLP layers
transformer_layers = 1
mlp_head_units = [128, 64] 

# Loading the different pre-training datasets with unaltered and altered classes
x_train_p, x_test_p, y_train_p, y_test_p = gen_pre_training_data("../data/scalograms_pt_untouched","../data/scalograms_inter", 0, 775, 775)
x_train_g, x_test_g, y_train_g, y_test_g = gen_pre_training_data("../data/scalograms_pt_untouched","../data/scalograms_noise", 0, 775, 775)
x_train_s, x_test_s, y_train_s, y_test_s = gen_pre_training_data("../data/scalograms_pt_untouched","../data/scalograms_intra", 0, 775, 775)

#Loading the fine tuning dataset
x_train, x_test, y_train, y_test = load_and_split("../data/scalograms_small", 0, 665)


#Model compilation======================================================================================================
inputs = tf.keras.layers.Input(shape = x_train[0].shape)
enc = ParallelEncoder(x_train.shape[1], transformer_layers,num_heads,projection_dim,transformer_units)(inputs)

logits = DecisionLayer(mlp_head_units, 1)(enc)

model = tf.keras.Model(inputs=inputs, outputs=logits)
model.save_weights('../tmp/initial_weights')
model.summary()
# ======================================================================================================================

# Prepare the performances arrays
swap_pt_loss = np.empty((n_exp, num_epochs))
swap_pt_val_loss = np.empty((n_exp, num_epochs))
swap_pt_val_acc = np.empty((n_exp, num_epochs))
swap_pt_val_AUC = np.empty((n_exp, num_epochs))

no_pt_loss = np.empty((n_exp, num_epochs))
no_pt_val_loss = np.empty((n_exp, num_epochs))
no_pt_val_acc = np.empty((n_exp, num_epochs))
no_pt_val_AUC = np.empty((n_exp, num_epochs))

gauss_pt_loss = np.empty((n_exp, num_epochs))
gauss_pt_val_loss = np.empty((n_exp, num_epochs))
gauss_pt_val_acc = np.empty((n_exp, num_epochs))
gauss_pt_val_AUC = np.empty((n_exp, num_epochs))

shuffle_pt_loss = np.empty((n_exp, num_epochs))
shuffle_pt_val_loss = np.empty((n_exp, num_epochs))
shuffle_pt_val_acc = np.empty((n_exp, num_epochs))
shuffle_pt_val_AUC = np.empty((n_exp, num_epochs))

hybrid_pt_loss = np.empty((n_exp, num_epochs))
hybrid_pt_val_loss = np.empty((n_exp, num_epochs))
hybrid_pt_val_acc = np.empty((n_exp, num_epochs))
hybrid_pt_val_AUC = np.empty((n_exp, num_epochs))

# For each experiment, pre-train 5 new models on all different pre-trainings, then fine tune them on the had-hoc set, then report the performances
for i in range(n_exp):
    print("EXP {} ==================================================================================================".format(i+1))
    
    #Pre-training and fine tuning of the models
    run_experiment(model, "../tmp/checkpoint_pre_training", x_train_p, x_test_p, y_train_p, y_test_p,'../tmp/initial_weights')
    history_swap_pt = run_experiment(model, "../tmp/checkpoint_1", x_train, x_test, y_train, y_test, "../tmp/checkpoint_pre_training")
    
    history_no_pt = run_experiment(model, "../tmp/checkpoint_2", x_train, x_test, y_train, y_test,'../tmp/initial_weights')

    run_experiment(model, "../tmp/checkpoint_pre_training_noise", x_train_g, x_test_g, y_train_g, y_test_g,'../tmp/initial_weights')
    history_gauss_pt = run_experiment(model, "../tmp/checkpoint_3", x_train, x_test, y_train, y_test, "../tmp/checkpoint_pre_training_noise")

    run_experiment(model, "../tmp/checkpoint_pre_training_shuffle", x_train_s, x_test_s, y_train_s, y_test_s,'../tmp/initial_weights')
    history_shuffle_pt = run_experiment(model, "../tmp/checkpoint_4", x_train, x_test, y_train, y_test, "../tmp/checkpoint_pre_training_shuffle")

    num_epochs//=2
    run_experiment(model, "../tmp/checkpoint_pre_training_hybrid", x_train_g, x_test_g, y_train_g, y_test_g,'../tmp/initial_weights')
    run_experiment(model, "../tmp/checkpoint_pre_training_hybrid", x_train_s, x_test_s, y_train_s, y_test_s,'../tmp/checkpoint_pre_training_hybrid')
    num_epochs*=2
    history_hybrid_pt = run_experiment(model, "../tmp/checkpoint_5", x_train, x_test, y_train, y_test, "../tmp/checkpoint_pre_training_hybrid")

    #Record the performances in a dictionnary
    swap_pt_loss[i] = history_swap_pt.history['loss']
    swap_pt_val_loss[i] = history_swap_pt.history['val_loss']
    swap_pt_val_acc[i] = history_swap_pt.history['val_accuracy']
    swap_pt_val_AUC[i] = history_swap_pt.history['val_AUC']

    no_pt_loss[i] = history_no_pt.history['loss']
    no_pt_val_loss[i] = history_no_pt.history['val_loss']
    no_pt_val_acc[i] = history_no_pt.history['val_accuracy']
    no_pt_val_AUC[i] = history_no_pt.history['val_AUC']

    gauss_pt_loss[i] = history_gauss_pt.history['loss']
    gauss_pt_val_loss[i] = history_gauss_pt.history['val_loss']
    gauss_pt_val_acc[i] = history_gauss_pt.history['val_accuracy']
    gauss_pt_val_AUC[i] = history_gauss_pt.history['val_AUC']

    shuffle_pt_loss[i] = history_shuffle_pt.history['loss']
    shuffle_pt_val_loss[i] = history_shuffle_pt.history['val_loss']
    shuffle_pt_val_acc[i] = history_shuffle_pt.history['val_accuracy']
    shuffle_pt_val_AUC[i] = history_shuffle_pt.history['val_AUC']

    hybrid_pt_loss[i] = history_hybrid_pt.history['loss']
    hybrid_pt_val_loss[i] = history_hybrid_pt.history['val_loss']
    hybrid_pt_val_acc[i] = history_hybrid_pt.history['val_accuracy']
    hybrid_pt_val_AUC[i] = history_hybrid_pt.history['val_AUC']

    raw_results = {
            "hybrid":{"loss" : hybrid_pt_loss[:i+1],
                      "val_loss": hybrid_pt_val_loss[:i+1],
                      "val_acc": hybrid_pt_val_acc[:i+1],
                      "val_AUC": hybrid_pt_val_AUC[:i+1]},

            "gauss" :{"loss" : gauss_pt_loss[:i+1],
                      "val_loss": gauss_pt_val_loss[:i+1],
                      "val_acc": gauss_pt_val_acc[:i+1],
                      "val_AUC": gauss_pt_val_AUC[:i+1]},

           "shuffle":{"loss" : shuffle_pt_loss[:i+1],
                      "val_loss": shuffle_pt_val_loss[:i+1],
                      "val_acc": shuffle_pt_val_acc[:i+1],
                      "val_AUC": shuffle_pt_val_AUC[:i+1]},
            "swap":{"loss" : swap_pt_loss[:i+1],
                  "val_loss": swap_pt_val_loss[:i+1],
                  "val_acc": swap_pt_val_acc[:i+1],
                  "val_AUC": swap_pt_val_AUC[:i+1]},

           "no_pre" :{"loss" : no_pt_loss[:i+1],
                      "val_loss": no_pt_val_loss[:i+1],
                      "val_acc": no_pt_val_acc[:i+1],
                      "val_AUC": no_pt_val_AUC[:i+1]}
            }

    # Dump performances in a pickle file
    with open('../results/pre_trainings_performances_{}.pickle'.format(i+1), 'wb') as file:
        pickle.dump(raw_results, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Delete models before next cycle
    for file in listdir("../tmp"):
        remove("{}/{}".format('../tmp',file))
    tf.keras.backend.clear_session()
