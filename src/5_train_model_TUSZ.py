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

    # If no improvement is observed in the minimum validation loss for 5 epochs, stop the training and take the best weights so far.
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode='min',
        restore_best_weights=True,
        verbose=1,
        )

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    # [batch_size] and [num_epochs] are global variables.
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback,early_stopping_callback],
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

# Model parameters
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer MLP layers
transformer_layers = 8
mlp_head_units = [512, 256] 

# Move to where the TUSZ dataset is located.
chdir(PATH)

# Loading the desired dataset: either pre-training or fine tuning (by changing the files in the line below)
x_train, x_test, y_train, y_test = gen_pre_training_data("../data/scalograms_ft_seiz", "../data/scalograms_ft_bkgd", 0, 8842,9264)

#Model compilation======================================================================================================
inputs = tf.keras.layers.Input(shape = x_train[0].shape)
enc = ParallelEncoder(x_train.shape[1], transformer_layers,num_heads,projection_dim,transformer_units)(inputs)

logits = DecisionLayer(mlp_head_units, 1)(enc)

model = tf.keras.Model(inputs=inputs, outputs=logits)
model.save_weights('../tmp/initial_weights')
model.summary()
# ======================================================================================================================

# Train the model (if fine tuning, do not forget to set the pre-trained weights to the ones of the pre-trained model)
history = run_experiment(model, "../tmp/checkpoint_fine_tuning_shuffle_pt_big_lr", x_train, x_test, y_train, y_test, pretrained_weights="../tmp/checkpoint_shuffle_pre_training")

# Dump the performances in a pickle file
with open('../raw_performances_ft_shuffle_big_lr.pickle', 'wb') as file:
    pickle.dump(history.history, file, protocol=pickle.HIGHEST_PROTOCOL)
