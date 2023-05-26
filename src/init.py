from library import *

# Prevent TensorFlow to print information and warning messages. Comment out to see them.
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Here are the variables that may be used accross multiple files:
# Set TUSZ_dataset to True to set up the parameters used for the TUSZ dataset
TUSZ_dataset = False

if TUSZ_dataset:

# Solely for big dataset ========================================================================================

	PATH = 'E:/TUEG_data/train' # Localization of the TUSZ dataset

	n_files = 1000 # Number of files in the dataset to consider (selected randomly).

# ===============================================================================================================

	pre_training_fraction = 0.7 # Fraction of dataset used for pre-training. Rest is used for fine-tuning.

	frequency = 250 #Hz, signal sampling frequency.

	transition_period = 1.5 #s

	segment_size = 5 #s, size 

	num_patches = 5  # Size of the patches to be extract from the input images

	projection_dim = 40 # Size of the projection space for the patches

else:
	pre_training_fraction = 0.7 # Fraction of dataset used for pre-training. Rest is used for fine-tuning.

	frequency = 1000 #Hz, signal sampling frequency.

	transition_period = 1.5 #s

	segment_size = .2 #s, size 

	num_patches = 5  # Size of the patches to be extract from the input images

	projection_dim = 8 # Size of the projection space for the patches