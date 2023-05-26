from init import *

#Loading and pre-processing the EO/EC dataset
ds = Dataset(frequency)
ds.initialize("../data/EEG_data.csv","../data/target.csv")
ds.discard_delay(transition_period)
ds.split_and_clean(segment_size)

#Splitting the dataset to pre-training and fine tuning, splitting the pre-training subset in half for alterations
pre_training_data, fine_tuning_data, _, fine_tuning_target = train_test_split(ds.data, ds.target, test_size = 1-pre_training_fraction, random_state=0)
pre_training_untouched, pre_training_modify, _, _ = train_test_split(pre_training_data, _, test_size=0.5, random_state=0)

print("Real data:")
#Turning fine tuning data to scalograms
ds.initialize_processed(fine_tuning_data, fine_tuning_target)
ds.gen_inputs(path='../data/scalograms_small', n_patches=num_patches, projection_dim=projection_dim, wavelet='morl', upper_freq=80, show=False, overwrite=True)

#Turning unaltered pre-training data to scalograms
ds.initialize_processed(pre_training_untouched)
ds.gen_inputs(path='../data/scalograms_pt_untouched', n_patches=num_patches, projection_dim=projection_dim, wavelet='morl', upper_freq=80, show=False, overwrite=True)

ds.initialize_processed(pre_training_modify)
#Applying all three alterations to copies of the set of data to be altered, then turning them to scalograms
print("White noise:")
gaus = gaussian_white_noise_eeg_generator(776, pre_training_modify, 5)
ds.initialize_processed(gaus)
ds.gen_inputs(path='../data/scalograms_noise', n_patches=num_patches, projection_dim=projection_dim, wavelet='morl', upper_freq=80, show=False, overwrite=True)

print("Shuffling channels:")
intra_swap = channel_swap_intradata(pre_training_modify, 776)
ds.initialize_processed(intra_swap)
ds.gen_inputs(path='../data/scalograms_intra', n_patches=num_patches, projection_dim=projection_dim, wavelet='morl', upper_freq=80, show=False, overwrite=True)

print("Swapping channels between data:")
inter_swap = channel_swap_interdata(pre_training_modify, 776, 5)
ds.initialize_processed(inter_swap)
ds.gen_inputs(path='../data/scalograms_inter', n_patches=num_patches, projection_dim=projection_dim, wavelet='morl', upper_freq=80, show=False, overwrite=True)
