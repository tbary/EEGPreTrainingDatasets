from init import *

np.random.seed(0)
# Move to where the TUSZ dataset is located.
chdir(PATH)

# Find all .edf files and shuffle them
all_files = np.random.permutation(np.array(glob.glob('*/*/*/*.edf')))

# Label the file depending on whether they contain a seizure or not
with_seizure = bool_files_with_seizure(all_files)
files_with_seizure = all_files[with_seizure][:n_files]
files_without_seizure = all_files[~with_seizure][:n_files]

# Separate files into pre-training and fine tuning
n_pt_samples = int(len(files_with_seizure)*pre_training_fraction)
pre_training_data = np.random.permutation(np.concatenate((files_with_seizure[:n_pt_samples], files_without_seizure[:n_pt_samples])))
fine_tuning_seiz = files_with_seizure[n_pt_samples:]
fine_tuning_bkgd = files_without_seizure[n_pt_samples:]

ds = Dataset(frequency)
shift = 0
# Generate the scalograms of pre-ictal segments.
for file_name in tqdm(fine_tuning_seiz, desc='Generating FT pre-ictal EEG scalograms', ascii=False):
	edf_data = edf.EdfReader(file_name)
	# The type of unipolar montage is explicited higher in the file tree.
	montage = file_name.split('\\')[2]
	# Obtain events data linked to EEG signals
	edf_events_filename = file_name.replace(".edf", ".csv_bi")
	limit = pre_ictal_limit(edf_events_filename, frequency, 5)

	# If the pre-ictal segment is shorter than [segment_size], ignore it.
	if limit > segment_size*frequency:
		pre_ictal_data = edf_to_csv(edf_data, montage, limit)
		pd.DataFrame(np.zeros(pre_ictal_data.shape[0])).to_csv("target.csv", index=False)
		ds.initialize("temp.csv","target.csv")
		ds.split_and_clean(segment_size)
		ds.gen_inputs(path='../data/scalograms_ft_seiz', shift=shift, n_patches=num_patches, projection_dim=projection_dim, wavelet='morl', upper_freq=80, disable_progress_bar=True)
		shift+=ds.target.shape[0]

n_seiz_data = shift
shift = 0
# Generate the scalograms of inter-ictal segments.
for file_name in tqdm(fine_tuning_bkgd, desc='Generating FT inter-ictal EEG scalograms', ascii=False):
	# To achieve balance between clases, and since there are much more inter-ictal than pre-ictal data, stop once balance is reached.
	if shift>n_seiz_data:
		break
	edf_data = edf.EdfReader(file_name)
	montage = file_name.split('\\')[2]
	edf_events_filename = file_name.replace(".edf", ".csv_bi")
	inter_ictal_data = edf_to_csv(edf_data, montage)
	pd.DataFrame(np.zeros(inter_ictal_data.shape[0])).to_csv("target.csv", index=False)

	ds.initialize("temp.csv","target.csv")
	ds.split_and_clean(segment_size)
	ds.gen_inputs(path='../data/scalograms_ft_bkgd', shift=shift, n_patches=num_patches, projection_dim=projection_dim, wavelet='morl', upper_freq=80, disable_progress_bar=True)
	shift+=ds.target.shape[0]

shift = 0
# Apply an alteration on half the unlabelled data to create the positive cases of the pre-training dataset. Then turn into scalograms
for file_name in tqdm(pre_training_data[:len(pre_training_data)//2], desc='Generating fake PT EEG scalograms', ascii=False):
	edf_data = edf.EdfReader(file_name)
	montage = file_name.split('\\')[2]
	edf_events_filename = file_name.replace(".edf", ".csv_bi")
	data = edf_to_csv(edf_data, montage)
	pd.DataFrame(np.zeros(data.shape[0])).to_csv("target.csv", index=False)
	
	# Alteration chosen from the results of step 3: shuffling
	fake_eeg = channel_swap_intradata(ds.data)
	ds.initialize_processed(fake_eeg)
	ds.gen_inputs(path='../data/scalograms_pt_shuffle', shift=shift, n_patches=num_patches, projection_dim=projection_dim, wavelet='morl', upper_freq=80, disable_progress_bar=True)
	shift+=ds.target.shape[0]

shift = 0
# Turn the other half of (unmodified) data into scalograms for the negative cases of the pre training dataset.
for file_name in tqdm(pre_training_data[len(pre_training_data)//2:], desc='Generating real PT EEG scalograms', ascii=False):
	edf_data = edf.EdfReader(file_name)
	montage = file_name.split('\\')[2]
	edf_events_filename = file_name.replace(".edf", ".csv_bi")
	data = edf_to_csv(edf_data, montage)
	pd.DataFrame(np.zeros(data.shape[0])).to_csv("target.csv", index=False)
	
	ds.initialize("temp.csv","target.csv")
	ds.split_and_clean(segment_size)
	ds.gen_inputs(path='../data/scalograms_pt_untouched', shift=shift, n_patches=num_patches, projection_dim=projection_dim, wavelet='morl', upper_freq=80, disable_progress_bar=True)
	shift+=ds.target.shape[0]

# Files used for compatibility of a function, can be removed once pre-processing is achieved.
remove('./temp.csv')
remove('./target.csv')