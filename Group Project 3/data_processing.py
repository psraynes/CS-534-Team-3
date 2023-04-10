import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


# Function to load a datafile. Requires a target column for the column containing the classification in the original
# dataset. Returns a pandas DataFrame object of the data sampled to be balanced through undersampling the majority class
def load_and_sample(filename, classification_label, id_labels, rand_state=534):
    data_raw = pd.read_csv(filename)
    # replacement flag can be flipped without impacting function
    # instantiate the random under sampler
    rus = RandomUnderSampler(sampling_strategy='majority', random_state=rand_state, replacement=False)
    # undersample the provided data
    features, classification = data_raw.drop(id_labels + [classification_label], axis=1), data_raw[classification_label]

    # feat_sampled is an array containing the observations of the dataset after sampling, excluding the classification
    # class_sampled is an array containing the classifications of the observation after sampling.
    feat_sampled, class_sampled = rus.fit_resample(features, classification) # type: ignore
    # combine the arrays as a dataframe again to return the
    sampled_data = pd.DataFrame(feat_sampled)
    sampled_data['Machine failure'] = class_sampled
    return sampled_data, feat_sampled, class_sampled

# Function to encode L, M, H to numbers
def encodeLMH(value):
    if value == 'L':
        return 0
    elif value == 'M':
        return 1
    elif value == 'H':
        return 2
    else:
        return value
    
