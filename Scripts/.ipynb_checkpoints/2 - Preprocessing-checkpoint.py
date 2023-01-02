import pandas as pd
from tqdm import tqdm
import os
import json
import essentia
import essentia.standard as es
# Write the aggregated features into a temporary directory.



tqdm.pandas()

base_path = "/mnt/disks/data/fma"

metadata_path = os.path.join(base_path,"fma_metadata")

dataset_path =  os.path.join(base_path,"fma_large")


df_tracks = pd.read_csv(os.path.join(metadata_path,"tracks_genres_id_full.csv"))

import os
def preprocess_file_essentia(filename,num_segment=10,n_fft=2048,sample_rate=22050,hop_legth=512,n_mfcc=40):
    try:
        # Compute all features.
        # Aggregate 'mean' and 'stdev' statistics for all low-level, rhythm, and tonal frame features.
        features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                                      rhythmStats=['mean', 'stdev'],
                                                      tonalStats=['mean', 'stdev'])(filename)

        # See all feature names in the pool in a sorted order
        #print(sorted(features.descriptorNames()))
        feature_file_name = filename.replace('fma_large','fma_large_features').replace('mp3','json')

        # Create dir
        file_dir = '/'.join(feature_file_name.split('/')[:-1])


        # checking if the directory demo_folder2 
        # exist or not.
        if not os.path.isdir(file_dir):

            # if the demo_folder2 directory is 
            # not present then create it.
            os.makedirs(file_dir)
        es.YamlOutput(filename=feature_file_name, format="json")(features)
    except:
        feature_file_name = "CORRUPTED"
    return feature_file_name



df_tracks['feature_file_path'] = df_tracks.file_path.progress_apply(lambda x: preprocess_file_essentia(x))

df_tracks.to_csv(os.path.join(metadata_path,"tracks_genres_id_full_features.csv"))