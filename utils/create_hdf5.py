import h5py
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def process_subject(args):
    subID, contents = args
    print("Started Subject: ", subID)
    filename = f'features_{subID}.hdf5'
    with h5py.File(filename, 'w') as f:
        sub_grp = f.create_group(subID)  # Create a subgroup for each subject
        for media_type, paths in contents.items():
            if isinstance(paths, dict):  # For nested media types like 'gallery'
                media_grp = sub_grp.create_group(media_type)  # Create a group for 'probe' or 'gallery'
                for item_type, item_paths in paths.items():
                    item_grp = media_grp.create_group(item_type)  # Create a group for 'images' or 'video'
                    for idx, path in enumerate(tqdm(item_paths)):
                        df = pd.read_pickle(path)
                        features = np.stack(df['feature'], axis=0)
                        item_grp.create_dataset(str(idx), data=features)
            else:  # For non-nested media types like 'probe'
                for idx, path in enumerate(paths):
                    df = pd.read_pickle(path)
                    features = np.stack(df['feature'], axis=0)
                    sub_grp.create_dataset(f"{media_type}_{idx}", data=features)
    print("Finished Subject: ", subID)

def main():
    with open('../retinaface_subID_to_media.pkl', 'rb') as f:
        subID_to_media_map = pickle.load(f)

    # List of tuples for multiprocessing
    items = list(subID_to_media_map.items())
    
    # Number of processes
    pool = Pool(processes=25)  # Adjust the number of processes based on your CPU
    
    # Using multiprocessing to process each subject in parallel
    for _ in tqdm(pool.imap_unordered(process_subject, items), total=len(items)):
        pass
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
