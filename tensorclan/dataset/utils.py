import os
import hashlib
from collections import defaultdict
import torch
import torch.utils.data


def split_dataset(dataset, div_factor: int = 1, train_ratio: float = 0.7):
    # 70 - 30 split
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_subset, test_subset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    train_subset = torch.utils.data.Subset(
        train_subset, range(0, len(train_subset)//div_factor))
    test_subset = torch.utils.data.Subset(
        train_subset, range(0, len(test_subset)//div_factor))

    return train_subset, test_subset


def _get_file_hash(path):
    """
    Generate hash of the file, using specific set of blocks of the file instead of whole file
    #TODO : Check out pHash library
        Args:
            path: str, file path

        Returns:
            hash digest string
    """    
    num_bytes_to_read_per_sample = 1024 # ideal should be device block size
    total_bytes = os.path.getsize(path)
    hasher = hashlib.sha512()

    with open(path, 'rb') as f:
        # If the file is too short to take 3 samples, hash the entire file
        if total_bytes < num_bytes_to_read_per_sample * 3:
            hasher.update(f.read())
        else:
            num_bytes_between_samples = (
                (total_bytes - num_bytes_to_read_per_sample * 3) / 2
            )

            # Read first, middle, and last bytes
            for offset_multiplier in range(3):
                start_of_sample = (
                    offset_multiplier
                    * (num_bytes_to_read_per_sample + num_bytes_between_samples)
                )
                f.seek(int(start_of_sample))
                sample = f.read(num_bytes_to_read_per_sample)
                hasher.update(sample)

    return hasher.hexdigest()


def get_duplicates(dir_path, recursive=False):
    """
    Get duplicate files in given directory path
        Args:
            dir_path: str, directory path
            recursive: Bool, if the directory has to be traversed recursively

        Returns:
            dict, int : dict of hashmaps, Number of files hashed
    """
    files_visited = 0
    hashmaps = defaultdict(list)

    if recursive:
        for root, dirs, files in os.walk(dir_path):
            for fname in files:
                file_path = os.path.join(root, fname)
                file_hash = _get_file_hash(file_path)
                files_visited += 1
                hashmaps[file_hash].append(file_path)
    else:
        for fname in os.listdir(dir_path):
            file_path = os.path.join(dir_path, fname)
            if os.path.isfile(file_path):
                file_hash = _get_file_hash(file_path)
                files_visited += 1
                hashmaps[file_hash].append(file_path)

    return dict(hashmaps), files_visited

def remove_duplicates(dir_path, recursive=False):
    """
    Remove duplicate files in given directory path
        Args:
            dir_path: str, directory path
            recursive: Bool, if the directory has to be traversed recursively
    """
    print('Generating hashmaps')
    hashmaps, count = get_duplicates(dir_path, recursive=recursive)
    if hashmaps:
        # Compatible for python 3.5
        print('Found {count} files'.format(count=count))
        print('Checking for duplicates')
        for _, files in hashmaps.items():
            if len(files) > 1:
                print('Found {count} duplicates of {fname}'.format(count=len(files)-1, fname=files[0]))
                for fname in files[1:]:
                    print('\tRemoving : {fname}'.format(fname=fname))
                    os.remove(fname)
        return 0
    print('No files found')
    return 1
    # sys.exit(1)