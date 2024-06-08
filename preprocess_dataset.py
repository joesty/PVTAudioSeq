import h5py
import os

#complete video_name information
def add_video_name(dataset_path, dataset_name):
    with h5py.File(dataset_path, "r+") as dataset:
        if dataset_name == "ovp":
            offset = 20
        elif dataset_name == "youtube":
            offset = 60
        else:
            return
        for video_idx in dataset.keys():
            idx = int(video_idx.split("_")[-1])
            dataset[video_idx].create_dataset("video_name", data="v{}".format(offset + idx))

        dataset.close()

def get_dataset_name(dataset_path):
    print(dataset_path)
    dataset_name = dataset_path.split("_")[-3]
    return dataset_name

def main():
    dataset_folder_path = "datasets"
    dataset_paths_list = os.listdir(dataset_folder_path)
    print(dataset_paths_list)
    for dataset_path in dataset_paths_list:
        if dataset_path.endswith(".h5"):
            dataset_name = get_dataset_name(os.path.join(dataset_folder_path, dataset_path))
            add_video_name(os.path.join(dataset_folder_path, dataset_path), dataset_name)

if __name__ == "__main__":
    main()

