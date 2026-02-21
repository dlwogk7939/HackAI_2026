import kagglehub
import shutil

# path = kagglehub.dataset_download("ayuraj/asl-dataset")

# video download, 5gb
path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")

destination = "./dataset"

shutil.move(path, destination)

print("Moved to:", destination)
print("Path to dataset files:", path)