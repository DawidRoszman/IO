import os
import shutil

# Define source and target directories
src_dir = "dogs-cats-mini"  # replace with your directory path
train_dir = os.path.join(src_dir, "train")
test_dir = os.path.join(src_dir, "test")

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the number of test images
num_test_images = 2500

# Move the first 2500 cat and dog images to the test directory
for animal in ["cat", "dog"]:
    for i in range(num_test_images):
        filename = f"{animal}.{i}.jpg"
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(test_dir, filename)
        shutil.move(src_file, dst_file)

# Move the remaining cat and dog images to the train directory
for filename in os.listdir(src_dir):
    if filename.startswith("cat") or filename.startswith("dog"):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(train_dir, filename)
        shutil.move(src_file, dst_file)
