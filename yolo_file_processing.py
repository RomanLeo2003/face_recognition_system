import os
from shutil import copyfile

train_dir = [file[:-4] for file in os.listdir(r"C:\Users\user\Downloads\phone_labels\train")]
valid_dir = [file[:-4] for file in os.listdir(r"C:\Users\user\Downloads\phone_labels\valid")]
test_dir = [file[:-4] for file in os.listdir(r"C:\Users\user\Downloads\phone_labels\test")]
image_path = r"C:\Users\user\Downloads\phones"

for filename in os.listdir(image_path):
    if filename[:-4] in train_dir:
        copyfile(image_path + "/" + filename, rf'C:\Users\user\Downloads\phone_labels\train\{filename}')
    elif filename[:-4] in valid_dir:
        copyfile(image_path + "/" + filename, rf'C:\Users\user\Downloads\phone_labels\valid\{filename}')
    elif filename[:-4] in test_dir:
        copyfile(image_path + "/" + filename, rf'C:\Users\user\Downloads\phone_labels\test\{filename}')