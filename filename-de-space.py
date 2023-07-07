import os
import shutil

def get_filename_ext(file_path):
    name_ext = os.path.basename(file_path)
    name, ext = os.path.splitext(name_ext)
    ext = "" if not ext else ext.lower()[1:]
    return name, ext

def file_copy_de_space_name(src_dir):

    print(f"Removing spaces from filename: src_dir={src_dir}")
    
    count = 0
    for p, dirs, files in os.walk(src_dir):
        for f in files:
            name, ext = get_filename_ext(f)
            if ' ' in name:
                count += 1
                dest_name = f"{name.replace(' ', '_')}{'.'+ext if ext else ''}"
                
                dest_file_path = os.path.join(p, dest_name)
                src_file_path = os.path.join(p, f)
                if os.path.exists(dest_file_path):
                    raise ValueError(f"Dest file path exists, abort to avoid file overwriting: {dest_file_path}")
                shutil.copy(src_file_path, dest_file_path)
                print(f"  Renamed file: {src_file_path} \n    => {dest_file_path}")

    print(f"Total number of files renamed: {count}")
    return count

if __name__ == '__main__':
    src_dir = "/home/yun/Pictures/Screenshots"
    file_copy_de_space_name(src_dir=src_dir)