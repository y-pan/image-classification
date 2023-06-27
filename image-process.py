import os
import shutil
from PIL import Image

def images_copy_resize(src_dir, 
                       dest_dir=None,
                       width=500, 
                       height=500, 
                       padcolor=255):
    if not dest_dir:
        dest_dir, ts = dir_path_like(src_dir, tag='RESIZED')
    os.makedirs(dest_dir)

    print(f"Copy & resized images: src_dir={src_dir}, destDir={dest_dir}")
    
    count = 0
    for p, dirs, files in os.walk(src_dir):
        for f in files:
            img_src_path = os.path.join(p, f)
           
            count += image_copy_resize(
                src_img_path=img_src_path, 
                dest_dir=dest_dir, 
                width=width,
                height=height,
                dest_ext="png",
                padcolor=padcolor,
                copies=10
            )
    print(f"Done, total out files count: {count}")
    return dest_dir

def image_copy_resize(src_img_path, 
                      dest_dir, 
                      width, 
                      height, 
                      dest_ext, 
                      padcolor=255, 
                      copies=5):
    supported_exts = ['png', 'jpg', 'jpeg']
    # src
    srcname = os.path.basename(src_img_path) # "a/b/c/t4.png" => "t4.png"
    src_name, src_ext = os.path.splitext(srcname) #  "t4.png" => ("t4", ".png")
    src_ext = "" if not src_ext else src_ext.lower()[1:]
    # out
    if src_ext.lower() not in supported_exts:
        print(f'Abort due to unsupported file extension: "{src_ext}"')
        return 0

    if dest_ext and dest_ext.lower() not in supported_exts:
        print(f'Abort due to unsupported file extension: "{dest_ext}"')
        return 0

    os.makedirs(dest_dir, exist_ok=True)
    assert os.path.isdir(dest_dir)

    dest_file_type = dest_ext if dest_ext else src_ext[1:]
    
    
    img_src = Image.open(src_img_path).convert('L')
    img_src.thumbnail((width, height))

    wdif = width - img_src.size[0]
    hdif = height - img_src.size[1]

    # At least one of them is zero: wdif, hdif
    if wdif == 0 and hdif == 0:
        img_dest_path = f"{os.path.join(dest_dir, src_name)}.{dest_file_type}"
        img_src.save(img_dest_path, dest_file_type)
        return 1 # 1 out file
    else:
        copies = max(1, copies)
        if hdif != 0:
            step = hdif if copies <= 1 else (hdif // (copies-1))
            if step == 0:
                step = hdif
            ypos = list(range(0, hdif, step))
            if len(ypos) != copies:
                ypos.append(hdif)
            xpos = [0] * len(ypos)
        else:
            step = wdif if copies <= 1 else (wdif // (copies-1))
            if step == 0:
                step = wdif
            xpos = list(range(0, wdif, step))
            if len(xpos) != copies:
                xpos.append(wdif)
            ypos = [0] * len(xpos)

        for i, (x, y) in enumerate(zip(xpos, ypos)):
            out = Image.new("L", (width, height), padcolor)
            out.paste(img_src, (x, y))
            out_copy_name = f"{src_name}_r{i+1}"
            img_dest_path = f"{os.path.join(dest_dir, out_copy_name)}.{dest_file_type}"
            out.save(img_dest_path, dest_file_type)
        return len(xpos)
    
def dir_path_like(src_dir, tag="tag"):
    ts = timestamp()
    parent_dir, src_name = os.path.split(src_dir)
    out_dir = parent_dir
    if out_dir:
        out_dir += f"/{src_name}_{tag}_{ts}"
    else:
        out_dir = f"{src_name}_{tag}_{ts}"
    return out_dir, ts

def files_copy_rename(src_dir, prefix=""):
    dest_dir, ts = dir_path_like(src_dir, tag='RENAMED')
    os.makedirs(dest_dir)

    print(f"Copy & rename files: src_dir={src_dir}, destDir={dest_dir}")
    
    count = 0
    for p, dirs, files in os.walk(src_dir):
        for f in files:
            count += 1
            name, ext = get_filename_ext(f)
            dest_name = f"{prefix + '_' if prefix else ''}{ts}_{count}{'.'+ext if ext else ''}"
            dest_file_path = os.path.join(dest_dir, dest_name)
            src_file_path = os.path.join(p, f)
            shutil.copy(src_file_path, dest_file_path)
    return dest_dir

def timestamp():
    import datetime 
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_filename_ext(file_path):
    name_ext = os.path.basename(file_path)
    name, ext = os.path.splitext(name_ext)
    ext = "" if not ext else ext.lower()[1:]
    return name, ext

def abspath(path_from_current):
  return os.path.join(os.path.dirname(__file__), path_from_current)  

def rename_folder_backup_if_exist(path_may_exist):
    if not os.path.exists(path_may_exist):
        print(f"Nothing happens, path is not occupied: {path_may_exist}")
        return False
    path_new = f"{path_may_exist}_backup{timestamp()}"
    os.rename(path_may_exist, path_new)
    print(f"Backed up path: old path={path_may_exist}, renamed={path_new}")
    return True

def test():
    out_dir, ts = dir_path_like(src_dir="/home/yun/Documents/code/ml/image-process/test-data", tag='Resize')
    assert f"/home/yun/Documents/code/ml/image-process/test-data_Resize_{ts}" == out_dir

    out_dir, ts = dir_path_like(src_dir="./test-data", tag='TAG')
    assert f"./test-data_TAG_{ts}" == out_dir

    out_dir, ts = dir_path_like(src_dir="test-data", tag='Resize')
    assert f"test-data_Resize_{ts}" == out_dir

if __name__ == '__main__':
    test()

    # # Optional: rename if needed
    # dir_renamed = files_copy_rename(src_dir="/home/yun/Documents/code/static/data/origin/noa_yes", prefix="NOA")
    # print(f"renamed to dest_dir: {dir_renamed}") ## ./test-images_Renamed_20230617_082714

    # # Main logic: copy/resize images of the folder

    # t4
    # src_dir = "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_t4"
    # dest_dir = "/home/yun/Documents/code/static/noa-t4-multi/train/t4"
    # rename_folder_backup_if_exist(dest_dir)
    # images_copy_resize(src_dir=src_dir, dest_dir=dest_dir, width=300, height=300)
    
    # noa
    # src_dir = "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_noa"
    # dest_dir = "/home/yun/Documents/code/static/noa-t4-multi/train/noa"
    # rename_folder_backup_if_exist(dest_dir)
    # images_copy_resize(src_dir=src_dir, dest_dir=dest_dir, width=300, height=300)

    # # raw_closing_statement
    # src_dir = "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_closing_statement"
    # dest_dir = "/home/yun/Documents/code/static/noa-t4-multi/train/closing_statement"
    # rename_folder_backup_if_exist(dest_dir)
    # images_copy_resize(src_dir=src_dir, dest_dir=dest_dir, width=300, height=300)

    # # raw_bank_statement
    # src_dir = "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_bank_statement"
    # dest_dir = "/home/yun/Documents/code/static/noa-t4-multi/train/bank_statement"
    # rename_folder_backup_if_exist(dest_dir)
    # images_copy_resize(src_dir=src_dir, dest_dir=dest_dir, width=300, height=300)

    fr_to_dirs = [
        {
            "fr": "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_noa",
            "to": "/home/yun/Documents/code/static/noa-t4-multi/train/noa"
        },
        {
            "fr": "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_t4",
            "to": "/home/yun/Documents/code/static/noa-t4-multi/train/t4"
        },
        {
            "fr": "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_bank_statement",
            "to": "/home/yun/Documents/code/static/noa-t4-multi/train/bank_statement"
        },
        {
            "fr": "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_paystub",
            "to": "/home/yun/Documents/code/static/noa-t4-multi/train/paystub"
        },
        {
            "fr": "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_closing_statement",
            "to": "/home/yun/Documents/code/static/noa-t4-multi/train/closing_statement"
        }

    ]

    for fr_to in fr_to_dirs:
        rename_folder_backup_if_exist(fr_to["to"])
        images_copy_resize(src_dir=fr_to["fr"], dest_dir=fr_to["to"], width=300, height=300)
