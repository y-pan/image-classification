import os

from bing_image_downloader import downloader


# output_dir = os.path.join(os.path.dirname(__file__), "bankStatement-canada")  
output_dir = "/home/yun/Documents/code/static/noa-t4-multi/raw"
os.makedirs(output_dir, exist_ok=True)

downloader.download("T4 Statement of Remuneration Paid", 
                    limit=500, 
                    output_dir=output_dir, 
                    adult_filter_off=False, 
                    force_replace=False, 
                    timeout=60, 
                    verbose=True)

print(f"=== end ===")