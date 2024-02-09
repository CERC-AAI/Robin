import multiprocessing
import concurrent.futures
import os
import json
import urllib.request as ureq

# BASE_PATH = "images"
BASE_PATH = "/app/LLaVA-Finetune"

def download_dataset(folder_name, urls):
    if folder_name == "json":
        os.system(f"wget -q https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json -P {BASE_PATH}")
        return

    if isinstance(urls, str):
        urls = [urls]

    for url in urls:
        name = url.split("/")[-1]
        os.makedirs(f"{BASE_PATH}/{folder_name}", exist_ok=True)
        os.system(f"wget -q {url} -P {BASE_PATH}/{folder_name}")
        os.system(f"unzip -q {BASE_PATH}/{folder_name}/{name} -d {BASE_PATH}/{folder_name}")
        os.system(f"rm {BASE_PATH}/{folder_name}/*.zip")

def download_image(k, url, path):
    ext = os.path.splitext(url)[1]
    outputFile = f'{path}/images/%s%s' % (k, ext)
    ureq.urlretrieve(url, outputFile)

def download_ocr_data():
    folder_name = "ocr_vqa"
    os.makedirs(f"{BASE_PATH}/{folder_name}", exist_ok=True)

    ### DOWNLOAD DATASET.JSON
    meta_url = "https://drive.usercontent.google.com/download?id=1r0tyZUwGCc4wIG4RkiglCGNL_nFJjR6Q&export=download&authuser=0&confirm=t&at=APZUnTW8fGOfgvS7p_RjJKw6sXyU:1707402060685"
    ureq.urlretrieve(meta_url, f'{BASE_PATH}/{folder_name}/dataset.json')

    with open(f'{BASE_PATH}/{folder_name}/dataset.json', 'r') as fp:
            data = json.load(fp)

    os.makedirs(f'{BASE_PATH}/{folder_name}/images', exist_ok=True)
    
    # for k in data.keys():
    #     ext = os.path.splitext(data[k]['imageURL'])[1]
    #     outputFile = f'{BASE_PATH}/{folder_name}/images/%s%s' % (k, ext)
    #     ureq.urlretrieve(data[k]['imageURL'], outputFile)

    pool = multiprocessing.Pool(100)

    # Call the run function in parallel using the pool
    inputs = [(k, data[k]['imageURL'], f"{BASE_PATH}/{folder_name}") for k in data.keys()]
    results = pool.starmap(download_image, inputs)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    os.system(f"mogrify -format jpg {BASE_PATH}/{folder_name}/images/*.gif")
    os.system(f"mogrify -format jpg {BASE_PATH}/{folder_name}/images/*.png")

if __name__ == "__main__":
    # Define your arguments
    datasets = [
        ("json", ""),
        ("coco", "images.cocodataset.org/zips/train2017.zip"),
        ("gqa", "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"),
        ("textvqa", "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"),
        ("vg", ("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
               "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip") )
    ]

    os.makedirs(f"{BASE_PATH}", exist_ok=True)
    
    # Create a multiprocessing pool
    pool = multiprocessing.Pool()

    # Call the run function in parallel using the pool
    results = pool.starmap(download_dataset, datasets)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Download OCR Data
    download_ocr_data()