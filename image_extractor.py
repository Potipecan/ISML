import argparse
from src.image_processor import ImageProcessor
from src.stored_data_manager import StoredDataManager
from PIL import Image
import os
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Image Extractor')
    parser.add_argument("-d", "--database", default="./database")
    parser.add_argument("-o", "--output-dest", default="./output")
    args = parser.parse_args()
    
    d_man = StoredDataManager(args.database)
    d_man.load_schemas()
    
    if not os.path.exists(args.output_dest):
        os.mkdir(args.output_dest)
        
    # init output dirs
    for tag in d_man.tag_map:
        path = os.path.join(args.output_dest, tag)
        if not os.path.exists(path) or not os.path.isdir(path):
            os.mkdir(path)
    
    # iterate mapping data
    for i, d, files in enumerate(d_man.mapping_data.items()):
        for j, file, data in enumerate(files.items()):
            print(f"Processing directory {d}, {i + 1}/{len(d_man.mapping_data)}; file {file}, {j + 1}/{len(files)}")
            image = Image.open(os.path.join(d, file)).convert('RGB')
            processor = ImageProcessor(image)
            processor.process_image(data, d_man.schemas, d_man.tag_map, args.output_dest, Path(file).stem)
            
            
    
    
if __name__ == "__main__":
    main()