import argparse
import json
import numpy as np
import cv2
import ast
import os
import csv
from src.stored_data_manager import StoredDataManager


def shape_tuple(arg_str):
    try:
        retval = ast.literal_eval(arg_str)
    except ValueError:
        raise argparse.ArgumentTypeError("%s is not a valid tuple string" % arg_str)
    if len(retval) != 2:
        raise argparse.ArgumentTypeError("%s is not a valid shape (must be have 2 items)" % arg_str)
    return retval


def get_characteristics(im_data: np.ndarray, shape: tuple) -> np.ndarray:
    im_p = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)        # grayscale
    im_p = cv2.GaussianBlur(im_p, (5, 5), 1)  # blurring to remove noise
    # adaptive threshold to create a binary image
    im_p = cv2.adaptiveThreshold(im_p, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow("im", im_p)
    # cv2.waitKey(0)
    
    retval = np.zeros(shape)
    
    x_coords = np.linspace(0, im_data.shape[1], shape[0] + 1, dtype=float).round().astype(int)
    y_coords = np.linspace(0, im_data.shape[0], shape[1] + 1, dtype=float).round().astype(int)
    for x in range(shape[0]):
        x_from = x_coords[x]
        x_to = x_coords[x + 1]
        for y in range(shape[1]):
            y_from = y_coords[y]
            y_to = y_coords[y + 1]
            sub_im = im_p[y_from:y_to, x_from:x_to]
            n_white = np.sum(sub_im) // 255 # since threshed values are either 0 or 255, normalizing the sum with 255 gets us the exact number of white pixels
            n_black = sub_im.shape[0] * sub_im.shape[1] - n_white
            retval[x, y] = n_black / (sub_im.shape[0] * sub_im.shape[1]) # return ratio of black pixels / area for sector x, y
            
    return retval
            


def main():
    parser = argparse.ArgumentParser(description='Extract characteristics from images')
    parser.add_argument('-d', '--database', type=str, default='./database')
    parser.add_argument('-s', '--shape', type=shape_tuple, default="(2, 2)")
    parser.add_argument('-p', '--output_dir', type=str, required=False)
    parser.add_argument('letter_dir', type=str)
    parser.add_argument('output_name', type=str)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.letter_dir
    elif not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    d_man = StoredDataManager(args.database)
    settings = {
        "shape": list(args.shape),
        "result_file": os.path.join(args.output_dir, f"{args.output_name}.csv")
    }

    # save settings
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w', encoding='utf-8') as f:
        json.dump(settings, f)

    with open(settings["result_file"], 'w', encoding='utf-8') as f:
        csv_w = csv.writer(f, delimiter=';')
        # iterate through sorted letter directories
        for d in os.scandir(args.letter_dir):
            if not d.is_dir():
                continue
            
            n_files = len(os.listdir(d.path))
            tag = d.name  # directory name is also the letters' tag
            if tag not in d_man.tag_map:
                continue
                
            print(f"Processing dir: {tag}:")
            for j, pic in enumerate(os.scandir(d.path)):
                im_data = cv2.imread(pic.path)
                ch = get_characteristics(im_data, args.shape)

                # save data
                #format: dir/file;tag code;flattened characteristics list
                csv_w.writerow([f'{tag}/{pic.name}', d_man.tag_map[tag]] + ch.flatten().tolist())
                signs = int((j + 1) / n_files * 10)
                print("\r" + "#" * signs + "_" * (10 - signs) + f' {(j + 1) / n_files * 100:0.1f} %', end="")
            print()


if __name__ == "__main__":
    main()
