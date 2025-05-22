import argparse
from src.stored_data_manager import StoredDataManager
import os
import numpy as np
from pathlib import Path
import cv2


def align_image(image, corners):
    pts_src = np.array(corners[[0, 1, 3, 2]],
                       dtype=np.float32)  # bottom corners must be swapped to get a clockwise order

    # Compute the width and height of the destination rectangle        
    width_top = np.linalg.norm(pts_src[0] - pts_src[1])
    width_bottom = np.linalg.norm(pts_src[3] - pts_src[2])
    width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(pts_src[0] - pts_src[3])
    height_right = np.linalg.norm(pts_src[1] - pts_src[2])
    height = max(int(height_left), int(height_right))

    # destination points for the rectangle
    pts_dst = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    # cv2 magic I found online
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def crop_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)

    angle_tolerance = np.pi / 45
    #find vertical lines
    v_lines = np.squeeze(cv2.HoughLines(edges, 1, np.pi / 180, img.shape[0] // 2, 
                                        min_theta=-angle_tolerance, 
                                        max_theta=angle_tolerance))
    # find horizontal lines
    h_lines = np.squeeze(cv2.HoughLines(edges, 1, np.pi / 180, img.shape[1] // 2,
                                        min_theta=np.pi / 2 - angle_tolerance,
                                        max_theta=np.pi / 2 + angle_tolerance))
    
    if len(h_lines.shape) == 1:
        h_lines = h_lines.reshape((1, h_lines.shape[0]))    
    if len(v_lines.shape) == 1:
        v_lines = v_lines.reshape((1, v_lines.shape[0]))
        
    h, w = img.shape[0: 2]
    if len(v_lines.shape) > 0 and len(v_lines) > 0:
        l_lines = np.where(v_lines[:, 0] <= w * 0.2)[0]
        r_lines = np.where(v_lines[:, 0] >= w * 0.8)[0]
        l = int(np.max(v_lines[l_lines, 0]) + 1 if len(l_lines) > 0 else 0)
        r = int(np.min(v_lines[r_lines, 0]) if len(r_lines) > 0 else w)
    else:
        l = 0
        r = w
        
    if len(h_lines.shape) > 0 and len(h_lines) > 0:
        t_lines = np.where(h_lines[:, 0] <= h * 0.2)[0]
        b_lines = np.where(h_lines[:, 0] >= h * 0.8)[0]
        t = int(np.max(h_lines[t_lines, 0]) + 1 if len(t_lines) > 0 else 0)
        b = int(np.min(h_lines[b_lines, 0]) if len(b_lines > 0) else h)
    else:
        t = 0
        b = h
    
    # ic = img.copy()
    # cv2.line(ic, (l, 0), (l, h), (0, 0, 255), 1)
    # cv2.line(ic, (r, 0), (r, h), (0, 0, 255), 1)
    # cv2.line(ic, (0, t), (w, t), (0, 0, 255), 1)
    # cv2.line(ic, (0, b), (w, b), (0, 0, 255), 1)
    # 
    # cv2.imshow("Crop", ic)
    # cv2.waitKey(0)
    
    # crop
    return img[t: b, l: r]


def process_image(image, settings, schemas, output_dir, file_stem):
    corners = np.array(settings['corners'])
    rotation = settings.get('rotation', 0)
    tags = schemas[settings['schema_key']]['tags']

    aligned = align_image(image, corners)

    x_coords = np.linspace(0, aligned.shape[1], 26, dtype=float).round().astype(int)
    y_coords = np.linspace(0, aligned.shape[0], 51, dtype=float).round().astype(int)

    x_tags = tags if len(tags) == 25 else None
    y_tags = tags if len(tags) == 50 else None
    im_index = 0

    for x in range(25):
        x_from = x_coords[x]
        x_to = x_coords[x + 1]

        tag = x_tags[x] if x_tags is not None else None

        for y in range(50):
            y_from = y_coords[y]
            y_to = y_coords[y + 1]

            tag = y_tags[y] if y_tags is not None else tag
            sub_img = aligned[y_from:y_to, x_from:x_to]

            if rotation != 0:
                match rotation:
                    case 90:
                        rotate_code = cv2.ROTATE_90_CLOCKWISE
                    case 180:
                        rotate_code = cv2.ROTATE_180
                    case 270:
                        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
                    case _:
                        raise ValueError('Invalid rotation')
                sub_img = cv2.rotate(sub_img, rotate_code)

            sub_img = crop_lines(sub_img)

            cv2.imwrite(str(os.path.join(output_dir, tag, f"{file_stem}_{im_index:04}.png")), sub_img)
            im_index += 1


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
    for i, (d, files) in enumerate(d_man.mapping_data.items()):
        for j, (file, data) in enumerate(files.items()):
            print(f"Processing directory {d}, {i + 1}/{len(d_man.mapping_data)}; file {file}, {j + 1}/{len(files)}")
            image = cv2.imread(str(os.path.join(d, file)))
            process_image(image, data, d_man.schemas, args.output_dest, Path(file).stem)


if __name__ == "__main__":
    main()
