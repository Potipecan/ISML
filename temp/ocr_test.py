import pytesseract as pyt
import cv2
from PIL import Image, ImageDraw, ImageOps
import os

for im_path in os.scandir("../data/raw"):

    img = (Image.open(im_path))
    boxes = pyt.image_to_boxes(img, 'slv')
    boxes = [list(map(lambda x: int(x), s.split(' ')[1 : 5])) for s in boxes.split('\n')]
    boxes = [b for b in boxes if all(b)]
    img = ImageOps.grayscale(img)
    draw = ImageDraw.Draw(img)
    for box in boxes:
        if len(box) != 4:
            continue
        draw.rectangle(xy=box, outline=(0,), width=2)
        
    img.show()
        
    
    
