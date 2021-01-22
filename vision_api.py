# +
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './keyFile.json'
print(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
from google.cloud import vision
client = vision.ImageAnnotatorClient()

import io
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw


# -

def draw_box(image, vertices, color, normalized=False, text = None):
    # Draw a border around the image using the hints in the vector list
    border = 3
    draw = ImageDraw.Draw(image)

    if normalized:
        draw.line([
            vertices[0].x * image.size[0], vertices[0].y * image.size[1],
            vertices[1].x * image.size[0], vertices[1].y * image.size[1],
            vertices[2].x * image.size[0], vertices[2].y * image.size[1],
            vertices[3].x * image.size[0], vertices[3].y * image.size[1],
            vertices[0].x * image.size[0], vertices[0].y * image.size[1]], fill=color, width=border)

        if text is not None:
            draw.text((vertices[0].x * image.size[0] + 2, vertices[0].y * image.size[1] + 2), text=text)

    else:
        draw.line([
            vertices[0].x, vertices[0].y,
            vertices[1].x, vertices[1].y,
            vertices[2].x, vertices[2].y,
            vertices[3].x, vertices[3].y,
            vertices[0].x, vertices[0].y], fill=color, width=border)

        if text is not None:
            try:
                draw.text((vertices[0].x + border + 1, vertices[0].y + border), text=re.sub(r'\n\W', '', text=text))
            except:
                PERMITTED_CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-\n ' 
                text = ''.join(c for c in text if c in PERMITTED_CHARS)
                draw.text((vertices[0].x + border + 1, vertices[0].y + border), text=text)

    return image


def mark_items(frame, return_frame=False):
    is_success, buf = cv2.imencode('.jpg', frame)
    content = buf.tobytes()

    image = vision.Image(content=content)
    response_obj = client.object_localization(image=image)
    objects = response_obj.localized_object_annotations
    object_names = [object_.name for object_ in objects]

    response_text = client.text_detection(image=image)
    texts = response_text.text_annotations
    text_desc = ''
    if len(texts) > 0:
        text_desc = texts[0].description
    
    if return_frame:
        pil_img = Image.fromarray(frame)
        for object_ in objects:
            pil_img = draw_box(pil_img, object_.bounding_poly.normalized_vertices, 'blue', normalized=True, text=object_.name)
        for text in texts[:1]:
            pil_img = draw_box(pil_img, text.bounding_poly.vertices, 'red', text=text.description)
        frame = np.array(pil_img)
        return frame, text_desc, object_names
    else:
        return text_desc, object_names
