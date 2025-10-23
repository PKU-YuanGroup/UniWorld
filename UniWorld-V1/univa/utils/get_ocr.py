import pandas as pd
import numpy as np
import pandas as pd
from PIL import ImageDraw
from datasets import load_dataset, Image
from PIL import Image
try:
    from paddleocr import PaddleOCR
except:
    PaddleOCR = None
          

def ocr_with_paddle(img):
    if PaddleOCR is None:
        raise ValueError('sudo apt install swig -y && pip install paddleocr==2.7.0.3 paddle-bfloat==0.1.7 paddlepaddle==2.5.2 protobuf==3.20.2')
    ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)
    result = ocr.ocr(img)
    new_result = []
    if result[0] is None:
        return new_result
    for i in result[0]:
        new_result.append(i[:-1] + [i[-1][0], i[-1][1]])
    return new_result

def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image
       

def calculate_position(box, width, height):
    """Calculates the position of a bounding box within a 9-grid.

    Args:
        box: A list of coordinates representing the bounding box (e.g., [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).
        width: The width of the image.
        height: The height of the image.

    Returns:
        A string representing the position of the box (e.g., "top-left", "center", "bottom-right").
    """
    x_coords = [coord[0] for coord in box]
    y_coords = [coord[1] for coord in box]

    # Calculate the center of the bounding box
    center_x = (min(x_coords) + max(x_coords)) / 2
    center_y = (min(y_coords) + max(y_coords)) / 2

    # Determine the row and column position
    if center_y < height / 3:
        row = "top"
    elif center_y < 2 * height / 3:
        row = "middle"
    else:
        row = "bottom"

    if center_x < width / 3:
        col = "left"
    elif center_x < 2 * width / 3:
        col = "center"
    else:
        col = "right"

    return f"{row}-{col}"


def process_dataframe(df, image_width, image_height):
    """Processes the DataFrame to filter by score and add a position column.

    Args:
        df: The input Pandas DataFrame with 'box', 'text', and 'score' columns.
        image_width: The width of the image.
        image_height: The height of the image.

    Returns:
        A Pandas DataFrame filtered by score and with an added 'position' column.
    """

    # Filter the DataFrame by score
    df_filtered = df[df['score'] > 0.9].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Apply the position calculation and create the 'position' column
    df_filtered['position'] = df_filtered['box'].apply(lambda box: calculate_position(box, image_width, image_height))

    return df_filtered



def format_for_text_to_image_condensed(df, image_number):
    """Formats the DataFrame into a condensed sentence for text-to-image models,
    grouping text at the same position, and includes the image number (full spelling)."""
    if len(df) == 0:
        return ''
    ordinal_map = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
        15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
        19: "nineteenth", 20: "twentieth"
    }

    ordinal = ordinal_map.get(image_number, None)  # Use number as string if not in map
    assert ordinal is not None, "ordinal is not None"
    position_to_texts = {}
    for index, row in df.iterrows():
        position = row['position']
        text = row['text']
        if position in position_to_texts:
            position_to_texts[position].append(text)
        else:
            position_to_texts[position] = [text]

    sentences = [f'In the {ordinal} image: (']
    for position, texts in position_to_texts.items():
        quoted_texts = [f"\"{text}\"" for text in texts]  # Quote each text
        text_string = ", ".join(quoted_texts)  # Join with commas
        sentences.append(f"The texts {text_string} are located at the {position} of the {ordinal} image.")
    return " ".join(sentences) + ' )'

def get_ocr_result(img_path: str, img_index: int = 0):
    img_index = img_index + 1
    ocr_result = ocr_with_paddle(img_path)
    ocr_result_df = pd.DataFrame(ocr_result, columns=['box', 'text', 'score'])
    image_width, image_height = Image.open(img_path).size 
    df_processed = process_dataframe(ocr_result_df, image_width, image_height)
    formatted_sentence = format_for_text_to_image_condensed(df_processed, image_number=img_index)
    return formatted_sentence