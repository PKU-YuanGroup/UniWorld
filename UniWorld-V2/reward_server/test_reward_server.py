import requests
import pickle
from PIL import Image
import io

ref_path = "assets/ref_image.png"
img_path = "assets/image.png"

def test_cot_continue():
    ref_image = Image.open(ref_path).convert("RGB")
    ref_image_io = io.BytesIO()
    ref_image.save(ref_image_io, format="JPEG")
    ref_image_io.seek(0)

    image = Image.open(img_path).convert("RGB")
    image_io = io.BytesIO()
    image.save(image_io, format="JPEG")
    image_io.seek(0)

    data = {
        "images": [image_io.getvalue(), image_io.getvalue()],
        "ref_images": [ref_image_io.getvalue(), ref_image_io.getvalue()],
        "prompts": ["Change the color of the snowman's hat from black to green", "Change the color of the snowman's hat from black to red"],
        "metadatas": [
            {"requirement": "None"},
            {"requirement": "None"},
        ],
    }

    payload = pickle.dumps(data)
    url = "http://127.0.0.1:12341/mode/logits_non_cot"
    proxies = {
        "http": None,
        "https": None,
    }
    response = requests.post(
        url,
        data=payload,
        proxies=proxies,
        headers={"Content-Type": "application/octet-stream"},
    )

    if response.status_code == 200:
        result = pickle.loads(response.content)
        print("Scores:", result["scores"])
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    test_cot_continue()
