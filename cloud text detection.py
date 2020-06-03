import io
import os
import pandas as pd

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io

    client = vision.ImageAnnotatorClient()

    with io.open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    units = pd.DataFrame(
        columns=[
            "word",
            "upper_left_x",
            "upper_right_x",
            "bottom_right_x",
            "bottom_left_x",
            "upper_left_y",
            "upper_right_y",
            "bottom_right_y",
            "bottom_left_y",
        ]
    )

    for i, text in enumerate(texts):
        print('\n"{}"'.format(text.description))

        vertices = [
            "({},{})".format(vertex.x, vertex.y)
            for vertex in text.bounding_poly.vertices
        ]
        print("bounds: {}".format(",".join(vertices)))
        if i > 0:
            units.loc[i] = [
                text.description,
                text.bounding_poly.vertices[0].x,
                text.bounding_poly.vertices[1].x,
                text.bounding_poly.vertices[2].x,
                text.bounding_poly.vertices[3].x,
                text.bounding_poly.vertices[0].y,
                text.bounding_poly.vertices[1].y,
                text.bounding_poly.vertices[2].y,
                text.bounding_poly.vertices[3].y,
            ]

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return units


units = detect_text("opnecvcdtest01.jpg")
units.to_csv("cds.csv", index=False)
