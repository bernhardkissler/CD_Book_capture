try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

import cv2
from pytesseract import Output

img = cv2.imread("opnecvcdtest_einzeln_02.jpg")

h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img, lang="deu")
for b in boxes.splitlines():
    b = b.split(" ")
    img = cv2.rectangle(
        img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2
    )

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

d = pytesseract.image_to_data(img, output_type=Output.DICT, lang="deu")
print(d.keys())

n_boxes = len(d["text"])
for i in range(n_boxes):
    if int(d["conf"][i]) > 60:
        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

x, y, w, h = cv2.boundingRect(cnt)
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)


test_picture = "opnecvcdtest_einzeln_02.jpg"

print(pytesseract.image_to_string(Image.open(test_picture), lang="deu",))

print("done")
