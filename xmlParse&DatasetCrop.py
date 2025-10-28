import xml.etree.ElementTree as ET
import cv2


def parseDataset(sourcePath: str, resultPath: str, xmlPath: str) -> None:
    doc = ET.parse(xmlPath)
    root = doc.getroot()
    images = root.findall("image")
    counter = 0
    for img in images:
        for box in img.findall("box"):
            x, y, x2, y2 = int(round(float(box.get("xtl")))), int(round(float(box.get("ytl")))), int(round(float(box.get("xbr")))), int(
                round(float(box.get("ybr"))))
            image = cv2.imread(f"{sourcePath}{img.get('name')}")
            cropped_image = image[y:y2, x:x2]
            cv2.imwrite(f"{resultPath}{box.get('label')}{counter}.jpg", cropped_image)
            counter += 1


parseDataset("./trainSource/", "./train/", "./trainSource/annotations.xml")
parseDataset("./testSource/", "./test/", "./testSource/annotations.xml")
