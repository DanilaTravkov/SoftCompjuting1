"""Brojanje vrednosti banana pomocu segmentacije boje i osobina regiona.

Digitalni asistent: OpenAI Codex je pomogao u analizi materijala i slika,
pisanju koda, komentara i lokalnoj proveri MAE.
Izvori metoda (materijali predmeta):
https://github.com/ftn-ai-lab/sc-2026/blob/master/v1-uvod/sc-v1-cv-basics.ipynb
https://github.com/ftn-ai-lab/sc-2026/blob/master/v2-segmentation/sc-v2-segmentation.ipynb
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import mean_absolute_error


def color_channels(image):
    """BGR kanali kao float"""
    return image.astype(np.float32).transpose(2, 0, 1)


def prepare_mask(mask):
    """Uklanja HUD, spaja male prekide i uklanja sitan sum."""
    mask = mask.copy()
    # Iste zone interfejsa na svakoj slici.
    mask[:145, :275] = False
    mask[:80, :320] = False
    mask[190:260, :190] = False
    mask = np.uint8(mask)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) # если сразу выполнить открытие, то некоторые объекты могут исчезнуть
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask

# «После цветовой сегментации внутри объекта могут появиться разрывы, а на фоне — мелкие белые пятна. Сначала закрытие с ядром 3×3 помогает соединить близкие фрагменты и заполнить небольшие отверстия. Затем открытие с ядром 2×2 удаляет мелкие белые структуры. Операции не являются обратными. Выбранный порядок позволяет сначала попытаться сохранить цельность кандидатов, но может объединять соседние объекты — это ограничение решения»


def external_contours(mask):
    """Samo spoljne (external) konture"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def valid_size(width, height):
    """Zajednicki opseg dimenzija kandidata u referentnoj rezoluciji."""
    return 4 <= width <= 45 and 7 <= height <= 45


def segment(image):
    """Isti pragovi i iste operacije primenjuju se na svaku sliku."""
    b, g, r = color_channels(image)
    yellow = (b < 110) & (g > 1.8*b) & (r > .95*g)
    broad = yellow & (r > 150) & (g > 120) & (r-g < 85)
    mask = yellow & (r > 165) & (g > 140) & (r-g < 75)
    core = (r > 200) & (g > 170) & (b < 90) & (r-g < 65)
    return prepare_mask(mask), prepare_mask(broad), core


def detect_objects(image):
    """Vraca detekcije iz piksela"""
    if image.shape[:2] != (720, 1280):
        image = cv2.resize(image, (1280, 720))
    mask, broad, core = segment(image)
    contours = external_contours(mask)
    broad_contours = external_contours(broad)
    broad_regions = []
    for contour in broad_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if valid_size(w, h):
            broad_regions.append((x, y, w, h, cv2.contourArea(contour)))
    detections = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 20 or not valid_size(w, h):
            continue
        local = np.zeros((h, w), np.uint8)
        cv2.drawContours(local, [contour - np.array([[[x, y]]])], -1, 255, -1)
        bright = np.count_nonzero(core[y:y+h, x:x+w] & (local > 0))
        if bright < 10 or bright/area < .45:
            continue
        fill = area/(w*h)
        if fill < .24:
            continue
        # Zeleni listovi oko zutog regiona cesto predstavljaju dekoraciju.
        roi = image[max(0,y-5):y+h+5, max(0,x-5):x+w+5]
        b, g, r = color_channels(roi)
        green = np.mean((g > 1.12*r) & (g > 1.3*b))
        if green > .25:
            continue
        # Sira maska cuva tamnije delove grozda koje strozi prag izostavlja.
        # Koristi se samo kompaktan region koji obuhvata centar kandidata.
        shape_area, shape_fill = area, fill
        cx, cy = x+w/2, y+h/2
        for bx, by, bw, bh, ba in broad_regions:
            if bx <= cx < bx+bw and by <= cy < by+bh:
                shape_area, shape_fill = ba, ba/(bw*bh)
                break
        if area > 600 and fill > .60:
            kind, value = 'coin', 10
        elif area >= 150 and shape_area >= 180 and shape_fill >= .38:
            kind, value = 'bunch', 5
        else:
            kind, value = 'banana', 1
        detections.append({'box': (x,y,w,h), 'kind': kind, 'value': value})
    return sorted(detections, key=lambda d:(d['box'][1], d['box'][0]))


def predict_value(image):
    return sum(d['value'] for d in detect_objects(image))


def evaluate(dataset):
    dataset = Path(dataset)
    with (dataset/'count.csv').open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('count.csv je prazan')
    results = []
    for row in rows:
        name = row['image_name']
        image = cv2.imread(str(dataset/'images'/name))
        if image is None:
            raise ValueError(f'Ne mogu da ucitam sliku: {name}')
        predicted = predict_value(image)
        expected = int(row['bananas_value'])
        results.append((name, expected, predicted, abs(predicted-expected)))
    expected_values = [row[1] for row in results]
    predicted_values = [row[2] for row in results]
    mae = mean_absolute_error(expected_values, predicted_values)
    return float(mae), results


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Upotreba: python solution.py <dataset_folder>')
    mae, _ = evaluate(sys.argv[1])
    print(f'{mae:.10f}')
