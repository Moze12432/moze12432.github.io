from PIL import Image, ImageChops, ImageEnhance
import os

def perform_ela(image_path, quality=90):

    original = Image.open(image_path).convert('RGB')

    temp_filename = "temp_ela.jpg"

    original.save(temp_filename, 'JPEG', quality=quality)

    compressed = Image.open(temp_filename)

    diff = ImageChops.difference(original, compressed)

    extrema = diff.getextrema()

    max_diff = max([ex[1] for ex in extrema])

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(diff).enhance(scale)

    os.remove(temp_filename)

    return ela_image
