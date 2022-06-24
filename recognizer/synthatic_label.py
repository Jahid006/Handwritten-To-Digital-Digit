from PIL import Image, ImageFont, ImageDraw
from config import FONT_DIR,FONT_SIZE


def get_label(N=10):
    ''' Generates Synthetic image of digit 0 to 9'''

    synthetic_labels = []
    for n in range(N):
        image = Image.new('L', (28, 28))
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(FONT_DIR, FONT_SIZE)

        text = str(n)
        draw.text((7, -2), text,fill ="white", font = font, align ="center")

        synthetic_labels.append(image)

    return synthetic_labels