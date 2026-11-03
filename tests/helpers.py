from pathlib import Path

from PIL import Image, ImageDraw


class FakeProvider:
    """Stands in for a model. vision() returns text derived from the file name so tests
    can check wiring; it never inspects pixels."""
    name, model = "fake", "fake-1"

    def __init__(self, text_replies=None):
        self.text_replies = list(text_replies or [])
        self.calls = []

    def vision(self, system, prompt, image_path, max_tokens):
        self.calls.append(("vision", Path(image_path).stem))
        return f"SLIDE: text of {Path(image_path).stem}\nIMAGE: none\nINPHOTO: none"

    def text(self, system, user, max_tokens, *, thinking=False):
        self.calls.append(("text", user[:30]))
        return self.text_replies.pop(0) if self.text_replies else '{"verdict":"PLAUSIBLE","why":""}'


def make_slide(path: Path, text: str, size=(1280, 720), bg=255, lines=()):
    """A slide with LARGE text and solid bars, like a real deck. Tiny default-font text
    changes almost no pixels, so two such 'different' slides look identical to MAE."""
    from PIL import ImageFont
    img = Image.new("RGB", size, (bg, bg, bg))
    d = ImageDraw.Draw(img)
    d.text((60, 40), text, fill=(0, 0, 0), font=ImageFont.load_default(size=72))
    for i, ln in enumerate(lines):
        d.text((60, 180 + 70 * i), ln, fill=(0, 0, 0), font=ImageFont.load_default(size=48))
        d.rectangle((60, 235 + 70 * i, 60 + 40 * len(ln), 245 + 70 * i), fill=(30, 30, 30))
    img.save(path, quality=95)
    return path
