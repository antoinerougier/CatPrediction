import io
import pytest
from PIL import Image


@pytest.fixture
def sample_cat_image_bytes():
    img = Image.new("RGB", (224, 224), color=(120, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()
