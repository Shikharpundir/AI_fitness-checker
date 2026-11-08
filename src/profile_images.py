from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_IMAGE_PIXELS = 20_000_000
PROFILE_IMAGE_SIZE = (512, 512)


def normalize_profile_image(file_storage):
    """Validate an uploaded image and return a resized JPEG representation."""
    try:
        image = Image.open(file_storage.stream)
        if image.format not in ALLOWED_IMAGE_FORMATS:
            raise ValueError("Choose a JPEG, PNG, or WebP image.")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("The image dimensions are too large.")
        image.verify()

        file_storage.stream.seek(0)
        image = Image.open(file_storage.stream)
        image = ImageOps.exif_transpose(image)
        image.thumbnail(PROFILE_IMAGE_SIZE, Image.Resampling.LANCZOS)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            foreground = image.convert("RGBA")
            background = Image.new("RGB", foreground.size, "white")
            background.paste(foreground, mask=foreground.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

        output = BytesIO()
        image.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()
    except ValueError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a valid supported image.") from error
