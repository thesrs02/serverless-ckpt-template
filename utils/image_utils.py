from PIL import Image, ImageFilter


def resize_large_images(image: Image.Image, target_size: int = 1024) -> Image.Image:
    width, height = image.size

    if width <= target_size and height <= target_size:
        return image

    aspect_ratio = width / height

    if width > height:
        new_height = target_size
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = target_size
        new_height = int(new_width / aspect_ratio)

    return image.resize((new_width, new_height), Image.LANCZOS)


def crop_to_target(image: Image.Image, target_size: int = 1024) -> Image.Image:
    width, height = image.size

    if width < target_size or height < target_size:
        return image

    left_margin = (width - target_size) // 2
    top_margin = (height - target_size) // 2
    crop_box = (
        left_margin,
        top_margin,
        left_margin + target_size,
        top_margin + target_size,
    )

    return image.crop(crop_box)


def sharpen_image(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.SHARPEN)
