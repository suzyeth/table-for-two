"""Multi-camera video frames for rollout recordings.

The operator camera sits behind the arms, so the jaws are hidden exactly when
they close on an object. ``compose_views`` renders several cameras from the same
simulation step and tiles them into one frame, each tile labelled with its
camera name, so a single run can be watched from the front, from above and from
both wrists at once.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

TILE_SIZE = (360, 480)  # (height, width) of one camera tile
LABEL_FILL = (240, 194, 51)


def tile_frames(tiles, labels=None, cols=None):
    """Tile equally sized HxWx3 uint8 images into one grid image, row by row.

    ``cols`` defaults to a near-square grid. Empty cells are black. With
    ``labels`` each tile gets its label drawn in the top-left corner.
    """
    if not tiles:
        raise ValueError("no tiles to compose")
    height, width = tiles[0].shape[:2]
    if any(t.shape[:2] != (height, width) for t in tiles):
        raise ValueError("all tiles must have the same size")
    if labels is not None and len(labels) != len(tiles):
        raise ValueError("one label per tile")
    cols = cols or math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * width, rows * height))
    draw = ImageDraw.Draw(sheet)
    for i, tile in enumerate(tiles):
        x, y = (i % cols) * width, (i // cols) * height
        sheet.paste(Image.fromarray(np.asarray(tile, dtype=np.uint8)), (x, y))
        if labels is not None:
            draw.text((x + 8, y + 6), labels[i], fill=LABEL_FILL)
    return np.asarray(sheet)


def compose_views(env, cameras, tile_size=TILE_SIZE):
    """One video frame: a single camera as-is, or several cameras tiled and labelled."""
    if len(cameras) == 1:
        return env.render(cameras[0])
    return tile_frames([env.render(cam, tile_size) for cam in cameras], labels=list(cameras))
