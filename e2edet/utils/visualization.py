from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_color(idx) -> List[int]:
    colors = [
        (111, 74, 0),
        (81, 0, 81),
        (128, 64, 128),
        (244, 35, 232),
        (250, 170, 160),
        (230, 150, 140),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (180, 165, 180),
        (150, 100, 100),
        (150, 120, 90),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 0, 90),
        (0, 0, 110),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
        (0, 0, 142),
    ]

    color = colors[idx % len(colors)]

    return color


def draw_bbox3d(image, bbox_vertices, color=(0, 200, 200), thickness=1):
    for idx in range(bbox_vertices.shape[0] - 1):
        v1 = (bbox_vertices[idx][0].item(), bbox_vertices[idx][1].item())
        v2 = (bbox_vertices[idx + 1][0].item(), bbox_vertices[idx + 1][1].item())

        image = cv2.line(image, v1, v2, color, thickness)

    return image


def draw_line(image, v1, v2, color=(0, 200, 200), thickness=1) -> None:
    return cv2.line(image, v1, v2, color, thickness)


def draw_circle(
    image, position, radius=5, color=(250, 100, 100), thickness=1, fill=True
) -> None:
    if fill:
        thickness = -1
    center = (int(position[0]), int(position[1]))
    return cv2.circle(image, center, radius, color=color, thickness=thickness)


def draw_bbox2d(image, bbox2d, color=(0, 200, 200), thickness=1) -> None:
    v1 = (int(bbox2d[0].item()), int(bbox2d[1].item()))
    v2 = (int(bbox2d[2].item()), int(bbox2d[3].item()))

    return cv2.rectangle(image, v1, v2, color, thickness)


def draw_text(
    image,
    text,
    position,
    scale=0.4,
    color=(0, 0, 0),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    bg_color=(255, 255, 255),
    blend=0.33,
    lineType=1,
) -> None:
    position = [int(position[0]), int(position[1])]
    if bg_color is not None:
        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(position[0], a_min=0, a_max=image.shape[1]))
        x_e = int(
            np.clip(position[0] + text_size[0] - 1 + 4, a_min=0, a_max=image.shape[1])
        )
        y_s = int(
            np.clip(position[1] - text_size[1] - 2, a_min=0, a_max=image.shape[0])
        )
        y_e = int(np.clip(position[1] + 1 - 2, a_min=0, a_max=image.shape[0]))

        image[y_s : y_e + 1, x_s : x_e + 1, 0] = image[
            y_s : y_e + 1, x_s : x_e + 1, 0
        ] * blend + bg_color[0] * (1 - blend)
        image[y_s : y_e + 1, x_s : x_e + 1, 1] = image[
            y_s : y_e + 1, x_s : x_e + 1, 1
        ] * blend + bg_color[1] * (1 - blend)
        image[y_s : y_e + 1, x_s : x_e + 1, 2] = image[
            y_s : y_e + 1, x_s : x_e + 1, 2
        ] * blend + bg_color[2] * (1 - blend)

        position[0] = int(np.clip(position[0] + 2, a_min=0, a_max=image.shape[1]))
        position[1] = int(np.clip(position[1] - 2, a_min=0, a_max=image.shape[0]))

    return cv2.putText(image, text, tuple(position), font, scale, color, lineType)


def draw_attn(image, attn) -> None:
    attn = cv2.applyColorMap(attn[None], cv2.COLORMAP_JET)

    return cv2.addWeighted(attn, 0.6, image, 0.3, 0)


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c],
        )
    return image


def imshow(image, attn=None, figure_num=None) -> None:
    if figure_num is not None:
        plt.figure(figure_num)
    else:
        f, axs = plt.subplots(1, 1, figsize=(15, 15))

    if len(image.shape) == 2:
        image = np.tile(image, [3, 1, 1]).transpose([1, 2, 0])

    plt.tick_params(labelbottom="off", labelleft="off")
    plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))

    if attn is not None:
        plt.imshow(attn, cmap=plt.cm.viridis, interpolation="nearest", alpha=0.9)
    plt.show(block=False)


def imsave(file_name, image, figure_num=None) -> None:
    if figure_num is not None:
        plt.figure(figure_num)
    else:
        f, axs = plt.subplots(1, 1, figsize=(15, 15))

    if len(image.shape) == 2:
        image = np.tile(image, [3, 1, 1]).transpose([1, 2, 0])

    plt.tick_params(labelbottom="off", labelleft="off")
    plt.imsave(file_name, cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))