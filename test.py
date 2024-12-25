import cv2
from PIL import ImageGrab, Image
import time
import numpy
import torch
import torchvision
import imageio
import os
from pathlib import Path

animation_dir = Path(__file__).parent / "animation"


def test():
    model = torch.jit.load("best.pt").to("cpu")
    fps = 0
    start = time.time_ns()
    transform = torchvision.transforms.ToTensor()
    cv2.namedWindow("test")
    cv2.setWindowProperty("test", cv2.WND_PROP_TOPMOST, 1)

    animation = False
    animation_frame = 0
    animation_fps = 5
    animation_last_time = 0
    if animation:
        os.makedirs(animation_dir)

    while True:
        img = ImageGrab.grab()

        arrow_bbox = (1819, 74, 1819 + 32, 74 + 32)
        arrow_img = img.crop(box=arrow_bbox)

        tensor_image = transform(arrow_img)
        x, y = model(torch.stack([tensor_image])).detach().numpy()[0]
        predicted_angle = numpy.arctan2(y, x)

        radius = 16
        dx = radius * numpy.cos(predicted_angle)
        dy = radius * numpy.sin(predicted_angle)

        minimap_bbox = (1760, 21, 1760 + 144, 21 + 144)
        minimap_img = img.crop(box=minimap_bbox)
        minimap_img = numpy.array(minimap_img)
        minimap_img = minimap_img[:, :, ::-1].copy()
        pt1 = (int(144/2)+2, int(144/2)-3)
        pt2 = (int(144/2 + 2 + dx),
               int(144/2 - 3 + (-dy)))
        minimap_img_with_pred_angle = cv2.line(
            minimap_img,
            pt1=pt1,
            pt2=pt2,
            color=(0, 255, 0),
            thickness=2)

        cv2.imshow("test", minimap_img_with_pred_angle)
        if animation and (time.time_ns() - animation_last_time) > 1_000_000_000/animation_fps:
            cv2.imwrite(animation_dir /
                        f"{animation_frame}.png", minimap_img_with_pred_angle)
            animation_last_time = time.time_ns()
            animation_frame += 1

        fps += 1
        if time.time_ns() - start >= 1_000_000_000:
            start = time.time_ns()
            print("Fps:", fps)
            fps = 0

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break


def create_image_grid():
    rows = 10
    cols = 10
    imgs = []
    img_dir = Path(__file__).parent / "dataset" / "images"
    for file_name in os.listdir(img_dir):
        imgs.append(Image.open(img_dir / file_name))
        if len(imgs) == rows * cols:
            break

    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))

    grid.show()


def compile_animation():
    images = []
    file_names = os.listdir(animation_dir)
    file_names.sort(key=lambda f: int(f.split(".png")[0]))
    for filename in file_names:
        images.append(imageio.v2.imread(animation_dir / filename))

    imageio.mimsave('animation.gif', images, fps=5)


if __name__ == "__main__":
    test()
    # create_image_grid()
    # compile_animation()
