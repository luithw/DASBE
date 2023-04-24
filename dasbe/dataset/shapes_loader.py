import numpy as np
from torch.utils.data import DataLoader, IterableDataset

np.random.seed(104174)
data_dir = "./tmp_data/"

square = np.array(
    [[1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1]])

triangle = np.array(
    [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

shapes = [square, triangle, triangle[::-1, :].copy()]


def generate_shapes_image(width, height, nr_shapes=3):
    img = np.zeros((height, width))
    grp = np.zeros_like(img)
    k = 1

    for i in range(nr_shapes):
        shape = shapes[np.random.randint(0, len(shapes))]
        sy, sx = shape.shape
        print(shape, sy, sx)
        print(width - sx + 1)
        x = np.random.randint(0, width - sx + 1)
        y = np.random.randint(0, height - sy + 1)
        region = (slice(y, y + sy), slice(x, x + sx))
        img[region][shape != 0] += 1
        grp[region][shape != 0] = k
        k += 1

    grp[img > 1] = 0
    img = img != 0
    return img, grp

def gen_dataset(num_samples, width, height, num_shapes):
    for _ in range(num_samples):
        yield generate_shapes_image(width, height, num_shapes)

class ShapesDataset(IterableDataset):
    def __init__(self, num_samples, width, height, num_shapes):
        super(ShapesDataset, self).__init__()
        self.width = width
        self.height = height
        self.num_shapes = num_shapes
        self.num_samples = num_samples

    def __iter__(self):
        return gen_dataset(self.num_samples, self.width, self.height, self.num_shapes)


#dataset = ShapesDataset(2, 28, 28, 2)
#dataloader = DataLoader(dataset)

#for images, group in dataloader:
#    print(images.shape, group.shape)