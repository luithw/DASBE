import numpy as np
import matplotlib.pyplot as plt
from plot_tools import plot_groups, plot_input_image
import h5py
import os
import os.path
import numpy as np
from torch.utils.data import DataLoader, IterableDataset

np.random.seed(9825619)

data_dir = "./tmp_data"

# Load the MNIST Dataset as prepared by the brainstorm data script
# You will need to run brainstorm/data/create_mnist.py first
with h5py.File(os.path.join(data_dir, 'MNIST.hdf5'), 'r') as f:
    mnist_digits = f['normalized_full/training/default'][0, :]
    mnist_targets = f['normalized_full/training/targets'][:]
    mnist_digits_test = f['normalized_full/test/default'][0, :]
    mnist_targets_test = f['normalized_full/test/targets'][:]


def crop(d):
    return d[np.sum(d, 1) != 0][:, np.sum(d, 0) != 0]


def generate_multi_mnist_img(digit_nrs, size=(60, 60), test=False, binarize_threshold=0.5):
    if not test:
        digits = [crop(mnist_digits[nr].reshape(28, 28)) for nr in digit_nrs]
    else:
        digits = [crop(mnist_digits_test[nr].reshape(28, 28)) for nr in digit_nrs]

    flag = False
    while not flag:
        img = np.zeros(size)
        grp = np.zeros(size)
        mask = np.zeros(size)
        k = 1

        for i, digit in enumerate(digits):
            h, w = size
            sy, sx = digit.shape
            x = np.random.randint(0, w - sx + 1)
            y = np.random.randint(0, h - sy + 1)
            region = (slice(y, y + sy), slice(x, x + sx))
            m = digit >= binarize_threshold
            img[region][m] = 1
            mask[region][m] += 1
            grp[region][m] = k
            k += 1
        if len(digit_nrs) <= 1 or (mask[region][m] > 1).sum() / (mask[region][m] >= 1).sum() < 0.2:
            flag = True

    grp[mask > 1] = 0  # ignore overlap regions
    return img, grp

def gen_dataset(num_samples, size, num_digits):
    for _ in range(num_samples):
        digit_nrs = np.random.randint(0, 60000, num_digits)
        yield generate_multi_mnist_img(digit_nrs, (size, size))

class MNISTDataset(IterableDataset):
    def __init__(self, num_samples, size, num_digits):
        super(MNISTDataset, self).__init__()
        self.size = size
        self.num_digits = num_digits
        self.num_samples = num_samples

    def __iter__(self):
        return gen_dataset(self.num_samples, self.size, self.num_digits)


#dataset = MNISTDataset(2, 28, 2)
#dataloader = DataLoader(dataset)

#for images, group in dataloader:
#    print(images, group)