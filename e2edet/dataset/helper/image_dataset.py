import torch

from e2edet.dataset.reader import ImageReader


class BaseImageDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(BaseImageDataset, self).__init__()


class ImageDataset(BaseImageDataset):
    def __init__(self, *args, **kwargs):
        super(ImageDataset, self).__init__()
        self.image_readers = []
        self.image_dict = {}

        for image_dir in kwargs["directories"]:
            image_reader = ImageReader(
                base_path=image_dir,
                reader_type=kwargs["reader_type"],
            )
            self.image_readers.append(image_reader)
        self.imdb = kwargs["imdb"]
        self.max_cache = kwargs.get("max_img_cache", 500)
        self.kwargs = kwargs

    def _fill_cache(self, image_file):
        images = self._read_image_and_info(image_file)
        if len(self.image_dict) < self.max_cache:
            self.image_dict[image_file] = images

        return images

    def _read_image_and_info(self, image_file):
        images = []
        for image_reader in self.image_readers:
            image = image_reader.read(image_file)

            images.append(image)

        return images

    def _get_image_and_info(self, image_file):
        images = self.image_dict.get(image_file, None)

        if images is None:
            images = self._fill_cache(image_file)
            # images = self._read_image_and_info(image_file)

        return images

    def __len__(self):
        return len(self.imdb) - 1

    def __getitem__(self, idx):
        image_info = self.imdb[idx]
        image_file_name = image_info.get("img_path", None)

        if image_file_name is None:
            raise AttributeError("Missing 'img_path' field in imdb")

        image_file = image_file_name
        images = self._get_image_and_info(image_file)

        item = {"image": images[0]}

        return item
