import os

from e2edet.utils.det3d.general import read_from_file


class PointReader:
    def __init__(self, base_path, reader_type):
        self.base_path = base_path
        self.reader_type = reader_type
        self.point_reader = None

    def _init_reader(self):
        if self.reader_type == "waymo":
            self.point_reader = WaymoReader()
        else:
            raise TypeError("unknown lidar reader type")

    def read(self, lidar_path):
        lidar_path = os.path.join(self.base_path, lidar_path)
        assert os.path.exists(lidar_path)

        if self.point_reader is None:
            self._init_reader()

        return self.point_reader.read(lidar_path)


class WaymoReader:
    def read(self, info, nsweeps=1):
        return read_from_file(info, nsweeps=nsweeps)
