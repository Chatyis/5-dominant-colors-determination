from threading import Lock


class ConfigMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=ConfigMeta):
    _cluster_amt: int
    _maximum_file_size: int
    _rag_threshold: int
    _is_rag_enabled: bool
    _image_path = 'Select image before continuing...'

    def __init__(self) -> None:
        self.change_config()

    def change_config(self, cluster_amt=32, maximum_file_size=250, rag_threshold=32, is_rag_enabled=False):
        self._cluster_amt = cluster_amt
        self._maximum_file_size = maximum_file_size
        self._rag_threshold = rag_threshold
        self._is_rag_enabled = is_rag_enabled

    def cluster_amt(self):
        return self._cluster_amt

    def maximum_file_size(self):
        return self._maximum_file_size

    def rag_threshold(self):
        return self._rag_threshold

    def is_rag_enabled(self):
        return self._is_rag_enabled

    def image_path(self):
        return self._image_path

    def set_image_path(self, image_path):
        self._image_path = image_path
