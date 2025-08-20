import h5py, numpy as np, torch
from torch.utils.data import Dataset

class H5LazyDataset(Dataset):
    def __init__(self, path, feature_paths, label_path, dtype=torch.float32):
        self.path = path
        self.feature_paths = feature_paths
        self.label_path = label_path
        with h5py.File(path, "r") as f:
            self.n = f[label_path].shape[0]
            for p in feature_paths:
                assert f[p].shape[0] == self.n
        self._fh = None

    def _open(self):
        if self._fh is None:
            self._fh = h5py.File(self.path, "r")
            self._feat = {p: self._fh[p] for p in self.feature_paths}
            self._label = self._fh[self.label_path]

    def __len__(self): return self.n

    def __getitem__(self, idx):
        self._open()
        xs = [torch.from_numpy(np.asarray(self._feat[p][idx])).float().reshape(-1)
              for p in self.feature_paths]
        x = torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]
        y = torch.tensor(self._label[idx]).float()
        return x, y

