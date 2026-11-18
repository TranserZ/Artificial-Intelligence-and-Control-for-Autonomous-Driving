"""torch 없이 torch.save(state_dict) 파일을 numpy로 읽는다."""
import pickle, zipfile, collections
import numpy as np
DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64,
      "IntStorage": np.int32, "BoolStorage": np.bool_, "HalfStorage": np.float16}
def load(path):
    z = zipfile.ZipFile(path); pre = z.namelist()[0].split("/")[0]
    class _S:
        def __init__(self, n): self.n = n
    def rebuild(storage, offset, size, stride, *a):
        arr, = storage
        if not size: return np.array(arr[offset])
        return np.array(np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                        strides=tuple(x * arr.itemsize for x in stride)))
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2": return rebuild
            if name == "_rebuild_parameter": return lambda t, *a: t
            if name == "OrderedDict": return collections.OrderedDict
            if mod.startswith("torch") and name.endswith("Storage"): return _S(name)
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            _, st, key, loc, numel = pid
            return (np.frombuffer(z.read(f"{pre}/data/{key}"), dtype=DT[st.n]),)
    return U(z.open(f"{pre}/data.pkl")).load()
