import numpy as np

def DictArray(shapes_dict, axis=0, dtype=np.float32):
    keys = sorted(list(shapes_dict.keys())) 
    shapes = np.array([shapes_dict[k] for k in keys])
    sizes = [np.prod(shape) for shape in shapes]
    total_size = np.sum(sizes)

    key_to_slice = {}
    key_to_shape = {}
    key_to_size = {}
    prev_idx = 0
    for i, k in enumerate(keys):
        size = sizes[i]
        key_to_slice[k] = slice(prev_idx, prev_idx+size)
        key_to_shape[k] = shapes[i]
        key_to_size[k] = size
        prev_idx += size
        

    class Template(np.ndarray):
        def __getitem__(self, k):
            if isinstance(k, str):
                slice_ = key_to_slice[k]
                shape = key_to_shape[k]
                return np.reshape(self[slice_], shape)
            else:
                return super(Template, self).__getitem__(k)

    def template(**kwargs):
        arr = np.zeros(total_size, dtype=dtype)
        for k in kwargs:
            slice_ = key_to_slice[k]
            arr[slice_] = kwargs[k].reshape(key_to_size[k])
        return arr.view(Template)

    return template


if __name__ == "__main__":
    TestArr = DictArray(shapes_dict={'a': (5,3), 'b': (4)}, axis=1)
    a1 = TestArr(a=np.random.randn(5,3), b=np.ones(4))
    a2 = TestArr(a=np.random.randn(5,3), b=np.ones(4))

    print(a1, type(a1))
    print(a1['a'])
    print(a1['b'])
    print((a1+a2)['a'])
