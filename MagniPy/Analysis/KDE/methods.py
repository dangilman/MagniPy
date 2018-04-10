import numpy as np

fastFFT = np.array(
    [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 144, 160, 192, 256, 288, 320, 384, 432, 480,
     512, 576, 640, 720, 768, 864, 960, 1024, 1152, 1280, 1440, 1536, 1728, 1920, 2048, 2304, 2560, 2880, 3072,
     3456, 3840, 4096, 4608, 5120, 5760, 6144, 6912, 7680, 8192, 9216, 10240, 11520, 12288, 13824, 15360, 16384,
     18432, 20480, 23040, 24576, 27648, 30720, 32768, 36864, 40960, 46080, 49152, 55296, 61440, 65536, 73728, 81920,
     92160, 98304, 110592, 122880, 131072, 147456, 163840, 184320, 196608, 221184, 245760, 262144, 294912, 327680,
     368640, 393216, 442368, 491520, 524288, 589824, 655360, 737280, 786432, 884736, 983040, 1048576, 1179648,
     1310720, 1474560, 1572864, 1769472, 1966080, 2097152, 2359296, 2621440, 2949120, 3145728, 3538944, 3932160,
     4194304, 4718592, 5242880, 5898240, 6291456, 7077888, 7864320, 8388608, 9437184, 10485760, 11796480, 12582912,
     14155776, 15728640, 16777216, 18874368, 20971520, 23592960, 25165824, 28311552, 31457280, 33554432, 37748736,
     41943040, 47185920, 50331648, 56623104, 62914560, 67108864, 75497472, 83886080, 94371840, 100663296, 113246208,
     125829120, 134217728, 150994944, 167772160, 188743680, 201326592, 226492416, 234881024, 251658240, 268435456,
     301989888, 335544320, 377487360, 402653184, 452984832, 503316480, 536870912, 603979776, 671088640, 754974720,
     805306368, 905969664, 1006632960, 1207959552, 1342177280, 1358954496, 1509949440, 1610612736, 1811939328,
     2013265920], dtype=np.int)


def nearestFFTnumber(x):
    return np.maximum(x, fastFFT[np.searchsorted(fastFFT, x)])

def convolveFFTn(in1, in2, mode="same", largest_size=0, cache=None, yfft=None, xfft=None, cache_args=[1, 2]):
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    size = s1 + s2 - 1
    fsize = nearestFFTnumber(np.maximum(largest_size, size))
    if cache is not None:
        if xfft is None and 1 in cache_args:
            key = (tuple(fsize), tuple(in1.shape), id(in1))
            xfft = cache.get(key)
        if yfft is None and 2 in cache_args:
            key2 = (tuple(fsize), tuple(in2.shape), id(in2))
            yfft = cache.get(key2)
    if xfft is None:
        xfft = np.fft.rfftn(in1, fsize)
        if cache is not None and 1 in cache_args: cache[key] = xfft
    if yfft is None:
        yfft = np.fft.rfftn(in2, fsize)
        if cache is not None and 2 in cache_args: cache[key2] = yfft

    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = np.fft.irfftn(xfft * yfft, fsize)[fslice]

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    startind = (np.array(arr.shape) - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
