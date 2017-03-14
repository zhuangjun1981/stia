import numpy as np
import matplotlib.pyplot as plt
import basic as bas


def generate_filter(length, fs, f_low=None, f_high=None, mode='box', is_plot=False):
    """
    generate one dimensional filter on Fourier domain, with symmetrical structure

    :param length: length of filter, int
    :param fs: sampling frequency
    :param f_low: low cutoff frequency, None: low-pass
    :param f_high: high cutoff frequency, None: high-pass
    :param mode: filter mode, '1/f' or 'box'
    :return: a array describing the filter
    """

    freqs = np.fft.fftfreq(int(length), d=(1. / float(fs)))

    filter_array = np.ones(length)

    if f_low is None and f_high is None:
        print('no filtering required!')
    elif f_low is None and f_high is not None:
        print('low-pass fileter')
        if f_high <= 0:
            raise(ValueError, 'Higher cutoff frquency should be positive!')
        filter_array[freqs >= f_high] = 0.
        filter_array[freqs <= -f_high] = 0.
    elif f_low is not None and f_high is None:
        print('high-pass fileter')
        if f_low < 0:
            raise (ValueError, 'Lower cutoff frquency should be non-negative!')
        filter_array[np.logical_and((freqs >= -f_low), (freqs <= f_low))] = 0.
    else:
        print('band-pass filter')
        if f_high <= 0:
            raise (ValueError, 'Higher cutoff frquency should be positive!')
        if f_low < 0:
            raise (ValueError, 'Lower cutoff frquency should be non-negative!')
        filter_array[freqs >= f_high] = 0.
        filter_array[freqs <= -f_high] = 0.
        filter_array[np.logical_and((freqs >= -f_low), (freqs <= f_low))] = 0.

    if mode == '1/f':
        filter_array[1:] = filter_array[1:] / abs(freqs[1:])
        filter_array[0] = 0
        filter_array = bas.array_nor(filter_array)
    elif mode == 'box':
        filter_array[0] = 0
    else:
        raise(NameError, 'Variable "mode" should be either "1/f" or "box"!')

    if is_plot:
        plot_array = zip(freqs, filter_array)
        plot_array.sort(key=lambda x: x[0])
        plot_array = zip(*plot_array)

        _ = plt.figure(figsize=(10, 3))
        plt.plot(plot_array[0], plot_array[1])
        plt.xlabel('frequency (Hz)')
        plt.ylim([-0.1, 1.1])
        plt.show()

    return freqs, filter_array


if __name__ == "__main__":

    # -------------------------------------------------------------------------------------
    fil = generate_filter(100, 10., f_low=0, f_high=4., mode='box', is_plot=True)
    # -------------------------------------------------------------------------------------
