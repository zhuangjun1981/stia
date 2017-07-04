import os
import numpy as np
import scipy.ndimage as ni
import matplotlib.pyplot as plt
import random
import tifffile as tf
import h5py

import stia.utility.basic
import utility.image_analysis as ia
import utility.basic as bas
import utility.signal_processing as sp

DTYPE = np.uint16
FRAME_SHAPE = (256, 256) # pixel
PIXEL_SIZE = (0.5, 0.5) # micron
TEMPORAL_SAMPLING_RATE = 31.  # Hz
DURATION = 100. # second


def get_random_number(distribution, shape):
    """
    get a random number from given distribution
    :param distribution: tuple in the format, (distribution type, parameter1, parameter2, ...)
                        supported: ('flat', mean, range)
                                   ('gaussian, mean, sigma)
                                   ('exponential', mean)
    :param shape: output shape
    :return: a random number
    """
    if distribution is None:
        output = np.zeros(shape, dtype=np.float64)
    elif distribution[0] == 'flat':
        output = np.random.rand(*shape) * float(distribution[2]) - 0.5 * distribution[2] + distribution[1]
    elif distribution[0] == 'gaussian':
        output = np.random.randn(*shape) * float(distribution[2]) + float(distribution[1])
    elif distribution[0] == 'exponential':
        if distribution[1] <= 0:
            raise(ValueError, 'The mean of the exponential distribution should be larger than 0!')
        output = np.random.exponential(float(distribution[1]), shape)
    else:
        raise (LookupError, 'the first element of "noise" should be "gaussian", "flat" or "exponential"!')

    return output


class ArtificialDataset(object):

    def __init__(self,
                 dtype=DTYPE,
                 frame_shape=FRAME_SHAPE,
                 pixel_size=PIXEL_SIZE,
                 duration=DURATION,
                 fs=TEMPORAL_SAMPLING_RATE,
                 motion=None
                 ):
        """
        :param dtype: data type, usually np.uint16
        :param frame_shape: (height, width)
        :param duration: total duration (sec)
        :param fs: sampling rate (Hz)
        :param pixel_size: microns, (height width)
        :param motion: should be MotionTrace object
        """
        self._dtype = dtype
        self._frame_shape = tuple([int(p) for p in frame_shape])
        self._duration = float(duration)
        self._fs = float(fs)
        self._pixel_size = tuple([float(m) for m in pixel_size])
        self._independent_components = {}
        self._interactive_components = {}
        if motion is None:
            self._motion = MotionTrace(amplitude=0.0)
        else:
            self._motion = motion

    @property
    def _total_frame_num(self):
        """
        total number of points of the whole simulated trace
        """
        return int(round(self._duration * self._fs))

    def set_motion(self, motion_obj):
        self._motion = motion_obj

    # @property
    # def _motion_amplitude(self):
    #     if self._motion is None:
    #         return 0.0
    #     else:
    #         return self._motion.get_amplitude()
    #
    # @property
    # def _motion_max_height_pix(self):
    #     return self._motion_amplitude / self._pixel_size[0]
    #
    # @property
    # def _motion_max_width_pix(self):
    #     return self._motion_amplitude / self._pixel_size[1]

    def _get_motion_traces(self):
        """
        get motion in x and y
        :return: a list of tuple, each tuple is x and y translation of motion
        """

        motion_height = TemporalFrequencyTrace(fs=TEMPORAL_SAMPLING_RATE, duration=self._duration,
                                               peak_dff=self._motion_max_height_pix * 2, noise=None,
                                               cutoff_freq_low=self._motion_cutoff_freq_low,
                                               cutoff_freq_high=self._motion_cutoff_freq_high, filter_mode='box')

        motion_height_trace = (motion_height.generate_trace() - self._motion_max_height_pix).astype(np.int)

        motion_width = TemporalFrequencyTrace(fs=TEMPORAL_SAMPLING_RATE, duration=self._duration,
                                              peak_dff=self._motion_max_width_pix * 2, noise=None,
                                              cutoff_freq_low=self._motion_cutoff_freq_low,
                                              cutoff_freq_high=self._motion_cutoff_freq_high, filter_mode='box')

        motion_width_trace = (motion_width.generate_trace() - self._motion_max_width_pix).astype(np.int)

        print motion_width_trace.shape

        return zip(motion_height_trace, motion_width_trace)

    def get_frame_shape(self):
        return self._frame_shape

    def get_pixel_size(self):
        return self._pixel_size

    def get_duration(self):
        return self._duration

    def get_fs(self):
        return self._fs

    def add_independent_component(self, name, spatial_component, temporal_component):
        spatial_component.set_frame_shape([self._frame_shape[0] + 2 * int(self._motion.get_max_height_pix()),
                                           self._frame_shape[1] + 2 * int(self._motion.get_max_width_pix())])
        self._independent_components.update({name: {'spatial': spatial_component,
                                                    'temporal': temporal_component}})

    def add_interactive_component(self, name, component):
        component.set_frame_shape([self._frame_shape[0] + 2 * int(self._motion.get_max_height_pix()),
                                   self._frame_shape[1] + 2 * int(self._motion.get_max_width_pix())])
        self._interactive_components.update({name: component})

    @staticmethod
    def _generate_mov_from_frame_and_trace(frame, trace, is_plot=False):
        """
        generate a movie given a static frame and static trace, assuming there is no interactions between spatial and
        temporal components
        """

        height = frame.shape[0]
        width = frame.shape[0]
        length = len(trace)

        df_trace = np.array(trace) + 1.

        mov = np.dot(df_trace.reshape((length, 1)),
                     frame.reshape((1, height * width))).reshape((length, height, width))

        # mov = np.zeros((len(trace), frame.shape[0], frame.shape[1]))
        # for i in range(mov.shape[0]):
        #     mov[i] = frame * (1. + trace[i])

        if is_plot:
            tf.imshow(mov, cmap='gray')
            plt.show()

        return mov

    def to_h5(self, data_path, is_plot=False):

        if os.path.isfile(data_path) or os.path.isdir(data_path):
            err_str = 'The defined path already exists, please set a new path.\n'+str(data_path)
            raise LookupError(err_str)

        dfile = h5py.File(data_path)

        large_mov = np.empty((self._total_frame_num,
                              self._frame_shape[0] + 2 * int(self._motion.get_max_height_pix()),
                              self._frame_shape[1] + 2 * int(self._motion.get_max_width_pix())))

        for name, component in self._independent_components.iteritems():
            spatial_component = component['spatial']
            temporal_component = component['temporal']

            # if not np.array_equal(spatial_component.get_frame_shape(), self._frame_shape):
            #     raise(ValueError, 'component ' + name + ': frame shape inconsistent with dataset frame shape!')
            #
            # if not np.array_equal(spatial_component.get_pixel_size(), self._pixel_size):
            #     raise (ValueError, 'component ' + name + ': pixel size inconsistent with dataset pixel size!')

            if temporal_component.get_duration() != self._duration:
                raise (ValueError, 'component ' + name + ': duration inconsistent with dataset duration!')

            if temporal_component.get_fs() != self._fs:
                raise (ValueError, 'component ' + name + ': fs inconsistent with dataset fs!')

            curr_group = dfile.create_group(name)
            curr_spatial_group = curr_group.create_group('spatial')
            frame = spatial_component.to_h5(curr_spatial_group)
            curr_temporal_group = curr_group.create_group('temporal')
            trace = temporal_component.to_h5(curr_temporal_group)

            large_mov += self._generate_mov_from_frame_and_trace(frame, trace)

        for name, component in self._interactive_components.iteritems():

            # if not np.array_equal(component.get_frame_shape(), self._frame_shape):
            #     raise (ValueError, 'component ' + name + ': frame shape inconsistent with dataset frame shape!')
            #
            # if not np.array_equal(component.get_pixel_size(), self._pixel_size):
            #     raise (ValueError, 'component ' + name + ': pixel size inconsistent with dataset pixel size!')

            if component.get_duration() != self._duration:
                raise (ValueError, 'component ' + name + ': duration inconsistent with dataset duration!')

            if component.get_fs() != self._fs:
                raise (ValueError, 'component ' + name + ': fs inconsistent with dataset fs!')

            curr_group = dfile.create_group(name)
            large_mov += component.to_h5(curr_group)

        mov = np.zeros((self._total_frame_num, self._frame_shape[0], self._frame_shape[1]))

        motion_group = dfile.create_group('motion')
        motion_trace = self._motion.to_h5(motion_group)
        print '\n'.join([str(p) for p in motion_trace])

        for i, curr_motion in enumerate(motion_trace):
            curr_x = self._motion.get_max_width_pix() + curr_motion[0]
            curr_y = self._motion.get_max_height_pix() + curr_motion[1]
            mov[i, :, :] = large_mov[i, :, :][curr_y:curr_y + self._frame_shape[0],
                                              curr_x:curr_x + self._frame_shape[1]]

        if np.issubdtype(self._dtype, np.integer):
            mov.clip(min=np.iinfo(self._dtype).min, max=np.iinfo(self._dtype).max)
        elif np.issubdtype(self._dtype, np.float):
            mov.clip(min=np.finfo(self._dtype).min, max=np.finfo(self._dtype).max)
        else:
            raise (ValueError, 'data type should be integer of float!')

        mov = mov.astype(self._dtype)

        dfile.create_dataset('final_movie', data=mov, compression='lzf')

        dfile.close()

        if is_plot:
            tf.imshow(mov, cmap='gray')
            plt.show()

        return mov


class SpatialComponent(object):
    """
    2-dimensional spatial component, the generate_mask() method will return a 2D matrix with mean pixel value equals
    the baseline.
    When implemented into ArtificialDataset all pixels in this matrix should change proportionally over time.
    """

    def __init__(self,
                 frame_shape=FRAME_SHAPE,
                 pixel_size=PIXEL_SIZE,
                 noise=('gaussian', 0, 0.2),
                 baseline=1000.
                 ):
        """
        :param frame_shape: (height, width)
        :param pixel_size: microns, (height width)
        :param noise: noise distribution ('gaussian',"mean","sigma") or ('flat',"mean","amplitude")
        :param baseline: baseline fluorscence, artificial unit (usually count of analog to digital range)
        """
        self._frame_shape = [int(p) for p in frame_shape]
        self._pixel_size = [float(m) for m in pixel_size]
        self._noise = noise
        self._baseline = float(baseline)

    @property
    def _pixel_area(self):
        """
        :return: pixel area in square micirons
        """
        return self._pixel_size[0] * self._pixel_size[1]

    def set_frame_shape(self, frame_shape):
        self._frame_shape = frame_shape

    def get_frame_shape(self):
        return self._frame_shape

    def get_pixel_size(self):
        return self._pixel_size

    def generate_mask(self):
        """
        generate two dimensional mask
        """
        print "This is a method place holder, should be over writen by subclasses."

    def to_h5(self, h5_group):
        """
        place holder for a method generating a hdf5 group for saving
        """

        h5_group.attrs['frame_shape_height_pixel'] = self._frame_shape[0]
        h5_group.attrs['frame_shape_width_pixel'] = self._frame_shape[1]
        h5_group.attrs['pixel_size_height_micron'] = self._pixel_size[0]
        h5_group.attrs['pixel_size_width_micron'] = self._pixel_size[1]
        h5_group.attrs['noise_type'] = self._noise[0]
        h5_group.attrs['noise_mean_normalized'] = self._noise[1]
        h5_group.attrs['noise_err_normalized'] = self._noise[2]
        h5_group.attrs['baseline'] = self._baseline


class FilledSoma(SpatialComponent):
    """
    class to generate simulated mask for filled neurons, final mask has the mean intensity of all pixels in ROI equals
    baseline, inherited from SpatialComponent Class
    """

    def __init__(self,
                 center,
                 approx_area=50.,
                 filter_size=1.,
                 threshold=0.1,
                 **kwargs):
        """
        :param center: tuple (y, x)
        :param approx_area: approximate area of the neuron cell body, square microns
        :param filter_size: 2-d gaussian filter size, micron
        :param threshold: threshold to generate binary mask, [0, 1]
        """

        super(FilledSoma, self).__init__(**kwargs)

        if center[0] >= self._frame_shape[0] or center[1] >= self._frame_shape[1]:
            raise(ValueError, 'center should be within the field of view.')

        self._center = center
        self._approx_area = float(approx_area)
        self._filter_size = float(filter_size)
        self._threshold = threshold

    def __str__(self):
        return 'A spatial component of a filled soma. ' + \
               'FilledSoma class inherited from stia.artificial_dataset.SpatialComponent class.'

    @property
    def _pixel_num(self):
        return bas.round_int(self._approx_area / self._pixel_area)

    @property
    def _filter_pixel_size(self):
        return self._filter_size / np.mean(self._pixel_size, axis=0)

    def get_center(self):
        return self._center

    def get_approx_area(self):
        return self._approx_area

    def get_filter_size(self):
        return self._filter_size

    def get_threshold(self):
        return self._threshold

    def generate_mask(self, is_plot=False):
        binary = ia.random_filled_shape(self._frame_shape, self._center, self._pixel_num)
        weighted = ni.filters.gaussian_filter(binary.astype(np.float), self._filter_pixel_size).astype(np.float)
        weighted = stia.utility.basic.array_nor(weighted)

        mask = weighted > self._threshold

        noise = get_random_number(self._noise, self._frame_shape)
        weighted = stia.utility.basic.array_nor(weighted) + noise
        weighted[weighted < 0] = 0
        weighted = stia.utility.basic.array_nor(np.multiply(weighted, mask))

        pixel_mean = ia.Mask(weighted).get_mean()
        weighted = self._baseline * weighted / pixel_mean

        if is_plot:
            _ = plt.figure()
            plt.imshow(weighted, cmap='gray', interpolation='nearest')
            plt.colorbar()
            plt.show()

        return weighted

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """
        
        super(FilledSoma, self).to_h5(h5_group)

        h5_group.attrs['type'] = self.__str__()
        h5_group.attrs['center_height_pixel'] = self._center[0]
        h5_group.attrs['center_width_pixel'] = self._center[1]
        h5_group.attrs['approx_area_square_micron'] = self._approx_area
        h5_group.attrs['filter_size_micron'] = self._filter_size
        h5_group.attrs['threshold_normalized'] = self._threshold

        mask = self.generate_mask()

        mask_group = h5_group.create_group('mask')
        ia.Mask(mask).to_h5(mask_group)

        return mask


class DoughnutSoma(SpatialComponent):
    """
    class to generate simulated mask for doughnut-shaped neurons, final mask has the amplitude of [0., 1.]
    inherited from SpatialComponent Class
    """

    def __init__(self,
                 center,
                 first_area=50.,
                 second_area=50.,
                 filter_size=1.,
                 threshold=0.2,
                 **kwargs):
        """

        :param center: tuple (y, x)
        :param first_area: approximate area for the first component, square microns
        :param second_area: approximate area for the second component, square microns
        :param filter_size: 2-d gaussian filter size, micron
        :param threshold: threshold to generate binary mask, [0, 1]
        """

        super(DoughnutSoma, self).__init__(**kwargs)

        if center[0] >= self._frame_shape[0] or center[1] >= self._frame_shape[1]:
            raise (ValueError, 'center should be within the field of view.')

        self._center = center
        self._first_area = first_area
        self._second_area = second_area
        self._filter_size = filter_size
        self._threshold = threshold

    def __str__(self):
        return 'A spatial component of a doughnut soma. ' + \
               'DoughnutSoma class inherited from stia.artificial_dataset.SpatialComponent class.'

    @property
    def _first_pixel_num(self):
        return bas.round_int(self._first_area / self._pixel_area)

    @property
    def _second_pixel_num(self):
        return bas.round_int(self._second_area / self._pixel_area)

    @property
    def _filter_pixel_size(self):
        return self._filter_size / np.mean(self._pixel_size, axis=0)

    def get_center(self):
        return self._center

    def get_first_area(self):
        return self._first_area

    def get_second_area(self):
        return self._second_area

    def get_filter_size(self):
        return self._filter_size

    def get_threshold(self):
        return self._threshold

    def generate_mask(self, is_plot=False):
        large_binary = ia.random_filled_shape(self._frame_shape, self._center, self._first_pixel_num).astype(np.int)
        small_binary = ia.random_filled_shape(self._frame_shape, self._center, self._second_pixel_num).astype(np.int)
        binary = np.abs(large_binary - small_binary).astype(np.bool)

        # plt.imshow(large_binary - small_binary, cmap='gray', interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        weighted = ni.filters.gaussian_filter(binary.astype(np.float), self._filter_pixel_size).astype(np.float)
        weighted = stia.utility.basic.array_nor(weighted)
        mask = weighted > self._threshold
        noise = get_random_number(self._noise, self._frame_shape)

        weighted = stia.utility.basic.array_nor(weighted) + noise
        weighted[weighted < 0] = 0
        weighted = stia.utility.basic.array_nor(np.multiply(weighted, mask))

        pixel_mean = ia.Mask(weighted).get_mean()
        weighted = self._baseline * weighted / pixel_mean

        if is_plot:
            _ = plt.figure()
            plt.imshow(weighted, cmap='gray', interpolation='nearest')
            plt.colorbar()
            plt.show()

        return weighted

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """

        super(DoughnutSoma, self).to_h5(h5_group)

        h5_group.attrs['type'] = self.__str__()
        h5_group.attrs['center_height_pixel'] = self._center[0]
        h5_group.attrs['center_width_pixel'] = self._center[1]
        h5_group.attrs['first_area_square_micron'] = self._first_area
        h5_group.attrs['second_area_square_micron'] = self._second_area
        h5_group.attrs['filter_size_micron'] = self._filter_size
        h5_group.attrs['threshold_normalized'] = self._threshold

        mask = self.generate_mask()

        mask_group = h5_group.create_group('mask')
        ia.Mask(mask).to_h5(mask_group)

        return mask


class Process(SpatialComponent):
    """
    class to generate the morphology masks of neural processes
    inherited from SpatialComponent Class
    """

    def __init__(self,
                 center,
                 approx_area=300.,
                 branch_distribution=([1, 2, 3], [0.8, 0.1, 0.1]),
                 distance_distribution=('gaussian', 10, 5),
                 angle_distribution=('flat', 0, 2 * np.pi),
                 filter_size=0.5,
                 threshold=0.2,
                 **kwargs
                 ):
        """
        :param center: tuple (y, x)
        :param approx_area: approximate total area of skeleton, square microns
        :param branch_distribution: distribution of number of branches from each node, tuple of two lists
                                    first list: number of branches, int
                                    second list: probability of each branch, float
        :param distance_distribution: distribution of distances of segments, (distribution type, mean, err)
                                      mean and err are in micorns.
                                      i.e. ('gaussian', 5, 0.1) or ('flat', 5, 0.1)
        :param angle_distribution: distribution of angles, ('flat', 0, 2*np.pi)
        :param filter_size: 2-d gaussian filter size, micron
        :param threshold: threshold to generate binary mask, [0, 1]
        """

        super(Process, self).__init__(**kwargs)

        if center[0] >= self._frame_shape[0] or center[1] >= self._frame_shape[1]:
            raise (ValueError, 'center should be within the field of view.')

        self._center = center
        self._approx_area = float(approx_area)
        self._branch_distribution = branch_distribution
        self._distance_distribution = distance_distribution
        self._angle_distribution = angle_distribution
        self._filter_size = filter_size
        self._threshold = threshold

    def __str__(self):
        return 'A spatial component of a process arbor from a single neuron. ' + \
               'Process class inherited from stia.artificial_dataset.SpatialComponent class.'

    @property
    def _approx_pixel_area(self):
        return bas.round_int(self._approx_area / self._pixel_area)

    @property
    def _filter_pixel_size(self):
        return self._filter_size / np.mean(self._pixel_size, axis=0)

    def _get_angle(self):
        return get_random_number(self._angle_distribution, (1,))[0]

    def _get_pixel_distance(self):
        """
        :return: a distance following the defined distribution in pixels
        """

        distance = 0
        mean_distance_pixel = self._distance_distribution[1] / np.mean(self._pixel_size)
        var_distance_pixel = self._distance_distribution[2] / np.mean(self._pixel_size)
        distance_pixel_distribution = (self._distance_distribution[0], mean_distance_pixel, var_distance_pixel)

        while distance < np.sqrt(2):
            distance = get_random_number(distance_pixel_distribution, (1,))[0]

        return distance

    def _get_branch_number(self):
        return np.random.choice(self._branch_distribution[0], 1, p=self._branch_distribution[1])[0]

    def get_approx_area(self):
        return self._approx_area

    def get_branch_distribution(self):
        return self._branch_distribution

    def get_distance_distribution(self):
        return self._distance_distribution

    def get_angle_distribution(self):
        return self._angle_distribution

    def get_filter_size(self):
        return self._filter_size

    def get_threshold(self):
        return self._threshold

    # def _get_decay_envelope(self):
    #     """
    #     square gaussian kernel.
    #     """
    #
    #     col, row = np.meshgrid(np.arange(self._frame_shape[0]), np.arange(self._frame_shape[1]))
    #     row_c = self._center[0]
    #     col_c = self._center[1]
    #
    #     envelope =  np.exp(-4 * np.log(2) * ((row - row_c) ** 2 + (col - col_c) ** 2) / self._decay_sigma ** 2)
    #
    #     return ia.array_nor(envelope)

    def generate_mask(self, is_plot=False):

        total_area = 0
        nodes = [self._center]
        skeleton = np.zeros(self._frame_shape)

        while nodes and total_area <= self._approx_pixel_area:

            curr_node = random.choice(nodes)
            curr_branch_number = self._get_branch_number()

            # print 'all nodes:', nodes

            for i in range(curr_branch_number):

                curr_angle = self._get_angle()
                curr_distance = self._get_pixel_distance()
                branch_pixels = ia.binary_ray(curr_node, curr_angle, curr_distance, self._frame_shape)

                while len(branch_pixels[0]) == 1:
                    curr_angle = self._get_angle()
                    curr_distance = self._get_pixel_distance()
                    branch_pixels = ia.binary_ray(curr_node, curr_angle, curr_distance, self._frame_shape)

                # print('curr_node:', curr_node, ', curr_distance:', curr_distance, ', curr_angle:',
                #       curr_angle * 180 / np.pi)

                # print branch_pixels
                branch = np.zeros(self._frame_shape)
                branch[branch_pixels] = 1
                skeleton += branch
                # if is_plot:
                #     f = plt.figure(figsize=(13, 10))
                #     plt.imshow(ia.array_nor(mask), interpolation='nearest')
                #     plt.colorbar()
                #     plt.show()
                nodes.append((branch_pixels[0][-1], branch_pixels[1][-1]))

            nodes.remove(curr_node)
            total_area = np.sum(skeleton[skeleton > 0])

        skeleton = stia.utility.basic.array_nor(skeleton)
        weighted = stia.utility.basic.array_nor(ni.filters.gaussian_filter(skeleton.astype(np.float),
                                                                           self._filter_pixel_size))
        mask = weighted > self._threshold

        noise = get_random_number(self._noise, self._frame_shape)

        weighted = stia.utility.basic.array_nor(weighted) + noise
        weighted[weighted < 0] = 0
        weighted = stia.utility.basic.array_nor(np.multiply(weighted, mask))

        pixel_mean = ia.Mask(weighted).get_mean()
        weighted = self._baseline * weighted / pixel_mean

        if is_plot:
            _ = plt.figure(figsize=(13, 10))
            plt.imshow(weighted, cmap='gray', interpolation='nearest')
            plt.colorbar()
            plt.show()

        return weighted

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """

        super(Process, self).to_h5(h5_group)

        h5_group.attrs['type'] = self.__str__()
        h5_group.attrs['center_height_pixel'] = self._center[0]
        h5_group.attrs['center_width_pixel'] = self._center[1]
        h5_group.attrs['approx_area_square_micron'] = self._approx_area
        h5_group.attrs['branch_distribution_samples'] = self._branch_distribution[0]
        h5_group.attrs['branch_distribution_frequencies'] = self._branch_distribution[1]
        h5_group.attrs['distance_distribution_type'] = self._distance_distribution[0]
        h5_group.attrs['distance_distribution_mean_micron'] = self._distance_distribution[1]
        h5_group.attrs['distance_distribution_err_micron'] = self._distance_distribution[2]
        h5_group.attrs['angle_distribution_type'] = self._angle_distribution[0]
        h5_group.attrs['angle_distribution_mean_arc'] = self._angle_distribution[1]
        h5_group.attrs['angle_distribution_err_arc'] = self._angle_distribution[2]
        h5_group.attrs['filter_size_micron'] = self._filter_size
        h5_group.attrs['threshold_normalized'] = self._threshold

        mask = self.generate_mask()

        mask_group = h5_group.create_group('mask')
        ia.Mask(mask).to_h5(mask_group)

        return mask


class FilledSomaWithProcess(SpatialComponent):
    """
    class to generate the morphology masks of a filled soma with its processes
    inherited from SpatialComponent Class
    """

    def __init__(self,
                 center,
                 soma_approx_area=50.,
                 soma_filter_size=1.,
                 soma_threshold=0.1,
                 process_approx_area=300.,
                 process_branch_distribution=([1, 2, 3], [0.8, 0.1, 0.1]),
                 process_distance_distribution=('gaussian', 10, 5),
                 process_angle_distribution=('flat', 0, 2 * np.pi),
                 process_filter_size=0.5,
                 process_threshold=0.2,
                 **kwargs
                 ):
        """
        :param center: tuple (y, x)
        :param soma_approx_area: approximate area for the soma, square microns
        :param soma_filter_size: 2-d gaussian filter size for the soma, micron
        :param soma_threshold: threshold to generate binary mask for the soma, [0, 1]
        :param process_approx_area: approximate total area of process skeleton, square microns
        :param process_branch_distribution: distribution of number of branches from each node, tuple of two lists
                                            first list: number of branches, int
                                            second list: probability of each branch, float
        :param process_distance_distribution: distribution of distances of segments, (distribution type, mean, err)
                                              mean and err are in micorns.
                                              i.e. ('gaussian', 5, 0.1) or ('flat', 5, 0.1)
        :param process_angle_distribution: distribution of angles, ('flat', 0, 2*np.pi)
        :param process_filter_size: 2-d gaussian filter size for the process, micron
        :param process_threshold: threshold to generate binary mask for the process, [0, 1]
        :param **kwargs: inputs to the super class, SpatialComponent
        """

        super(FilledSomaWithProcess, self).__init__(**kwargs)

        if center[0] >= self._frame_shape[0] or center[1] >= self._frame_shape[1]:
            raise (ValueError, 'center should be within the field of view.')

        self._soma = FilledSoma(center, approx_area=soma_approx_area, filter_size=soma_filter_size,
                                threshold=soma_threshold, **kwargs)

        self._process = Process(center, approx_area=process_approx_area,
                                branch_distribution=process_branch_distribution,
                                distance_distribution=process_distance_distribution,
                                angle_distribution=process_angle_distribution, filter_size=process_filter_size,
                                threshold=process_threshold, **kwargs)

    def __str__(self):
        return 'A spatial component of a neuron with filled soma and a process arbor. ' + \
               'Process class inherited from stia.artificial_dataset.SpatialComponent class.'

    def set_frame_shape(self, frame_shape):
        self._soma.set_frame_shape(frame_shape)
        self._process.set_frame_shape(frame_shape)

    def generate_mask(self, is_plot=False):
        soma_mask = self._soma.generate_mask()
        process_mask = self._process.generate_mask()
        # soma_mask_binary_not = np.logical_not(ia.Mask(soma_mask).get_binary_dense_mask())
        # process_mask = np.multiply(process_mask, soma_mask_binary_not.astype(np.float))
        weighted = soma_mask + process_mask
        pixel_mean = ia.Mask(weighted).get_mean()
        weighted = self._baseline * weighted / pixel_mean

        if is_plot:
            _ = plt.figure(figsize=(13, 10))
            plt.imshow(weighted, cmap='gray', interpolation='nearest')
            plt.colorbar()
            plt.show()

        return weighted

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """

        super(FilledSomaWithProcess, self).to_h5(h5_group)

        h5_group.attrs['type'] = self.__str__()
        h5_group.attrs['center_height_pixel'] = self._soma.get_center()[0]
        h5_group.attrs['center_width_pixel'] = self._soma.get_center()[1]
        h5_group.attrs['soma_approx_area_square_micron'] = self._soma.get_approx_area()
        h5_group.attrs['soma_filter_size_micron'] = self._soma.get_filter_size()
        h5_group.attrs['soma_threshold_normalized'] = self._soma.get_threshold()

        h5_group.attrs['process_approx_area_square_micron'] = self._process.get_approx_area()
        h5_group.attrs['process_branch_distribution_samples'] = self._process.get_branch_distribution()[0]
        h5_group.attrs['process_branch_distribution_frequencies'] = self._process.get_branch_distribution()[1]
        h5_group.attrs['process_distance_distribution_type'] = self._process.get_distance_distribution()[0]
        h5_group.attrs['process_distance_distribution_mean_micron'] = self._process.get_distance_distribution()[1]
        h5_group.attrs['process_distance_distribution_err_micron'] = self._process.get_distance_distribution()[2]
        h5_group.attrs['process_angle_distribution_type'] = self._process.get_angle_distribution()[0]
        h5_group.attrs['process_angle_distribution_mean_arc'] = self._process.get_angle_distribution()[1]
        h5_group.attrs['process_angle_distribution_err_arc'] = self._process.get_angle_distribution()[2]
        h5_group.attrs['process_filter_size_micron'] = self._process.get_filter_size()
        h5_group.attrs['process_threshold_normalized'] = self._process.get_threshold()

        mask = self.generate_mask()

        mask_group = h5_group.create_group('mask')
        ia.Mask(mask).to_h5(mask_group)

        return mask


class DoughnutSomaWithProcess(SpatialComponent):
    """
    class to generate the morphology masks of a doughnut soma with its processes
    inherited from SpatialComponent Class
    """

    def __init__(self,
                 center,
                 soma_first_area=50.,
                 soma_second_area=50.,
                 soma_filter_size=1.,
                 soma_threshold=0.2,
                 process_approx_area=300.,
                 process_branch_distribution=([1, 2, 3], [0.8, 0.1, 0.1]),
                 process_distance_distribution=('gaussian', 10, 5),
                 process_angle_distribution=('flat', 0, 2 * np.pi),
                 process_filter_size=0.5,
                 process_threshold=0.2,
                 **kwargs
                 ):
        """
        :param center: tuple (y, x)
        :param soma_first_area: approximate area for the first component, square microns
        :param soma_second_area: approximate area for the second component, square microns
        :param soma_filter_size: 2-d gaussian filter size for the soma, micron
        :param soma_threshold: threshold to generate binary mask for the soma, [0, 1]
        :param process_approx_area: approximate total area of process skeleton, square microns
        :param process_branch_distribution: distribution of number of branches from each node, tuple of two lists
                                            first list: number of branches, int
                                            second list: probability of each branch, float
        :param process_distance_distribution: distribution of distances of segments, (distribution type, mean, err)
                                              mean and err are in micorns.
                                              i.e. ('gaussian', 5, 0.1) or ('flat', 5, 0.1)
        :param process_angle_distribution: distribution of angles, ('flat', 0, 2*np.pi)
        :param process_filter_size: 2-d gaussian filter size for the process, micron
        :param process_threshold: threshold to generate binary mask for the process, [0, 1]
        :param **kwargs: inputs to the super class, SpatialComponent
        """

        super(DoughnutSomaWithProcess, self).__init__(**kwargs)

        if center[0] >= self._frame_shape[0] or center[1] >= self._frame_shape[1]:
            raise (ValueError, 'center should be within the field of view.')

        self._soma = DoughnutSoma(center, first_area=soma_first_area, second_area=soma_second_area,
                                  filter_size=soma_filter_size, threshold=soma_threshold, **kwargs)

        self._process = Process(center, approx_area=process_approx_area,
                                branch_distribution=process_branch_distribution,
                                distance_distribution=process_distance_distribution,
                                angle_distribution=process_angle_distribution, filter_size=process_filter_size,
                                threshold=process_threshold, **kwargs)

    def __str__(self):
        return 'A spatial component of a neuron with doughnut soma and a process arbor. ' + \
               'Process class inherited from stia.artificial_dataset.SpatialComponent class.'

    def generate_mask(self, is_plot=False):
        soma_mask = self._soma.generate_mask()
        process_mask = self._process.generate_mask()

        soma_mask_binaary = ia.Mask(soma_mask).get_binary_dense_mask()
        closing_steps = len(ia.Mask(soma_mask).get_data()) / 2
        soma_filled_mask_binary = ni.binary_closing(soma_mask_binaary, iterations=closing_steps)
        process_mask = np.multiply(process_mask, np.logical_not(soma_filled_mask_binary))

        weighted = soma_mask + process_mask
        pixel_mean = ia.Mask(weighted).get_mean()
        weighted = self._baseline * weighted / pixel_mean

        if is_plot:
            _ = plt.figure(figsize=(13, 10))
            plt.imshow(weighted, cmap='gray', interpolation='nearest')
            plt.colorbar()
            plt.show()

        return weighted

    def set_frame_shape(self, frame_shape):
        self._soma.set_frame_shape(frame_shape)
        self._process.set_frame_shape(frame_shape)

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """

        super(DoughnutSomaWithProcess, self).to_h5(h5_group)

        h5_group.attrs['type'] = self.__str__()
        h5_group.attrs['center_height_pixel'] = self._soma.get_center()[0]
        h5_group.attrs['center_width_pixel'] = self._soma.get_center()[1]
        h5_group.attrs['soma_first_area_square_micron'] = self._soma.get_first_area()
        h5_group.attrs['soma_second_area_square_micron'] = self._soma.get_second_area()
        h5_group.attrs['soma_filter_size_micron'] = self._soma.get_filter_size()
        h5_group.attrs['soma_threshold_normalized'] = self._soma.get_threshold()

        h5_group.attrs['process_approx_area_square_micron'] = self._process.get_approx_area()
        h5_group.attrs['process_branch_distribution_samples'] = self._process.get_branch_distribution()[0]
        h5_group.attrs['process_branch_distribution_frequencies'] = self._process.get_branch_distribution()[1]
        h5_group.attrs['process_distance_distribution_type'] = self._process.get_distance_distribution()[0]
        h5_group.attrs['process_distance_distribution_mean_micron'] = self._process.get_distance_distribution()[1]
        h5_group.attrs['process_distance_distribution_err_micron'] = self._process.get_distance_distribution()[2]
        h5_group.attrs['process_angle_distribution_type'] = self._process.get_angle_distribution()[0]
        h5_group.attrs['process_angle_distribution_mean_arc'] = self._process.get_angle_distribution()[1]
        h5_group.attrs['process_angle_distribution_err_arc'] = self._process.get_angle_distribution()[2]
        h5_group.attrs['process_filter_size_micron'] = self._process.get_filter_size()
        h5_group.attrs['process_threshold_normalized'] = self._process.get_threshold()

        mask = self.generate_mask()

        mask_group = h5_group.create_group('mask')
        ia.Mask(mask).to_h5(mask_group)

        return mask


class BackgroundSpatialFrequency(SpatialComponent):
    """
    A background frame defined by spatial frequency, all pixels should change proportionally over time.
    inherited from SpatialComponent Class
    """

    def __init__(self, cutoff_freq_low=None, cutoff_freq_high=0.02, **kwargs):
        """
        :param cutoff_freq_low, cycle per micron, if None, low-pass
        :param cutoff_freq_high, cycle per micron, if None, high-pass
        :param kwargs: other inputs to SpatialComponent super class
        """

        super(BackgroundSpatialFrequency, self).__init__(**kwargs)
        self._cutoff_freq_low = cutoff_freq_low
        self._cutoff_freq_high = cutoff_freq_high

    def __str__(self):
        return 'A spatial component of neuropil background confined within certain spatial frequency range. ' + \
               'BackgroundSpatialFrequency class inherited from stia.artificial_dataset.SpatialComponent class.'

    def generate_mask(self, is_plot=False):

        _, row_filter = sp.generate_filter(self._frame_shape[0], 1. / self._pixel_size[0],
                                           self._cutoff_freq_low, self._cutoff_freq_high, mode='1/f')

        _, col_filter = sp.generate_filter(self._frame_shape[1], 1. / self._pixel_size[1],
                                           self._cutoff_freq_low, self._cutoff_freq_high, mode='1/f')

        raw_frame = np.random.rand(len(row_filter), len(col_filter))
        raw_frame_fft = np.fft.fftn(raw_frame)

        filter_fft = np.dot(np.array([row_filter]).transpose(), np.array([col_filter]))
        # filter_x = np.repeat(np.array([col_filter]), len(row_filter), axis=0)
        # filter_y = np.repeat(np.transpose(np.array([row_filter])), len(col_filter), axis=1)
        # filter_fft = filter_x * filter_y

        frame_filtered_fft = raw_frame_fft * filter_fft

        frame_filtered = bas.array_nor(np.real(np.fft.ifftn(frame_filtered_fft)))

        noise = get_random_number(self._noise, self._frame_shape)

        weighted = bas.array_nor(frame_filtered + noise)

        weighted = self._baseline * weighted / np.mean(weighted)

        if is_plot:
            _ = plt.figure()
            plt.imshow(weighted, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
            plt.colorbar()
            plt.show()

        return weighted

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """

        super(BackgroundSpatialFrequency, self).to_h5(h5_group)

        h5_group.attrs['type'] = self.__str__()
        if self._cutoff_freq_low is None:
            h5_group.attrs['cutoff_freq_low_cycle_per_micron'] = np.nan
        else:
            h5_group.attrs['cutoff_freq_low_cycle_per_micron'] = self._cutoff_freq_low

        if self._cutoff_freq_high is None:
            h5_group.attrs['cutoff_freq_high_cycle_per_micron'] = np.nan
        else:
            h5_group.attrs['cutoff_freq_high_cycle_per_micron'] = self._cutoff_freq_high

        mask = self.generate_mask()
        mask_group = h5_group.create_group('mask')
        mask_group.attrs['shape'] = mask.shape
        mask_group.attrs['type'] = 'dense_mask'
        mask_group.create_dataset('mask', data=mask)

        return mask


class BackgroundGaussianFiltering(SpatialComponent):
    """
    A background frame defined by spatial frequency, all pixels should change proportionally over time.
    inherited from SpatialComponent Class
    """

    def __init__(self, sigma=(20., 20.), **kwargs):
        """
        :param sigma: sigma of 2d gaussian filter, micron
        :param kwargs: inputs to super class, SpatialComponent
        """
        super(BackgroundGaussianFiltering, self).__init__(**kwargs)

        self._sigma = sigma

    def __str__(self):
        return 'A spatial component of neuropil background filtered within certain guassian filter. ' + \
               'BackgroundGaussianFiltering class inherited from stia.artificial_dataset.SpatialComponent class.'

    @property
    def _pixel_sigma(self):
        """
        sigma in pixel value
        """
        return self._sigma[0] / self._pixel_size[0], self._sigma[1] / self._pixel_size[1]

    def generate_mask(self, is_plot=False):
        frame_raw = np.random.rand(*self._frame_shape)
        frame_filtered = ni.filters.gaussian_filter(frame_raw, self._sigma)
        noise = get_random_number(self._noise, self._frame_shape)
        weighted = bas.array_nor(bas.array_nor(frame_filtered) + noise)
        weighted = self._baseline * weighted / np.mean(weighted)

        if is_plot:
            _ = plt.figure()
            plt.imshow(weighted, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
            plt.colorbar()
            plt.show()

        return weighted

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """

        super(BackgroundGaussianFiltering, self).to_h5(h5_group)

        h5_group.attrs['type'] = self.__str__()
        h5_group.attrs['sigma_height_micron'] = self._sigma[0]
        h5_group.attrs['sigma_width_micron'] = self._sigma[1]

        mask = self.generate_mask()
        mask_group = h5_group.create_group('mask')
        mask_group.attrs['shape'] = mask.shape
        mask_group.attrs['type'] = 'dense_mask'
        mask_group.create_dataset('mask', data=mask)

        return mask


class Spine(SpatialComponent):
    # todo: add spine component
    pass


class Bouton(SpatialComponent):
    # todo: add bouton component
    pass


class TemporalComponent(object):
    """
    1-dimensional temporal component, the generate_trace() method will return a 1D array representing DF/F
    """

    def __init__(self,
                 fs=TEMPORAL_SAMPLING_RATE,
                 duration=DURATION,
                 amplitude=1.,
                 noise=('gaussian', 0., 0.01), ):
        """
        :param fs: temporal sampling rate, Hz
        :param duration: temporal duration, sec
        :param amplitude: maximum range of dF/F, float
        :param noise: noise distribution ('gaussian',"mean","sigma") or ('flat',"mean","amplitude")
        """

        self._fs = float(fs)
        self._duration = float(duration)
        self._amplitude = float(amplitude)
        self._noise = noise

    @property
    def _total_point_num(self):
        """
        total number of points of the whole simulated trace
        """
        return int(round(self._duration * self._fs))

    def get_duration(self):
        return self._duration

    def get_fs(self):
        return self._fs

    def generate_trace(self):
        """
        generate one dimensional trace
        """
        print "This is a method place holder, should be over writen by subclasses."

    def to_h5(self, h5_group):
        """
        place holder for a method generating a hdf5 group for saving
        """

        h5_group.attrs['temporal_sampling_rate_hz'] = self._fs
        h5_group.attrs['duration_sec'] = self._duration
        # h5_group.attrs['amplitude'] = self._amplitude
        h5_group.attrs['noise_type'] = self._noise[0]
        h5_group.attrs['noise_mean_dff'] = self._noise[1]
        h5_group.attrs['noise_err_dff'] = self._noise[2]
        h5_group.attrs['total_point_num'] = self._total_point_num


class PossionCalciumTrace(TemporalComponent):
    """
    Simulated 1D calcium trace with possion firing process and linear rising and exponential decaying calcium response
    to each single action potential
    """

    def __init__(self,
                 firing_rate=0.2,
                 refactor_dur=0.002,
                 delay=0.05,
                 rising_dur=0.2,
                 decaying_tau=2.,
                 **kwarg):
        """
        :param firing_rate: the firing rate of the neuron, Hz
        :param refactor_dur: refactory duration of the neuron, sec
        :param delay: the delay from action potential to the start of rising of calcium signal, sec
        :param rising_dur: rising duration, sec
        :param decaying_tau: tau (1/lambda) of the exponential decay of the calcium signal to each action potential
        :param kwarg: other inputs for super class, TemporalComponent
        """

        super(PossionCalciumTrace, self).__init__(**kwarg)
        self._firing_rate = float(firing_rate)
        self._refactor_dur = float(refactor_dur)
        self._delay = float(delay)
        self._rising_dur = float(rising_dur)
        self._decaying_tau = float(decaying_tau)

    def __str__(self):
        return 'A temporal component simulating calcium response to a possion spike train. ' + \
               'PossionCalciumTrace class inherited from stia.artificial_dataset.TemporalComponent class.'

    @property
    def _irf_delay_point_num(self):
        """
        number of data points in delay phase in the impulse response function
        """
        return int(round(self._delay * self._fs))

    @property
    def _irf_rising_point_num(self):
        """
        number of data points in rising phase in the impulse response function
        """
        return int(round(self._rising_dur * self._fs))

    @property
    def _irf_decaying_point_num(self):
        """
        number of data points in decaying phase in the impulse response function, calculated as six times of tau
        """
        return int(round(self._decaying_tau * 6 * self._fs))

    @property
    def snr(self):
        """
        signal to noise ratio, ((range of noise) / (2 * peak dF/F))
        """
        return self._noise[2] / (2. * self._amplitude)

    def generate_irf(self, is_plot=False):
        """
        generate the impulse response function (calcium response to a single action potential)
        """

        # delay phase
        irf = [0.] * self._irf_delay_point_num

        # rising phase
        rising = np.linspace(0, self._amplitude, self._irf_rising_point_num, endpoint=False)
        irf += list(rising)

        # decaying phase
        decaying_time = np.arange(self._irf_decaying_point_num) / self._fs
        decaying = self._amplitude * np.exp((-1 / self._decaying_tau) * decaying_time)

        irf += list(decaying)

        if is_plot:
            f = plt.figure(figsize=(6, 3))
            ax = f.add_axes([0.15, 0.2, 0.8, 0.7])
            ax.set_xlabel('time after AP onset (sec)')
            ax.set_ylabel('dF/F')
            ax.plot(np.arange(len(irf)) / self._fs, irf)
            plt.show()

        return np.array(irf, dtype=np.float)

    def generate_spike_train(self, is_plot=False):
        """
        return binary spike train of the whole simluated period
        """

        # increase the total number of time points to remove the boundary effects when convolving
        total_point_num = self._total_point_num + self._irf_delay_point_num + \
            self._irf_rising_point_num + self._irf_decaying_point_num - 1

        spike_train = np.zeros(total_point_num, dtype=np.uint8)

        curr_ind = 0
        while curr_ind < self._total_point_num:

            curr_isi = np.random.exponential(1 / self._firing_rate)

            while curr_isi <= self._refactor_dur:
                curr_isi = np.random.exponential(1 / self._firing_rate)

            next_ind = curr_ind + int(round(curr_isi * self._fs))

            if next_ind < self._total_point_num:
                spike_train[next_ind] = 1

            curr_ind = next_ind

        print('Expected firing rage: ' + str(self._firing_rate) + ' Hz')
        print('Real firing rate: ' + str(len(np.where(spike_train)[0]) / self._duration) + ' Hz')

        if is_plot:
            f = plt.figure(figsize=(15, 3))
            ax = f.add_axes([0.1, 0.2, 0.8, 0.7])
            ax.set_xlabel('time (sec)')
            spike_timing = np.where(spike_train)[0] / self._fs
            ax.plot(spike_timing, np.ones(spike_timing.shape), '.b')
            ax.set_xlim([0, self._duration])
            ax.yaxis.set_ticks([])
            ax.set_ylabel('spikes')
            plt.show()

        return spike_train

    def generate_trace(self, spike_train=None, is_plot=False):
        """
        generate calcium traces
        """

        if spike_train is None:
            spike_train = self.generate_spike_train()

        irf = self.generate_irf()

        clean_trace = np.convolve(spike_train, irf, 'full')
        clean_trace = clean_trace[0:self._total_point_num]

        noise = get_random_number(self._noise, clean_trace.shape)

        noise_trace = clean_trace + noise

        if is_plot:
            f = plt.figure(figsize=(15, 5))

            spike_timing = np.where(spike_train)[0] / self._fs
            ax1 = f.add_axes([0.1, 0.8, 0.8, 0.1])
            ax1.plot(spike_timing, np.ones(spike_timing.shape), '.b')
            ax1.set_xlim([0, self._duration])
            ax1.yaxis.set_ticks([])
            ax1.set_ylabel('spikes')

            ax2 = f.add_axes([0.1, 0.45, 0.8, 0.25])
            ax2.plot(np.arange(len(clean_trace)) / self._fs, clean_trace)
            ax2.set_xlim([0, self._duration])
            ax2.set_ylabel('dF/F')

            ax3 = f.add_axes([0.1, 0.1, 0.8, 0.25])
            ax3.plot(np.arange(len(noise_trace)) / self._fs, noise_trace)
            ax3.set_xlim([0, self._duration])
            ax3.set_xlabel('time (sec)')
            ax3.set_ylabel('dF/F')
            plt.show()

        return noise_trace

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """

        super(PossionCalciumTrace, self).to_h5(h5_group)

        h5_group.attrs['peak_dff'] = self._amplitude
        h5_group.attrs['type'] = self.__str__()
        h5_group.attrs['firing_rate_hz'] = self._firing_rate
        h5_group.attrs['refactor_dur_sec'] = self._refactor_dur
        h5_group.attrs['delay_sec'] = self._delay
        h5_group.attrs['rising_dur_sec'] = self._rising_dur
        h5_group.attrs['decaying_tau_sec'] = self._decaying_tau

        spike_train = self.generate_spike_train()
        trace = self.generate_trace(spike_train)

        trace_group = h5_group.create_group('trace')
        trace_group.create_dataset('trace_dff', data=trace)
        trace_group.create_dataset('spike_indices', data=np.where(spike_train==True)[0])

        return trace


class TemporalFrequencyTrace(TemporalComponent):
    """
    A random trace defined by cutoff temporal frequncies
    """

    def __init__(self,
                 cutoff_freq_low=None,
                 cutoff_freq_high=8.,
                 filter_mode='1/f',
                 **kwargs):
        """
        :param cutoff_freq_low: low cutoff temporal frequency, Hz
        :param cutoff_freq_high: high cutoff temporal frequency, Hz
        :param filter_mode: type of filter, 'box' or '1/f'
        :param kwargs:
        """
        super(TemporalFrequencyTrace, self).__init__(**kwargs)

        if cutoff_freq_low is None:
            self._cutoff_freq_low = None
        else:
            self._cutoff_freq_low = float(cutoff_freq_low)

        if cutoff_freq_high is None:
            self._cutoff_freq_high = None
        else:
            self._cutoff_freq_high = float(cutoff_freq_high)

        self._filter_mode = filter_mode

    def __str__(self):
        return 'A temporal component filtered by certain cutoff temporal frequencies. ' + \
               'TemporalFrequencyTrace class inherited from stia.artificial_dataset.TemporalComponent class.'

    def generate_trace(self, is_plot=False):
        """
        generate a random trace with defined temporal frequency
        """
        _, temporal_filter = sp.generate_filter(self._total_point_num, self._fs, self._cutoff_freq_low,
                                                self._cutoff_freq_high, mode=self._filter_mode)

        trace_raw = np.random.rand(self._total_point_num)
        trace_fft = np.fft.fft(trace_raw)
        trace_filtered_fft = trace_fft * temporal_filter
        trace_filtered = bas.array_nor(np.real(np.fft.ifft(trace_filtered_fft))) * self._amplitude
        noise = get_random_number(self._noise, trace_filtered.shape)
        trace = trace_filtered + noise

        if is_plot:
            _ = plt.figure(figsize=(15, 5))
            plt.plot(np.arange(self._total_point_num) / self._fs, trace)
            plt.ylabel("dF/F")
            plt.xlabel("time (sec)")
            plt.show()

        return trace

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """

        super(TemporalFrequencyTrace, self).to_h5(h5_group)

        h5_group.attrs['peak_dff'] = self._amplitude

        h5_group.attrs['type'] = self.__str__()
        if self._cutoff_freq_low is None:
            h5_group.attrs['cutoff_freq_low_hz'] = np.nan
        else:
            h5_group.attrs['cutoff_freq_low_hz'] = self._cutoff_freq_low

        if self._cutoff_freq_high is None:
            h5_group.attrs['cutoff_freq_high_hz'] = np.nan
        else:
            h5_group.attrs['cutoff_freq_high_hz'] = self._cutoff_freq_high

        h5_group.attrs['filter_mode'] = self._filter_mode

        trace = self.generate_trace()

        trace_group = h5_group.create_group('trace')
        trace_group.create_dataset('trace_dff', data=trace)

        return trace


class GaussianFilteredTrace(TemporalComponent):
    """
    A random trace defined by random numbers filtered by gaussian filter
    """

    def __init__(self,
                 sigma=5.,
                 **kwargs):
        """
        :param sigma: sigma of guassian filter, sec
        :param kwargs: inputs to superclass, TemporalComponent
        """

        super(GaussianFilteredTrace, self).__init__(**kwargs)
        self._sigma = float(sigma)

    def __str__(self):
        return 'A temporal component filtered by a gaussian filter. ' + \
               'GaussianFilteredTrace class inherited from stia.artificial_dataset.TemporalComponent class.'

    @property
    def _sigma_pixel(self):
        return self._sigma * self._fs

    def generate_trace(self, is_plot=False):
        trace_raw = np.random.rand(self._total_point_num)
        trace_filtered = bas.array_nor(ni.filters.gaussian_filter1d(trace_raw, self._sigma_pixel)) * self._amplitude
        noise = get_random_number(self._noise, trace_filtered.shape)
        trace = trace_filtered + noise

        if is_plot:
            _ = plt.figure(figsize=(15, 5))
            plt.plot(np.arange(self._total_point_num) / self._fs, trace)
            plt.ylabel("dF/F")
            plt.xlabel("time (sec)")
            plt.show()

        return trace

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """

        super(GaussianFilteredTrace, self).to_h5(h5_group)

        h5_group.attrs['peak_dff'] = self._amplitude

        h5_group.attrs['type'] = self.__str__()
        h5_group.attrs['sigma_sec'] = self._sigma

        trace = self.generate_trace()

        trace_group = h5_group.create_group('trace')
        trace_group.create_dataset('trace_dff', data=trace)

        return trace


class MotionTrace(TemporalComponent):
    """
    A class to generate random motion traces, subclass of TemporalComponent class
    """

    def __init__(self,
                 rate=1.,
                 amplitude=3.,
                 temporal_filter_sigma=0.1,
                 pixel_size=PIXEL_SIZE,
                 **kwarg):
        """

        :param rate: rate of motion onset, Hz
        :param amplitude: peak amplitude, plus and minus, micron
        :param temporal_filter_sigma: sigma of temporal gaussian filter, sec
        :param pixel_size: microns, (height width)
        :param kwarg: other inputs to superclass, TemporalComponent
        """

        super(MotionTrace, self).__init__(amplitude=amplitude,**kwarg)

        self._temporal_filter_sigma=temporal_filter_sigma
        self._rate = rate
        self._pixel_size = pixel_size

    def __str__(self):
        return 'A temporal component simulating random motion. ' + \
               'PossionCalciumTrace class inherited from stia.artificial_dataset.TemporalComponent class.'

    def get_max_height_pix(self):
        return int(self._amplitude / self._pixel_size[0])

    def get_max_width_pix(self):
        return int(self._amplitude / self._pixel_size[1])

    def _generate_motion_onsets(self):
        """
        generate a list of indices of motion onsets
        """

        onset_num = int(self._duration * self._rate)
        onset_ind = random.sample(set(range(self._total_point_num)), onset_num)
        onset_ind.sort()
        return onset_ind

    def generate_trace(self):

        onset_ind = self._generate_motion_onsets()

        segments = zip([0] + onset_ind, onset_ind + [self._total_point_num])
        trace_height = np.empty((self._total_point_num,))
        trace_width = np.empty((self._total_point_num,))
        random_height_pos = np.random.rand(len(segments)) * 2 * self.get_max_height_pix() - self.get_max_height_pix()
        random_width_pos = np.random.rand(len(segments)) * 2 * self.get_max_width_pix() - self.get_max_width_pix()

        for i, segment in enumerate(segments):
            trace_height[segment[0]:segment[1]] = random_height_pos[i]
            trace_width[segment[0]:segment[1]] = random_width_pos[i]

        sigma_point = self._temporal_filter_sigma * self._fs

        trace_height = trace_height + get_random_number(self._noise, trace_height.shape)
        trace_width = trace_height + get_random_number(self._noise, trace_width.shape)

        trace_height = ni.gaussian_filter1d(trace_height, sigma_point)
        trace_width = ni.gaussian_filter1d(trace_width, sigma_point)

        return zip(trace_height.astype(np.int), trace_width.astype(np.int))

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """

        super(MotionTrace, self).to_h5(h5_group)

        h5_group.attrs['amplitude_micron'] = self._amplitude
        h5_group.attrs['rate_hz'] = self._rate
        h5_group.attrs['temporal_filter_sigma'] = self._temporal_filter_sigma

        trace = self.generate_trace()
        trace_group = h5_group.create_group('trace')
        trace_group.create_dataset('trace_pixel', data=trace)

        return trace


class SpatialTemporalNoiseComponent(object):
    """
    a 3D matrix with defined shape (frame, height, width), signal in each dimension are filtered by defined filters.
    temporal filter is applied in frequency domain by defining low cutoff frequency and high cutoff requency
    spatial filter is applied as 2d gaussian with defined sigma
    Caution: generated movies could be large
    """

    def __init__(self,
                 frame_shape=FRAME_SHAPE,
                 pixel_size=PIXEL_SIZE,
                 duration=DURATION,
                 fs=TEMPORAL_SAMPLING_RATE,
                 temporal_cutoff_freq_low=None,
                 temporal_cutoff_freq_high=8.,
                 temporal_filter_mode='1/f',
                 spatial_filter_sigma=(15., 15),
                 peak_dff=1.,
                 noise=('gaussian', 0, 0.1),
                 baseline = 1000.
                 ):
        """
        :param frame_shape: the shape of a single frame, tuple of 2 ints, (height, width)
        :param duration: total duration of simulation, sec
        :param fs: temporal sampling rate, Hz
        :param pixel_size: physical size of each pixel, (height width), micron
        :param temporal_cutoff_freq_low: low cutoff temporal frequency, Hz
        :param temporal_cutoff_freq_high: high cutoff temporal frequency, Hz
        :param temporal_filter_mode: type of filter, 'box' or '1/f'
        :param spatial_filter_sigma: sigma of 2d spatial gaussian filter, micron
        :param peak_dff: maximum range of dF/F, float
        :param noise: noise distribution ('gaussian',"mean","sigma") or ('flat',"mean","amplitude")
        :param baseline: baseline fluorscence, artificial unit (usually count of analog to digital range)
        """

        self._frame_shape = (int(frame_shape[0]), int(frame_shape[1]))
        self._duration = float(duration)
        self._fs = float(fs)
        self._pixel_size = (float(pixel_size[0]), float(pixel_size[1]))

        if temporal_cutoff_freq_low is None:
            self._temporal_cutoff_freq_low = None
        else:
            self._temporal_cutoff_freq_low = float(temporal_cutoff_freq_low)

        if temporal_cutoff_freq_high is None:
            self._temporal_cutoff_freq_high = None
        else:
            self._temporal_cutoff_freq_high = float(temporal_cutoff_freq_high)

        self._temporal_filter_mode = temporal_filter_mode
        self._spatial_filter_sigma = (float(spatial_filter_sigma[0]), float(spatial_filter_sigma[1]))
        self._peak_dff = float(peak_dff)
        self._noise = noise
        self._baseline = baseline

    def __str__(self):
        return 'A spatial temporal component limited within certain spatial temporal frequencies range. This is not ' \
               'a linear product of a static spatial and static temporal component, but contains higher order ' \
               'interactions between spatial and temporal components.'

    @property
    def _total_point_num(self):
        """
        total number of points of the whole simulated trace
        """
        return int(round(self._duration * self._fs))

    @property
    def _filter_size_pixel(self):
        return self._spatial_filter_sigma[0] / self._pixel_size[0], self._spatial_filter_sigma[1] / self._pixel_size[1]

    def set_frame_shape(self, frame_shape):
        self._frame_shape = frame_shape

    def get_frame_shape(self):
        return self._frame_shape

    def get_pixel_size(self):
        return self._pixel_size

    def get_duration(self):
        return self._duration

    def get_fs(self):
        return self._fs

    def generate_movie(self, is_plot=False):

        mov_raw = np.random.rand(self._total_point_num, self._frame_shape[0], self._frame_shape[1])

        mov_spatial_filtered = ni.gaussian_filter(mov_raw,
                                                  (0., self._spatial_filter_sigma[0], self._spatial_filter_sigma[1]))

        del mov_raw

        mov_spatial_filtered_fft = np.fft.fftn(mov_spatial_filtered, axes=[0])

        del mov_spatial_filtered

        _, temporal_filter = sp.generate_filter(self._total_point_num, self._fs, self._temporal_cutoff_freq_low,
                                                self._temporal_cutoff_freq_high, mode=self._temporal_filter_mode)

        mov_spatial_filtered_fft = mov_spatial_filtered_fft.transpose((1, 2, 0))

        mov_spatial_filtered_fft_filtered = np.multiply(mov_spatial_filtered_fft, temporal_filter)

        del mov_spatial_filtered_fft

        mov_spatial_filtered_fft_filtered = mov_spatial_filtered_fft_filtered.transpose((2, 0, 1))

        mov_filtered = np.real(np.fft.ifftn(mov_spatial_filtered_fft_filtered, axes=[0]))

        del mov_spatial_filtered_fft_filtered

        mov_filtered = bas.array_nor(mov_filtered) * self._peak_dff

        noise = get_random_number(self._noise, mov_filtered.shape)

        mov = mov_filtered + noise

        del mov_filtered, noise

        mov = (mov + 1.) * self._baseline

        if is_plot:
            tf.imshow(mov, cmap='gray')
            plt.show()

        return mov

    def to_h5(self, h5_group):
        """
        generate a hdf5 group for saving
        """
        h5_group.attrs['type'] = self.__str__()
        h5_group.attrs['frame_shape_height_pixel'] = self._frame_shape[0]
        h5_group.attrs['frame_shape_width_pixel'] = self._frame_shape[1]
        h5_group.attrs['pixel_size_height_micron'] = self._pixel_size[0]
        h5_group.attrs['pixel_size_width_micron'] = self._pixel_size[1]
        h5_group.attrs['duration_sec'] = self._duration
        h5_group.attrs['temporal_sampling_rate_hz'] = self._fs

        if self._temporal_cutoff_freq_low is None:
            h5_group.attrs['temporal_cutoff_freq_low_hz'] = np.nan
        else:
            h5_group.attrs['temporal_cutoff_freq_low_hz'] = self._temporal_cutoff_freq_low

        if self._temporal_cutoff_freq_high is None:
            h5_group.attrs['temporal_cutoff_freq_high_hz'] = np.nan
        else:
            h5_group.attrs['temporal_cutoff_freq_high_hz'] = self._temporal_cutoff_freq_high

        h5_group.attrs['temporal_filter_mode'] = self._temporal_filter_mode
        h5_group.attrs['spatial_filter_sigma_micron'] = self._spatial_filter_sigma
        h5_group.attrs['peak_dff'] = self._peak_dff
        h5_group.attrs['noise_type'] = self._noise[0]
        h5_group.attrs['noise_mean_dff'] = self._noise[1]
        h5_group.attrs['noise_err_dff'] = self._noise[2]
        h5_group.attrs['baseline'] = self._baseline

        mov = self.generate_movie()

        movie_group = h5_group.create_group('movie')
        movie_group.create_dataset('movie', data=mov, compression='lzf')

        return mov


def run():

    # the path to save
    test_data_path = "E:/data/python_temp_folder/artificial_dataset.hdf5"
    test_movie_path = "E:/data/python_temp_folder/artificial_dataset.tif"
    # test_data_path = "E:/data/python_temp_folder/stia_motion_correction_test/artificial_dataset.hdf5"
    # test_movie_path = "E:/data/python_temp_folder/stia_motion_correction_test/artificial_dataset.tif"

    # create dataset object
    dset = ArtificialDataset()

    # set motion
    motion = MotionTrace(noise=('gaussian', 0., 1.))
    dset.set_motion(motion)

    #initialize cell number
    cell_number = 0

    # add 3 filled soma
    filled_soma_num = 3
    for i in range(filled_soma_num):
        name = 'cell' + bas.int2str(cell_number, 4)
        center = (random.choice(range(20, dset.get_frame_shape()[0]-20)),
                  random.choice(range(20, dset.get_frame_shape()[1] - 20)))
        firing_rate = np.random.rand() * 0.8 + 0.1
        baseline = np.random.rand() * 2000. + 1000.
        peak_dff = np.random.rand() + 0.5
        dset.add_independent_component(name, FilledSoma(center, baseline=baseline),
                                       PossionCalciumTrace(firing_rate=firing_rate, amplitude=peak_dff))
        cell_number += 1

    # add 3 filled soma with processes
    filled_soma_process_num = 3
    for i in range(filled_soma_process_num):
        name = 'cell' + bas.int2str(cell_number, 4)
        center = (random.choice(range(20, dset.get_frame_shape()[0] - 20)),
                  random.choice(range(20, dset.get_frame_shape()[1] - 20)))
        firing_rate = np.random.rand() * 0.8 + 0.1
        baseline = np.random.rand() * 2000. + 1000.
        peak_dff = np.random.rand() + 0.5
        dset.add_independent_component(name, FilledSomaWithProcess(center, baseline=baseline,
                                                                   process_approx_area=300., process_threshold=0.15,
                                                                   process_branch_distribution=([1, 2, 3], [0.8, 0.1, 0.1]),
                                                                   process_distance_distribution=('flat', 8., 8.)),
                                       PossionCalciumTrace(firing_rate=firing_rate, amplitude=peak_dff))
        cell_number += 1

    # add 8 doughnut soma
    doughnut_soma_num = 8
    for i in range(doughnut_soma_num):
        name = 'cell' + bas.int2str(cell_number, 4)
        center = (random.choice(range(20, dset.get_frame_shape()[0] - 20)),
                  random.choice(range(20, dset.get_frame_shape()[1] - 20)))
        firing_rate = np.random.rand() * 0.8 + 0.1
        baseline = np.random.rand() * 2000. + 1000.
        peak_dff = np.random.rand() + 0.5
        dset.add_independent_component(name, DoughnutSoma(center, baseline=baseline),
                                       PossionCalciumTrace(firing_rate=firing_rate, amplitude=peak_dff))
        cell_number += 1

    # add 3 doughnut soma with processes
    doughnut_soma_process_num = 3
    for i in range(doughnut_soma_process_num):
        name = 'cell' + bas.int2str(cell_number, 4)
        center = (random.choice(range(20, dset.get_frame_shape()[0] - 20)),
                  random.choice(range(20, dset.get_frame_shape()[1] - 20)))
        firing_rate = np.random.rand() * 0.8 + 0.1
        baseline = np.random.rand() * 2000. + 1000.
        peak_dff = np.random.rand() + 0.5
        dset.add_independent_component(name, DoughnutSomaWithProcess(center, baseline=baseline,
                                                                     process_approx_area=300., process_threshold=0.2,
                                                                     process_branch_distribution=([1, 2, 3], [0.8, 0.1, 0.1]),
                                                                     process_distance_distribution=('flat', 8., 8.)),
                                       PossionCalciumTrace(firing_rate=firing_rate, amplitude=peak_dff))
        cell_number += 1

    # add linear background
    linear_bg_frame = BackgroundGaussianFiltering(sigma=(50., 50.), baseline=5000.)
    linear_bg_trace = TemporalFrequencyTrace(cutoff_freq_high=0.25, amplitude=0.5, noise=('gaussian', 0., 0.05))
    dset.add_independent_component('linear_background', linear_bg_frame, linear_bg_trace)

    # add nonlinear background and random noise (contains interaction between spatial and temporal interaction)
    nonlinear_bg = SpatialTemporalNoiseComponent(peak_dff=0.5, noise=('flat', 0., 1.5), baseline=3000.)
    dset.add_interactive_component('nonlinear_background', nonlinear_bg)

    # create movie and save hdf5 file
    mov = dset.to_h5(test_data_path, is_plot=True)

    # save movie as a tif file
    tf.imsave(test_movie_path, mov)


if __name__ == "__main__":

    run()

    # -----------------------------------------------------------------
    # filled_neuron = FilledSoma(frame_shape=FRAME_SHAPE, center=(23, 46))
    # filled_neuron.set_frame_shape((300,300))
    # mask = filled_neuron.generate_mask(is_plot=False)
    # print mask.shape
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # doughnut_neuron = DoughnutSoma(frame_shape=FRAME_SHAPE, center=(23, 46))
    # doughnut_neuron.generate_mask(is_plot=True)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # dendrite = Process(frame_shape=FRAME_SHAPE, center=(23, 46))
    # dendrite.generate_mask(is_plot=True)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # calcium1 = PossionCalciumTrace()
    # _ = calcium1.generate_irf(is_plot=True)
    # _ = calcium1.generate_spike_train(is_plot=True)
    # calcium1.generate_trace(is_plot=True)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # steady_bg = BackgroundSpatialFrequency()
    # steady_bg.generate_mask(is_plot=True)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # steady_bg = BackgroundGaussianFiltering()
    # steady_bg.generate_mask(is_plot=True)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # trace_tf = TemporalFrequencyTrace()
    # trace_tf.generate_trace(is_plot=True)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # trace_gaussian = GaussianFilteredTrace()
    # trace_gaussian.generate_trace(is_plot=True)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # interactive_bg = InteractiveSpatialTemporalComponent(frame_shape=(128, 128),
    #                                                      duration=50.,
    #                                                      fs=30.,
    #                                                      pixel_size=(0.5, 0.5),
    #                                                      temporal_cutoff_freq_low=None,
    #                                                      temporal_cutoff_freq_high=4.,
    #                                                      temporal_filter_mode='1/f',
    #                                                      spatial_filter_sigma=(20., 20),
    #                                                      peak_dff=1.,
    #                                                      noise=('gaussian', 0, 0.3),
    #                                                      baseline=1000.)
    # mov = interactive_bg.generate_movie()
    # tf.imsave(r"E:\data\python_temp_folder\test_interactive_bg.tif", mov.astype(np.float32))
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # neuron_spatial = DoughnutSoma(center=(47, 58), threshold=0.15)
    # dendrite_spatial = Process(center=(47, 58), threshold=0.15,
    #                            branch_distribution=([1, 2, 3, 4], [0.85, 0.05, 0.05, 0.05]),
    #                            distance_distribution=('gaussian', 5., 5.))
    # temporal = PossionCalciumTrace(peak_dff=2., noise=('gaussian', 0., 0.01))
    # spatial_frame = bas.array_nor(neuron_spatial.generate_mask() + dendrite_spatial.generate_mask())
    # temporal_trace = temporal.generate_trace()
    # neuron = IndependentSpatialTemporalComponent(spatial_frame, temporal_trace, baseline=2000.)
    # mov = neuron.generate_movie()
    #
    # neuron2_spatial = DoughnutSoma(center=(180, 63), first_area=70., second_area=70., threshold=0.2)
    # neuron2_temporal = PossionCalciumTrace(firing_rate=0.5, peak_dff=1.5, noise=('gaussian', 0., 0.01))
    # neuron2_frame = neuron2_spatial.generate_mask()
    # neuron2_trace = neuron2_temporal.generate_trace()
    # neuron2 = IndependentSpatialTemporalComponent(neuron2_frame, neuron2_trace, baseline=1500.)
    # mov += neuron2.generate_movie()
    #
    # neuron3_spatial = FilledSoma(center=(114, 204), filter_size=2., approx_area=70., threshold=0.3)
    # neuron3_temporal = PossionCalciumTrace(firing_rate=0.3, peak_dff=1., noise=('gaussian', 0., 0.01))
    # neuron3_frame = neuron3_spatial.generate_mask()
    # neuron3_trace = neuron3_temporal.generate_trace()
    # neuron3 = IndependentSpatialTemporalComponent(neuron3_frame, neuron3_trace, baseline=1000.)
    # mov += neuron3.generate_movie()
    #
    # dendrite2_spatial = Process(center=(87, 193), approx_area=500., threshold=0.2,
    #                             branch_distribution=([1, 2, 3], [0.95, 0.03, 0.02]),
    #                             distance_distribution=('gaussian', 5., 5.))
    # dendrite2_temporal = PossionCalciumTrace(firing_rate=0.1, peak_dff=5., noise=('gaussian', 0., 0.01))
    # dendrite2_frame = dendrite2_spatial.generate_mask()
    # dendrite2_trace = dendrite2_temporal.generate_trace()
    # dendrite2 = IndependentSpatialTemporalComponent(dendrite2_frame, dendrite2_trace, baseline=800.)
    # mov += dendrite2.generate_movie()
    #
    # bg_spatial = BackgroundGaussianFiltering(sigma=(50., 50.))
    # bg_temporal = TemporalFrequencyTrace(cutoff_freq_high=0.25, peak_dff=0.1, noise=('gaussian', 0., 0.01))
    # bg_frame = bg_spatial.generate_mask()
    # bg_trace = bg_temporal.generate_trace()
    # background = IndependentSpatialTemporalComponent(bg_frame, bg_trace, baseline=3000.)
    # mov += background.generate_movie()
    #
    # interactive_bg = InteractiveSpatialTemporalComponent(peak_dff=1., noise=('flat', 0., 1.), baseline=2000.)
    # mov += interactive_bg.generate_movie()
    #
    # mov = mov.clip(min=0., max=65535.)
    #
    # tf.imsave(r"E:\data\python_temp_folder\test_mov.tif", mov.astype(np.uint16))
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # motion = MotionTrace()
    # print motion.get_max_height_pix()
    # traces = motion.generate_trace()
    # print '\n'.join([str(p) for p in traces])
    # -----------------------------------------------------------------

