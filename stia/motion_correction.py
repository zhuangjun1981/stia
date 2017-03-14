"""
This is the module to do simple motion correction of a 2-photon data set. It use open cv phase coorelation function to
find parameters of x, y rigid transformation frame by frame iteratively. The input dataset should be a set of tif
files. This files should be small enough to be loaded and manipulated in your memory.

@Jun Zhuang May 27, 2016
"""

import tifffile as tf
import os
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
import utility.image_analysis as ia
import utility.basic as bas



def phase_correlation(img_match, img_ref):
    """
    openCV phase correction wrapper, as one of the align_func to perform motion correction. Open CV phaseCorrelate
    function returns (x_offset, y_offset). This wrapper switches the order of result and returns (height_offset,
    width_offset) to be more consistent with numpy indexing convention.

    :param img_match: the matching image
    :param img_ref: the reference image
    :return: rigid_transform coordinates of alignment (height_offset, width_offset)
    """

    x_offset, y_offset =  cv2.phaseCorrelate(img_match.astype(np.float32), img_ref.astype(np.float32))
    return [y_offset, x_offset]


def align_single_chunk(chunk, img_ref, max_offset=(10., 10.), align_func=phase_correlation, fill_value=0.,
                       verbose=True):
    """
    align the frames in a single chunk of movie to the img_ref.

    all operations will be applied with np.float32 format

    If the translation is larger than max_offset, the translation of that particular frame will be set as zero,
    and it will not be counted during the calculation of average projection image.

    :param chunk: a movie chunk, should be small enough to be managed in memory, 3d numpy.array
    :param img_ref: reference image, 2d numpy.array
    :param max_offset: maximum offset, (height, width), if single value, will be applied to both height and width
    :param align_func: function to align two image frames, return rigid transform offset (height, width)
    :param fill_value: value to fill empty pixels
    :param verbose:
    :return: alignment offset list, aligned movie chunk (same data type of original chunk), updated mean projection
    image np.float32
    """

    data_type = chunk.dtype
    img_ref = img_ref.astype(np.float32)

    # handle max_offset
    try:
        max_offset_height = float(max_offset[0])
        max_offset_width = float(max_offset[1])
    except TypeError:
        max_offset_height = float(max_offset)
        max_offset_width = float(max_offset)

    offset_list = []
    aligned_chunk = np.empty(chunk.shape, dtype=np.float32)

    # sum of valid frames, will be used to calculate updated mean project
    sum_img = np.zeros((chunk.shape[1], chunk.shape[2]), dtype=np.float32)

    # number of valid frames, will be used to calculate updated mean project
    valid_frame_num = 0

    # total number of frames of the chunk
    total_frame_num = chunk.shape[0]

    for i in range(chunk.shape[0]):

        if verbose:
            if i % (total_frame_num // 10) == 0:
                print 'Motion correction progress:', int(round(float(i) * 100 / total_frame_num)), '%'

        curr_frame = chunk[i, :, :].astype(np.float32)

        curr_offset = align_func(curr_frame, img_ref)

        if curr_offset[0] <= max_offset_height and curr_offset[1] <= max_offset_width:
            aligned_frame = ia.rigid_transform_cv2_2d(curr_frame, offset=curr_offset[::-1], fill_value=fill_value)
            aligned_chunk[i, :, :] = aligned_frame
            offset_list.append(curr_offset)
            sum_img = sum_img + aligned_frame
            valid_frame_num += 1
        else:
            aligned_chunk[i, :, :] = curr_frame
            offset_list.append([0., 0.])

    new_mean_img = sum_img / float(valid_frame_num)

    return offset_list, aligned_chunk.astype(data_type), new_mean_img


def align_single_chunk_iterate(chunk, iteration=2, max_offset=(10., 10.), align_func=phase_correlation, fill_value=0.,
                               verbose=True):
    """
    align the frames in a single chunk of movie to its mean projection iteratively. for each iteration, the reference
    image (mean projection) will be updated based on the aligned chunk

    all operations will be applied with np.float32 format

    If the translation is larger than max_offset, the translation of that particular frame will be set as zero,
    and it will not be counted during the calculation of average projection image.

    :param chunk: a movie chunk, should be small enough to be managed in memory, 3d numpy.array
    :param iteration: number of iterations, int
    :param max_offset: maximum offset, (height, width), if single value, will be applied to both height and width
    :param align_func: function to align two image frames, return rigid transform offset (height, width)
    :param fill_value: value to fill empty pixels
    :param verbose:
    :return: alignment offset list, aligned movie chunk (same data type of original chunk), updated mean projection
    image np.float32
    """

    if iteration < 1:
        raise ValueError('iteration should be an integer larger than 0.')

    img_ref = np.mean(chunk.astype(np.float32), axis=0)
    offset_list = None
    aligned_chunk = None

    for i in range(iteration):

        print "\nMotion Correction, iteration " + str(i)
        offset_list, aligned_chunk, img_ref = align_single_chunk(chunk, img_ref, max_offset=max_offset,
                                                                 align_func=align_func, fill_value=fill_value,
                                                                 verbose=verbose)

    return offset_list, aligned_chunk, img_ref


def align_single_chunk_iterate2(chunk, iteration=2, max_offset=(10., 10.), align_func=phase_correlation, fill_value=0.,
                               verbose=True):
    """
    align the frames in a single chunk of movie to its mean projection iteratively. for each iteration, the reference
    image (mean projection) will be updated based on the aligned chunk

    all operations will be applied with np.float32 format

    If the translation is larger than max_offset, the translation of that particular frame will be set as zero,
    and it will not be counted during the calculation of average projection image.

    the difference between align_single_chunk_iterate and align_single_chunk_iterate2 is during iteration, the first
    method only updates img_ref and for each iteration it aligns the original movie to the updated img_ref. it returns
    the offsets generated by the last iteration. But the second method updates both img_ref and the movie itself, and
    for each iteration it aligns the corrected movie form last iteration to the updated img_ref, and return the
    accumulated offsets of all iterations.

    :param chunk: a movie chunk, should be small enough to be managed in memory, 3d numpy.array
    :param iteration: number of iterations, int
    :param max_offset: maximum offset, (height, width), if single value, will be applied to both height and width
    :param align_func: function to align two image frames, return rigid transform offset (height, width)
    :param fill_value: value to fill empty pixels
    :param verbose:
    :return: alignment offset list, aligned movie chunk (same data type of original chunk), updated mean projection
    image np.float32
    """

    if iteration < 1:
        raise ValueError('iteration should be an integer larger than 0.')

    img_ref = np.mean(chunk.astype(np.float32), axis=0)
    offset_list = None
    aligned_chunk = chunk

    for i in range(iteration):

        print "\nMotion Correction, iteration " + str(i)
        curr_offset_list, aligned_chunk, img_ref = align_single_chunk(aligned_chunk, img_ref, max_offset=max_offset,
                                                                      align_func=align_func, fill_value=fill_value,
                                                                      verbose=verbose)

        if offset_list is None:
            offset_list = curr_offset_list
        else:
            for i in range(len(offset_list)):
                offset_list[i] = [offset_list[i][0] + curr_offset_list[i][0],
                                  offset_list[i][1] + curr_offset_list[i][1]]

    return offset_list, aligned_chunk, img_ref


def correct_movie(mov, offset, fill_value=0., verbose=True):
    """
    correcting a movie with given offset list, whole process will be operating on np.float32 data format.

    :param mov: movie to be corrected, should be a 3-d np.array managable by the computer memory
    :param offset: list of correction offsets for each frame of the movie
    :param fill_value: value to fill empty pixels
    :param verbose:
    :return: corrected movie, with same data type as original movie
    """

    if isinstance(offset, np.ndarray):
        if len(offset.shape) != 2:
            raise ValueError('offset should be 2-dimensional.')
        elif offset.shape[0] != mov.shape[0]:
            raise ValueError('offset should have same length as the number of frames in the movie!')
        elif offset.shape[1] != 2:
            raise ValueError('each item in offset should contain 2 values, (offset_height, offset_width)!')
    elif isinstance(offset, list):
        if len(offset) != mov.shape[0]:
            raise ValueError('offset should have same length as the number of frames in the movie!')
        else:
            for single_offset in offset:
                if len(single_offset) != 2:
                    raise ValueError('each item in offset should contain 2 values, (offset_height, offset_width)!')

    total_frame_num = mov.shape[0]
    corrected_mov = np.empty(mov.shape, dtype=np.float32)

    for i in range(mov.shape[0]):

        if verbose:
            if i % (total_frame_num // 10) == 0:
                print 'Correction progress:', int(round(float(i) * 100 / total_frame_num)), '%'

        curr_frame = mov[i, :, :].astype(np.float32)
        curr_offset = offset[i]
        corrected_frame = ia.rigid_transform_cv2_2d(curr_frame, offset=curr_offset[::-1], fill_value=fill_value)
        corrected_mov[i, :, :] = corrected_frame

    return corrected_mov.astype(mov.dtype)


def align_multiple_files_iterate(paths, output_folder=None, is_output_mov=True, iteration=2, max_offset=(10., 10.),
                                 align_func=phase_correlation, fill_value=0., verbose=True, offset_file_name=None,
                                 mean_projection_file_name=None):

    """
    Motion correct a list of movie files (currently only support .tif format, designed for ScanImage output files.
    each files will be first aligned to its own mean projection iteratively. for each iteration, the reference
    image (mean projection) will be updated based on the aligned result. Then all files will be aligned based on their
    final mean projection images.

    all operations will be applied with np.float32 format

    :param paths: list of paths of data files (currently only support .tif format), they should have same height and
                  width dimensions.
    :param output_folder: folder to save output, if None, a subfolder named "motion_correction" will be created in the
                          folder of the first paths in paths
    :param is_output_mov: bool, if True, aligned movie will be saved, if False, only correction offsets and final mean
                          projection image of all files will be saved
    :param iteration: int, number of iterations to correct each file
    :param max_offset: If the correction is larger than max_offset, the correction of that particular frame will be
                       set as zero, and it will not be counted during the calculation of average projection image.
    :param align_func: function to align two image frames, return rigid transform offset (height, width)
    :param fill_value: value to fill empty pixels
    :param verbose:
    :param offset_file_name: str, the file name of the saved offsets hdf5 file (without extension), if None,
                             default will be 'correction_offsets.hdf5'
    :param mean_projection_file_name: str, the file name of the saved mean projection image (without extension), if
                                      None, default will be 'corrected_mean_projection.tif'
    :return: offsets, dictionary of correction offsets. Key: path of file; value: list of tuple with correction offsets,
             (height, width)
    """

    if output_folder is None:
        main_folder, _ = os.path.split(paths[0])
        output_folder = os.path.join(main_folder, 'motion_correction')

    if not os.path.isdir(output_folder):
        print "\n\nOutput folder: " + str(output_folder) + "does not exist. Create new folder."
        os.mkdir(output_folder)
    else:
        print "\n\nOutput folder: " + str(output_folder) + "already exists. Write into this folder."
    os.chdir(output_folder)

    offsets = [] # list of local correction for each file
    mean_projections=[] # final mean projection of each file

    for path in paths:

        if verbose:
            print '\nCorrecting file: ' + str(path) + ' ...'

        curr_mov = tf.imread(path)
        offset, _, mean_projection = align_single_chunk_iterate(curr_mov, iteration=iteration, max_offset=max_offset,
                                                                align_func=align_func, fill_value=fill_value,
                                                                verbose=verbose)

        offsets.append(offset)
        mean_projections.append(mean_projection)

    mean_projections = np.array(mean_projections, dtype=np.float32)

    print '\n\nCorrected mean projection images of all files ...'
    mean_projection_offset, _, final_mean_projection = align_single_chunk_iterate(mean_projections, iteration=iteration,
                                                                                  max_offset=max_offset,
                                                                                  align_func=align_func,
                                                                                  fill_value=fill_value,
                                                                                  verbose=False)

    print '\nAdding global correction offset to local correction offsets and save.'
    if offset_file_name is None:
        h5_file = h5py.File(os.path.join(output_folder, 'correction_offsets.hdf5'))
    else:
        h5_file = h5py.File(os.path.join(output_folder, offset_file_name + '.hdf5'))
    offset_dict = {}
    for i in range(len(offsets)):
        curr_offset = offsets[i]
        curr_global_offset = mean_projection_offset[i]
        offsets[i] = [[offset[0] + curr_global_offset[0],
                       offset[1] + curr_global_offset[1]] for offset in curr_offset]
        curr_h5_dset = h5_file.create_dataset('file' + bas.int2str(i, 4), data=offsets[i])
        curr_h5_dset.attrs['path'] = str(paths[i])
        curr_h5_dset.attrs['format'] = ['height', 'width']
        offset_dict.update({str(paths[i]):offsets[i]})

    print '\nSaving final mean projection image.'
    if mean_projection_file_name is None:
        tf.imsave(os.path.join(output_folder, 'corrected_mean_projection.tif'), final_mean_projection)
    else:
        tf.imsave(os.path.join(output_folder, mean_projection_file_name + '.tif'), final_mean_projection)

    if is_output_mov:
        for i, curr_path in enumerate(paths):
            print '\nFinal Correction of file: ' + curr_path
            curr_mov = tf.imread(curr_path)
            curr_offset = offsets[i]
            curr_corrected_mov = correct_movie(curr_mov, curr_offset, fill_value=fill_value, verbose=verbose)
            _, curr_file_name = os.path.split(curr_path)
            curr_save_name = bas.add_suffix(curr_file_name, '_corrected')
            print 'Saving corrected file: ' + curr_save_name + ' ...'
            tf.imsave(os.path.join(output_folder, curr_save_name), curr_corrected_mov)


    f = plt.figure(figsize = (10, 10))
    ax = f.add_subplot(111)
    ax.imshow(final_mean_projection, cmap='gray', interpolation='nearest')
    plt.show()

    return offset_dict










