import random
import cv2

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ni
import scipy.sparse as sparse
import tifffile as tf
import h5py

import basic as bas

CONNECTIVITY = 4


def find_border(pixel_ind, frame_shape=(128, 128)):
    """
    return a list of pixel indices bordering the given mask
    :param pixel_ind: the pixel indices of the mask
    :param frame_shape: shape of the whole frame, (height, width)
    :return: pixel indices of the border
    """
    mask = np.zeros(frame_shape, dtype=np.bool)
    mask[pixel_ind] = True
    big_mask = ni.binary_dilation(mask)
    border = big_mask - mask
    return np.where(border)


def grow_one_connected_pixel(mask):
    """
    increase one pixel on the border with existing binary mask
    :param mask: input binary mask, np.bool
    :return: binary mask
    """

    pixels = np.where(mask)
    pixels = [list(p) for p in pixels]
    # print pixels
    border = find_border(pixels, frame_shape=mask.shape)
    random_ind = random.randrange(len(border[0]))
    pixels[0].append(border[0][random_ind])
    pixels[1].append(border[1][random_ind])

    new_mask = np.zeros(mask.shape, dtype=np.bool)
    new_mask[pixels] = True

    return new_mask


def is_pixel_convex(mask, pixel, is_plot=False):
    """
    check if the pixel is added to the mask, is the convexity of the mask get reduced
    :param mask: input binary mask, np.bool
    :param pixel: index of the pixel, [y, x]
    :param is_plot: mostly plot for debug
    :return: bool
    """

    if mask[pixel[0], pixel[1]]:
        raise(LookupError, 'pixel is inside the mask!')

    if is_plot:
        _ = plt.figure()
        plot_mask = np.array(mask, dtype=np.uint8)
        plot_mask[pixel[0], pixel[1]] = 2
        plt.imshow(plot_mask, cmap='gray', interpolation='nearest')
        plt.show()

    # check up direction
    if pixel[0] > 1 and mask[pixel[0]-1, pixel[1]] == False:
        rest_line = mask[:pixel[0], pixel[1]]
        if any(rest_line):
            return False

    # check down direction
    if pixel[0] < mask.shape[0] - 2 and mask[pixel[0] + 1, pixel[1]] == False:
        rest_line = mask[pixel[0]+1:, pixel[1]]
        if any(rest_line):
            return False

    # check left direction
    if pixel[1] > 1 and mask[pixel[0], pixel[1] - 1] == False:
        rest_line = mask[pixel[0], :pixel[1]]
        if any(rest_line):
            return False

    # check left direction
    if pixel[1] < mask.shape[1] - 2 and mask[pixel[0], pixel[1] + 1] == False:
        rest_line = mask[pixel[0], pixel[1] + 1:]
        if any(rest_line):
            return False

    return True


def binary_line(point1, point2):
    """
    generate a list of pixels connecting to points
    :param point1: (row, col) of the first points
    :param point2: (row, col) of the second points
    :return: pixels in the line ([rows], [cols])
    """
    data_points = max([abs(point1[0] - point2[0]), abs(point1[1] - point2[1])]) * 2
    rows = np.linspace(point1[0], point2[0], num=data_points, endpoint=False)
    cols = np.linspace(point1[1], point2[1], num=data_points, endpoint=False)

    rows = [bas.round_int(r) for r in rows]
    cols = [bas.round_int(c) for c in cols]

    new_rows = [rows[0]]
    new_cols = [cols[0]]

    for i in range(1, len(rows)):
        if not (rows[i] == new_rows[-1] and cols[i] == new_cols[-1]):
            new_rows.append(rows[i])
            new_cols.append(cols[i])

    return new_rows, new_cols


def binary_ray(point, angle, distance, frame_shape=None):
    """
    generate a list of pixels representing a ray shooting from the start point with defined angle and distance
    :param point: start point (y, x)
    :param angle: angle of the ray, arc
    :param distance: length of the ray, unit is pixel
    :param frame_shape: the size of frame, (number of rows, number of columns), (int, int). If None, assume no
                        limitation of the frame. If not None, only the pixels inside the frame will be returned
    :return: pixels in the ray ([rows], [cols])
    """
    end_point = [bas.round_int(point[0] - np.sin(angle) * distance), bas.round_int(point[1] + np.cos(angle) * distance)]
    pixels = binary_line(point, end_point)
    if frame_shape is not None:
        pixels = zip(*pixels)
        pixels = [p for p in pixels if 0 <= p[0] < frame_shape[0] and 0 <= p[1] < frame_shape[1]]
        pixels = zip(*pixels)
    return pixels


def grow_one_convex_pixel(mask):
    """
    increase one pixel on the border with existing binary mask, this added pixel should not add convexity of the
    originalmask
    :param mask: input binary mask, np.bool
    :return: binary mask
    """

    pixels = np.where(mask)
    pixels = [list(p) for p in pixels]
    # print pixels
    border = find_border(pixels)
    random_ind = random.randrange(len(border[0]))
    curr_pixel = (border[0][random_ind], border[1][random_ind])

    while not is_pixel_convex(mask, curr_pixel):
        random_ind = random.randrange(len(border[0]))
        curr_pixel = (border[0][random_ind], border[1][random_ind])

    pixels[0].append(curr_pixel[0])
    pixels[1].append(curr_pixel[1])

    new_mask = np.zeros(mask.shape, dtype=np.bool)
    new_mask[pixels] = True

    return new_mask


def check_hole(mask, connectivity=CONNECTIVITY):
    """
    check if there are holes in a binary image
    :param mask: input binary mask, np.bool
    :param connectivity: 4 or 8
    :return: bool
    """
    reverse_mask = np.invert(mask)
    if connectivity == 4:
        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    elif connectivity == 8:
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    else:
        raise (ValueError, 'connectivity should be 4 or 8.')
    _, patch_num = ni.label(reverse_mask, structure)
    return patch_num > 1


def collapse_line(arr):
    """
    move one pixel to its nearest True element toward lower indexing direction
    :param arr: input 1-d array, np.bool
    :return: new collapsed array
    """
    true_num = np.sum(arr)
    for i, p in enumerate(arr):
        if p:
            break
    new_arr = np.zeros(arr.shape, dtype=np.bool)
    new_arr[i: i + true_num] = True
    return new_arr


def collapse_shape(mask, direction):
    """
    collapse a binary shape into a convex shape towards a certain direction

    :param mask: input binary mask, np.bool
    :param direction: direction to collapse, ('u','d','l','r')
    :return: collapsed mask, np.bool
    """

    mask_shape = mask.shape
    new_mask = np.zeros(mask_shape, dtype=np.bool)

    if direction in ['u', 'd']:
        for x in range(mask_shape[1]):
            if direction == 'u':
                new_mask[:, x] = collapse_line(mask[:, x])
            if direction == 'd':
                new_mask[:, x] = collapse_line(mask[:, x][::-1])[::-1]

    if direction in ['l', 'r']:
        for y in range(mask_shape[0]):
            if direction == 'l':
                new_mask[y, :] = collapse_line(mask[y, :])
            if direction == 'r':
                new_mask[y, :] = collapse_line(mask[y, :][::-1])[::-1]

    return new_mask


def random_filled_shape(frame_shape=(128, 128), center=(38, 59), pixel_num=20, is_plot=False):
    """
    generating binary random shape with no holes
    :param frame_shape: shape of the whole frame, (height, width)
    :param center: center pixel
    :param pixel_num: total pixel number of the shape
    :param is_plot
    :return: bianry map of the shape
    """
    directions = np.array(['r', 'u', 'l', 'd'])

    mask = np.zeros(frame_shape, dtype=np.bool)
    mask[center] = True

    for i in range(pixel_num - 1):
        curr_mask = grow_one_connected_pixel(mask)
        mask = np.array(curr_mask)

    changed = True

    while changed and (not mask.all()):
        # generating sequence of collapse directions
        start_dir = random.choice(directions)
        ind = np.where(directions == start_dir)[0][0]
        collapse_dirs = []
        turn_dir = random.choice([1, -1])
        for i in range(ind, ind + turn_dir * 4, turn_dir):
            collapse_dirs.append(directions[i % 4])
        for i in range(ind, ind + (-turn_dir) * 4, (-turn_dir)):
            collapse_dirs.append(directions[i % 4])

        # print collapse_dirs

        for collapse_dir in collapse_dirs:
            new_mask = collapse_shape(mask, collapse_dir)

        changed = not(np.array_equal(new_mask, mask))
        mask = new_mask

    if is_plot:
        plt.figure()
        plt.imshow(mask, cmap='gray', interpolation='nearest')
        plt.show()

    return mask.astype(np.bool)


def noise_movie(frame_filter, width_filter, height_filter, is_plot=False):
    """
    creating a numpy array with shape [len(frame_filter), len(height_filter), len(width_filter)]

    this array is random noise filtered by these three filters in Fourier domain
    each pixel of the movie have the value in [0. - 1.]
    """

    raw_mov = np.random.rand(len(frame_filter), len(height_filter), len(width_filter))

    raw_mov_fft = np.fft.fftn(raw_mov)

    filter_x = np.repeat(np.array([width_filter]), len(height_filter), axis=0)
    filter_y = np.repeat(np.transpose(np.array([height_filter])), len(width_filter), axis=1)

    filter_xy = filter_x * filter_y

    for i in xrange(raw_mov_fft.shape[0]):
        raw_mov_fft[i] = frame_filter[i] * (raw_mov_fft[i] * filter_xy)

    filtered_mov = np.real(np.fft.ifftn(raw_mov_fft))

    movie = bas.array_nor(filtered_mov)

    if is_plot:
        tf.imshow(movie, vmin=0, vmax=1, cmap='gray')

    return movie


def distance(p0, p1):
    """
    calculate distance between two points, can be multi-dimensinal
    p0 and p1 should be a 1d array, with each element for each dimension
    """

    if not isinstance(p0, np.ndarray):
        p0 = np.array(p0)
    if not isinstance(p1, np.ndarray):
        p1 = np.array(p1)
    if len(p0.shape) != 1:
        raise(ValueError, 'p0 should be one dimensional.')
    if len(p1.shape) != 1:
        raise(ValueError, 'p1 should be one dimensional.')
    return np.sqrt(np.mean(np.square(p0-p1).flatten()))


def center_image_cv2(img, center_pixel, new_size=512, fill_value=0):
    """
    Center a certain image in a new canvas. The pixel defined by 'center_pixel' in the original image will be at the
    center of the output image. The size of output image is defined by 'new_size'. Empty pixels will be filled with
    'fill_value'

    :param img: original image, 2d ndarray
    :param center_pixel: the coordinates of center pixel in original image, [col, row]
    :param new_size: the size of output image
    :param fill_value:
    :return:
    """

    x = new_size / 2 - center_pixel[1]
    y = new_size / 2 - center_pixel[0]

    mat = np.float32([[1, 0, x], [0, 1, y]])

    new_img = cv2.warpAffine(img, mat, (new_size, new_size), borderValue=fill_value)

    return new_img


def resize_image(img, output_shape, fill_value=0.):
    """
    resize every frame of a 3-d matrix to defined output shape
    if the original image is too big it will be truncated
    if the original image is too small, value defined as fillValue will filled in. default: 0
    """

    width = output_shape[1]
    height = output_shape[0]

    if width < 1:
        raise(ValueError, 'width should be bigger than 0!')

    if height < 1:
        raise(ValueError, 'height should be bigger than 0!')

    if len(img.shape) == 2:  # 2-d image
        start_width = img.shape[-1]
        start_height = img.shape[-2]
        new_img = np.array(img)
        if start_width > width:
            new_img = new_img[:, 0:width]
        elif start_width < width:
            attach_right = np.zeros((start_height, width - start_width))
            attach_right[:] = fill_value
            attach_right.astype(img.dtype)
            new_img = np.hstack((new_img, attach_right))

        if start_height > height:
            new_img = new_img[0:height, :]
        elif start_height < height:
            attach_bottom = np.zeros((height - start_height, width))
            attach_bottom[:] = fill_value
            attach_bottom.astype(img.dtype)
            new_img = np.vstack((new_img, attach_bottom))

    elif len(img.shape) == 3:  # 3-d matrix
        start_depth = img.shape[0]
        start_width = img.shape[-1]
        start_height = img.shape[-2]
        new_img = np.array(img)
        if start_width > width:
            new_img = new_img[:, :, 0:width]
        elif start_width < width:
            attach_right = np.zeros((start_depth, start_height, width - start_width))
            attach_right[:] = fill_value
            attach_right.astype(img.dtype)
            new_img = np.concatenate((img, attach_right), axis=2)

        if start_height > height:
            new_img = new_img[:, 0:height, :]
        elif start_height < height:
            attach_bottom = np.zeros((start_depth, height - start_height, width))
            attach_bottom[:] = fill_value
            attach_bottom.astype(img.dtype)
            new_img = np.concatenate((new_img, attach_bottom), axis=1)
            
    else:
        raise(ValueError, 'input image should be a 2-d or 3-d array!')

    return new_img


def expand_image_cv2(img, fill_value=0.):
    """
    expand a given image into a square shape, with length of each dimension equals to the length of the diagonal line 
    of the original image
    """

    if len(img.shape) != 2:
        raise(ValueError, 'Input image should be 2d!')

    dtype = img.dtype
    img = img.astype(np.float32)
    rows, cols = img.shape
    diagonal = int(np.sqrt(rows ** 2 + cols ** 2))
    mat = np.float32([[1, 0, (diagonal-cols) / 2], [0, 1, (diagonal - rows) / 2]])
    new_img = cv2.warpAffine(img, mat, (diagonal, diagonal), borderValue=fill_value)
    return new_img.astype(dtype)


def expand_image(img):
    """
    expand a given image into a square shape, with length of each dimension equals to the length of the diagonal line
    of the original image
    """

    if len(img.shape) == 2:
        rows, cols = img.shape
        diagonal = int(np.sqrt(rows**2+cols**2))
        top = np.zeros(((diagonal-rows)/2, cols), dtype=img.dtype)
        down = np.zeros((diagonal-img.shape[0]-top.shape[0], cols), dtype=img.dtype)
        tall = np.vstack((top, img, down))
        left = np.zeros((tall.shape[0], (diagonal-cols)/2), dtype=img.dtype)
        right = np.zeros((tall.shape[0], diagonal-img.shape[1]-left.shape[1]), dtype=img.dtype)
        new_img = np.hstack((left, tall, right))
        return new_img
    elif len(img.shape) == 3:
        frames, rows, cols = img.shape
        diagonal = int(np.sqrt(rows**2+cols**2))
        top = np.zeros((frames, (diagonal-rows)/2, cols), dtype=img.dtype)
        down = np.zeros((frames, diagonal-img.shape[1]-top.shape[1], cols), dtype=img.dtype)
        tall = np.concatenate((top, img, down), axis=1)
        left = np.zeros((frames, tall.shape[1], (diagonal-cols)/2), dtype=img.dtype)
        right = np.zeros((frames, tall.shape[1], diagonal-img.shape[2]-left.shape[2]), dtype=img.dtype)
        new_img = np.concatenate((left, tall, right), axis=2)
        return new_img
    else:
        raise(ValueError, 'Input image should be 2d or 3d!')


def zoom_image(img, zoom, interpolation='cubic'):
    """
    zoom a 2d image using open cv
    :param img: input image
    :param zoom: if zoom is a single value, it will apply to both axes, if zoom has two values it will be applied to
                 height and width respectively, zoom[0]: height; zoom[1]: width
    :param interpolation: 'cubic','linear','area','nearest','lanczos4'
    :return: zoomed image
    """
    if len(img.shape) != 2:
        raise(ValueError, 'Input image should be 2d!')

    try:
        zoom_h = float(zoom[0])
        zoom_w = float(zoom[1])
    except TypeError:
        zoom_h = float(zoom)
        zoom_w = float(zoom)

    if interpolation == 'cubic':
        interpo = cv2.INTER_CUBIC
    elif interpolation == 'linear':
        interpo = cv2.INTER_LINEAR
    elif interpolation == 'area':
        interpo = cv2.INTER_AREA
    elif interpolation == 'nearest':
        interpo = cv2.INTER_NEAREST
    elif interpolation == 'lanczos4':
        interpo = cv2.INTER_LANCZOS4
    else:
        raise(LookupError, 'unrecognized interpolation method, should be one of "cubic","linear","area","nearest",'
                           '"lanczos4".')

    new_img = cv2.resize(img.astype(np.float), dsize=(int(img.shape[1] * zoom_w), int(img.shape[0] * zoom_h)),
                         interpolation=interpo)
    return new_img


def move_image(img, xoffset, yoffset, width, height, fill_value=0.0):
    """
    move image defined by xoffset and yoffset using open cv

    new canvas size is defined by width and height

    empty pixels will be filled with fill_value
    """

    if len(img.shape) != 2:
        raise(ValueError, 'Input image should be 2d!')

    mat = np.float32([[1, 0, xoffset], [0, 1, yoffset]])

    new_img = cv2.warpAffine(img, mat, (width, height), borderValue=fill_value)

    return new_img.astype(img.dtype)


def rotate_image(img, angle, fill_value=0.0):
    """
    rotate an image conter-clockwise by an angle around its center

    :param img: input image
    :param angle: in degree
    :param fill_value: value for empty pixels
    :return: rotated image
    """

    if len(img.shape) != 2:
        raise(ValueError, 'Input image should be 2d!')

    rows, cols = img.shape

    mat = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    new_img = cv2.warpAffine(img, mat, (cols, rows), borderValue=fill_value)

    return new_img


def rigid_transform(img, zoom=None, rotation=None, offset=None, output_shape=None, mode='constant', fill_value=0.0):

    """
    rigid transformation of a 2d-image or 3d-matrix by using scipy
    :param img: input image/matrix
    :param zoom: if zoom is a single value, it will apply to both axes, if zoom has two values it will be applied to
                 height and width respectively, zoom[0]: height; zoom[1]: width
    :param rotation: in degree, counterclock wise
    :param offset: tuple (xoffset, yoffset) pixel value of starting point of output image
    :param output_shape: the shape of output image, (height, width)
    :param mode: str, optional, Points outside the boundaries of the input are filled according to the given mode
                 ('constant', 'nearest', 'reflect' or 'wrap'). Default is 'constant'.
    :param fill_value: scalar, optional, Value used for points outside the boundaries of the input if mode='constant'.
                 Default is 0.0
    :return: new image or matrix after transformation
    """

    if len(img.shape) != 2 and len(img.shape) != 3:
        raise(ValueError, 'Input image is not a 2d or 3d array!')

    new_img = img.astype(np.float32)

    if zoom:
        if len(img.shape) == 2:
            new_zoom = (zoom, zoom)
        elif len(img.shape) == 3:
            new_zoom = (1, zoom, zoom)
        else:
            raise(ValueError, 'Input image is not a 2d or 3d array!')
        new_img = ni.zoom(new_img, zoom=new_zoom, mode=mode, cval=fill_value)

    if rotation:
        new_img = expand_image(new_img)
        if len(img.shape) == 2:
            new_img = ni.rotate(new_img, angle=rotation, reshape=False, mode=mode, cval=fill_value)
        elif len(img.shape) == 3:
            new_img = ni.rotate(new_img, angle=rotation, axes=(1, 2), reshape=False, mode=mode, cval=fill_value)

    if offset:
        if len(img.shape) == 2:
            new_img = ni.shift(new_img, (offset[1], offset[0]), mode=mode, cval=fill_value)
        if len(img.shape) == 3:
            new_img = ni.shift(new_img, (0, offset[1], offset[0]), mode=mode, cval=fill_value)

    if output_shape:
        new_img = resize_image(new_img, output_shape)

    return new_img.astype(img.dtype)


def rigid_transform_cv2_2d(img, zoom=None, rotation=None, offset=None, output_shape=None, fill_value=None):
    """
    rigid transformation of a 2d-image by using opencv
    :param img: input image/matrix
    :param zoom: if zoom is a single value, it will apply to both axes, if zoom has two values it will be applied to
                 height and width respectively, zoom[0]: height; zoom[1]: width
    :param rotation: in degree, counterclock wise
    :param offset: tuple (xoffset, yoffset) pixel value of starting point of output image
    :param output_shape: the shape of output image, (height, width)
    :param fill_value: value to fill empty pixels, default: minimum value of input image
    :return: new image or matrix after transformation
    """

    if len(img.shape) != 2:
        raise(LookupError, 'Input image is not a 2d or 3d array!')

    new_img = np.array(img).astype(np.float)

    if fill_value is None:
        fill_value = np.amin(new_img)

    if zoom:
        new_img = zoom_image(img, zoom=zoom)

    if rotation:
        new_img = expand_image_cv2(new_img)
        new_img = rotate_image(new_img, rotation, fill_value=fill_value)

    if (output_shape is None) and (offset is None):
        return new_img
    else:
        if output_shape is None:
            output_shape = new_img.shape
        if offset is None:
            offset = (0, 0)
        new_img = move_image(new_img, offset[0], offset[1], output_shape[1], output_shape[0], fill_value=fill_value)

        return new_img.astype(img.dtype)


def rigid_transform_cv2_3d(img, zoom=None, rotation=None, offset=None, output_shape=None, fill_value=None):
    """
    rigid transform a 3d matrix by using rigid_transform_cv2_2d function above
    """

    if len(img.shape) != 3:
        raise(LookupError, 'Input image is not a 3d array!')

    if not output_shape:
        if zoom:
            new_height = int(img.shape[1] * zoom)
            new_width = int(img.shape[2] * zoom)
        else:
            new_height = img.shape[1]
            new_width = img.shape[2]
    else:
        new_height = output_shape[0]
        new_width = output_shape[1]
    new_img = np.empty((img.shape[0], new_height, new_width), dtype=img.dtype)

    for i in range(img.shape[0]):
        new_img[i, :, :] = rigid_transform_cv2_2d(img[i, :, :], zoom=zoom, rotation=rotation, offset=offset,
                                                  output_shape=output_shape, fill_value=fill_value)

    return new_img


def rigid_transform_cv2(img, zoom=None, rotation=None, offset=None, output_shape=None, fill_value=None):
    """
    rigid transformation of a 2d-image or 3d-matrix by using opencv
    :param img: input image/matrix
    :param zoom: if zoom is a single value, it will apply to both axes, if zoom has two values it will be applied to
                 height and width respectively, zoom[0]: height; zoom[1]: width
    :param rotation: in degree, counterclock wise
    :param offset: tuple (xoffset, yoffset) pixel value of starting point of output image
    :param output_shape: the shape of output image, (height, width)
    :param fill_value: value to fill empty pixels, default: minimum value of input image
    :return: new image or matrix after transformation
    """

    if len(img.shape) == 2:
        return rigid_transform_cv2_2d(img, zoom=zoom, rotation=rotation, offset=offset, output_shape=output_shape,
                                      fill_value=fill_value)
    elif len(img.shape) == 3:
        return rigid_transform_cv2_3d(img, zoom=zoom, rotation=rotation, offset=offset, output_shape=output_shape,
                                      fill_value=fill_value)
    else:
        raise(ValueError, 'Input image is not a 2d or 3d array!')



class Mask(sparse.coo_matrix):
    """
    2-d mask class, a sub class of scipy.sparse.coo_matrix
    """

    def __init__(self, *args, **kwargs):
        super(Mask, self).__init__(*args, **kwargs)

    def get_row(self):
        return self.row

    def get_col(self):
        return self.col

    def get_data(self):
        return self.data

    def get_triplets(self):
        return np.array([self.row, self.col, self.data]).transpose().astype(np.float)

    def get_indices(self):
        return np.array([self.row, self.col]).transpose().astype(np.int)

    def get_mean(self):
        """
        :return: the mean value of all pixels in the mask
        """
        return np.mean(self.data)

    def get_binary_dense_mask(self):
        binary_mask = np.zeros(self.get_shape(), dtype=np.bool)
        binary_mask[(self.row, self.col)] = True
        return binary_mask

    def get_binary_center(self):
        return np.mean(self.row.astype(np.float)), np.mean(self.col.astype(np.float))

    def get_weighted_center(self):
        i = np.sum(np.multiply(self.row, self.data.astype(np.float))) / np.sum(self.data.astype(np.float))
        j = np.sum(np.multiply(self.col, self.data.astype(np.float))) / np.sum(self.data.astype(np.float))
        return i, j

    def check_hole(self, connectivity=CONNECTIVITY):
        """
        check if there are holes in the binary image
        :return: bool
        """
        return check_hole(self.get_binary_dense_mask(), connectivity)

    def grow_pixels(self, coors, values):
        """
        grow pixels defined by coors and value, if there are duplications the values of the same pixel will be sumed
        :param coors: coordinates for the pixels to be added, tuple of two lists, ([rows], [cols])
        :param values: values for the pixels to be added, list
        """
        row = np.hstack((self.row, coors[0]))
        col = np.hstack((self.col, coors[1]))
        data = np.hstack((self.data, values))
        self.__init__((data, (row, col)), shape=self.get_shape())

    def find_border_pixels(self, connectivity=CONNECTIVITY):
        """
        return a list of pixel indices bordering the given mask
        :param connectivity: 4 or 8
        :return: pixel indices of the border
        """
        if connectivity == 4:
            structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        elif connectivity == 8:
            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        else:
            raise (ValueError, 'connectivity should be 4 or 8.')
        mask = self.get_binary_dense_mask()
        big_mask = ni.binary_dilation(mask, structure)
        border = big_mask - mask
        return np.where(border)

    def grow_one_connected_pixel(self, value=1., connectivity=CONNECTIVITY):
        """
        increase one pixel on the border with existing binary mask
        :connectivity: 4 or 8
        :return: binary mask
        """
        row = list(self.get_row())
        col = list(self.get_col())
        data = list(self.get_data())

        border_pixels = self.find_border_pixels(connectivity)
        random_ind = random.randrange(len(border_pixels[0]))
        row.append(border_pixels[0][random_ind])
        col.append(border_pixels[1][random_ind])
        data.append(value)

        self.__init__((data, (row, col)), shape=self.get_shape())

    def reorder(self):
        """
        reorder the indices of every element in the mask
        """
        new_mask = sparse.coo_matrix(self.todense())
        self.row = new_mask.row
        self.col = new_mask.col
        self.data = new_mask.data

    def to_h5(self, h5_group, is_reorder=False):
        """
        Generate a hdf5 group for saving
        """

        if is_reorder:
            self.reorder()

        h5_group.attrs['type'] = 'coo_mask'
        h5_group.attrs['shape'] = self._shape
        h5_group.create_dataset('row', data=self.get_row())
        h5_group.create_dataset('col', data=self.get_col())
        h5_group.create_dataset('data', data=self.get_data())

    @staticmethod
    def from_h5(h5_group):
        """
        generate the Mask object from a saved h5_group
        """

        shape = h5_group.attrs['shape']
        row = h5_group['row'][:]
        col = h5_group['col'][:]
        data = h5_group['data'][:]

        mask = Mask((data, (row, col)), shape=shape)

        return mask


if __name__ == '__main__':

    # -----------------------------------------------------------
    # mask = np.zeros((50,50),dtype=np.bool)
    # mask[20, 13] = True
    # for i in range(5):
    #     mask = grow_one_connected_pixel(mask)
    #     plt.imshow(mask,cmap='gray',interpolation='nearest')
    #     plt.show()
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # border_ind = find_border(pixel_ind=([38,39],[59,59]),frame_shape=(128,128))
    # print border_ind
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # img = np.zeros((50,50),dtype=np.bool)
    # img[(5, 6, 5, 4), (5, 6, 7, 6)] = True
    # plt.imshow(img,interpolation='nearest')
    # plt.show()
    # print check_hole(img)
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # mask = random_filled_shape2(pixel_num=500, is_plot=True)
    # print np.sum(mask)
    # print check_hole(mask)
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # mask = random_filled_shape(pixel_num=30, is_plot=True)
    # print np.sum(mask)
    # print check_hole(mask)
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # arr = np.zeros(5, dtype=np.bool)
    # arr[2] = True
    # arr[4] = True
    # arr2 = collapse_line(arr)
    # print arr
    # print arr2
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # mask = np.zeros((50,50),dtype=np.bool)
    # mask[20, 13] = True
    # for i in range(20):
    #     mask = grow_one_connected_pixel(mask)
    #
    # mask2 = collapse_shape(mask, 'l')
    #
    # f = plt.figure(figsize=(15, 6))
    # ax1 = f.add_subplot(121)
    # ax1.imshow(mask,cmap='gray',interpolation='nearest')
    # ax2 = f.add_subplot(122)
    # ax2.imshow(mask2, cmap='gray', interpolation='nearest')
    # plt.show()
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # mat = np.zeros((10, 10))
    # mat[(6,7,3,4,3,5), (2,3,5,8,0,8)] = 2
    # print mat
    # mask = sparse.coo_matrix(mat)
    # mask = Mask(mat)
    # print mask.get_indices()
    # print mask.get_binary_dense_mask()
    # print mask.get_binary_center()
    # print mask.get_weighted_center()
    # mask.grow_pixels(([7, 7], [8, 4]), [5.5, 6])
    # print mask.get_row()
    # print mask.get_col()
    # print mask.get_data()
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # p1 = [2, 8]
    # p2 = [9, 3]
    # mask = np.zeros((10, 10))
    # mask[binary_line(p1, p2)] = 1
    # mask[p1[0], p1[1]] = 2
    # mask[p2[0], p2[1]] = 2
    # plt.imshow(mask, interpolation='nearest')
    # plt.show()
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # p1 = [8, 2]
    # angle = np.pi / 3
    # distance = 2
    # mask = np.zeros((10, 10))
    # mask[binary_ray(p1, angle, distance)] = 1
    # mask[p1[0], p1[1]] = 2
    # plt.imshow(mask, interpolation='nearest')
    # plt.show()
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # p1 = [8, 2]
    # angle = np.pi / 3
    # distance = 20
    # mask = np.zeros((10, 10))
    # pixels = binary_ray(p1, angle, distance)
    # pixels = zip(*pixels)
    # pixels = [p for p in pixels if p[0] < 10 and p[1] < 10]
    # pixels = zip(*pixels)
    # mask[pixels] = 1
    # mask[p1[0], p1[1]] = 2
    # plt.imshow(mask, interpolation='nearest')
    # plt.show()
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    print binary_line([2, 8], [2, 4])
    # -----------------------------------------------------------
