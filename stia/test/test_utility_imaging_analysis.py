import os
import stia.utility.image_analysis as ia
import matplotlib.pyplot as plt
import numpy as np
import h5py

TEST_DATA_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
MASK_MATRIX = np.load(os.path.join(TEST_DATA_FOLDER, 'mask_matrix.npy'))


def test_check_hole_1():
    img = np.zeros((50, 50), dtype=np.bool)
    img[(5, 6, 5, 4), (5, 6, 7, 6)] = True
    assert(ia.check_hole(img) == True)


def test_check_hole_2():
    img = np.zeros((50, 50), dtype=np.bool)
    img[(5, 6,), (5, 6)] = True
    assert(ia.check_hole(img) == False)


def test_random_filled_shape():
    mask = ia.random_filled_shape(pixel_num=20)
    assert(np.sum(mask) == 20)
    assert(ia.check_hole(mask) == False)


def test_collapse_line():
    arr = np.zeros(5, dtype=np.bool)
    assert (np.array_equal(ia.collapse_line(arr), [False, False, False, False, False]))
    arr[2] = True
    assert (np.array_equal(ia.collapse_line(arr), [False, False, True, False, False]))
    arr[4] = True
    assert (np.array_equal(ia.collapse_line(arr), [False, False, True, True, False]))


def test_collapse_shape():
    arr = np.zeros((5, 5), dtype=np.bool)
    arr[1, 1] = True
    arr[1, 2] = True
    arr[2, 1] = True
    arr[2, 3] = True
    arr[3, 2] = True
    assert(np.array_equal(ia.collapse_shape(arr, 'u'), [[False, False, False, False, False],
                                                        [False,  True,  True, False, False],
                                                        [False,  True,  True,  True, False],
                                                        [False, False,  False, False, False],
                                                        [False, False, False, False, False]]))

    assert (np.array_equal(ia.collapse_shape(arr, 'd'), [[False, False, False, False, False],
                                                         [False,  True, False, False, False],
                                                         [False,  True,  True,  True, False],
                                                         [False, False,  True, False, False],
                                                         [False, False, False, False, False]]))

    assert (np.array_equal(ia.collapse_shape(arr, 'l'), [[False, False, False, False, False],
                                                         [False,  True,  True, False, False],
                                                         [False,  True,  True, False, False],
                                                         [False, False,  True, False, False],
                                                         [False, False, False, False, False]]))

    assert (np.array_equal(ia.collapse_shape(arr, 'r'), [[False, False, False, False, False],
                                                         [False,  True,  True, False, False],
                                                         [False, False,  True,  True, False],
                                                         [False, False,  True, False, False],
                                                         [False, False, False, False, False]]))


def test_is_pixel_convex():
    mask = np.zeros((10, 10), dtype=np.bool)
    mask[(4, 5, 6, 5, 6, 7, 5, 6, 5, 5), (4, 4, 4, 5, 5, 5, 6, 6, 7, 8)] = True

    # plt.imshow(mask, cmap='gray', interpolation='nearest')
    # plt.show()

    assert(ia.is_pixel_convex(mask, (3, 4)) == True)
    assert(ia.is_pixel_convex(mask, (3, 3)) == True)
    assert(ia.is_pixel_convex(mask, (4, 3)) == True)
    assert(ia.is_pixel_convex(mask, (5, 3)) == True)
    assert(ia.is_pixel_convex(mask, (6, 3)) == True)
    assert(ia.is_pixel_convex(mask, (7, 6)) == True)
    assert(ia.is_pixel_convex(mask, (7, 4)) == True)
    assert(ia.is_pixel_convex(mask, (8, 5)) == True)
    assert(ia.is_pixel_convex(mask, (6, 7)) == True)
    assert(ia.is_pixel_convex(mask, (5, 9)) == True)
    assert(ia.is_pixel_convex(mask, (4, 5)) == True)
    assert(ia.is_pixel_convex(mask, (8, 2)) == True)
    assert(ia.is_pixel_convex(mask, (9, 2)) == True)
    assert(ia.is_pixel_convex(mask, (3, 9)) == True)

    assert(ia.is_pixel_convex(mask, (2, 4)) == False)
    assert(ia.is_pixel_convex(mask, (4, 2)) == False)
    assert(ia.is_pixel_convex(mask, (5, 2)) == False)
    assert(ia.is_pixel_convex(mask, (6, 2)) == False)
    assert(ia.is_pixel_convex(mask, (7, 2)) == False)
    assert(ia.is_pixel_convex(mask, (8, 4)) == False)
    assert(ia.is_pixel_convex(mask, (9, 4)) == False)
    assert(ia.is_pixel_convex(mask, (9, 5)) == False)
    assert(ia.is_pixel_convex(mask, (8, 6)) == False)
    assert(ia.is_pixel_convex(mask, (9, 6)) == False)
    assert(ia.is_pixel_convex(mask, (7, 7)) == False)
    assert(ia.is_pixel_convex(mask, (8, 7)) == False)
    assert(ia.is_pixel_convex(mask, (9, 7)) == False)
    assert(ia.is_pixel_convex(mask, (7, 8)) == False)
    assert(ia.is_pixel_convex(mask, (8, 8)) == False)
    assert(ia.is_pixel_convex(mask, (9, 8)) == False)
    assert(ia.is_pixel_convex(mask, (4, 9)) == False)
    assert(ia.is_pixel_convex(mask, (3, 8)) == False)
    assert(ia.is_pixel_convex(mask, (3, 7)) == False)
    assert(ia.is_pixel_convex(mask, (3, 6)) == False)
    assert(ia.is_pixel_convex(mask, (3, 5)) == False)
    assert(ia.is_pixel_convex(mask, (6, 8)) == False)
    assert(ia.is_pixel_convex(mask, (4, 7)) == False)
    assert(ia.is_pixel_convex(mask, (4, 6)) == False)
    assert(ia.is_pixel_convex(mask, (4, 8)) == False)


def test_binary_line():
    p1 = (2, 8)
    p2 = (9, 3)
    pixels = ia.binary_line(p1, p2)
    assert(np.array_equal(pixels, ([2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 9], [8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3])))
    assert(np.array_equal(ia.binary_line([2, 8], [2, 4]), ([2, 2, 2, 2], [8, 7, 6, 5])))


def test_binary_ray():
    p1 = [2, 8]
    ray1 = ia.binary_ray(p1, np.pi, 4)
    assert(np.array_equal(ray1, ([2, 2, 2, 2], [8, 7, 6, 5])))

    ray2 = ia.binary_ray(p1, 1.5 * np.pi, 18, frame_shape=(10, 10))
    assert(np.array_equal(ray2, [(2, 3, 4, 5, 6, 7, 8, 9), (8, 8, 8, 8, 8, 8, 8, 8)]))

    p2 = [9, 5]
    ray2 = ia.binary_ray(p2, 1.5 * np.pi, 18, frame_shape=(10, 10))
    assert (np.array_equal(ray2, [(9,), (5,)]))


def test_mask_get_data():
    mask = ia.Mask(MASK_MATRIX)
    assert(np.array_equal(mask.get_row(), [1, 3, 3, 4, 5, 6, 7]))
    assert(np.array_equal(mask.get_col(), [9, 0, 5, 8, 8, 2, 3]))
    assert(np.array_equal(mask.get_data(), [8., 2., 2., 2., 2., 2., 2.]))


def test_mask_get_indices():
    mask = ia.Mask(MASK_MATRIX)
    assert(np.array_equal(mask.get_indices(), [[1, 9], [3, 0], [3, 5], [4, 8], [5, 8], [6, 2], [7, 3]]))


def test_mask_get_binary_dense_mask():
    mask = ia.Mask(MASK_MATRIX)
    binary_dense_mask = mask.get_binary_dense_mask()
    assert(np.array_equal(binary_dense_mask, MASK_MATRIX != 0))


def test_mask_get_binary_center():
    mask = ia.Mask(MASK_MATRIX)
    binary_center = mask.get_binary_center()
    assert(abs(binary_center[0] - 4.1428571428571432) < 10e-16)
    assert(abs(binary_center[1] - 5.0) < 10e-16)


def test_mask_get_weight_center():
    mask = ia.Mask(MASK_MATRIX)
    weight_center = mask.get_weighted_center()
    assert(abs(weight_center[0] - 3.2) < 10e-16)
    assert(abs(weight_center[1] - 6.2) < 10e-16)


def test_mask_grow_pixels():
    mask = ia.Mask(MASK_MATRIX)
    mask.grow_pixels(([8, 4], [7, 7]), [5.5, 6])
    assert(np.array_equal(mask.get_row(), [1, 3, 3, 4, 5, 6, 7, 8, 4]))
    assert(np.array_equal(mask.get_col(), [9, 0, 5, 8, 8, 2, 3, 7, 7]))
    assert(np.array_equal(mask.get_data(), [8., 2., 2., 2., 2., 2., 2., 5.5, 6.]))
    assert(mask.todense()[8, 7] == 5.5)
    assert(mask.todense()[4, 7] == 6)


def test_mask_check_hole():
    img = np.zeros((50, 50), dtype=np.bool)
    img[(5, 6, 5, 4), (5, 6, 7, 6)] = True
    assert(ia.Mask(img).check_hole() == True)

    img = np.zeros((50, 50), dtype=np.bool)
    img[(5, 6,), (5, 6)] = True
    assert (ia.Mask(img).check_hole() == False)


def test_mask_reorder():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[(5, 6, 5, 4), (5, 6, 7, 6)] = [1, 2, 3, 4]
    mask = ia.Mask(img)
    mask.row = np.hstack((mask.row, [3]))
    mask.col = np.hstack((mask.col, [4]))
    mask.data = np.hstack((mask.data, [9]))

    mask.reorder()

    assert(np.array_equal(mask.get_row(), [3, 4, 5, 5, 6]))
    assert (np.array_equal(mask.get_col(), [4, 6, 5, 7, 6]))
    assert (np.array_equal(mask.get_data(), [9, 4, 1, 3, 2]))


def test_mask_h5():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[(5, 6, 5, 4), (5, 6, 7, 6)] = [1, 2, 3, 4]
    mask = ia.Mask(img)

    data_path = os.path.join(TEST_DATA_FOLDER, 'mask_h5.hdf5')

    mask_h5 = h5py.File(data_path)
    mask_h5.create_group('mask')
    mask.to_h5(mask_h5['mask'])
    mask_h5.close()

    mask_h5 = h5py.File(data_path)
    loaded_mask = ia.Mask.from_h5(mask_h5['mask'])
    assert(np.array_equal(loaded_mask.row, [4, 5, 5, 6]))
    assert(np.array_equal(loaded_mask.col, [6, 5, 7, 6]))
    assert(np.array_equal(loaded_mask.data, [4, 1, 3, 2]))
    assert(np.array_equal(loaded_mask.shape, [50, 50]))
    mask_h5.close()

    os.remove(data_path)


def run_test():
    test_check_hole_1()
    test_check_hole_2()
    test_random_filled_shape()
    test_collapse_line()
    test_collapse_shape()
    test_is_pixel_convex()
    test_binary_line()
    test_binary_ray()

    # test for Mask class
    test_mask_get_data()
    test_mask_get_indices()
    test_mask_get_binary_dense_mask()
    test_mask_get_binary_center()
    test_mask_get_weight_center()
    test_mask_grow_pixels()
    test_mask_check_hole()
    test_mask_reorder()
    test_mask_h5()


if __name__ == '__main__':

    run_test()
