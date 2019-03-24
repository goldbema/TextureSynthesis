'''
This module is a Python implementation of:

    A. Efros and T. Leung, "Texture Synthesis by Non-parametric Sampling,"
    Proceedings of the Seventh IEEE International Conference on Computer
    Vision, September 1999.

Specifically, this module implements texture synthesis by growing a 3x3 texture patch 
pixel-by-pixel. Please see the authors' project page for additional algorithm details: 

    https://people.eecs.berkeley.edu/~efros/research/EfrosLeung.html

Example:

    Generate a 50x50 texture patch from a texture available at the input path and save it to
    the output path. Also, visualize the synthesis process:

        $ python synthesis.py --sample_path=[input path] --out_path=[output path] --visualize

'''

__author__ = 'Maxwell Goldberg'

import argparse
import cv2
import numpy as np

EIGHT_CONNECTED_NEIGHBOR_KERNEL = np.array([[1., 1., 1.],
                                            [1., 0., 1.],
                                            [1., 1., 1.]], dtype=np.float64)
SIGMA_COEFF = 6.4      # The denominator for a 2D Gaussian sigma used in the reference implementation.
ERROR_THRESHOLD = 0.1  # The default error threshold for synthesis acceptance in the reference implementation.


def normalized_ssd(sample, window, mask):
    wh, ww = window.shape
    sh, sw = sample.shape

    # Get sliding window views of the sample, window, and mask.
    strided_sample = np.lib.stride_tricks.as_strided(sample, shape=((sh-wh+1), (sw-ww+1), wh, ww), 
                        strides=(sample.strides[0], sample.strides[1], sample.strides[0], sample.strides[1]))
    strided_sample = strided_sample.reshape(-1, wh, ww)

    # Note that the window and mask views have the same shape as the strided sample, but the kernel is fixed
    # rather than sliding for each of these components.
    strided_window = np.lib.stride_tricks.as_strided(window, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, window.strides[0], window.strides[1]))
    strided_window = strided_window.reshape(-1, wh, ww)

    strided_mask = np.lib.stride_tricks.as_strided(mask, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, mask.strides[0], mask.strides[1]))
    strided_mask = strided_mask.reshape(-1, wh, ww)

    # Form a 2D Gaussian weight matrix from symmetric linearly separable Gaussian kernels and generate a 
    # strided view over this matrix.
    sigma = wh / SIGMA_COEFF
    kernel = cv2.getGaussianKernel(ksize=wh, sigma=sigma)
    kernel_2d = kernel * kernel.T

    strided_kernel = np.lib.stride_tricks.as_strided(kernel_2d, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, kernel_2d.strides[0], kernel_2d.strides[1]))
    strided_kernel = strided_kernel.reshape(-1, wh, ww)

    # Take the sum of squared differences over all sliding sample windows and weight it so that only existing neighbors
    # contribute to error. Use the Gaussian kernel to weight central values more strongly than distant neighbors.
    squared_differences = ((strided_sample - strided_window)**2) * strided_kernel * strided_mask
    ssd = np.sum(squared_differences, axis=(1,2))
    ssd = ssd.reshape(sh-wh+1, sw-ww+1)

    # Normalize the SSD by the maximum possible contribution.
    total_ssd = np.sum(mask * kernel_2d)
    normalized_ssd = ssd / total_ssd

    return normalized_ssd

def get_candidate_indices(normalized_ssd, error_threshold=ERROR_THRESHOLD):
    min_ssd = np.min(normalized_ssd)
    min_threshold = min_ssd * (1. + error_threshold)
    indices = np.where(normalized_ssd <= min_threshold)
    return indices

def select_pixel_index(normalized_ssd, indices, method='uniform'):
    N = indices[0].shape[0]

    if method == 'uniform':
        weights = np.ones(N) / float(N)
    else:
        weights = normalized_ssd[indices]
        weights = weights / np.sum(weights)

    # Select a random pixel index from the index list.
    selection = np.random.choice(np.arange(N), size=1, p=weights)
    selected_index = (indices[0][selection], indices[1][selection])
    
    return selected_index

def get_neighboring_pixel_indices(pixel_mask):
    # Taking the difference between the dilated mask and the initial mask
    # gives only the 8-connected neighbors of the mask frontier.
    kernel = np.ones((3,3))
    dilated_mask = cv2.dilate(pixel_mask, kernel, iterations=1)
    neighbors = dilated_mask - pixel_mask

    # Recover the indices of the mask frontier.
    neighbor_indices = np.nonzero(neighbors)

    return neighbor_indices

def permute_neighbors(pixel_mask, neighbors):
    N = neighbors[0].shape[0]

    # Generate a permutation of the neigboring indices
    permuted_indices = np.random.permutation(np.arange(N))
    permuted_neighbors = (neighbors[0][permuted_indices], neighbors[1][permuted_indices])

    # Use convolution to count the number of existing neighbors for all entries in the mask.
    neighbor_count = cv2.filter2D(pixel_mask, ddepth=-1, kernel=EIGHT_CONNECTED_NEIGHBOR_KERNEL, borderType=cv2.BORDER_CONSTANT)

    # Sort the permuted neighboring indices by quantity of existing neighbors descending.
    permuted_neighbor_counts = neighbor_count[permuted_neighbors]

    sorted_order = np.argsort(permuted_neighbor_counts)[::-1]
    permuted_neighbors = (permuted_neighbors[0][sorted_order], permuted_neighbors[1][sorted_order])

    return permuted_neighbors

def texture_can_be_synthesized(mask):
    # The texture can be synthesized while the mask has unfilled entries.
    mh, mw = mask.shape[:2]
    num_completed = np.count_nonzero(mask)
    num_incomplete = (mh * mw) - num_completed
    
    return num_incomplete > 0

def initialize_texture_synthesis(original_sample, window_size, kernel_size):
    # Convert original to sample representation.
    sample = cv2.cvtColor(original_sample, cv2.COLOR_BGR2GRAY)
    
    # Convert sample to floating point and normalize to the range [0., 1.]
    sample = sample.astype(np.float64)
    sample = sample / 255.

    # Generate window
    window = np.zeros(window_size, dtype=np.float64)

    # Generate output window
    if original_sample.ndim == 2:
        result_window = np.zeros_like(window, dtype=np.uint8)
    else:
        result_window = np.zeros(window_size + (3,), dtype=np.uint8)

    # Generate window mask
    h, w = window.shape
    mask = np.zeros((h, w), dtype=np.float64)

    # Initialize window with random seed from sample
    sh, sw = original_sample.shape[:2]
    ih = np.random.randint(sh-3+1)
    iw = np.random.randint(sw-3+1)
    seed = sample[ih:ih+3, iw:iw+3]

    # Place seed in center of window
    ph, pw = (h//2)-1, (w//2)-1
    window[ph:ph+3, pw:pw+3] = seed
    mask[ph:ph+3, pw:pw+3] = 1
    result_window[ph:ph+3, pw:pw+3] = original_sample[ih:ih+3, iw:iw+3]

    # Obtain padded versions of window and mask
    win = kernel_size//2
    padded_window = cv2.copyMakeBorder(window, 
                                       top=win, bottom=win, left=win, right=win, borderType=cv2.BORDER_CONSTANT, value=0.)
    padded_mask = cv2.copyMakeBorder(mask,
                                     top=win, bottom=win, left=win, right=win, borderType=cv2.BORDER_CONSTANT, value=0.)
    
    # Obtain views of the padded window and mask
    window = padded_window[win:-win, win:-win]
    mask = padded_mask[win:-win, win:-win]

    return sample, window, mask, padded_window, padded_mask, result_window
    
def synthesize_texture(original_sample, window_size, kernel_size, visualize):
    global gif_count
    (sample, window, mask, padded_window, 
        padded_mask, result_window) = initialize_texture_synthesis(original_sample, window_size, kernel_size)

    # Synthesize texture until all pixels in the window are filled.
    while texture_can_be_synthesized(mask):
        # Get neighboring indices
        neighboring_indices = get_neighboring_pixel_indices(mask)

        # Permute and sort neighboring indices by quantity of 8-connected neighbors.
        neighboring_indices = permute_neighbors(mask, neighboring_indices)
        
        for ch, cw in zip(neighboring_indices[0], neighboring_indices[1]):

            window_slice = padded_window[ch:ch+kernel_size, cw:cw+kernel_size]
            mask_slice = padded_mask[ch:ch+kernel_size, cw:cw+kernel_size]

            # Compute SSD for the current pixel neighborhood and select an index with low error.
            ssd = normalized_ssd(sample, window_slice, mask_slice)
            indices = get_candidate_indices(ssd)
            selected_index = select_pixel_index(ssd, indices)

            # Translate index to accommodate padding.
            selected_index = (selected_index[0] + kernel_size // 2, selected_index[1] + kernel_size // 2)

            # Set windows and mask.
            window[ch, cw] = sample[selected_index]
            mask[ch, cw] = 1
            result_window[ch, cw] = original_sample[selected_index[0], selected_index[1]]

            if visualize:
                cv2.imshow('synthesis window', result_window)
                key = cv2.waitKey(1) 
                if key == 27:
                    cv2.destroyAllWindows()
                    return result_window

    if visualize:
        cv2.imshow('synthesis window', result_window)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_window


def validate_args(args):
    wh, ww = args.window_height, args.window_width
    if wh < 3 or ww < 3:
        raise ValueError('window_size must be greater than or equal to (3,3).')

    if args.kernel_size <= 1:
        raise ValueError('kernel size must be greater than 1.')

    if args.kernel_size % 2 == 0:
        raise ValueError('kernel size must be odd.')

    if args.kernel_size > min(wh, ww):
        raise ValueError('kernel size must be less than or equal to the smaller window_size dimension.')

def parse_args():
    parser = argparse.ArgumentParser(description='Perform texture synthesis')
    parser.add_argument('--sample_path', type=str, required=True, help='Path to the texture sample')
    parser.add_argument('--out_path', type=str, required=False, help='Output path for synthesized texture')
    parser.add_argument('--window_height', type=int,  required=False, default=50, help='Height of the synthesis window')
    parser.add_argument('--window_width', type=int, required=False, default=50, help='Width of the synthesis window')
    parser.add_argument('--kernel_size', type=int, required=False, default=11, help='One dimension of the square synthesis kernel')
    parser.add_argument('--visualize', required=False, action='store_true', help='Visualize the synthesis process')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    sample = cv2.imread(args.sample_path)
    if sample is None:
        raise ValueError('Unable to read image from sample_path.')

    validate_args(args)

    synthesized_texture = synthesize_texture(original_sample=sample, 
                                             window_size=(args.window_height, args.window_width), 
                                             kernel_size=args.kernel_size, 
                                             visualize=args.visualize)

    if args.out_path is not None:
        cv2.imwrite(args.out_path, synthesized_texture)

if __name__ == '__main__':
    main()