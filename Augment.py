import numpy as np
import random as rand

class Compose():
    def __init__(self, matrix, **kwargs):
        self.result = matrix
        available_funcs = ['flipHorizontal', 'flipVertical', 'rotate', 
                           'perspectiveShift', 'cropResize', 'cropPaste',
                           'gaussianBlur', 'gaussianNoise', 'adjustSharpness',
                           'adjustBrightness']
        
        for key in kwargs:
            if key not in available_funcs:
                print(f'{key} is not an available function. Exiting')
                break
            params = kwargs[key]
            function = getattr(self, key)
            if callable(function):
                output = function(self.result, params)
                self.result = output

    def flipHorizontal(self, matrix, params):
        prob = params[0]
        if 1-prob < rand.random():
            matrix = matrix[:, ::-1]
        return matrix

    def flipVertical(self, matrix, params):
        prob = params[0]
        if 1-prob < rand.random():
            matrix = matrix[::-1, :]
        return matrix

    def rotate(self, matrix, params):
        degrees_range, prob = params
        # rotation matrix
        if 1-prob < rand.random():
            theta = rand.randint(degrees_range[0], degrees_range[1]) * (np.pi/180) # rotation in radians
            
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            
            rotated_matrix = np.zeros_like(matrix)
            h, w = matrix.shape
            cx, cy = w//2, h//2
            for i in range(h):
                for j in range(w):
                    x, y = j - cx, i - cy
                    rotated_point = np.dot(np.array([x, y]), rot)
                    new_x, new_y = rotated_point + np.array([cx, cy])
                    new_x, new_y = int(round(new_x)), int(round(new_y))

                    if (0<=new_y<h) and (0<=new_x<w):
                        rotated_matrix[new_y, new_x] = matrix[i, j]

            return rotated_matrix
        
        else:
            return matrix

    def perspectiveShift(self, matrix, params):
        rows, cols = matrix.shape
        row_shift, col_shift = int(rows*0.2), int(cols*0.2)

        # only 2 perspectives it can shift between
        if rand.random() > 0.5:
            # src = [[top left], [top right], [bottom left], [bottom right]]
            src = np.array([[0, 0], [0, cols-1], [rows-1, 0], [rows-1, cols-1]], dtype=np.float32)
            # dst pinch top left and bottom left in. Leave right side unchanged
            dst = np.array([[row_shift, col_shift], [1, cols-2], [(rows-1)-row_shift, col_shift], [rows-2, cols-2]], dtype=np.float32)
        else:
            src = np.array([[0, 0], [0, cols-1], [rows-1, 0], [rows-1, cols-1]], dtype=np.float32)
            # dst pinch top left and bottom left in. Leave right side unchanged
            dst = np.array([[1, 1], [1, (cols-1)-col_shift], [rows-2, 1], [(rows-1)-row_shift, (cols-1)-col_shift]], dtype=np.float32)
        
        eq_mat = []
        for (sx, sy), (dx, dy) in zip(src, dst):
            eq_mat.append([-sx, -sy, -1, 0, 0, 0, sx*dx, sy*dx, dx])
            eq_mat.append([0, 0, 0, -sx, -sy, -1, sx*dy, sy*dy, dy])

        eq_mat = np.array(eq_mat, dtype=np.float32)

        b = eq_mat[:, -1]
        A = eq_mat[:, :-1]
        M = np.linalg.solve(A, b)
        M = np.append(M, -1) 
        M = np.reshape(M, (3,3)) * -1

        # shifted_image = cv.warpPerspective(matrix, M, dsize=(rows, cols))
        shifted_image = np.zeros_like(matrix, dtype=float)
        
        
        for row in range(rows):
            for col in range(cols):
                start = np.array([col, row, 1])
                end = M @ start.T
                end = end/end[2]
                final_col, final_row = int(end[0]), int(end[1])

                if 0 <= final_row < rows and 0 <= final_col < cols:
                    shifted_image[row, col] = matrix[final_row, final_col]

        return shifted_image

    def cropResize(self, matrix, params):
        prob = params[0]
        if 1-prob < rand.random():
            h, w = matrix.shape
            crop_h, crop_w = int(h*0.7), int(w*0.7)

            
            top_left = (rand.randint(0, h-crop_h-1), rand.randint(0, w-crop_w-1))
            bottom_right = (top_left[0]+crop_h, top_left[1]+crop_w)
            cropped = matrix[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

            scale_col, scale_row = h/crop_h, w/crop_w
            cropped_resize = np.zeros((h, w))

            # do nearest neighbor interp
            for row in range(h):
                for column in range(w):
                    cropped_resize[row, column] = cropped[int(row/scale_row), int(column/scale_col)]

            return cropped_resize
        
        else:
            return matrix

    def cropPaste(self, matrix, params):
        rows, cols = matrix.shape
        crop_rows, crop_cols = int(rows*0.2), int(cols*0.2)

        start_row, end_row = rand.randint(0, (rows-crop_rows-1)), rand.randint(0, (rows-crop_rows-1))
        start_col, end_col = rand.randint(0, (cols-crop_cols-1)), rand.randint(0, (cols-crop_cols-1))

        cropped = np.copy(matrix)
        cropped[end_row:(end_row+crop_rows), end_col:(end_col+crop_cols)] = cropped[start_row:(start_row+crop_rows), start_col:(start_col+crop_cols)]
        cropped[start_row:(start_row+crop_rows), start_col:(start_col+crop_cols)] = 0

        return cropped

    def gaussianBlur(self, matrix, params):
        size, variance = params
        lim = size//2
        val = np.linspace(-lim, lim, size)
        kernel = np.zeros((size, size), dtype=float)

        for row in range(size):
            for col in range(size):
                kernel[row, col] = np.exp(- (val[row]**2 + val[col]**2) / (2 * variance**2))
        kernel = kernel / np.sum(kernel)

        padded = np.pad(matrix, (lim, lim), 'edge')
        convolved_matrix = np.zeros_like(matrix, dtype=float)

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                for krow in range(size):
                    for kcol in range(size):
                        convolved_matrix[row, col] += (kernel[krow, kcol] * padded[row+krow, col+kcol])
        
        return np.clip(convolved_matrix, 0, 255)

    def gaussianNoise(self, matrix, params):
        if params[0] == None:
            scale = 10
        else:
            scale = params[0]

        h, w = matrix.shape

        noise = np.array([rand.gauss(mu=0, sigma=1) for i in range(h*w)])
        noise = noise.reshape((h, w))

        noisy_matrix = matrix + (scale*noise)
        return np.clip(noisy_matrix, 0, 255)

    def adjustSharpness(self, matrix, params):
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

        padded = np.pad(matrix, (1, 1), 'edge')
        convolved_matrix = np.zeros_like(matrix, dtype=float)

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                for krow in range(3):
                    for kcol in range(3):
                        convolved_matrix[row, col] += (kernel[krow, kcol] * padded[row+krow, col+kcol])

        return np.clip(convolved_matrix, 0, 255)

    def adjustBrightness(self, matrix, params):
        scalaing_factor = rand.randrange(50, 150)/100
        return np.clip(matrix*scalaing_factor, 0, 255)

if __name__ == '__main__':
    from PIL import Image, ImageOps
    import matplotlib.pyplot as plt

    test_image = Image.open('Project/me.jpeg')
    test_image = ImageOps.grayscale(test_image)
    test_image = test_image.resize([512,512])
    test_image = np.array(test_image)



    test_func_list = {'rotate':[(-15, 15), 1.0]}
    # test_func_list = {'gaussianBlur':[9, 1]}

    composition = Compose(test_image, **test_func_list)
    new = composition.result



    fig, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(test_image, cmap='gray')
    ax[1].imshow(new, cmap='gray')
    plt.show()
