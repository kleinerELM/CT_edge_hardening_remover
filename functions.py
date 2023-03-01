#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import tifffile, os, cv2

class FitCurveProfile:
    def __init__(self, data):
        self.y_data = data
        self.x_data = np.arange(0, len(self.y_data))

    def fit_function_exp(self, x, x0, b, c, d):
        return c + d**(d**((x-x0)/b))

    def plot_function(self):
        print( 'c + d^( d^( (x-x_0)/b ) )' )
        print( 'x_0 = {:.2f}, b = {:.2f}, c = {:.2f}, d = {:.2f}'.format(self.params[0], self.params[1], self.params[2], self.params[3]) )

    # deletes all values after the maximum
    def filter_values(self):
        max_y_value = self.y_data.max()
        filtered_values = np.zeros(0)
        for i in range(len(self.y_data)):
            filtered_values = np.append(filtered_values, self.y_data[i])
            if ( self.y_data[i] == max_y_value ):
                break
        self.y_data = filtered_values
        self.x_data = np.arange(0, len(self.y_data))

    def fit_data(self, p0=(1000, 700, 100, 1.5)):
        bounds = ((0, 0, 0, 1.0), (1e+8, 1e+5, 1e+4, 1.75) )
        self.params, self.cv = scipy.optimize.curve_fit(self.fit_function_exp, self.x_data, self.y_data, p0, bounds=bounds, maxfev=10000)

    def get_f_data(self):
        self.f_data = []
        for i in range(len(self.y_data)):
            self.f_data.append( self.fit_function_exp( self.x_data[i], self.params[0], self.params[1], self.params[2], self.params[3] ) )
        self.f_data = np.array(self.f_data)
        return self.f_data

    def plot_data(self):
        ax = plt.axes()
        ax.plot(self.x_data, self.y_data, label='raw data')
        ax.plot(self.x_data, self.f_data, label='fit')
        ax.set_title('background fit')
        ax.set_ylabel('grey value')
        ax.set_ylim((0, 255))
        ax.set_xlim((0, len(self.x_data)))
        ax.set_xlabel('circumference position')
        ax.legend()
        plt.show()
        #self.fit_df = pd.DataFrame(data, columns=['x', 'y', 'y_fit'])
        #self.fit_df.plot(x='x', y=['y', 'y_fit'], title='title')

class CTPreprocessor:
    def __init__(self, filepath):
        if os.path.isfile(filepath):
            self.dataset = tifffile.imread(filepath)
            self.z, self.h, self.w = self.dataset.shape
            print( "Dimensions: z = {:d}, h = {:d}, w = {:d} [px]".format(self.z, self.h, self.w) )
            #self.x_data = np.arange(0, len(self.y_data))
        else:
            raise Exception( '{} does not exist!'.format(filepath) )

    def select_slice(self, id):
        self.slice_id = id
        self.slice = self.dataset[ id ]
        self.center = (0,0)
        self.blur = []
        return self.slice

    def show_example_slices(self):
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle( "example images", fontsize=16 )

        ax[0].imshow( self.select_slice( int(self.z/3) ), cmap='gray' )
        ax[0].set_title( "slice id {:d}".format( self.slice_id ) )

        ax[1].imshow( self.select_slice( int(self.z/3*2) ), cmap='gray' )
        ax[1].set_title( "slice id {:d}".format( self.slice_id ) )

        plt.show()

    def identify_main_circle(self, threshold=20, gauss_kernel=11):
        if len(self.blur) == 0:
            self.blur = cv2.GaussianBlur(self.slice,(gauss_kernel,gauss_kernel),0)

        _, thresh = cv2.threshold( self.blur, threshold, 255, cv2.THRESH_BINARY_INV )

        edges = cv2.bitwise_not(thresh)
        contour, _ = cv2.findContours( edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )
        for cnt in contour:
            cv2.drawContours(edges,[cnt],0,255,-1)

        self.main_circle = cv2.bitwise_not(edges)

        # identify the center point
        M = cv2.moments(edges)
        self.center = ( int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) )

        print( "found the center point at ({:d}, {:d})".format(self.center[0], self.center[1]) )

    def get_main_circle( self ):
        if self.center != (0,0):
            return cv2.circle(self.main_circle, self.center, 3, (255, 255, 255), -1)
        else:
            raise Exception( 'Center point not found or not yet calculated. Call self.identify_main_circle() first!' )

    def identify_pores(self, threshold=70):

        if self.center != (0,0):
            _, self.pore_thresh = cv2.threshold( self.blur, threshold, 255, cv2.THRESH_BINARY_INV )

            self.inner_pores = (self.pore_thresh - self.main_circle)/255
        else:
            self.identify_main_circle()
            self.identify_pores(threshold=threshold)

    def remove_pores(self):
        self.removed_pores = self.slice * np.logical_not(self.inner_pores) + self.inner_pores * np.median( self.slice )
        #return self.removed_pores

if __name__ == "__main__":
    print('nothing to see here. Use in Jupyter notebook.')