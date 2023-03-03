#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import tifffile, os, cv2
from scipy.ndimage import median_filter

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

# find a fit function for the background
# expects a 1-dimensional array containing the mean brightness across the circle radius
# p0 - start values
def fit_background_function( mean_values, p0 = (1000, 700, 100, 1.5), remove_y_offset = True ):
    FCP = FitCurveProfile( mean_values )
    FCP.filter_values()
    FCP.fit_data(p0)
    FCP.plot_function()
    fit_data = FCP.get_f_data()
    FCP.plot_data()

    offset = 0
    # remove the y-offset to only get the background deviation
    if remove_y_offset:
        offset   = fit_data.min()
        fit_data = fit_data - offset

    return fit_data, offset


class CTPreprocessor:
    def __init__(self, filepath):
        if os.path.isfile(filepath):
            self.dataset = tifffile.imread(filepath)
            self.z, self.h, self.w = self.dataset.shape
            print( "Dimensions: z = {:d}, h = {:d}, w = {:d} [px]".format(self.z, self.h, self.w) )
        else:
            raise Exception( '{} does not exist!'.format(filepath) )

    def select_slice(self, id):
        self.slice_id   = id
        self.slice      = self.dataset[ id ]
        self.center     = (0,0)
        self.min_length = 0
        self.blur       = []
        self.mean_polar_brightness = []
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

        # get minimum from the circle center to the edge of the image
        self.min_length = np.array( [self.center[0], self.w - self.center[0], self.center[1], self.h - self.center[1]] ).min()

        print( "found the center point at ({:d}, {:d}). The circle has a maximum radius of {:d} px.".format(self.center[0], self.center[1], self.min_length) )

    # return an image displaying the main circular specimen and the center point
    def get_main_circle( self ):
        if self.center != (0,0):
            return cv2.circle(np.copy(self.main_circle), self.center, 3, (255, 255, 255), -1)
        else:
            raise Exception( 'Center point not found or not yet calculated. Call self.identify_main_circle() first!' )

    # set a threshold to obtain a mask for the pores within the specimen
    def identify_pores(self, threshold=70):
        if self.center != (0,0):
            _, self.pore_thresh = cv2.threshold( self.blur, threshold, 255, cv2.THRESH_BINARY_INV )

            self.inner_pores = (self.pore_thresh - self.main_circle)/255
        else:
            self.identify_main_circle()
            self.identify_pores(threshold=threshold)

    # replace (black) pores with the median value of the slice to improve future processing
    def remove_pores(self):
        if self.center != (0,0):
            self.removed_pores = self.slice * np.logical_not(self.inner_pores) + self.inner_pores * np.median( self.slice )
        else:
            raise Exception( 'Center point not found or not yet calculated. Call self.identify_main_circle() first!' )

    # convert linear image (containing a slice of the circular specimen) to polar coordinates
    def circle_to_polar(self, image=[], median_blur_kernel = 1):
        if self.center != (0,0):
            if image == []: image = self.removed_pores # usually this image is used...

            polar_image = cv2.linearPolar(image, (self.center[0], self.center[1]), self.min_length, cv2.WARP_FILL_OUTLIERS)

            # blur in the direction of the radius, if the kernel is larger than 1
            if median_blur_kernel > 1: polar_image = smooth_polar_image(polar_image, median_blur_kernel)

            self.polar_image = polar_image
            return self.polar_image
        else:
            raise Exception( 'Center point not found or not yet calculated. Call self.identify_main_circle() first!' )

    # convert polar image back to a linear image
    def polar_to_circle(self, polar_image = []):
        if self.center != (0,0):
            if polar_image == []: polar_image = self.polar_image # usually this image is used...

            unpolar = cv2.linearPolar(polar_image, (self.center[0], self.center[1]), self.min_length, cv2.WARP_INVERSE_MAP)
            return unpolar
        else:
            raise Exception( 'Center point not found or not yet calculated. Call self.identify_main_circle() first!' )


    def get_mean_polar_brightness(self, polar_image = []):
        if polar_image == []: polar_image = self.polar_image # usually this image is used...

        if polar_image != []:
            self.mean_polar_brightness = (polar_image.sum(0)/len(polar_image)).astype(np.uint8)

            self.polar_background = np.empty(shape=polar_image.shape, dtype=int)
            self.polar_background.fill(0)
            for i in range(len(polar_image)):
                self.polar_background[i] = self.mean_polar_brightness

            self.border_position = np.argmax(self.mean_polar_brightness)+2
            return self.mean_polar_brightness, self.polar_background
        else:
            raise Exception( 'No polar image found. Check call variable polar_image or call self.circle_to_polar() first!' )


    def get_border_deviation(self, polar_image = [], show_graph = True):
        if polar_image == []: polar_image = self.polar_image # usually this image is used...

        if polar_image != []:
            border_position = []
            for i in range(len(polar_image)):
                border_position.append(np.argmax(polar_image[i]))

            medial_filter_kernel = 101
            self.border_position = median_filter(  np.array(border_position).astype(np.uint16), medial_filter_kernel )

            self.border_deviation =(self.border_position - np.mean(self.border_position)).astype(np.int16)

            #show uncircularity
            if show_graph:
                ax = plt.axes()
                ax.plot(range(len(self.border_deviation)), self.border_deviation, label='position of max value in dataset')
                ax.set_title('deviation from circularity using max values')
                ax.set_ylabel('deviation from circularity (0)')
                ax.set_xlim((0, len(self.border_deviation)))
                ax.set_xlabel('circumference position')
                ax.legend()
                plt.show()

            return self.border_position, self.border_deviation
        else:
            raise Exception( 'No polar image found. Check call variable polar_image or call self.circle_to_polar() first!' )

    def get_fit_function_start_values(self):
        if self.mean_polar_brightness != []:
            p0 = (int(self.min_length/2),  int(self.min_length/3*2), np.min(self.mean_polar_brightness), 1.5)
            return p0
        else:
            raise Exception( 'Mean polar brightness is not calculated yet. Call self.get_mean_polar_brightness() first!' )


    def get_polar_background(self, polar_image=[]):
        if polar_image == []: polar_image = self.polar_image # usually this image is used...

        if polar_image != []:
            mean_values, polar_background = self.get_mean_polar_brightness(polar_image)
            p0 = self.get_fit_function_start_values()
            fit_data, offset = fit_background_function( mean_values, p0 )
            # process the background
            polar_background_fit = np.empty(shape=polar_image.shape, dtype=int)
            polar_background_fit.fill(0)

            for i in range(len(polar_image)):
                for j,v in enumerate(fit_data.astype(np.uint8)):
                    polar_background_fit[i][j] = v

            return polar_background, polar_background_fit, offset
        else:
            raise Exception( 'No polar image found. Check call variable polar_image or call self.circle_to_polar() first!' )

    def correct_circularity( self, show_result = False ):
        goal_length = self.border_position[0]-self.border_deviation[0] # get a constant goal length
        result = []
        for i, line in enumerate(self.polar_image):
            start_length = self.border_position[i]
            result.append( cv2.resize( line[:start_length], (1,goal_length), interpolation = cv2.INTER_LINEAR ).flatten() )

        result = np.pad(result, ((0,0), (0,self.polar_image.shape[1]-goal_length) ), mode='constant', constant_values=0 )
        if show_result:
            fig, ax = plt.subplots(1,2, figsize=(20,10))
            ax[0].imshow( self.polar_image, cmap='gray' )
            ax[0].plot( self.border_position,range(len(self.border_position)) )
            ax[0].set_title( "polar transformed circle & indicating the identified border" )
            ax[1].imshow( result, cmap='gray' )
            ax[1].set_title( "polar transformed circle, corrected circularity" )
            plt.show()

        return result

def smooth_polar_image(polar_image, median_blur_kernel = 21):
    for i in range(len(polar_image)):
        polar_image[i] = cv2.medianBlur(polar_image[i].astype(np.uint8), median_blur_kernel).flatten()

    return polar_image

if __name__ == "__main__":
    print('nothing to see here. Use in Jupyter notebook.')