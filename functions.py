#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tifffile, os, cv2, multiprocessing, time, sys
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


class _BackgroundFunction:
    def __init__(self, function=lambda x, a, b: None, function_string="not defined"):
        self.function = function
        self.params = np.ones(self.param_count - 1) * np.nan
        self.function_string = function_string

    @property
    def param_count(self):
        return self.function.__code__.co_argcount

    @property
    def param_names(self):
        return self.function.__code__.co_varnames[1:]

    def fit(self, x, y, *args, **kwargs):
        popt, pcov = scipy.optimize.curve_fit(self.function, x, y, *args, **kwargs)
        self.params = np.array(popt)
        return self

    def eval(self, x):
        return self.function(x, *self.params)

    # deletes all values after the maximum
    # def filter_values(self):
    #     max_y_value = self.y_data.max()
    #     filtered_values = np.zeros(0)
    #     for i in range(len(self.y_data)):
    #         filtered_values = np.append(filtered_values, self.y_data[i])
    #         if self.y_data[i] == max_y_value:
    #             break
    #     self.y_data = filtered_values
    #     self.x_data = np.arange(0, len(self.y_data))

    def __repr__(self) -> str:
        return (
            f"-- {self.__class__.__qualname__} --\n"
            + f"function: {self.function_string}\n"
            + ", ".join(
                [f"{var} = {val}" for var, val in zip(self.param_names, self.params)]
            )
        )


class LinearBackgroundFunction(_BackgroundFunction):
    def __init__(self):
        function = lambda x, a, b: b + a * x
        function_string = "b + a*x"
        super().__init__(function=function, function_string=function_string)


class QuadraticBackgroundFunction(_BackgroundFunction):
    def __init__(self):
        function = lambda x, a, b, c: c + b * x + a * x**2
        function_string = "c + b*x + a*x**2"
        super().__init__(function=function, function_string=function_string)


class ExponentialBackgroundFunction(_BackgroundFunction):
    def __init__(self):
        function = lambda x, x0, b, c, d: c + d ** (d ** ((x - x0) / b))
        function_string = "c + d ** (d ** ((x - x0) / b))"
        super().__init__(function=function, function_string=function_string)


class CTPreprocessor:
    def __init__(
        self,
        filepath,  # full path to dataset eg. 'C:\data\dataset.tif'
        pore_threshold=70,  # value between 0 and 255 to differentiate pores from material
        circle_threshold=20,  # value between 0 and 255 to differentiate the circle from the background
        gauss_kernel=11,  # size of the Gauß kernel to soften the images used to apply thresholds
        unit="px",  # unit has to be the same in all directions
        unit_factors=(
            1.0,
            1.0,
            1.0,
        ),  # factor to convert px to chosen unit as float (w, h, z)
    ):
        if os.path.isfile(filepath):
            self.unit = unit
            self.unit_factors = unit_factors
            self.pore_threshold = pore_threshold
            self.circle_threshold = circle_threshold
            self.gauss_kernel = gauss_kernel
            self.stack_statistics = {
                "mean_radius": 0.0,
                "mean_center_point": (0, 0),
                "mean_pore_area": 0.0,
            }

            self.dataset = tifffile.imread(filepath)
            self.z, self.h, self.w = self.dataset.shape
            print(
                "Dimensions: w = {:d}, h = {:d}, z = {:d} [px]".format(
                    self.w, self.h, self.z
                )
            )
            if unit != "px":
                print(
                    "            w = {:.1f}, h = {:.1f}, z = {:.1f} [{}]".format(
                        self.w * unit_factors[0],
                        self.h * unit_factors[1],
                        self.z * unit_factors[2],
                        unit,
                    )
                )
        else:
            raise Exception("{} does not exist!".format(filepath))

    def select_slice(self, id):
        self.slice_id = id
        self.slice = CTSlicePreprocessor(
            id,
            self.dataset[id],
            pore_threshold=self.pore_threshold,
            circle_threshold=self.circle_threshold,
            gauss_kernel=self.gauss_kernel,
            unit=self.unit,
            stack_statistics=self.stack_statistics,
        )
        return self.slice.slice

    def show_example_slices(self):
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle("example images", fontsize=16)

        ax[0].imshow(self.select_slice(int(self.z / 3)), cmap="gray")
        ax[0].set_title("slice id {:d}".format(self.slice_id))
        ax[0].set_xlabel("x-position in {}".format(self.unit))
        ax[0].set_ylabel("y-position in {}".format(self.unit))

        ax[1].imshow(self.select_slice(int(self.z / 3 * 2)), cmap="gray")
        ax[1].set_title("slice id {:d}".format(self.slice_id))
        ax[1].set_xlabel("x-position in {}".format(self.unit))
        ax[1].set_ylabel("y-position in {}".format(self.unit))

        plt.show()

    def process_slice_cb(self, r):
        id = r["id"]
        self.circle_max_radii[id] = r["min_length"]  # not really r
        self.circle_radii[id] = r["radius"]  # actual radius
        self.circle_centers[id] = r[
            "center"
        ]  # x and y position of the identified circle
        self.pore_areas[id] = r["pore_area"]  # porea area percentage
        self.fixed_volume[id] = r[
            "fixed_volume"
        ]  # the processed backgrounds as a volume
        self.bg_diff_volume[id] = r["bg_diff_volume"]  # corrected dataset
        if id % 50 == 0:
            print("slice {:d} done".format(id), flush=True)
        sys.stdout.flush()

        # os.system("slice {:d} done".format(id))

    def process_full_stack(self, max_processes=0, verbose=False):
        self.circle_radii = np.empty((self.z,), dtype=np.int16)
        self.circle_max_radii = np.empty((self.z,), dtype=np.int16)
        self.circle_centers = np.empty((self.z, 2), dtype=np.int32)
        self.pore_areas = np.empty((self.z,), dtype=np.float64)

        self.fixed_volume = np.empty(shape=self.dataset.shape, dtype=np.uint8)
        self.bg_diff_volume = np.empty(shape=self.dataset.shape, dtype=np.uint8)

        # benchmark a singe slice
        self.select_slice(0)
        time_start = time.time()
        self.process_slice_cb(process_slice(0, self.slice))
        duration = time.time() - time_start

        if verbose:
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))

            ax[0].imshow(self.slice.slice, cmap="gray")
            ax[0].set_title("slice{: 5d} of{: 5d}".format(0, self.z))
            ax[0].set_xlabel("x-position in {}".format(self.unit))
            ax[0].set_ylabel("y-position in {}".format(self.unit))

            ax[1].imshow(self.bg_diff_volume[0], cmap="gray")
            ax[1].set_title("background correction")
            ax[1].set_xlabel("x-position in {}".format(self.unit))
            ax[1].set_ylabel("y-position in {}".format(self.unit))

            ax[2].imshow(self.fixed_volume[0], cmap="gray")
            ax[2].set_title("corrected slice")
            ax[2].set_xlabel("x-position in {}".format(self.unit))
            ax[2].set_ylabel("y-position in {}".format(self.unit))

            plt.show()

        coreCount = 1  # multiprocessing.cpu_count()
        processCount = (coreCount - 1) if coreCount > 1 else 1
        if (max_processes > 1) & (processCount > max_processes):
            print(
                "reduced process count from {:d} to {:d}".format(
                    processCount, max_processes
                )
            )
            processCount = max_processes
        print("Splitting slice processing in {:d} processes.".format(processCount))
        print(
            "The processing of each slice may take around {:.1f} seconds".format(
                duration
            )
        )
        print(
            "The overall process may take {:.1f} minutes".format(
                duration * self.z / processCount / 60
            )
        )

        pool = multiprocessing.Pool(processCount)
        for i in range(1, self.z):
            self.select_slice(i)
            if processCount == 1:
                self.process_slice_cb(process_slice(i, self.slice))
                self.stack_statistics["mean_radius"] = np.mean(self.circle_radii)
                self.stack_statistics["mean_pore_area"] = np.mean(self.pore_areas)
                # self.stack_statistics['mean_center_point'] = (0,0)
            else:
                pool.apply_async(
                    process_slice, args=(i, self.slice), callback=self.process_slice_cb
                )

        pool.close()  # close the process pool
        pool.join()  # wait for all tasks to finish

        self.circle_centers = np.swapaxes(self.circle_centers, 0, 1)
        self.min_pore_pos = np.argmin(self.pore_areas)


class CTSlicePreprocessor:
    center = (0, 0)
    min_length = 0
    radius = 0
    blur = []
    mean_polar_brightness = []

    def __init__(
        self,
        id,
        slice,
        pore_threshold=70,
        circle_threshold=20,
        gauss_kernel=11,
        unit="px",
        stack_statistics={},
    ) -> None:
        self.slice_id = id
        self.slice = slice
        self.h, self.w = self.slice.shape  # set some global constants
        self.unit = unit
        self.gauss_kernel = gauss_kernel  # in px
        self.circle_threshold = circle_threshold  # brightness level
        self.pore_threshold = pore_threshold  # brightness level

        self.has_statistics = "mean_radius" in stack_statistics.keys()
        self.has_statistics = self.has_statistics & id > 3
        self.mean_radius = (
            stack_statistics["mean_radius"]
            if "mean_radius" in stack_statistics.keys()
            else 0
        )
        self.mean_pore_area = (
            stack_statistics["mean_pore_area"]
            if "mean_pore_area" in stack_statistics.keys()
            else 0
        )
        self.mean_center_point = (
            stack_statistics["mean_center_point"]
            if "mean_center_point" in stack_statistics.keys()
            else (0, 0)
        )

    def identify_main_circle(self, verbose=True):
        if len(self.blur) == 0:
            self.blur = cv2.GaussianBlur(
                self.slice, (self.gauss_kernel, self.gauss_kernel), 0
            )

        _, thresh = cv2.threshold(
            self.blur, self.circle_threshold, 255, cv2.THRESH_BINARY_INV
        )

        edges = cv2.bitwise_not(thresh)
        contour, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(edges, [cnt], 0, 255, -1)

        self.main_circle = cv2.bitwise_not(edges)

        # identify the center point
        M = cv2.moments(edges)
        self.center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # get minimum from the circle center to the edge of the image
        self.min_length = np.array(
            [
                self.center[0],
                self.w - self.center[0],
                self.center[1],
                self.h - self.center[1],
            ]
        ).min()

        if verbose:
            print(
                "found the center point at ({:d}, {:d}). The circle has a maximum radius of {:d} px.".format(
                    self.center[0], self.center[1], self.min_length
                )
            )

    # return an image displaying the main circular specimen and the center point
    def get_main_circle(self):
        if self.center != (0, 0):
            circle = (
                cv2.circle(
                    np.copy(self.main_circle), self.center, 3, (255, 255, 255), -1
                )
                / 255
            ).astype(np.uint8)
            self.circle_area = circle.sum()
            return circle
        else:
            raise Exception(
                "Center point not found or not yet calculated. Call self.identify_main_circle() first!"
            )

    def get_inner_pores(self, slice, do_blurring=True):
        if do_blurring:
            slice = cv2.GaussianBlur(slice, (self.gauss_kernel, self.gauss_kernel), 0)
        _, thresh = cv2.threshold(
            slice, self.pore_threshold, 255, cv2.THRESH_BINARY_INV
        )
        return ((np.array(thresh) - self.main_circle) / 255).astype(np.uint8)

    # set a threshold to obtain a mask for the pores within the specimen
    def identify_pores(self):
        if self.center != (0, 0):
            self.inner_pores = self.get_inner_pores(self.blur, do_blurring=False)
            self.pore_area = self.inner_pores.sum()
            self.pore_area_percent = self.inner_pores.sum() / self.circle_area.sum()
        else:
            raise Exception(
                "Center point not found or not yet calculated. Call self.identify_main_circle() first!"
            )

    # replace (black) pores with the median value of the slice to improve future processing
    def remove_pores(self, pore_mask=[], background=[]):
        if len(pore_mask) == 0:
            pore_mask = self.inner_pores
        if self.center != (0, 0):
            # use the median of the slice value to replace pores, if no other background is given
            if len(background) == 0:
                background = np.median(self.slice)
            self.removed_pores = (
                self.slice * np.logical_not(pore_mask) + pore_mask * background
            )

            self.polar_image = self.circle_to_polar(self.removed_pores)
            return self.removed_pores

        else:
            raise Exception(
                "Center point not found or not yet calculated. Call self.identify_main_circle() first!"
            )

    # convert linear image (containing a slice of the circular specimen) to polar coordinates
    def circle_to_polar(self, image, median_blur_kernel=1):
        if self.center != (0, 0):
            # is_main_polar = (image != [])
            # if is_main_polar: image = self.removed_pores # usually this image is used...

            polar_image = cv2.linearPolar(
                image,
                (self.center[0], self.center[1]),
                self.min_length,
                cv2.WARP_FILL_OUTLIERS,
            )

            # blur in the direction of the radius, if the kernel is larger than 1
            if median_blur_kernel > 1:
                polar_image = smooth_polar_image(polar_image, median_blur_kernel)
            # if is_main_polar: self.polar_image = polar_image

            return polar_image
        else:
            raise Exception(
                "Center point not found or not yet calculated. Call self.identify_main_circle() first!"
            )

    # convert polar image back to a linear image
    def polar_to_circle(self, polar_image=[]):
        if self.center != (0, 0):
            if len(polar_image) == 0:
                polar_image = self.polar_image  # usually this image is used...

            unpolar = cv2.linearPolar(
                polar_image,
                (self.center[0], self.center[1]),
                self.min_length,
                cv2.WARP_INVERSE_MAP,
            )
            return unpolar
        else:
            raise Exception(
                "Center point not found or not yet calculated. Call self.identify_main_circle() first!"
            )

    def get_mean_polar_brightness(self, polar_image=[]):
        is_main_polar = len(polar_image) == 0
        if is_main_polar:
            polar_image = self.polar_image  # usually this image is used...

        if len(polar_image) > 0:
            mean_polar_brightness = (polar_image.sum(0) / len(polar_image)).astype(
                np.uint8
            )

            polar_background = np.empty(shape=polar_image.shape, dtype=int)
            polar_background.fill(0)
            for i in range(len(polar_image)):
                polar_background[i] = mean_polar_brightness

            self.radius = np.argmax(mean_polar_brightness) + 2

            if self.has_statistics:
                if (
                    self.radius
                    < 0.9 * self.mean_radius | self.radius
                    > 1.1 * self.mean_radius
                ):
                    print(
                        "The calculated radius {:d} significantly differs from the last radii {:.1f}.".format(
                            self.radius, self.has_statistics
                        )
                    )
                    print("Replacing the value using the radius of the last slice.")
                    self.radius = int(self.mean_radius)

                    ax = plt.axes()
                    ax.plot(
                        range(len(polar_background)),
                        polar_background,
                        label="polar_background",
                    )
                    ax.set_title("-")
                    ax.set_ylabel("-")
                    ax.set_xlim((0, len(self.polar_background)))
                    ax.set_xlabel("-")
                    ax.legend()
                    plt.show()

            if is_main_polar:
                self.mean_polar_brightness = mean_polar_brightness
                self.polar_background = polar_background

            return mean_polar_brightness, polar_background
        else:
            raise Exception(
                "No polar image found. Check call variable polar_image or call self.circle_to_polar() first!"
            )

    def get_border_deviation(self, polar_image=[], show_graph=True):
        is_main_polar = len(polar_image) == 0
        if is_main_polar:
            polar_image = self.polar_image  # usually this image is used...

        if len(polar_image) > 0:
            border_position = []
            for i in range(len(polar_image)):
                border_position.append(np.argmax(polar_image[i]))

            medial_filter_kernel = 101
            border_position = median_filter(
                np.array(border_position).astype(np.uint16), medial_filter_kernel
            )

            border_deviation = (border_position - np.mean(border_position)).astype(
                np.int16
            )

            # show uncircularity
            if show_graph:
                ax = plt.axes()
                ax.plot(
                    range(len(self.border_deviation)),
                    self.border_deviation,
                    label="position of max value in dataset",
                )
                ax.set_title("deviation from circularity using max values")
                ax.set_ylabel("deviation from circularity (0)")
                ax.set_xlim((0, len(self.border_deviation)))
                ax.set_xlabel("circumference position")
                ax.legend()
                plt.show()

            if is_main_polar:
                self.border_deviation = border_deviation
                self.border_position = border_position

            return border_position, border_deviation
        else:
            raise Exception(
                "No polar image found. Check call variable polar_image or call self.circle_to_polar() first!"
            )

    def get_fit_function_start_values(self, mean_polar_brightness=[]):
        if len(mean_polar_brightness) == 0:
            mean_polar_brightness = self.mean_polar_brightness

        if len(mean_polar_brightness) > 0:
            p0 = (
                int(self.min_length / 2),
                int(self.min_length / 3 * 2),
                np.min(mean_polar_brightness),
                1.5,
            )
            return p0
        else:
            raise Exception(
                "Mean polar brightness is not calculated yet. Call self.get_mean_polar_brightness() first!"
            )

    def fit_to_polar(self, fit_data, polar_image):
        # process the background
        polar_background_fit = np.empty(shape=polar_image.shape, dtype=int)
        polar_background_fit.fill(0)

        for i in range(len(polar_image)):
            for j, v in enumerate(fit_data.astype(np.uint8)):
                polar_background_fit[i][j] = v

        return polar_background_fit

    def get_polar_background(self, polar_image=[], verbose_level=2):
        is_main_polar = len(polar_image) == 0
        if is_main_polar:
            polar_image = self.polar_image  # usually this image is used...

        if len(polar_image) > 0:
            if is_main_polar:
                mean_values, polar_background = self.get_mean_polar_brightness()
            else:
                mean_values, polar_background = self.get_mean_polar_brightness(
                    polar_image
                )
            p0 = self.get_fit_function_start_values(mean_values)
            fit_data, offset = fit_background_function(
                mean_values, p0, verbose_level=verbose_level
            )
            # process the background
            polar_background_fit = self.fit_to_polar(fit_data, polar_image)

            return fit_data, polar_background, polar_background_fit, offset
        else:
            raise Exception(
                "No polar image found. Check call variable polar_image or call self.circle_to_polar() first!"
            )

    def correct_circularity(
        self, border_position=[], border_deviation=[], show_result=False
    ):
        if len(border_position) == 0:
            border_position = self.border_position
        if len(border_deviation) == 0:
            border_deviation = self.border_deviation

        # ony metadata of self.polar_image is used
        goal_length = (
            border_position[0] - border_deviation[0]
        )  # get a constant goal length
        circular_polar_image = []

        # resize each line, data is cropped to the goal length
        for i, line in enumerate(self.polar_image):
            start_length = border_position[i]
            circular_polar_image.append(
                cv2.resize(
                    line[:start_length],
                    (1, goal_length),
                    interpolation=cv2.INTER_LINEAR,
                ).flatten()
            )

        # extend the dataset to fit the original self.polar_image.shape
        circular_polar_image = np.pad(
            circular_polar_image,
            ((0, 0), (0, self.polar_image.shape[1] - goal_length)),
            mode="constant",
            constant_values=0,
        )

        if show_result:
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(self.polar_image, cmap="gray")
            ax[0].plot(border_position, range(len(border_position)))
            ax[0].set_title(
                "polar transformed circle & indicating the identified border"
            )
            ax[1].imshow(circular_polar_image, cmap="gray")
            ax[1].set_title("polar transformed circle, corrected circularity")
            plt.show()

        return circular_polar_image

    # defining the process for a single correction step
    def fix_background_single(self, slice, background, verbose_level=0):
        # get binary mask of inner pores using a (previously fixed) slice
        inner_pores = self.get_inner_pores(slice)

        # replace inner pores by the background
        removed_pores = self.remove_pores(inner_pores, background)

        # transform to polar image
        polar_image_removed_pores = self.circle_to_polar(image=removed_pores)

        # try to correct circularity error
        border_position, border_deviation = self.get_border_deviation(
            polar_image_removed_pores, show_graph=(verbose_level == 2)
        )
        polar_image_corr = self.correct_circularity(
            border_position, border_deviation, show_result=(verbose_level == 2)
        )
        (
            fit_data,
            _,
            polar_background_fit,
            background_offset,
        ) = self.get_polar_background(polar_image_corr, verbose_level=verbose_level)

        polar_mean_background = self.polar_to_circle(
            polar_background_fit + background_offset
        )

        polar_fixed_image = (
            self.circle_to_polar(slice) - polar_background_fit
        )  # this can result in negative values
        polar_fixed_image = polar_fixed_image.clip(
            min=0
        )  # therefore the final image has to be clipped

        circular_fixed_image = self.polar_to_circle(polar_fixed_image) * np.logical_not(
            self.inner_pores
        )  # the pores have to be removed

        if verbose_level > 0:
            difference = polar_mean_background - background
            difference = difference - difference.min()  # * (-1*(self.main_circle-1))

            fig, ax = plt.subplots(1, 5, figsize=(30, 10))
            ax[0].imshow(slice, cmap="gray")
            ax[0].set_title("input raw")
            ax[1].imshow(removed_pores.clip(min=0), cmap="gray")
            ax[1].set_title("pores removed")
            ax[2].imshow(polar_mean_background, cmap="gray")
            ax[2].set_title("input background")
            ax[3].imshow(difference, cmap="gray")
            ax[3].set_title("background difference")
            ax[4].imshow(circular_fixed_image, cmap="gray")
            ax[4].set_title("output image")
            plt.show()

        return (
            fit_data,
            self.polar_to_circle(polar_background_fit),
            polar_mean_background,
            circular_fixed_image,
        )

    def fix_background(self, iterations=2, verbose_level=0):
        if verbose_level > 0:
            print("Calculating initial background.")
        _, _, background_difference, background_offset = self.get_polar_background(
            self.slice, verbose_level=verbose_level
        )
        background = (
            background_difference + background_offset
        )  # self.polar_to_circle( polar_background_fit + background_offset )

        for i in range(iterations):
            if verbose_level > 0:
                print("\nIteration #{:d} of {:d}".format(i + 1, iterations))
            (
                fit_data,
                background_difference,
                background,
                fixed,
            ) = self.fix_background_single(
                self.slice, background=background, verbose_level=verbose_level
            )

        self.background = background
        self.fixed = fixed
        return fit_data, background_difference, background, fixed


def process_slice(id, slice):
    # if id%50 == 0: print("slice {:d} start".format(id), flush=True)
    # sys.stdout.flush()

    slice.identify_main_circle(verbose=False)
    slice.get_main_circle()
    slice.identify_pores()

    slice.remove_pores()

    fit_data, bg_diff_volume, background, fixed_slice = slice.fix_background(
        iterations=2, verbose_level=0
    )

    # fixed_volume[i] = (CT.slice - CT.polar_to_circle( CT.fit_to_polar( fit_data, CT.polar_image ) )) * np.logical_not( CT.inner_pores )#CT.remove_pores( background = background_difference )

    return {
        "id": id,
        "min_length": slice.min_length,
        "radius": slice.radius,
        "center": slice.center,
        "pore_area": slice.pore_area_percent * 100,
        "bg_diff_volume": bg_diff_volume.astype(np.uint8),
        "fixed_volume": fixed_slice.astype(np.uint8),
    }


def smooth_polar_image(polar_image, median_blur_kernel=21):
    for i in range(len(polar_image)):
        polar_image[i] = cv2.medianBlur(
            polar_image[i].astype(np.uint8), median_blur_kernel
        ).flatten()

    return polar_image


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    # remove root windows
    root = tk.Tk()
    root.withdraw()

    home_dir = os.path.dirname(os.path.realpath(__file__))

    tiff_file = filedialog.askopenfilename(
        initialdir=home_dir,
        title="select 8-Bit 3D CT-Tiff stack",
        filetypes=(("Tif file", "*.tif"), ("Tiff file", "*.tiff")),
    )
    save_dir = os.path.dirname(tiff_file)
    file_name = os.path.basename(tiff_file).split(".", 1)[0]
    print('Loading "{}"'.format(tiff_file))

    CT = CTPreprocessor(tiff_file)

    CT.process_full_stack(verbose=False)

    tifffile.imwrite(save_dir + os.sep + file_name + "_fixed.tif", CT.fixed_volume)
    tifffile.imwrite(save_dir + os.sep + file_name + "_bg.tif", CT.bg_diff_volume)

    print("done...")
