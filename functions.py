#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tifffile, os, cv2, multiprocessing, time, sys
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


class _BackgroundFunction:
    """base class for functional relationships describing the intensity distribution
    in polar images
    """

    def __init__(
        self, function=lambda x, a, b: None, function_string="not defined"
    ) -> None:
        self.function = function
        self.params = np.ones(self.param_count - 1) * np.nan
        self.function_string = function_string

    @property
    def param_count(self) -> int:
        """Returns the number of parameters of the fitting function

        Returns
        -------
        int
            number of parameters of fitting function
        """
        return self.function.__code__.co_argcount

    @property
    def param_names(self) -> list[str]:
        """Returns the names of the parameters of the fitting function

        Returns
        -------
        list[str]
            list of strings containing the parameter names of the fitting function
        """
        return self.function.__code__.co_varnames[1:]

    def fit(self, x_data: np.ndarray, y_data: np.ndarray, *args, **kwargs):
        """Fits the underlying function to the data provided by x and y.
        args and kwargs are passed directly to scipy.optimize.curve_fit

        Parameters
        ----------
        x_data : np.ndarray
            x values used for curve fitting
        y_data : np.ndarray
            y values used for curve fitting

        Returns
        -------
        self
            returns its own instance
        """
        popt, _ = scipy.optimize.curve_fit(
            self.function, x_data, y_data, *args, **kwargs
        )
        self.params = np.array(popt)
        return self

    def eval_params(self, x_eval: np.ndarray, *params) -> np.ndarray:
        """Evaluates the function with the provided parameters at x_eval

        Parameters
        ----------
        x_eval : np.ndarray
            points to evaluate the function at

        Returns
        -------
        np.ndarray
            f(y) at x_eval
        """
        return self.function(x_eval, *params)

    def eval(self, x_eval: np.ndarray) -> np.ndarray:
        """Evaluates the function with the fitted parameters at x_eval

        Parameters
        ----------
        x_eval : np.ndarray
            points to evaluate the function at

        Returns
        -------
        np.ndarray
            f(y) at x_eval
        """
        return self.function(x_eval, *self.params)

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


class PolynomialBackgroundFunction(_BackgroundFunction):
    """Background function with polynomial relation ship of specified degree
    sum_i a_i * x**(degree - i)
    """

    def __init__(self, deg):
        assert isinstance(deg, int), "degree must be an integer"
        assert deg > 0, "degree can not be negative"
        assert deg < 10, "degree to large, maximum degree 10"
        self.deg = deg
        function = np.poly1d
        function_string = " + ".join(
            [f"{a} * x**{self.deg - i}" for i, a in enumerate(self.param_names)]
        )
        super().__init__(function=function, function_string=function_string)

    @property
    def param_count(self) -> int:
        return self.deg + 1

    @property
    def param_names(self) -> list[str]:
        return list([f"a_{i}" for i in range(self.param_count)])

    def fit(self, x_data: np.ndarray, y_data: np.ndarray, *args, **kwargs):
        self.params = np.polyfit(x_data, y_data, self.deg, *args, **kwargs)
        return self

    def eval(self, x_eval: np.ndarray) -> np.ndarray:
        return self.function(self.params)(x_eval)

    def eval_params(self, x_eval: np.ndarray, *params) -> np.ndarray:
        return self.function(*params)(x_eval)


class ExponentialBackgroundFunction(_BackgroundFunction):
    """Background function with exponential relationship
    c + d ** (d ** ((x - x0) / b))
    """

    def __init__(self):
        function = lambda x, x0, b, c, d: c + d ** (d ** ((x - x0) / b))
        function_string = "c + d ** (d ** ((x - x0) / b))"
        super().__init__(function=function, function_string=function_string)


class QuadraticExponentialBackgroundFunction(_BackgroundFunction):
    """Background function mixing quadratic and exponential relationship
    a + b*x + c*x**2 + d ** ((x - x0) / e))
    """

    def __init__(self):
        function = (
            lambda x, x0, a, b, c, d, e: a + b * x + c * x**2 - d ** ((x - x0) / e)
        )
        function_string = "a + b*x + c*x**2 + d ** ((x - x0) / e))"
        super().__init__(function=function, function_string=function_string)


class RootBackgroundFunction(_BackgroundFunction):
    """Background function with root relationship
    b + (a * x**n)
    """

    def __init__(self):
        function = lambda x, a, b, n: b + (a * x**n)
        function_string = "b + (a * x**n)"
        super().__init__(function=function, function_string=function_string)


class SaturationBackgroundFunction(_BackgroundFunction):
    """Background function with saturation relationship
    d - c * b ** (-a * x + d)
    """

    def __init__(self):
        function = lambda x, a, b, c, d: d - c * b ** (-a * x + d)
        function_string = "d - c * b ** (-a * x + d)"
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


from pathlib import Path
from scipy.interpolate import pchip_interpolate
from scipy.optimize import curve_fit, basinhopping, least_squares
import warnings
from skimage import draw


def circular_mask(cx, cy, r, shape, dtype=np.bool_):
    mask = np.zeros(shape, dtype=dtype)
    rr, cc = draw.disk((cx, cy), r, shape=shape)
    mask[rr, cc] = True
    return mask


def ring_mask(cx, cy, ri, ra, shape, dtype=np.bool_):
    mask_inner = circular_mask(cx, cy, ri, shape, dtype)
    mask_outer = circular_mask(cx, cy, ra, shape, dtype)
    return np.logical_xor(mask_inner, mask_outer)


def fit_circle(xs, ys):
    xs, ys = np.array(xs), np.array(ys)

    def circle(a, x, y):
        # a[0] -> xm, a[1] -> ym, a[2] -> r
        return (
            2 * x * a[0]
            + 2 * y * a[1]
            + a[2] ** 2
            - a[0] ** 2
            - a[1] ** 2
            - x**2
            - y**2
        )

    res = least_squares(circle, [0, 0, 10], args=(xs, ys))
    assert res.success, "Circle Fitting Failed!"
    xm, ym, r = res.x[0], res.x[1], res.x[2]
    return xm, ym, r


class CTSlice:
    find_center_default = "contour"
    filter_kernel_default = 5
    mask_dtype = np.uint8

    def __init__(
        self, image: np.ndarray, x_crop=slice(None), y_crop=slice(None), image_path=None
    ):
        self.source = Path(image_path)
        self.image = image[y_crop, x_crop]
        assert len(self.shape) == 2, "Images has more than one channel!"
        self.contour_threshold = (
            self.dtype_max * 0.355
        )  # threshold used to determine object contour, default 1/20 of max intensity
        self.contour_mask = self.calc_contour_mask().contour_mask
        self.pore_threshold = self.dtype_max * 0.47
        self.pore_mask = self.calc_pore_mask().pore_mask
        self.is_polar = False

    @classmethod
    def import_cv2(cls, image_path, *cv2_flags, x_crop=slice(None), y_crop=slice(None)):
        image = cv2.imread(str(image_path), *cv2_flags)
        return cls(image, x_crop=x_crop, y_crop=y_crop, image_path=image_path)

    @property
    def shape(self):
        return self.image.shape

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    @property
    def dtype(self):
        return self.image.dtype

    @property
    def dtype_max(self):
        return np.iinfo(self.dtype).max

    @property
    def dtype_min(self):
        return np.iinfo(self.dtype).min

    @property
    def min(self):
        return np.min(self.image)

    @property
    def max(self):
        return np.max(self.image)

    @property
    def mean(self):
        return np.mean(self.image)

    @property
    def median(self):
        median = np.median(self.image)
        # raise a warning if the median is 0 since it can lead to unwanted behaviour
        # in later processing
        if median == 0:
            warnings.warn(
                "Median is 0 which can lead to unexpected behaviour. Consider cropping the image prior to porcessing."
            )
        return np.median(self.image)

    def calc_contour_mask(self):
        """Detect object contour via simple thresholding. Threshold value can be adjusted via
        self.contour_threshold. Default 1/20 of maximum possible intensity"""
        self.contour_mask = scipy.ndimage.binary_fill_holes(
            self.image > self.contour_threshold
        ).astype(self.mask_dtype)
        return self

    def erode_contour_mask(self, *args, **kwargs):
        self.contour_mask = scipy.ndimage.binary_erosion(
            self.contour_mask, *args, **kwargs
        ).astype(self.mask_dtype)
        return self

    def dilate_contour_mask(self, *args, **kwargs):
        self.contour_mask = scipy.ndimage.binary_dilation(
            self.contour_mask, *args, **kwargs
        ).astype(self.mask_dtype)
        return self

    def calc_pore_mask(self, contour_mask=None):
        if contour_mask is None:
            contour_mask = self.contour_mask
        self.pore_mask = np.logical_and(
            contour_mask,
            self.image < self.pore_threshold,
        ).astype(self.mask_dtype)
        return self

    def hough_transform(self, blur=None):
        if blur is None:
            blur = self.filter_kernel_default
        tmp_mask = self.contour_mask.copy()
        # TODO needs to be removed
        tmp_mask[:1000, :] = 0
        tmp_mask[:, 500:1000] = 0
        blur = cv2.medianBlur(
            # self.contour_mask,
            tmp_mask,
            # ksize=blur,
            ksize=11,
        )
        c = cv2.HoughCircles(
            blur,
            method=cv2.HOUGH_GRADIENT,
            dp=0.5,
            minDist=1000,
            param1=1,
            param2=0.1,
            minRadius=650,
            maxRadius=750,
        )
        # check which circle is closest to image center
        if len(c[0]) > 1:
            c = sorted(
                c[0],
                # key=lambda x: np.hypot(x[0] - self.width // 2, x[1] - self.height // 2),
                key=lambda x: np.sum(
                    (
                        self.contour_mask.astype(np.bool_)
                        * circular_mask(
                            x[0], x[1], x[2], shape=self.shape, dtype=np.bool_
                        )
                    ).astype(np.int8)
                    * 4
                    - 5
                ),
            )
        elif len(c[0] == 1):
            c = c[0]
        else:
            raise RuntimeError("No circle found")
        *center, radius = c[-1]
        return center, radius

    def morphological_circle_contour(self):
        blur = cv2.medianBlur(self.contour_mask, self.filter_kernel_default)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            area = cv2.contourArea(c)
            if len(approx) > 5 and area > 1000 and area < 5000000:
                ((x, y), r) = cv2.minEnclosingCircle(c)

        return (x, y), r

    def get_center(self, method=None):
        if method is None:
            method = self.find_center_default
        if method == "contour":
            return [np.average(ind) for ind in np.where(self.contour_mask)]
        elif method == "hough":
            return self.hough_transform()[0][::-1]
        elif method == "morph":
            return self.morphological_circle_contour()[0]
        elif method == "basinhopping":
            penalty_image = -((self.image > 37500) * 2 + 1)

            def function(image, cx, cy, ri, ra):
                mask = ring_mask(cx, cy, ri, ra, shape=self.shape)
                return float(np.sum(image[mask]))

            res = basinhopping(
                lambda x, *args: function(-penalty_image / 1e6, x[0], x[1], 400, 600),
                x0=[750, 750],
            )
            return res.x
        elif method == "fit_circle":
            circle_contour = cv2.Canny(self.contour_mask, 0, 1)
            circle_contour[750:, :] = 0
            contour_cords = np.where(circle_contour > 0)
            cx, cy, r = fit_circle(*contour_cords)
            return cx, cy
        else:
            raise NotImplementedError(f"Unknown method: '{method}'")

    def shift_image(self, shift_x, shift_y):
        self.image = scipy.ndimage.shift(self.image, (shift_x, shift_y))
        return self

    def center_slice(self, method=None):
        if method is None:
            method = self.find_center_default
        cx_old, cy_old = self.get_center(method=method)
        cx_new, cy_new = self.width / 2, self.height / 2
        shift_x, shift_y = -cx_new + cx_old, -cy_new + cy_old
        self.shift_image(shift_x=-shift_x, shift_y=-shift_y)
        return self

    def get_shift(self, method=None):
        if method is None:
            method = self.find_center_default
        cx_old, cy_old = self.get_center(method=method)
        cx_new, cy_new = self.width / 2, self.height / 2
        shift_x, shift_y = cx_new - cx_old, cy_new - cy_old
        return shift_x, shift_y

    def correct_circularity(self, contour, *args, goal_length=None, **kwargs):
        if not self.is_polar:
            self.transform()

        def stretch_row(row, edge, goal, pad_to):
            x_old = np.arange(edge)
            x_new = np.linspace(0, edge, int(goal))
            return np.pad(
                pchip_interpolate(
                    x_old[:],
                    row[:edge],
                    x_new,
                ),
                (0, int(pad_to - goal)),
                mode="constant",
                constant_values=(0, 0),
            )

        if goal_length is None:
            goal_length = np.ceil(np.median(contour))

        for i in range(self.height):
            self.image[i, :] = stretch_row(
                self.image[i, :], contour[i], goal_length, self.width
            )

        self.contour_mask = np.ones_like(self.image, dtype=self.mask_dtype)
        self.contour_mask[:, int(goal_length) :] = 0
        self.erode_contour_mask(*args, **kwargs).calc_pore_mask().dilate_contour_mask(
            *args, **kwargs
        )
        return self

    def transform(self, center: tuple = None, to: str = None):
        """_summary_

        Parameters
        ----------
        center : tuple, optional
            _description_, by default None
        to : str, optional
            'polar' or 'linear' to specify transform, by default None

        Returns
        -------
        _type_
            _description_
        """
        assert to in [None, "polar", "linear"], "illegal <to> method"
        if center is None:
            center = self.width / 2, self.height / 2

        if (to != "linear") and (not self.is_polar):
            self.image = cv2.linearPolar(
                self.image, center, self.width // 2, cv2.WARP_FILL_OUTLIERS
            )
            self.contour_mask = cv2.linearPolar(
                self.contour_mask, center, self.width // 2, cv2.WARP_FILL_OUTLIERS
            )
            self.pore_mask = cv2.linearPolar(
                self.pore_mask, center, self.width // 2, cv2.WARP_FILL_OUTLIERS
            )
        elif (to != "polar") and self.is_polar:
            self.image = cv2.linearPolar(
                self.image,
                center,
                self.width // 2,
                cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS,
            )
            self.contour_mask = cv2.linearPolar(
                self.contour_mask,
                center,
                self.width // 2,
                cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS,
            )
            self.pore_mask = cv2.linearPolar(
                self.pore_mask,
                center,
                self.width // 2,
                cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS,
            )
        else:
            return self
        self.is_polar = ~self.is_polar
        return self

    def __repr__(self) -> str:
        return (
            f"-- {self.__class__.__qualname__} --\n"
            + f"   size            : {self.width} x {self.height} | {self.dtype}\n"
            + f"   min intensity   : {self.min}\n"
            + f"   max intensity   : {self.max}\n"
            + f"   mean intensity  : {self.mean}\n"
            + f"   median intensity: {self.median}\n"
            + f"source: {self.source!s}"
        )
