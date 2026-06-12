from math import cos, radians, sqrt
from numbers import Real
from pathlib import Path
from typing import Optional, Union

import cameratransform as ct
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cameratransform import Camera
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated


class FitConstraint(BaseModel):
    min: float
    max: float
    init: float


class RectilinearProjection(BaseModel):
    focallength_mm: Optional[float] = None
    view_x_deg: Optional[float] = None
    view_y_deg: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None


class SpatialOrientation(BaseModel):
    heading_deg: Union[float, FitConstraint] = Field(
        default_factory=lambda: FitConstraint(min=0, max=360, init=60)
    )
    tilt_deg: Union[float, FitConstraint] = Field(
        default_factory=lambda: FitConstraint(min=0, max=90, init=70)
    )
    roll_deg: Union[float, FitConstraint] = Field(
        default_factory=lambda: FitConstraint(min=-90, max=90, init=0)
    )
    elevation_m: Union[float, FitConstraint] = Field(
        default_factory=lambda: FitConstraint(min=2, max=20, init=6)
    )
    pos_x_m: float = 0
    pos_y_m: float = 0


class BrownLensDistortion(BaseModel):
    k1: Union[float, FitConstraint] = Field(
        default_factory=lambda: FitConstraint(min=-1, max=1, init=0)
    )
    k2: Union[float, FitConstraint] = Field(
        default_factory=lambda: FitConstraint(min=-0.4, max=0.4, init=0)
    )
    k3: Union[float, FitConstraint] = Field(
        default_factory=lambda: FitConstraint(min=-0.2, max=0.2, init=0)
    )


class GPSLocation(BaseModel):
    lat: float
    lon: float


class CameraParameters(BaseModel):
    rectilinear_projection: RectilinearProjection
    spatial_orientation: SpatialOrientation
    brown_lens_distortion: BrownLensDistortion = Field(default_factory=BrownLensDistortion)
    gps_location: GPSLocation


class ImageCoords(BaseModel):
    x_px: int
    y_px: int


class GPSCoords(BaseModel):
    lat: float
    lon: float
    elevation_m: float = 0


class Landmark(BaseModel):
    image: ImageCoords
    gps: GPSCoords

    @model_validator(mode="before")
    @classmethod
    def accept_compact_landmark(cls, value):
        """Accept either mapping landmarks or legacy [[x, y], [lat, lon, z]] lists."""
        if isinstance(value, (list, tuple)) and len(value) == 2:
            image, gps = value
            return {
                "image": {"x_px": image[0], "y_px": image[1]},
                "gps": {
                    "lat": gps[0],
                    "lon": gps[1],
                    "elevation_m": gps[2] if len(gps) > 2 else 0,
                },
            }
        return value


class TopView(BaseModel):
    do_plot: bool = True
    extent: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None
    m_per_pixel: float | None = None


class AutofitConfig(BaseModel):
    image_path: Path
    camera_parameters: CameraParameters
    landmarks: list[Landmark]
    iteration_num: int = 6000
    top_view: TopView = Field(default_factory=TopView)
    save_cam: bool = True

    @property
    def gps_locations(self):
        return np.array(
            [(lm.gps.lat, lm.gps.lon, lm.gps.elevation_m) for lm in self.landmarks]
        )

    @property
    def px_locations(self):
        return np.array([(lm.image.x_px, lm.image.y_px) for lm in self.landmarks])


def to_param(param: Optional[Union[float, FitConstraint]]) -> Optional[float]:
    if isinstance(param, Real):
        return float(param)
    return None


def to_fit_parameter(name: str, constraint: FitConstraint) -> ct.FitParameter:
    return ct.FitParameter(
        name=name,
        lower=constraint.min,
        upper=constraint.max,
        value=constraint.init,
    )


class CameraFit:
    def __init__(
        self,
        camerajson: Optional[str] = None,
        fitconfig: AutofitConfig | None = None,
    ) -> None:
        self._fitconfig = fitconfig
        self.img: np.ndarray | None = None

        if camerajson is not None:
            self.camera: Camera = ct.Camera.load(camerajson)
            print("Camera loaded with pre-defined parameters")
            return

        if fitconfig is None:
            raise ValueError("Either camerajson or fitconfig must be provided")

        self.camera = self._camera_fitting()

    def _camera_fitting(self) -> Camera:
        self.img = plt.imread(self._fitconfig.image_path)
        camera = self._initialize_camera()
        space_location = camera.spaceFromGPS(self._fitconfig.gps_locations)
        camera.addLandmarkInformation(
            self._fitconfig.px_locations,
            space_location,
            [0.5, 0.5, 0.2],
        )
        camera.metropolis(
            self._create_fit_parameters(),
            iterations=self._fitconfig.iteration_num,
            print_trace=False,
            disable_bar=True,
        )
        return camera

    def _initialize_camera(self) -> Camera:
        cam_params = self._fitconfig.camera_parameters
        rectilinear = cam_params.rectilinear_projection
        orientation = cam_params.spatial_orientation
        distortion = cam_params.brown_lens_distortion
        gps = cam_params.gps_location

        camera = ct.Camera(
            projection=ct.RectilinearProjection(
                image=self.img,
                focallength_mm=rectilinear.focallength_mm,
                view_x_deg=rectilinear.view_x_deg,
                view_y_deg=rectilinear.view_y_deg,
                sensor_width_mm=rectilinear.sensor_width_mm,
                sensor_height_mm=rectilinear.sensor_height_mm,
            ),
            orientation=ct.SpatialOrientation(
                heading_deg=to_param(orientation.heading_deg),
                tilt_deg=to_param(orientation.tilt_deg),
                roll_deg=to_param(orientation.roll_deg),
                elevation_m=to_param(orientation.elevation_m),
                pos_x_m=orientation.pos_x_m,
                pos_y_m=orientation.pos_y_m,
            ),
            lens=ct.BrownLensDistortion(
                k1=to_param(distortion.k1),
                k2=to_param(distortion.k2),
                k3=to_param(distortion.k3),
            ),
        )
        camera.setGPSpos(gps.lat, gps.lon)
        return camera

    def _create_fit_parameters(self) -> list[ct.FitParameter]:
        params = self._fitconfig.camera_parameters
        fit_parameters = []

        for name, value in params.spatial_orientation:
            if isinstance(value, FitConstraint):
                fit_parameters.append(to_fit_parameter(name, value))

        for name, value in params.brown_lens_distortion:
            if isinstance(value, FitConstraint):
                fit_parameters.append(to_fit_parameter(name, value))

        return fit_parameters

    def get_topview(self) -> np.ndarray | None:
        if not self._fitconfig.top_view.do_plot:
            return None
        if self._fitconfig.top_view.extent is None:
            raise ValueError("top_view.extent is required when top_view.do_plot is true")
        if self._fitconfig.top_view.m_per_pixel is None:
            raise ValueError("top_view.m_per_pixel is required when top_view.do_plot is true")

        topview_image = self.camera.getTopViewOfImage(
            self.img,
            extent=self._fitconfig.top_view.extent,
            scaling=self._fitconfig.top_view.m_per_pixel,
        )
        cv2.circle(
            topview_image,
            self._topview_origin_px(),
            10,
            color=(255, 128, 128),
            thickness=-1,
        )

        for landmark in self._fitconfig.landmarks:
            fit_point = self.camera.gpsFromImage(
                (landmark.image.x_px, landmark.image.y_px),
                Z=landmark.gps.elevation_m,
            )[:2]
            actual_point = (landmark.gps.lat, landmark.gps.lon)
            cv2.circle(
                topview_image,
                self._map_gps_to_topview(actual_point),
                5,
                color=(0, 0, 255),
                thickness=-1,
            )
            cv2.line(
                topview_image,
                self._map_gps_to_topview(fit_point),
                self._map_gps_to_topview(actual_point),
                color=(0, 0, 255),
                thickness=2,
            )
            cv2.circle(
                topview_image,
                self._map_gps_to_topview(fit_point),
                5,
                color=(0, 255, 0),
                thickness=-1,
            )

        return topview_image

    def _map_gps_to_topview(self, point_gps):
        origin_offset_px = self._topview_origin_px()
        origin = self._fitconfig.camera_parameters.gps_location
        diff_lat_m = self._distance_lat_m(origin.lat, point_gps[0])
        diff_lon_m = self._distance_lon_m(origin.lat, origin.lon, point_gps[1])
        m_per_pixel = self._fitconfig.top_view.m_per_pixel
        diff_lat_px = int(diff_lat_m // m_per_pixel)
        diff_lon_px = int(diff_lon_m // m_per_pixel)
        return (origin_offset_px[0] + diff_lon_px, origin_offset_px[1] - diff_lat_px)

    def _topview_origin_px(self):
        m_per_pixel = self._fitconfig.top_view.m_per_pixel
        extent = self._fitconfig.top_view.extent
        return (int(-extent[0] // m_per_pixel), int(-extent[2] // m_per_pixel))

    def save_cam(self, path: Path):
        if self._fitconfig.save_cam:
            self.camera.save(str(path))

    def get_perf(self) -> float:
        calculated_points = np.array(
            [
                self.camera.gpsFromImage(
                    (landmark.image.x_px, landmark.image.y_px),
                    Z=landmark.gps.elevation_m,
                )
                for landmark in self._fitconfig.landmarks
            ]
        )
        distances = self._calculate_distances(calculated_points, self._fitconfig.gps_locations)
        return sum(distances) / len(distances)

    def print_parameters(self):
        print("All Camera Parameters:")
        for attr, value in self.camera.__dict__.items():
            print(f"{attr}: {value}")

    def plot_trace(self, path: Path):
        plt.rcParams["figure.figsize"] = (10, 10)
        self.camera.plotTrace()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def plot_fit_information_image_space(self, path: Path):
        plt.rcParams["figure.figsize"] = (10, 10)
        self.camera.plotFitInformation(self.img)
        plt.legend()
        plt.savefig(path)
        plt.close()

    def get_undistorted_image(self):
        return self.camera.undistortImage(self.img, extent=(-2000, 4000, -2000, 4000))

    @classmethod
    def _calculate_distances(cls, calculated_points, groundtruth_points):
        return [
            cls._gps_distance_m(calc[0], calc[1], gt[0], gt[1])
            for calc, gt in zip(calculated_points, groundtruth_points)
        ]

    @staticmethod
    def _gps_distance_m(lat1, lon1, lat2, lon2):
        lat_dist_m = CameraFit._distance_lat_m(lat1, lat2)
        lon_dist_m = CameraFit._distance_lon_m(lat1, lon1, lon2)
        return sqrt(lat_dist_m**2 + lon_dist_m**2)

    @staticmethod
    def _distance_lat_m(lat1, lat2):
        return 111320 * (lat2 - lat1)

    @staticmethod
    def _distance_lon_m(lat, lon1, lon2):
        return 111320 * cos(radians(lat)) * (lon2 - lon1)


Camerafit = CameraFit
