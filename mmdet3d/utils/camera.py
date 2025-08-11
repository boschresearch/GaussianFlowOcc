from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import torch
from torch import nn


def fast_inverse(x):
    x_inv = torch.linalg.inv_ex(x)
    torch._assert_async(x_inv.info.eq(0).all())

    return x_inv.inverse


class CameraModel(nn.Module, ABC):
    def __init__(
        self,
        *,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_size: tuple[int, int] | None = None,
        downsample: int = 1,
    ):
        """Constructor of camera model class which stores batch size and number of cameras together with the intrinsic
        and extrinsic matrices. The inverted matrices (for intrinsics and rotations) are pre-computed.

        Parameters
        ----------
            intrinsics(torch.Tensor(B, N, 3, 3)): Intrinsic camera matrices
                                    for all batches B and number of cameras N.
            extrinsics(torch.Tensor(B, N, 4, 4)): Extrinsic camera matrices
                                    for all batches B and number of cameras N.
            image_size(tuple[int, int]): Optional image size as height + width (in that order).
            downsample(int): Optional down sample factor.
        """
        super().__init__()

        self.nr_batches = intrinsics.shape[0]
        self.nr_cameras = intrinsics.shape[1]
        self.device = intrinsics.device

        intrinsics = intrinsics.clone()
        intrinsics[..., :2, :] /= downsample

        # Add empty dimension for points
        self.intrinsics = intrinsics.reshape(self.nr_batches, self.nr_cameras, 1, 3, 3)
        self.extrinsics = extrinsics.reshape(self.nr_batches, self.nr_cameras, 1, 4, 4)

        self.rotation = self.extrinsics[..., :3, :3]
        self.translation = self.extrinsics[..., :3, 3]

        self.rotation_inv = fast_inverse(self.rotation)
        self.intrinsics_inv = fast_inverse(self.intrinsics)

        if image_size is None:
            self.image_size = None
        else:
            self.image_size = (image_size[0] // downsample, image_size[1] // downsample)

    @cached_property
    def grid_uv(self):
        return self._build_grid(batch_size=self.nr_batches, num_cameras=self.nr_cameras)

    def _build_grid(self, batch_size: int = 1, num_cameras: int = 1) -> torch.Tensor:
        """Build the uv image coordinate grid.

        Returns:
            Grid of image coordinates of shape [1, 1, H, W, 2].
        """
        assert self.image_size is not None

        image_height, image_width = self.image_size
        pixel_offset = 0.5
        vec_u = pixel_offset + torch.arange(image_width, dtype=torch.float32, device=self.device)
        vec_v = pixel_offset + torch.arange(image_height, dtype=torch.float32, device=self.device)
        v, u = torch.meshgrid(vec_v, vec_u, indexing="ij")
        grid_uv = torch.stack([u.flatten(), v.flatten()], dim=-1)
        grid_uv = grid_uv.reshape(1, 1, image_height, image_width, 2)  # Add batch & camera axes
        grid_uv = grid_uv.expand(batch_size, num_cameras, -1, -1, -1)
        return grid_uv

    def img2cam(self, *, depth: torch.Tensor | None = None, grid_uv: torch.Tensor | None = None):
        """Unproject image pixel to sight rays.

        Parameters
        ----------
            grid_uv: uv image coordinates of shape [B, N, ..., 2]. Defaults to None.
            depth: Depth of shape [B, N, D, ...]. Defaults to None.

        Returns
        -------
            sight_rays(torch.Tensor(B, N, P, D)): Sight rays for current camera
                    with B batches, N cameras, P points of dimension D (D=2 or D=3).
        """
        image_coordinates = self.grid_uv if grid_uv is None else grid_uv

        if image_coordinates.shape[-1] != 2:
            raise ValueError("Image coordinates need to be two-dimensional.")

        if image_coordinates.ndim >= 5:
            # Flatten all point dimensions
            image_coordinates = image_coordinates.flatten(2, -2)

        # Convert to homogeneous coordinates of shape [B, N, P, 3]
        image_coordinates = torch.cat((image_coordinates, torch.ones_like(image_coordinates[..., :1])), dim=-1)
        image_coordinates = image_coordinates.unsqueeze(dim=-3)  # [B, N, D=1, P, 3]

        if depth is not None:
            if depth.ndim >= 5:
                # Flatten all point dimensions
                depth = depth.flatten(3, -1)  # [_B, _N, D, _P]

            image_coordinates = image_coordinates * depth.unsqueeze(dim=-1)

        unprojected_points = self.unproject_points(image_coordinates.flatten(start_dim=-3, end_dim=-2))

        return self._img2cam(unprojected_points)  # [B, N, D*H*W, 3]

    @abstractmethod
    def _img2cam(self, grid_uv: torch.Tensor):
        """Unproject image pixel to camera coordinates (sight rays).

        Parameters
        ----------
            grid_uv(torch.Tensor(B, N, P, 3)): Homogeneous, depth-scaled, image coordinates
                    with B batches, N cameras and P points.

        Returns
        -------
            sight_rays(torch.Tensor(B, N, P, D)): Sight rays for current camera
                    with B batches, N cameras, P points of dimension D (D=2 or D=3).
        """

    @abstractmethod
    def cam2img(self, sight_rays_cam, skip_validity_check=False):
        """Function that projects sight rays into the image grid.

        Parameters
        ----------
            sight_rays_cam(torch.Tensor(B, N, P, D)): Sight rays for current camera
                    with B batches, N cameras, P points of dimension D (D=2 or D=3).

        Returns
        -------
            grid_uv(torch.Tensor(B, N, P, 2)): Pixel grid uv coordinates
                    with B batches, N cameras and P points.
            valid(torch.Tensor(B, N, P) bool): Flags that states which sight rays have valid projections.
            dist(torch.Tensor(B, N, P, 1)): In case of pinhole camera: z distance;
                    in case of cylindrical and spherical camera: radial distance to Y axis
                    (i.e. length of projection to X-Z-plane).
        """

    def img2world(self, *, depth: torch.Tensor | None = None, grid_uv: torch.Tensor | None = None) -> torch.Tensor:
        """Return world coordinates given optional depth values.

        Args:
            grid_uv: uv image coordinate grid of shape [B, N, H, W, 2]. Defaults to None.
            depth: Depth grid of shape [B, N, D, H, W]. Defaults to None.

        Returns:
            _type_: World coordinate grid.
        """
        # Unproject to world
        sight_rays_cam = self.img2cam(depth=depth, grid_uv=grid_uv)
        sight_rays_world = self.cam2world(sight_rays_cam)
        return sight_rays_world

    def world2img(self, sight_rays_world, skip_validity_check=False):
        # Project to image
        sight_rays_cam = self.world2cam(sight_rays_world)
        grid_uv, valid, dist = self.cam2img(sight_rays_cam, skip_validity_check=skip_validity_check)
        return grid_uv, valid, dist

    def cam2world(self, sight_rays_cam):
        # Apply (extrinsic) rotation and translation
        sight_rays_rot = self.cam2world_rot(sight_rays_cam)
        sight_rays_world = self.cam2world_tra(sight_rays_rot)
        return sight_rays_world

    def cam2world_rot(self, sight_rays):
        # Apply (extrinsic) rotation
        return (self.rotation @ sight_rays.unsqueeze(-1)).squeeze(-1)

    def cam2world_tra(self, sight_rays_cam):
        # Apply (extrinsic) translation
        sight_rays_tra = sight_rays_cam + self.translation
        return sight_rays_tra

    def world2cam(self, sight_rays_world):
        # Apply inverse (extrinsic) rotation and translation
        sight_rays_tra = self.world2cam_tra(sight_rays_world)
        sight_rays_cam = self.world2cam_rot(sight_rays_tra)
        return sight_rays_cam

    def world2cam_rot(self, sight_rays_rot):
        # Apply inverse (extrinsic) rotation
        return (self.rotation_inv @ sight_rays_rot.unsqueeze(-1)).squeeze(-1)

    def world2cam_tra(self, sight_rays_world):
        # Apply inverse (extrinsic) translation
        return sight_rays_world - self.translation

    def project_sight_rays(self, sight_rays):
        # Project camera world coordinates to image/vector pixel/array position
        return (self.intrinsics @ sight_rays.unsqueeze(-1)).squeeze(-1)

    def unproject_points(self, grid_uv_hom):
        # Unproject points to camera sight rays
        return (self.intrinsics_inv @ grid_uv_hom.unsqueeze(-1)).squeeze(-1)

    def normalize_homogeneous_coordinates(self, coor_hom, skip_validity_check=False):
        # Normalize homogeneous coordinates
        hom = coor_hom[..., 2]
        if not skip_validity_check and torch.any(hom == 0):
            return torch.zeros_like(coor_hom[..., :2]), torch.zeros_like(hom, dtype=bool)
        coor = coor_hom[..., :2] / hom.unsqueeze(-1)
        valid = hom > 0
        return coor, valid

    def is_point_in_image(self, points_uv: torch.Tensor):
        """Returns for each point if it is inside the image plane."""
        if self.image_size is None:
            raise ValueError("self.image_size must not be None")

        h, w = self.image_size

        mask = points_uv[..., 0] >= 0
        mask &= points_uv[..., 0] < w
        mask &= points_uv[..., 1] >= 0
        mask &= points_uv[..., 1] < h

        return mask

    def get_azimuth_for_sight_ray(self, sight_rays):
        azimuth = torch.atan2(sight_rays[..., 0], sight_rays[..., 2])
        return azimuth

    def get_camera_fov(self):
        pixel_left_extreme = torch.cat(
            [torch.zeros_like(self.intrinsics[..., 1, 2]), self.intrinsics[..., 1, 2]], dim=-1
        )
        pixel_right_extreme = torch.cat(
            [torch.full_like(self.intrinsics[..., 1, 2], self.image_size[1]), self.intrinsics[..., 1, 2]], dim=-1
        )
        sr = self.img2cam(grid_uv=torch.stack([pixel_left_extreme, pixel_right_extreme], dim=-2))
        fov_left_rad = torch.atan2(sr[..., 1, 0], sr[..., 1, 2])
        fov_right_rad = -torch.atan2(sr[..., 0, 0], sr[..., 0, 2])
        fov_rad = fov_left_rad + fov_right_rad
        return fov_rad, fov_left_rad, fov_right_rad


class PinholeCameraModel(CameraModel):
    def _img2cam(self, grid_uv):
        # Unproject points to camera sight rays
        # sight_rays_cam[..., 0] contains the X coordinate
        # sight_rays_cam[..., 1] contains the Y coordinate
        # sight_rays_cam[..., 2] contains the Z coordinate
        return grid_uv

    def cam2img(self, sight_rays_cam, skip_validity_check=False):
        # Sight rays have format (batchsize, n_camera, n_points, point_dimension)
        # sight_rays_cam[..., 0] contains the X coordinate
        # sight_rays_cam[..., 1] contains the Y coordinate
        # sight_rays_cam[..., 2] contains the Z coordinate
        dist_Z = sight_rays_cam[..., 2]  # distance in Z direction
        # Project camera world coordinates to image/vector pixel/array position
        grid_uv_hom = self.project_sight_rays(sight_rays_cam)
        # Normalize homogeneous coordinates
        grid_uv, valid = self.normalize_homogeneous_coordinates(grid_uv_hom, skip_validity_check=skip_validity_check)
        return grid_uv, valid, dist_Z


class PinholeCameraZoomModel(PinholeCameraModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # CV coordinate convention
        #    7 Z
        #   /
        #  +------> X
        #  |
        #  |
        #  V Y
        assert self.image_size is not None

        # Get image height
        image_height = self.image_size[0]
        # Factor to control strength of distortion (higher -> stronger distortion effect)
        self.mult_1 = 10
        # Fraction where function meets identity (i.e., zoom becomes unzoom)
        meet_identity = 0.75
        # Vertical position in meter where function meets identity (i.e., zoom becomes unzoom)
        meet_identity_point = image_height / 2 / self.intrinsics[0, 0, 0, 1, 1] * meet_identity
        self.mult_2 = meet_identity_point / torch.sinh(self.mult_1 * meet_identity_point)

    def project_sight_rays(self, sight_rays):
        # Apply zoom model
        sight_rays[..., 1] = torch.asinh(sight_rays[..., 1] / self.mult_2) / self.mult_1
        # Project camera world coordinates to image/vector pixel/array position
        sight_rays = super().project_sight_rays(sight_rays)
        return sight_rays

    def unproject_points(self, grid_uv_hom):
        # Unproject points to camera sight rays
        grid_uv_hom = super().unproject_points(grid_uv_hom)
        # Apply zoom model
        grid_uv_hom[..., 1] = torch.sinh(self.mult_1 * grid_uv_hom[..., 1]) * self.mult_2
        return grid_uv_hom


class CylindricalCameraModel(CameraModel):
    def _img2cam(self, grid_uv):
        # Unproject points to camera sight rays
        # un[..., 0] contains the azimuth, i.e. angle around Y axis
        # un[..., 1] contains the Y coordinate
        # un[..., 2] contains the radial distance
        # sight_rays_cam[..., 0] contains the X coordinate
        # sight_rays_cam[..., 1] contains the Y coordinate
        # sight_rays_cam[..., 2] contains the Z coordinate
        radial_distance = grid_uv[..., 2:3]
        normalized_image_coordinates = grid_uv[..., :2] / radial_distance
        sight_rays_cam = torch.stack(
            (
                normalized_image_coordinates[..., 0].sin(),
                normalized_image_coordinates[..., 1],
                normalized_image_coordinates[..., 0].cos(),
            ),
            dim=-1,
        )

        return radial_distance * sight_rays_cam

    def cam2img(self, sight_rays_cam, skip_validity_check=False):
        # Sight rays have format (batchsize, n_camera, n_points, point_dimension)
        # sight_rays_cam[..., 0] contains the X coordinate
        # sight_rays_cam[..., 1] contains the Y coordinate
        # sight_rays_cam[..., 2] contains the Z coordinate
        rad_dist_Y = torch.norm(
            sight_rays_cam[..., 0:3:2], dim=-1
        )  # radial distance to Y axis (i.e. length of projection to X-Z-plane)
        if not skip_validity_check and torch.any(rad_dist_Y == 0):
            return torch.zeros_like(sight_rays_cam[..., :2]), False, torch.ones_like(sight_rays_cam[..., 2])
        # un[..., 0] contains the azimuth, i.e. angle around Y axis
        # un[..., 1] contains the Y coordinate, scaled by inverse radial distance
        # un[..., 2] contains the radial distance, set to 1
        un = torch.stack(
            (
                torch.atan2(sight_rays_cam[..., 0], sight_rays_cam[..., 2]),
                sight_rays_cam[..., 1] / rad_dist_Y,
                torch.ones_like(rad_dist_Y),
            ),
            dim=-1,
        )
        # Project camera world coordinates to image/vector pixel/array position
        grid_uv_hom = self.project_sight_rays(un)
        # Normalize homogeneous coordinates
        grid_uv, valid = self.normalize_homogeneous_coordinates(grid_uv_hom, skip_validity_check=skip_validity_check)

        # Check validity
        if self.image_size is not None:
            valid &= self.is_point_in_image(grid_uv)

        return grid_uv, valid, rad_dist_Y


class CylindricalCameraZoomModel(CylindricalCameraModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # CV coordinate convention
        #    7 Z
        #   /
        #  +------> X
        #  |
        #  |
        #  V Y
        assert self.image_size is not None

        # Get image height
        image_height = self.image_size[0]
        # Factor to control strength of distortion (higher -> stronger distortion effect)
        self.mult_1 = 10
        # Fraction where function meets identity (i.e., zoom becomes unzoom)
        meet_identity = 0.7
        # Vertical position in meter where function meets identity (i.e., zoom becomes unzoom)
        meet_identity_point = image_height / 2 / self.intrinsics[0, 0, 0, 1, 1] * meet_identity
        self.mult_2 = meet_identity_point / torch.sinh(self.mult_1 * meet_identity_point)

    def project_sight_rays(self, sight_rays):
        # Apply zoom model
        sight_rays[..., 1] = torch.asinh(sight_rays[..., 1] / self.mult_2) / self.mult_1
        # Project camera world coordinates to image/vector pixel/array position
        sight_rays = super().project_sight_rays(sight_rays)
        return sight_rays

    def unproject_points(self, grid_uv_hom):
        # Unproject points to camera sight rays
        grid_uv_hom = super().unproject_points(grid_uv_hom)
        # Apply zoom model
        grid_uv_hom[..., 1] = torch.sinh(self.mult_1 * grid_uv_hom[..., 1]) * self.mult_2
        return grid_uv_hom


class SphericalCameraModel(CameraModel):
    def _img2cam(self, grid_uv):
        # Unproject points to camera sight rays
        # un[..., 0] contains the azimuth, i.e. angle around Y axis
        # un[..., 1] contains the elevation, i.e. angle around X axis
        # un[..., 2] contains the radial distance
        # sight_rays_cam[..., 0] contains the X coordinate
        # sight_rays_cam[..., 1] contains the Y coordinate
        # sight_rays_cam[..., 2] contains the Z coordinate

        radial_distance = grid_uv[..., 2:3]
        normalized_image_coordinates = grid_uv[..., :2] / radial_distance
        sight_rays_cam = torch.stack(
            (
                normalized_image_coordinates[..., 1].cos() * normalized_image_coordinates[..., 0].sin(),
                normalized_image_coordinates[..., 1].sin(),
                normalized_image_coordinates[..., 1].cos() * normalized_image_coordinates[..., 0].cos(),
            ),
            dim=-1,
        )
        return radial_distance * sight_rays_cam

    def cam2img(self, sight_rays_cam, skip_validity_check=False):
        # Sight rays have format (batchsize, n_camera, n_points, point_dimension)
        # sight_rays_cam[..., 0] contains the X coordinate
        # sight_rays_cam[..., 1] contains the Y coordinate
        # sight_rays_cam[..., 2] contains the Z coordinate
        rad_dist = torch.norm(sight_rays_cam, dim=-1)  # radial distance to camera center
        if not skip_validity_check and torch.any(rad_dist == 0):
            return torch.zeros_like(sight_rays_cam[..., :2]), False, torch.ones_like(sight_rays_cam[..., 2])
        rad_dist_Y = torch.norm(
            sight_rays_cam[..., [0, 2]], dim=-1
        )  # radial distance to Y axis (i.e. length of projection to X-Z-plane)
        # un[..., 0] contains the azimuth, i.e. angle around Y axis
        # un[..., 1] contains the elevation, i.e. angle around X axis
        # un[..., 2] contains the radial distance, set to 1
        un = torch.stack(
            (
                torch.atan2(sight_rays_cam[..., 0], sight_rays_cam[..., 2]),
                torch.atan2(sight_rays_cam[..., 1], rad_dist_Y),
                torch.ones_like(rad_dist_Y),
            ),
            dim=-1,
        )
        # Project camera world coordinates to image/vector pixel/array position
        grid_uv_hom = self.project_sight_rays(un)
        # Normalize homogeneous coordinates
        grid_uv, valid = self.normalize_homogeneous_coordinates(grid_uv_hom)
        return grid_uv, valid, rad_dist


class MeiR3CameraModel(CameraModel):
    def __init__(
        self,
        *,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_size: tuple[int, int],
        downsample: int = 1,
    ):
        intrinsics_pinhole = intrinsics.clone()

        # Unpack intrinsics
        self.D0 = intrinsics[..., 1, 0]
        self.D1 = intrinsics[..., 2, 0]
        self.D2 = intrinsics[..., 2, 1]
        self.xi = intrinsics[..., 2, 2]

        intrinsics_pinhole[..., 1, 0] = 0
        intrinsics_pinhole[..., 2, :2] = 0
        intrinsics_pinhole[..., 2, 2] = 1
        super().__init__(
            intrinsics=intrinsics_pinhole, extrinsics=extrinsics, image_size=image_size, downsample=downsample
        )

    def _img2cam(self, grid_uv: torch.Tensor):
        raise NotImplementedError()

    def cam2img(self, sight_rays_cam, skip_validity_check=False):
        # Sight rays have format (batchsize, n_camera, n_points, point_dimension)
        sight_vector_norm = sight_rays_cam.norm(dim=-1)
        xy = sight_rays_cam[..., :2]
        z = sight_rays_cam[..., 2]

        z_xi = z + self.xi.unsqueeze(-1) * sight_vector_norm

        un_undistorted = xy / z_xi.unsqueeze(-1)

        r_2 = un_undistorted.pow(2).sum(-1)
        m = 1 + self.D0 * r_2 + self.D1 * r_2.pow(2) + self.D2 * r_2.pow(3)
        un = torch.cat([un_undistorted * m.unsqueeze(-1), torch.ones_like(un_undistorted[..., :1])], dim=-1)

        # Project camera world coordinates to image/vector pixel/array position
        grid_uv_hom = self.project_sight_rays(un)

        # Check validity
        valid = (z_xi > 0) & self.is_point_in_image(grid_uv_hom[..., :2])

        # Normalize homogeneous coordinates
        return grid_uv_hom[..., :2], valid, z


class FisheyeEquiAngularCameraModel(CameraModel):
    def __init__(
        self,
        *,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_size: tuple[int, int],
        downsample: int = 1,
    ):
        intrinsics_pinhole = intrinsics.clone()

        # Unpack intrinsics
        self.rd0 = intrinsics[..., 0, 1]
        self.rd1 = intrinsics[..., 1, 0]
        self.rd2 = intrinsics[..., 2, 0]
        self.td0 = intrinsics[..., 2, 1]
        self.td1 = intrinsics[..., 2, 2]

        intrinsics_pinhole[..., 0, 1] = 0
        intrinsics_pinhole[..., 1, 0] = 0
        intrinsics_pinhole[..., 2, :2] = 0
        intrinsics_pinhole[..., 2, 2] = 1
        super().__init__(
            intrinsics=intrinsics_pinhole, extrinsics=extrinsics, image_size=image_size, downsample=downsample
        )

    def _img2cam(self, grid_uv: torch.Tensor):
        raise NotImplementedError()

    def cam2img(self, sight_rays_cam, skip_validity_check=False):
        # Sight rays have format (batchsize, n_camera, n_points, point_dimension)
        xy = sight_rays_cam[..., :2]
        z = sight_rays_cam[..., 2]
        xy_norm = xy.norm(dim=-1)

        un_undistorted = xy * (torch.arctan2(xy_norm, z) / xy_norm).unsqueeze(-1)

        # Distort
        x = un_undistorted[..., 0]
        y = un_undistorted[..., 1]
        xy = x * y
        x_2 = x.pow(2)
        y_2 = y.pow(2)
        r_2 = x_2 + y_2
        r_4 = r_2.pow(2)
        r_6 = r_2.pow(3)

        rad_dist = 1 + self.rd0 * r_2 + self.rd1 * r_4 + self.rd2 * r_6

        un = torch.stack(
            [
                rad_dist * x + self.td1 * (r_2 + 2.0 * x_2) + 2.0 * self.td0 * xy,
                rad_dist * y + self.td0 * (r_2 + 2.0 * y_2) + 2.0 * self.td1 * xy,
                torch.ones_like(x),
            ],
            dim=-1,
        )

        # Project camera world coordinates to image/vector pixel/array position
        grid_uv_hom = self.project_sight_rays(un)

        # Check validity
        valid = self.is_point_in_image(grid_uv_hom[..., :2])

        # Normalize homogeneous coordinates
        return grid_uv_hom[..., :2], valid, z