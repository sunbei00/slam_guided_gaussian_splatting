import viser
import numpy as np
import time
from pathlib import Path
from typing import Literal
from typing import Optional, Tuple, Callable
import viser.transforms as vt
from nerfview import Viewer, RenderTabState
from nerfview._renderer import Renderer, RenderTask


class GsplatRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # controlable parameters
    max_sh_degree: int = 5
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "rgb", "depth(accumulated)", "depth(expected)", "alpha"
    ] = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"


class GsplatViewer(Viewer):
    """
    Viewer for gsplat.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        initial_camera_c2w: Optional[np.ndarray] = None,
        camera_position_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        z_axis_move_scale: float = 1.0,
        orbit_pivot_distance: float = 1.0,
    ):
        self._initial_camera_c2w = (
            np.array(initial_camera_c2w, dtype=np.float64)
            if initial_camera_c2w is not None
            else None
        )
        self._camera_position_bounds = (
            (
                np.array(camera_position_bounds[0], dtype=np.float64),
                np.array(camera_position_bounds[1], dtype=np.float64),
            )
            if camera_position_bounds is not None
            else None
        )
        self._z_axis_move_scale = float(np.clip(z_axis_move_scale, 0.0, 1.0))
        self._orbit_pivot_distance = max(float(orbit_pivot_distance), 1e-3)
        self._camera_pose_initialized = {}
        self._camera_adjusting = {}
        self._last_camera_position = {}
        self._last_gui_stats_update_ts = 0.0
        self._gui_stats_update_interval_s = 0.25
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

    def _set_client_camera_pose(self, client: viser.ClientHandle, c2w: np.ndarray):
        position = c2w[:3, 3]
        forward = np.array(c2w[:3, 2], dtype=np.float64)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            forward = forward / forward_norm
        look_at = position + forward * self._orbit_pivot_distance
        up_direction = -c2w[:3, 1]
        with client.atomic():
            client.camera.position = position
            client.camera.look_at = look_at
            client.camera.up_direction = up_direction

    def _set_orbit_pivot_ahead(self, client: viser.ClientHandle):
        position = np.array(client.camera.position, dtype=np.float64)
        rotation = vt.SO3(client.camera.wxyz).as_matrix()
        forward = rotation[:, 2]
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return
        forward = forward / forward_norm
        client.camera.look_at = position + forward * self._orbit_pivot_distance

    def _adjust_camera_position(self, client: viser.ClientHandle, client_id: int) -> bool:
        position_before = np.array(client.camera.position, dtype=np.float64)
        adjusted = position_before.copy()
        prev = self._last_camera_position.get(client_id)
        if prev is not None and self._z_axis_move_scale < 1.0:
            adjusted[2] = prev[2] + (adjusted[2] - prev[2]) * self._z_axis_move_scale
        if self._camera_position_bounds is not None:
            bounds_min, bounds_max = self._camera_position_bounds
            adjusted = np.clip(adjusted, bounds_min, bounds_max)
        if np.allclose(adjusted, client.camera.position):
            self._last_camera_position[client_id] = np.array(
                client.camera.position, dtype=np.float64
            )
            return False
        look_at_before = np.array(client.camera.look_at, dtype=np.float64)
        delta = adjusted - position_before
        with client.atomic():
            client.camera.position = adjusted
            client.camera.look_at = look_at_before + delta
        self._last_camera_position[client_id] = np.array(
            client.camera.position, dtype=np.float64
        )
        return True

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        if client_id in self._camera_pose_initialized:
            self._camera_pose_initialized.pop(client_id)
        if client_id in self._camera_adjusting:
            self._camera_adjusting.pop(client_id)
        if client_id in self._last_camera_position:
            self._last_camera_position.pop(client_id)
        super()._disconnect_client(client)

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id] = Renderer(
            viewer=self, client=client, lock=self.lock
        )
        self._renderers[client_id].start()
        self._camera_pose_initialized[client_id] = False
        self._camera_adjusting[client_id] = False
        self._last_camera_position[client_id] = None

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            if self._camera_adjusting[client_id]:
                return

            self._camera_adjusting[client_id] = True
            try:
                if (
                    not self._camera_pose_initialized[client_id]
                ):
                    if self._initial_camera_c2w is not None:
                        self._set_client_camera_pose(client, self._initial_camera_c2w)
                    else:
                        self._set_orbit_pivot_ahead(client)
                    self._camera_pose_initialized[client_id] = True
                    self._last_camera_position[client_id] = np.array(
                        client.camera.position, dtype=np.float64
                    )

                self._adjust_camera_position(client, client_id)
            finally:
                self._camera_adjusting[client_id] = False

            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder("Gsplat"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                max_sh_degree_number = server.gui.add_number(
                    "Max SH",
                    initial_value=self.render_tab_state.max_sh_degree,
                    min=0,
                    max=5,
                    step=1,
                    hint="Maximum SH degree used",
                )

                @max_sh_degree_number.on_update
                def _(_) -> None:
                    self.render_tab_state.max_sh_degree = int(
                        max_sh_degree_number.value
                    )
                    self.rerender(_)

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @near_far_plane_vec2.on_update
                def _(_) -> None:
                    self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                    self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                    self.rerender(_)

                radius_clip_slider = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                @radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = radius_clip_slider.value
                    self.rerender(_)

                eps2d_slider = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to the egienvalues of projected 2D covariance matrices.",
                )

                @eps2d_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.eps2d = eps2d_slider.value
                    self.rerender(_)

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = backgrounds_slider.value
                    self.rerender(_)

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    if "depth" in render_mode_dropdown.value:
                        normalize_nearfar_checkbox.disabled = False
                        inverse_checkbox.disabled = False
                    else:
                        normalize_nearfar_checkbox.disabled = True
                        inverse_checkbox.disabled = True
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                normalize_nearfar_checkbox = server.gui.add_checkbox(
                    "Normalize Near/Far",
                    initial_value=self.render_tab_state.normalize_nearfar,
                    disabled=True,
                    hint="Normalize depth with near/far plane.",
                )

                @normalize_nearfar_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.normalize_nearfar = (
                        normalize_nearfar_checkbox.value
                    )
                    self.rerender(_)

                inverse_checkbox = server.gui.add_checkbox(
                    "Inverse",
                    initial_value=self.render_tab_state.inverse,
                    disabled=True,
                    hint="Inverse the depth.",
                )

                @inverse_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.inverse = inverse_checkbox.value
                    self.rerender(_)

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)

                rasterize_mode_dropdown = server.gui.add_dropdown(
                    "Anti-Aliasing",
                    ("classic", "antialiased"),
                    initial_value=self.render_tab_state.rasterize_mode,
                    hint="Whether to use classic or antialiased rasterization.",
                )

                @rasterize_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                    self.rerender(_)

                camera_model_dropdown = server.gui.add_dropdown(
                    "Camera",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for rendering.",
                )

                @camera_model_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.camera_model = camera_model_dropdown.value
                    self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
            }
        )
        super()._populate_rendering_tab()

    def _after_render(self):
        now = time.time()
        if now - self._last_gui_stats_update_ts < self._gui_stats_update_interval_s:
            return
        self._last_gui_stats_update_ts = now

        # These GUI updates are best-effort. On some viser versions, queueing
        # websocket messages from the render thread can intermittently raise:
        # RuntimeError("deque mutated during iteration").
        total_handle = self._rendering_tab_handles["total_gs_count_number"]
        rendered_handle = self._rendering_tab_handles["rendered_gs_count_number"]

        total_count = self.render_tab_state.total_gs_count
        rendered_count = self.render_tab_state.rendered_gs_count

        try:
            if total_handle.value != total_count:
                total_handle.value = total_count
            if rendered_handle.value != rendered_count:
                rendered_handle.value = rendered_count
        except RuntimeError as e:
            if "deque mutated during iteration" not in str(e):
                raise
