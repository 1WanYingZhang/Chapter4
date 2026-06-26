from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# =============================================================================
# Code config: edit here, then run this file directly.
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent
SHIP_MAP_FILE = ROOT_DIR.parent / "Baseline" / "邮轮地图.txt"
SHIP_BACKGROUND_IMAGE = ROOT_DIR.parent / "Baseline" / "D11.png"
OUTPUT_DIR = ROOT_DIR / "输出风险场"

# FDS/PyroSim CO slice is stored as volume fraction (mol/mol).
# The risk formula in the thesis uses ppm, so C_t(x,y) = X_CO * 1e6.
CO_VOLUME_FRACTION_TO_PPM = 1_000_000.0

AMBIENT_TEMPERATURE_C = 20.0
TEMPERATURE_RISK_THRESHOLD_C = 60.0
CO_RISK_THRESHOLD_PPM = 500.0
VISIBILITY_CLEAR_M = 10.0
VISIBILITY_CRITICAL_M = 5.0

# Ship evacuation grid is 0.5 m x 0.5 m, while the PyroSim/FDS slice grid is
# 0.25 m x 0.25 m. One ship cell is therefore aggregated from 2 x 2 FDS cells.
SHIP_GRID_CELL_SIZE_M = 0.5
PYROSIM_GRID_CELL_SIZE_M = 0.25

# None means use the minimum X/Y of the scenario's FDS meshes as the ship-grid
# physical origin. Set explicit values here if a later case uses another origin.
SHIP_ORIGIN_X: float | None = None
SHIP_ORIGIN_Y: float | None = None

# None means all frames. Use a small number such as 10 for a quick smoke test.
MAX_FRAMES: int | None = None
FRAME_STRIDE = 1

# Save intermediate component risks in addition to temperature, CO and life risk.
SAVE_COMPONENT_RISKS = True

# Preview PNGs are small sanity-check figures for the max-over-time life risk.
SAVE_PREVIEW_PNG = True

# Save time-varying risk-value text files. These are float grids, not 0/1/2 map
# files. This can create many files because the fire field changes over time.
SAVE_RISK_TXT_FRAMES = True
RISK_TXT_FMT = "%.6f"

# Dynamic overlay heatmaps. Life risk uses red alpha; visibility risk uses black
# alpha, so the ship-map background remains visible.
SAVE_HEATMAP_VIDEO = True
SAVE_HEATMAP_MAX_PNG = True
HEATMAP_FPS = 8
HEATMAP_DPI = 180
HEATMAP_MAX_ALPHA = 0.78
HEATMAP_FRAME_STRIDE = 1


@dataclass(frozen=True)
class MeshSpec:
    mesh_id: int
    ijk: tuple[int, int, int]
    xb: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    root: Path
    prefix: str


SCENARIOS = [
    ScenarioSpec(
        name="厨房火灾",
        root=ROOT_DIR / "厨房火灾切片数据",
        prefix="2658463932",
    ),
    ScenarioSpec(
        name="客舱火灾",
        root=ROOT_DIR / "客舱火灾切片数据",
        prefix="1831609035",
    ),
]


def load_txt_grid(path: Path) -> np.ndarray:
    rows: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if any(ch.isspace() for ch in line):
            rows.append([int(value) for value in line.split()])
        else:
            rows.append([int(ch) for ch in line])
    if not rows:
        raise ValueError(f"Empty map file: {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Map rows have inconsistent widths: {path}")
    return np.asarray(rows, dtype=np.int8)


def _parse_number_list(text: str, cast):
    return tuple(cast(value.strip()) for value in text.split(",") if value.strip())


def _mesh_id_from_line(line: str, fallback: int) -> int:
    id_match = re.search(r"ID\s*=\s*'([^']+)'", line, flags=re.IGNORECASE)
    if id_match:
        digits = re.findall(r"\d+", id_match.group(1))
        if digits:
            return int(digits[-1])
    return int(fallback)


def load_meshes_from_fds(path: Path) -> list[MeshSpec]:
    meshes: list[MeshSpec] = []
    current = ""
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        if line.upper().startswith("&MESH"):
            current = line
        elif current:
            current += " " + line
        if current and "/" in line:
            mesh_id = _mesh_id_from_line(current, len(meshes) + 1)
            ijk_match = re.search(r"IJK\s*=\s*([^/]+?)(?:,\s*[A-Z_]+\s*=|/)", current, flags=re.IGNORECASE)
            xb_match = re.search(r"XB\s*=\s*([^/]+?)(?:,\s*[A-Z_]+\s*=|/)", current, flags=re.IGNORECASE)
            if not ijk_match or not xb_match:
                raise ValueError(f"Could not parse MESH line in {path}: {current}")
            ijk = _parse_number_list(ijk_match.group(1), int)
            xb = _parse_number_list(xb_match.group(1), float)
            if len(ijk) != 3 or len(xb) != 6:
                raise ValueError(f"Invalid MESH IJK/XB in {path}: {current}")
            meshes.append(MeshSpec(mesh_id=mesh_id, ijk=ijk, xb=xb))
            current = ""
    if not meshes:
        raise ValueError(f"No &MESH entries found in {path}")
    return sorted(meshes, key=lambda mesh: mesh.mesh_id)


def scenario_fds_path(scenario: ScenarioSpec) -> Path:
    preferred = scenario.root / f"{scenario.prefix}.fds"
    if preferred.exists():
        return preferred
    candidates = sorted(scenario.root.glob("*.fds"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No .fds file found in {scenario.root}")
    raise ValueError(f"Multiple .fds files found in {scenario.root}; keep only the one for this scenario.")


def _read_fortran_record(f) -> bytes:
    raw_len = f.read(4)
    if len(raw_len) != 4:
        raise EOFError("Unexpected end of file while reading Fortran record length.")
    (size,) = struct.unpack("<i", raw_len)
    if size < 0:
        raise ValueError(f"Invalid Fortran record length: {size}")
    payload = f.read(size)
    if len(payload) != size:
        raise EOFError("Unexpected end of file while reading Fortran record payload.")
    raw_end = f.read(4)
    if len(raw_end) != 4:
        raise EOFError("Unexpected end of file while reading Fortran record terminator.")
    (end_size,) = struct.unpack("<i", raw_end)
    if end_size != size:
        raise ValueError(f"Fortran record length mismatch: {size} != {end_size}")
    return payload


def _decode_record_string(payload: bytes) -> str:
    return payload.decode("ascii", errors="ignore").strip()


@dataclass
class SliceHeader:
    quantity: str
    short_name: str
    unit: str
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int
    header_bytes: int

    @property
    def nx(self) -> int:
        return self.i2 - self.i1 + 1

    @property
    def ny(self) -> int:
        return self.j2 - self.j1 + 1

    @property
    def values_per_frame(self) -> int:
        return self.nx * self.ny


class FdsSliceFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            quantity = _decode_record_string(_read_fortran_record(f))
            short_name = _decode_record_string(_read_fortran_record(f))
            unit = _decode_record_string(_read_fortran_record(f))
            indices = struct.unpack("<6i", _read_fortran_record(f))
            header_bytes = f.tell()
        self.header = SliceHeader(
            quantity=quantity,
            short_name=short_name,
            unit=unit,
            i1=indices[0],
            i2=indices[1],
            j1=indices[2],
            j2=indices[3],
            k1=indices[4],
            k2=indices[5],
            header_bytes=header_bytes,
        )
        data_bytes = self.header.values_per_frame * 4
        self.frame_bytes = 12 + 8 + data_bytes
        remaining = self.path.stat().st_size - self.header.header_bytes
        if remaining % self.frame_bytes != 0:
            raise ValueError(
                f"{self.path.name} size is not an integer number of frames. "
                f"remaining={remaining}, frame_bytes={self.frame_bytes}"
            )
        self.frame_count = remaining // self.frame_bytes

    def read_frame(self, frame_index: int) -> tuple[float, np.ndarray]:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(frame_index)
        offset = self.header.header_bytes + frame_index * self.frame_bytes
        with self.path.open("rb") as f:
            f.seek(offset)
            time_payload = _read_fortran_record(f)
            if len(time_payload) == 4:
                (time_value,) = struct.unpack("<f", time_payload)
            elif len(time_payload) == 8:
                (time_value,) = struct.unpack("<d", time_payload)
            else:
                raise ValueError(f"Unexpected time record size in {self.path.name}: {len(time_payload)}")
            data_payload = _read_fortran_record(f)
        values = np.frombuffer(data_payload, dtype="<f4")
        expected = self.header.values_per_frame
        if values.size != expected:
            raise ValueError(f"{self.path.name}: expected {expected} values, got {values.size}.")
        return float(time_value), values.reshape((self.header.ny, self.header.nx)).copy()


@dataclass
class MeshSampler:
    mesh: MeshSpec
    rows: np.ndarray
    cols: np.ndarray
    x0_idx: np.ndarray
    x1_idx: np.ndarray
    y0_idx: np.ndarray
    y1_idx: np.ndarray
    wx: np.ndarray
    wy: np.ndarray

    def sample_into(self, source: np.ndarray, target: np.ndarray) -> None:
        if self.rows.size == 0:
            return
        v00 = source[self.y0_idx, self.x0_idx]
        v10 = source[self.y0_idx, self.x1_idx]
        v01 = source[self.y1_idx, self.x0_idx]
        v11 = source[self.y1_idx, self.x1_idx]
        values = (
            (1.0 - self.wx) * (1.0 - self.wy) * v00
            + self.wx * (1.0 - self.wy) * v10
            + (1.0 - self.wx) * self.wy * v01
            + self.wx * self.wy * v11
        )
        target[self.rows, self.cols] = values.mean(axis=1).astype(np.float32, copy=False)


def ship_cell_sample_offsets() -> tuple[np.ndarray, np.ndarray]:
    ratio = SHIP_GRID_CELL_SIZE_M / PYROSIM_GRID_CELL_SIZE_M
    samples_per_axis = int(round(ratio))
    if samples_per_axis <= 0 or not math.isclose(ratio, samples_per_axis, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "SHIP_GRID_CELL_SIZE_M must be an integer multiple of PYROSIM_GRID_CELL_SIZE_M "
            f"for sub-cell aggregation, got {SHIP_GRID_CELL_SIZE_M} / {PYROSIM_GRID_CELL_SIZE_M}."
        )
    offsets = (np.arange(samples_per_axis, dtype=np.float32) + 0.5) * PYROSIM_GRID_CELL_SIZE_M
    dx, dy = np.meshgrid(offsets, offsets)
    return dx.ravel(), dy.ravel()


def build_mesh_sampler(
    mesh: MeshSpec,
    header: SliceHeader,
    ship_shape: tuple[int, int],
    *,
    is_last_mesh: bool,
    origin_x: float,
    origin_y: float,
) -> MeshSampler:
    rows, cols = ship_shape
    row_grid, col_grid = np.indices((rows, cols), dtype=np.float32)
    center_x = origin_x + (col_grid + 0.5) * SHIP_GRID_CELL_SIZE_M
    center_y = origin_y + (row_grid + 0.5) * SHIP_GRID_CELL_SIZE_M

    x_min, x_max, y_min, y_max, _, _ = mesh.xb
    if is_last_mesh:
        inside_x = (center_x >= x_min) & (center_x <= x_max)
    else:
        inside_x = (center_x >= x_min) & (center_x < x_max)
    inside = inside_x & (center_y >= y_min) & (center_y <= y_max)

    target_rows, target_cols = np.nonzero(inside)
    if target_rows.size == 0:
        return MeshSampler(
            mesh=mesh,
            rows=target_rows.astype(np.intp),
            cols=target_cols.astype(np.intp),
            x0_idx=np.empty((0, 0), dtype=np.intp),
            x1_idx=np.empty((0, 0), dtype=np.intp),
            y0_idx=np.empty((0, 0), dtype=np.intp),
            y1_idx=np.empty((0, 0), dtype=np.intp),
            wx=np.empty((0, 0), dtype=np.float32),
            wy=np.empty((0, 0), dtype=np.float32),
        )

    sample_dx, sample_dy = ship_cell_sample_offsets()
    sample_x = origin_x + target_cols[:, None] * SHIP_GRID_CELL_SIZE_M + sample_dx[None, :]
    sample_y = origin_y + target_rows[:, None] * SHIP_GRID_CELL_SIZE_M + sample_dy[None, :]

    fx = (sample_x - x_min) / (x_max - x_min) * (header.nx - 1)
    fy = (sample_y - y_min) / (y_max - y_min) * (header.ny - 1)
    fx = np.clip(fx, 0.0, float(header.nx - 1))
    fy = np.clip(fy, 0.0, float(header.ny - 1))
    x0_idx = np.floor(fx).astype(np.intp)
    y0_idx = np.floor(fy).astype(np.intp)
    x1_idx = np.clip(x0_idx + 1, 0, header.nx - 1)
    y1_idx = np.clip(y0_idx + 1, 0, header.ny - 1)
    wx = (fx - x0_idx).astype(np.float32)
    wy = (fy - y0_idx).astype(np.float32)

    return MeshSampler(
        mesh=mesh,
        rows=target_rows.astype(np.intp),
        cols=target_cols.astype(np.intp),
        x0_idx=x0_idx,
        x1_idx=x1_idx,
        y0_idx=y0_idx,
        y1_idx=y1_idx,
        wx=wx,
        wy=wy,
    )


def scenario_file(scenario: ScenarioSpec, mesh_id: int, variable_id: int) -> Path:
    subdirs = {
        1: "能见度切片数据",
        2: "温度切片数据",
        3: "CO切片数据",
    }
    return scenario.root / subdirs[variable_id] / f"{scenario.prefix}_{mesh_id}_{variable_id}.sf"


def selected_frame_indices(frame_count: int) -> list[int]:
    indices = list(range(0, frame_count, max(1, int(FRAME_STRIDE))))
    if MAX_FRAMES is not None:
        indices = indices[: int(MAX_FRAMES)]
    if not indices:
        raise ValueError("No frames selected. Check MAX_FRAMES and FRAME_STRIDE.")
    return indices


def temperature_risk(temperature_c: np.ndarray) -> np.ndarray:
    value = ((temperature_c - AMBIENT_TEMPERATURE_C) / (TEMPERATURE_RISK_THRESHOLD_C - AMBIENT_TEMPERATURE_C)) ** 2
    return np.clip(value, 0.0, 1.0).astype(np.float32, copy=False)


def co_risk(co_ppm: np.ndarray) -> np.ndarray:
    return np.clip(co_ppm / CO_RISK_THRESHOLD_PPM, 0.0, 1.0).astype(np.float32, copy=False)


def visibility_risk(visibility_m: np.ndarray) -> np.ndarray:
    value = (VISIBILITY_CLEAR_M - visibility_m) / (VISIBILITY_CLEAR_M - VISIBILITY_CRITICAL_M)
    return np.clip(value, 0.0, 1.0).astype(np.float32, copy=False)


def make_output_arrays(
    scenario_dir: Path,
    n_frames: int,
    ship_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    shape = (n_frames, ship_shape[0], ship_shape[1])
    arrays = {
        "temperature_c": np.lib.format.open_memmap(
            scenario_dir / "temperature_c.npy", mode="w+", dtype=np.float32, shape=shape
        ),
        "co_ppm": np.lib.format.open_memmap(
            scenario_dir / "co_ppm.npy", mode="w+", dtype=np.float32, shape=shape
        ),
        "visibility_m": np.lib.format.open_memmap(
            scenario_dir / "visibility_m.npy", mode="w+", dtype=np.float32, shape=shape
        ),
        "life_risk": np.lib.format.open_memmap(
            scenario_dir / "life_risk.npy", mode="w+", dtype=np.float32, shape=shape
        ),
        "risk_visibility": np.lib.format.open_memmap(
            scenario_dir / "risk_visibility.npy", mode="w+", dtype=np.float32, shape=shape
        ),
    }
    if SAVE_COMPONENT_RISKS:
        arrays["risk_temperature"] = np.lib.format.open_memmap(
            scenario_dir / "risk_temperature.npy", mode="w+", dtype=np.float32, shape=shape
        )
        arrays["risk_co"] = np.lib.format.open_memmap(
            scenario_dir / "risk_co.npy", mode="w+", dtype=np.float32, shape=shape
        )
    return arrays


def save_preview_png(path: Path, grid: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    configure_matplotlib_fonts(plt)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 3.8))
    image = ax.imshow(grid, origin="upper", cmap="inferno", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("x / ship grid column")
    ax.set_ylabel("y / ship grid row")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, label="life risk")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_grid_txt(path: Path, grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, grid, fmt=RISK_TXT_FMT)


def configure_matplotlib_fonts(plt) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def heatmap_extent(ship_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    rows, cols = ship_shape
    return (-0.5, cols - 0.5, rows - 0.5, -0.5)


def load_ship_background(ship_grid: np.ndarray) -> np.ndarray:
    try:
        import matplotlib.pyplot as plt

        if SHIP_BACKGROUND_IMAGE.exists():
            configure_matplotlib_fonts(plt)
            image = plt.imread(SHIP_BACKGROUND_IMAGE)
            if image.ndim == 3 and image.shape[2] == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                image = rgb * alpha + (1.0 - alpha)
            return image
    except Exception:
        pass

    background = np.ones((ship_grid.shape[0], ship_grid.shape[1], 3), dtype=np.float32)
    background[ship_grid == 0] = np.array([0.50, 0.82, 0.96], dtype=np.float32)
    background[ship_grid == 1] = np.array([1.00, 1.00, 1.00], dtype=np.float32)
    background[ship_grid == 2] = np.array([0.05, 0.62, 0.20], dtype=np.float32)
    return background


def risk_rgba(grid: np.ndarray, rgb: tuple[float, float, float]) -> np.ndarray:
    clipped = np.clip(grid, 0.0, 1.0).astype(np.float32, copy=False)
    rgba = np.zeros((grid.shape[0], grid.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = clipped * HEATMAP_MAX_ALPHA
    return rgba


def transparent_cmap(name: str, rgb: tuple[float, float, float]):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        name,
        [(rgb[0], rgb[1], rgb[2], 0.0), (rgb[0], rgb[1], rgb[2], HEATMAP_MAX_ALPHA)],
    )


def save_overlay_png(
    path: Path,
    ship_grid: np.ndarray,
    risk_grid: np.ndarray,
    *,
    title: str,
    rgb: tuple[float, float, float],
    cmap_name: str,
    label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except Exception:
        return

    configure_matplotlib_fonts(plt)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 3.8))
    extent = heatmap_extent(tuple(ship_grid.shape))
    ax.imshow(load_ship_background(ship_grid), origin="upper", extent=extent)
    ax.imshow(risk_rgba(risk_grid, rgb), origin="upper", interpolation="nearest", extent=extent)
    ax.set_title(title)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=transparent_cmap(cmap_name, rgb), norm=Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=HEATMAP_DPI)
    plt.close(fig)


def save_overlay_video(
    path: Path,
    ship_grid: np.ndarray,
    risk_array: np.ndarray,
    times: np.ndarray,
    *,
    title: str,
    rgb: tuple[float, float, float],
    cmap_name: str,
    label: str,
) -> None:
    if not SAVE_HEATMAP_VIDEO:
        return
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
        from matplotlib.colors import Normalize
    except Exception as exc:
        print(f"  skip heatmap video {path.name}: {exc}")
        return

    configure_matplotlib_fonts(plt)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(risk_array.shape[0])
    frame_ids = list(range(0, n_frames, max(1, int(HEATMAP_FRAME_STRIDE))))
    if frame_ids and frame_ids[-1] != n_frames - 1:
        frame_ids.append(n_frames - 1)

    fig, ax = plt.subplots(figsize=(14, 3.8))
    extent = heatmap_extent(tuple(ship_grid.shape))
    ax.imshow(load_ship_background(ship_grid), origin="upper", extent=extent)
    overlay = ax.imshow(
        risk_rgba(risk_array[frame_ids[0]], rgb),
        origin="upper",
        interpolation="nearest",
        extent=extent,
    )
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=transparent_cmap(cmap_name, rgb), norm=Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label=label)
    fig.tight_layout()

    try:
        writer = FFMpegWriter(fps=HEATMAP_FPS)
        with writer.saving(fig, str(path), dpi=HEATMAP_DPI):
            for frame_id in frame_ids:
                overlay.set_data(risk_rgba(risk_array[frame_id], rgb))
                ax.set_title(f"{title}  t={float(times[frame_id]):.2f}s")
                writer.grab_frame()
    except Exception as exc:
        print(f"  skip heatmap video {path.name}: {exc}")
    finally:
        plt.close(fig)


def open_readers(scenario: ScenarioSpec, meshes: list[MeshSpec], variable_id: int) -> dict[int, FdsSliceFile]:
    readers = {}
    for mesh in meshes:
        path = scenario_file(scenario, mesh.mesh_id, variable_id)
        if not path.exists():
            raise FileNotFoundError(path)
        readers[mesh.mesh_id] = FdsSliceFile(path)
    return readers


def validate_readers(readers: Iterable[FdsSliceFile], variable_name: str) -> None:
    for reader in readers:
        header = reader.header
        if header.k1 != header.k2:
            raise ValueError(f"{reader.path.name}: expected a 2D slice, got k1={header.k1}, k2={header.k2}")
        if header.values_per_frame <= 0:
            raise ValueError(f"{reader.path.name}: invalid slice shape.")
        print(
            f"  {variable_name}: {reader.path.name}, "
            f"shape={header.ny}x{header.nx}, frames={reader.frame_count}, unit={header.unit}"
        )


def build_scenario(scenario: ScenarioSpec) -> Path:
    print(f"Building fire risk fields for {scenario.name}...")
    ship_grid = load_txt_grid(SHIP_MAP_FILE)
    ship_shape = tuple(int(v) for v in ship_grid.shape)
    print(f"  ship map shape: {ship_shape[0]}x{ship_shape[1]}")

    fds_path = scenario_fds_path(scenario)
    meshes = load_meshes_from_fds(fds_path)
    print(f"  fds file: {fds_path.name}")
    for mesh in meshes:
        print(f"  mesh {mesh.mesh_id}: IJK={mesh.ijk}, XB={mesh.xb}")
    origin_x = min(mesh.xb[0] for mesh in meshes) if SHIP_ORIGIN_X is None else float(SHIP_ORIGIN_X)
    origin_y = min(mesh.xb[2] for mesh in meshes) if SHIP_ORIGIN_Y is None else float(SHIP_ORIGIN_Y)
    samples_per_axis = int(round(SHIP_GRID_CELL_SIZE_M / PYROSIM_GRID_CELL_SIZE_M))
    print(
        f"  ship/FDS mapping: origin=({origin_x:.3f}, {origin_y:.3f}), "
        f"ship_cell={SHIP_GRID_CELL_SIZE_M}m, fds_cell={PYROSIM_GRID_CELL_SIZE_M}m, "
        f"samples_per_ship_cell={samples_per_axis * samples_per_axis}"
    )

    visibility_readers = open_readers(scenario, meshes, variable_id=1)
    temperature_readers = open_readers(scenario, meshes, variable_id=2)
    co_readers = open_readers(scenario, meshes, variable_id=3)
    validate_readers(visibility_readers.values(), "visibility")
    validate_readers(temperature_readers.values(), "temperature")
    validate_readers(co_readers.values(), "co")

    frame_count = min(
        min(reader.frame_count for reader in visibility_readers.values()),
        min(reader.frame_count for reader in temperature_readers.values()),
        min(reader.frame_count for reader in co_readers.values()),
    )
    frame_indices = selected_frame_indices(frame_count)
    n_frames = len(frame_indices)
    print(f"  selected frames: {n_frames} / {frame_count}, stride={FRAME_STRIDE}")

    last_mesh_id = meshes[-1].mesh_id
    samplers = {
        mesh.mesh_id: build_mesh_sampler(
            mesh,
            temperature_readers[mesh.mesh_id].header,
            ship_shape,
            is_last_mesh=mesh.mesh_id == last_mesh_id,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        for mesh in meshes
    }
    for mesh_id, sampler in samplers.items():
        print(f"  mesh {mesh_id}: mapped ship cells={len(sampler.rows)}")

    scenario_dir = OUTPUT_DIR / scenario.name
    arrays = make_output_arrays(scenario_dir, n_frames, ship_shape)
    times = np.lib.format.open_memmap(scenario_dir / "times_seconds.npy", mode="w+", dtype=np.float32, shape=(n_frames,))

    temp_field = np.empty(ship_shape, dtype=np.float32)
    co_field = np.empty(ship_shape, dtype=np.float32)
    visibility_field = np.empty(ship_shape, dtype=np.float32)
    max_life = np.zeros(ship_shape, dtype=np.float32)
    max_visibility_risk = np.zeros(ship_shape, dtype=np.float32)
    life_txt_dir = scenario_dir / "life_risk_txt"
    visibility_txt_dir = scenario_dir / "visibility_risk_txt"

    for out_idx, frame_idx in enumerate(frame_indices):
        temp_field.fill(AMBIENT_TEMPERATURE_C)
        co_field.fill(0.0)
        visibility_field.fill(VISIBILITY_CLEAR_M)
        frame_time = math.nan

        for mesh in meshes:
            visibility_time, visibility_slice = visibility_readers[mesh.mesh_id].read_frame(frame_idx)
            temp_time, temp_slice = temperature_readers[mesh.mesh_id].read_frame(frame_idx)
            co_time, co_slice = co_readers[mesh.mesh_id].read_frame(frame_idx)
            if math.isnan(frame_time):
                frame_time = temp_time
            if abs(temp_time - co_time) > 1e-4 or abs(temp_time - visibility_time) > 1e-4:
                raise ValueError(
                    f"Temperature/CO/visibility time mismatch at frame {frame_idx}, "
                    f"mesh {mesh.mesh_id}: T={temp_time}, CO={co_time}, V={visibility_time}"
                )
            sampler = samplers[mesh.mesh_id]
            sampler.sample_into(visibility_slice, visibility_field)
            sampler.sample_into(temp_slice, temp_field)
            sampler.sample_into(co_slice * CO_VOLUME_FRACTION_TO_PPM, co_field)

        r_temp = temperature_risk(temp_field)
        r_co = co_risk(co_field)
        r_visibility = visibility_risk(visibility_field)
        life = np.maximum(r_temp, r_co).astype(np.float32, copy=False)

        arrays["temperature_c"][out_idx] = temp_field
        arrays["co_ppm"][out_idx] = co_field
        arrays["visibility_m"][out_idx] = visibility_field
        arrays["life_risk"][out_idx] = life
        arrays["risk_visibility"][out_idx] = r_visibility
        if SAVE_COMPONENT_RISKS:
            arrays["risk_temperature"][out_idx] = r_temp
            arrays["risk_co"][out_idx] = r_co
        times[out_idx] = np.float32(frame_time)
        max_life = np.maximum(max_life, life)
        max_visibility_risk = np.maximum(max_visibility_risk, r_visibility)

        if SAVE_RISK_TXT_FRAMES:
            suffix = f"frame_{out_idx:06d}_t_{frame_time:.2f}s.txt"
            save_grid_txt(life_txt_dir / suffix, life)
            save_grid_txt(visibility_txt_dir / suffix, r_visibility)

        if out_idx == 0 or out_idx == n_frames - 1 or (out_idx + 1) % 100 == 0:
            print(
                f"  frame {out_idx + 1:04d}/{n_frames:04d}, "
                f"t={frame_time:.2f}s, "
                f"Tmax={float(temp_field.max()):.2f}C, "
                f"COmax={float(co_field.max()):.2f}ppm, "
                f"Vmin={float(visibility_field.min()):.2f}m, "
                f"life_risk_max={float(life.max()):.4f}, "
                f"visibility_risk_max={float(r_visibility.max()):.4f}"
            )

    for arr in arrays.values():
        arr.flush()
    times.flush()
    np.save(scenario_dir / "life_risk_max.npy", max_life.astype(np.float32))
    np.save(scenario_dir / "risk_visibility_max.npy", max_visibility_risk.astype(np.float32))
    if SAVE_RISK_TXT_FRAMES:
        save_grid_txt(scenario_dir / "life_risk_max.txt", max_life)
        save_grid_txt(scenario_dir / "risk_visibility_max.txt", max_visibility_risk)

    metadata = {
        "scenario": scenario.name,
        "source_root": str(scenario.root),
        "fds_file": str(fds_path),
        "ship_map_file": str(SHIP_MAP_FILE),
        "ship_map_shape": list(ship_shape),
        "selected_frame_count": n_frames,
        "source_frame_count": frame_count,
        "frame_stride": FRAME_STRIDE,
        "max_frames": MAX_FRAMES,
        "co_input_unit": "mol/mol",
        "co_output_unit": "ppm",
        "co_volume_fraction_to_ppm": CO_VOLUME_FRACTION_TO_PPM,
        "temperature_unit": "C",
        "visibility_unit": "m",
        "life_risk_formula": "max(clip(((T-20)/(60-20))^2,0,1), clip(C_ppm/500,0,1))",
        "visibility_risk_formula": "clip((10-V)/(10-5),0,1)",
        "ship_grid_mapping": {
            "method": "mean of PyroSim/FDS sub-cell center samples inside each ship grid cell",
            "ship_grid_cell_size_m": SHIP_GRID_CELL_SIZE_M,
            "pyrosim_grid_cell_size_m": PYROSIM_GRID_CELL_SIZE_M,
            "samples_per_axis": samples_per_axis,
            "samples_per_ship_cell": samples_per_axis * samples_per_axis,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "x_samples": "origin_x + col * ship_grid_cell_size_m + (k + 0.5) * pyrosim_grid_cell_size_m",
            "y_samples": "origin_y + row * ship_grid_cell_size_m + (k + 0.5) * pyrosim_grid_cell_size_m",
        },
        "meshes": [
            {
                "mesh_id": mesh.mesh_id,
                "ijk": list(mesh.ijk),
                "xb": list(mesh.xb),
                "mapped_ship_cells": int(len(samplers[mesh.mesh_id].rows)),
            }
            for mesh in meshes
        ],
        "outputs": {
            "times_seconds": "times_seconds.npy",
            "temperature_c": "temperature_c.npy",
            "co_ppm": "co_ppm.npy",
            "visibility_m": "visibility_m.npy",
            "life_risk": "life_risk.npy",
            "life_risk_max": "life_risk_max.npy",
            "risk_visibility": "risk_visibility.npy",
            "risk_visibility_max": "risk_visibility_max.npy",
            "life_risk_txt": "life_risk_txt/*.txt" if SAVE_RISK_TXT_FRAMES else None,
            "visibility_risk_txt": "visibility_risk_txt/*.txt" if SAVE_RISK_TXT_FRAMES else None,
            "risk_temperature": "risk_temperature.npy" if SAVE_COMPONENT_RISKS else None,
            "risk_co": "risk_co.npy" if SAVE_COMPONENT_RISKS else None,
            "life_risk_heatmap_max": "life_risk_heatmap_max.png" if SAVE_HEATMAP_MAX_PNG else None,
            "visibility_risk_heatmap_max": "visibility_risk_heatmap_max.png" if SAVE_HEATMAP_MAX_PNG else None,
            "life_risk_heatmap_video": "life_risk_heatmap.mp4" if SAVE_HEATMAP_VIDEO else None,
            "visibility_risk_heatmap_video": "visibility_risk_heatmap.mp4" if SAVE_HEATMAP_VIDEO else None,
        },
    }
    (scenario_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if SAVE_PREVIEW_PNG:
        save_preview_png(scenario_dir / "life_risk_max.png", max_life, f"{scenario.name} max life risk")
    if SAVE_HEATMAP_MAX_PNG:
        save_overlay_png(
            scenario_dir / "life_risk_heatmap_max.png",
            ship_grid,
            max_life,
            title=f"{scenario.name} max life risk",
            rgb=(1.0, 0.0, 0.0),
            cmap_name="transparent_red",
            label="life risk",
        )
        save_overlay_png(
            scenario_dir / "visibility_risk_heatmap_max.png",
            ship_grid,
            max_visibility_risk,
            title=f"{scenario.name} max visibility risk",
            rgb=(0.0, 0.0, 0.0),
            cmap_name="transparent_black",
            label="visibility risk",
        )
    if SAVE_HEATMAP_VIDEO:
        save_overlay_video(
            scenario_dir / "life_risk_heatmap.mp4",
            ship_grid,
            arrays["life_risk"],
            times,
            title=f"{scenario.name} life risk",
            rgb=(1.0, 0.0, 0.0),
            cmap_name="transparent_red",
            label="life risk",
        )
        save_overlay_video(
            scenario_dir / "visibility_risk_heatmap.mp4",
            ship_grid,
            arrays["risk_visibility"],
            times,
            title=f"{scenario.name} visibility risk",
            rgb=(0.0, 0.0, 0.0),
            cmap_name="transparent_black",
            label="visibility risk",
        )

    print(f"  saved to: {scenario_dir}")
    return scenario_dir


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        build_scenario(scenario)


if __name__ == "__main__":
    main()
