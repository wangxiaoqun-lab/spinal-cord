from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
_axon = importlib.import_module("02_axon_metrics")
LONGITUDINAL_SCALE = _axon.LONGITUDINAL_SCALE
OUTPUT_DIR = _axon.OUTPUT_DIR
SWC_DIR = _axon.SWC_DIR
TARGETS = _axon.TARGETS
infer_neurites = _axon.infer_neurites
LABEL_PATH = OUTPUT_DIR / 'private_gray_region_labels.npz'
OUTPUT_PATH = OUTPUT_DIR / 'gray_region_axon_extended_analysis.json'
TARGET_LOWER = 530.0
TARGET_UPPER = 6000.0
BIN_WIDTH = 100.0
DENSITY_SCALE = 10000.0
VOLUME_DENSITY_SCALE = 1000000.0
DISPLAY_SCALE_XYZ = np.asarray([7.1, 6.0, 7.1], dtype=np.float64)
REGION_NAMES = {1: 'Dorsal', 2: 'Intermediate', 3: 'Ventral'}
REGION_ORDER = [1, 2, 3]

def gray_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()

def distribution_summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {'count': 0, 'mean': None, 'sd': None, 'min': None, 'p10': None, 'median': None, 'p90': None, 'max': None}
    return {'count': int(values.size), 'mean': float(np.mean(values)), 'sd': float(np.std(values)), 'min': float(np.min(values)), 'p10': float(np.percentile(values, 10)), 'median': float(np.percentile(values, 50)), 'p90': float(np.percentile(values, 90)), 'max': float(np.max(values))}

def clip_segment_to_box(p0: np.ndarray, p1: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[float, float] | None:
    delta = p1 - p0
    enter = 0.0
    leave = 1.0
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            if p0[axis] < lower[axis] or p0[axis] > upper[axis]:
                return None
            continue
        t0 = (lower[axis] - p0[axis]) / delta[axis]
        t1 = (upper[axis] - p0[axis]) / delta[axis]
        if t0 > t1:
            t0, t1 = (t1, t0)
        enter = max(enter, float(t0))
        leave = min(leave, float(t1))
        if leave <= enter:
            return None
    return (enter, leave)

class LabelField:

    def __init__(self, label_path: Path) -> None:
        cached = np.load(label_path)
        self.labels = cached['labels']
        self.z = cached['z'].astype(np.float64)
        self.y = cached['y'].astype(np.float64)
        self.x = cached['x'].astype(np.float64)
        self.starts = np.asarray([self.x[0], self.y[0], self.z[0]], dtype=np.float64)
        self.ends = np.asarray([self.x[-1], self.y[-1], self.z[-1]], dtype=np.float64)
        self.steps = np.asarray([self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0]], dtype=np.float64)
        if not np.allclose(self.steps, self.steps[0]):
            raise ValueError(f'Expected equal XYZ grid steps, got {self.steps}')
        self.step = float(self.steps[0])
        self.shape_xyz = np.asarray([len(self.x), len(self.y), len(self.z)], dtype=np.int64)

    def _index_xyz(self, point_xyz: np.ndarray) -> np.ndarray | None:
        if np.any(point_xyz < self.starts) or np.any(point_xyz > self.ends):
            return None
        indices = np.floor((point_xyz - self.starts) / self.step + 0.5).astype(np.int64)
        indices = np.minimum(np.maximum(indices, 0), self.shape_xyz - 1)
        return indices

    def point_label(self, point_xyz: np.ndarray) -> int:
        indices = self._index_xyz(point_xyz)
        if indices is None:
            return 0
        ix, iy, iz = (int(value) for value in indices)
        return int(self.labels[iz, iy, ix])

    def segment_parts(self, p0: np.ndarray, p1: np.ndarray, display_length: float) -> list[tuple[int, float, float, float]]:
        clipped = clip_segment_to_box(p0, p1, self.starts, self.ends)
        if clipped is None or display_length <= 0:
            return []
        enter, leave = clipped
        delta = p1 - p0
        breakpoints = [enter, leave]
        for axis in range(3):
            axis_delta = float(delta[axis])
            if abs(axis_delta) < 1e-12:
                continue
            c_enter = float(p0[axis] + axis_delta * enter)
            c_leave = float(p0[axis] + axis_delta * leave)
            c_min, c_max = sorted((c_enter, c_leave))
            first_boundary_index = int(np.ceil((c_min - self.starts[axis]) / self.step - 0.5))
            last_boundary_index = int(np.floor((c_max - self.starts[axis]) / self.step - 0.5))
            for boundary_index in range(first_boundary_index, last_boundary_index + 1):
                boundary = self.starts[axis] + (boundary_index + 0.5) * self.step
                t = (boundary - p0[axis]) / axis_delta
                if enter + 1e-12 < t < leave - 1e-12:
                    breakpoints.append(float(t))
        values = np.unique(np.round(np.asarray(breakpoints, dtype=np.float64), 14))
        parts: list[tuple[int, float, float, float]] = []
        for t0, t1 in zip(values[:-1], values[1:]):
            if t1 <= t0:
                continue
            midpoint = p0 + delta * ((t0 + t1) / 2.0)
            code = self.point_label(midpoint)
            if code not in REGION_NAMES:
                continue
            part_length = display_length * float(t1 - t0)
            z0 = float(p0[2] + delta[2] * t0)
            z1 = float(p0[2] + delta[2] * t1)
            if parts and parts[-1][0] == code and (abs(parts[-1][3] - z0) < 1e-09):
                previous = parts.pop()
                parts.append((code, previous[1] + part_length, previous[2], z1))
            else:
                parts.append((code, part_length, z0, z1))
        return parts

def length_in_target_range(lengths: np.ndarray, positions0: np.ndarray, positions1: np.ndarray) -> float:
    total = 0.0
    for length, q0, q1 in zip(lengths, positions0, positions1):
        delta = float(q1 - q0)
        if abs(delta) < 1e-12:
            if TARGET_LOWER <= q0 <= TARGET_UPPER:
                total += float(length)
            continue
        t0 = (TARGET_LOWER - q0) / delta
        t1 = (TARGET_UPPER - q0) / delta
        enter = max(0.0, min(t0, t1))
        leave = min(1.0, max(t0, t1))
        if leave > enter:
            total += float(length) * float(leave - enter)
    return total

def add_part_to_bins(bin_lengths: np.ndarray, part_length: float, z0: float, z1: float) -> None:
    if part_length <= 0:
        return
    if abs(z1 - z0) < 1e-12:
        if TARGET_LOWER <= z0 <= TARGET_UPPER:
            index = min(int((z0 - TARGET_LOWER) // BIN_WIDTH), len(bin_lengths) - 1)
            if index >= 0:
                bin_lengths[index] += part_length
        return
    low = max(min(z0, z1), TARGET_LOWER)
    high = min(max(z0, z1), TARGET_UPPER)
    if high <= low:
        return
    start_index = max(0, int((low - TARGET_LOWER) // BIN_WIDTH))
    end_index = min(len(bin_lengths) - 1, int((high - TARGET_LOWER) // BIN_WIDTH))
    full_span = abs(z1 - z0)
    for index in range(start_index, end_index + 1):
        bin_low = TARGET_LOWER + index * BIN_WIDTH
        bin_high = min(bin_low + BIN_WIDTH, TARGET_UPPER)
        overlap = max(0.0, min(high, bin_high) - max(low, bin_low))
        if overlap > 0:
            bin_lengths[index] += part_length * overlap / full_span

def analyze_gray_neuron(name: str, path: Path, field: LabelField, region_volume: dict[int, dict]) -> dict:
    neuron = infer_neurites(path)
    ids = neuron['ids']
    xyz = neuron['xyz']
    parent = neuron['parent_index']
    inferred = neuron['inferred_type']
    raw_xyz = np.column_stack([xyz[:, 0] / 7.1, xyz[:, 2] / 6.0, xyz[:, 1] / LONGITUDINAL_SCALE])
    axon_children = np.flatnonzero((parent >= 0) & (inferred == 3))
    axon_parents = parent[axon_children]
    display_lengths = np.linalg.norm(xyz[axon_children] - xyz[axon_parents], axis=1)
    target_length_530_6000 = length_in_target_range(display_lengths, raw_xyz[axon_parents, 2], raw_xyz[axon_children, 2])
    bin_starts = np.arange(TARGET_LOWER, TARGET_UPPER, BIN_WIDTH, dtype=np.float64)
    bin_ends = np.minimum(bin_starts + BIN_WIDTH, TARGET_UPPER)
    region_lengths = {code: 0.0 for code in REGION_ORDER}
    region_moment1 = {code: 0.0 for code in REGION_ORDER}
    region_moment2 = {code: 0.0 for code in REGION_ORDER}
    region_min = {code: np.inf for code in REGION_ORDER}
    region_max = {code: -np.inf for code in REGION_ORDER}
    bin_lengths = {code: np.zeros(len(bin_starts), dtype=np.float64) for code in REGION_ORDER}
    all_parts: list[tuple[int, float, float, float]] = []
    intersecting_edge_count = 0
    for edge_index, (child, parent_index, display_length) in enumerate(zip(axon_children, axon_parents, display_lengths), start=1):
        parts = field.segment_parts(raw_xyz[parent_index], raw_xyz[child], float(display_length))
        if parts:
            intersecting_edge_count += 1
        for code, part_length, z0, z1 in parts:
            region_lengths[code] += part_length
            region_moment1[code] += part_length * (z0 + z1) / 2.0
            region_moment2[code] += part_length * (z0 * z0 + z0 * z1 + z1 * z1) / 3.0
            region_min[code] = min(region_min[code], z0, z1)
            region_max[code] = max(region_max[code], z0, z1)
            add_part_to_bins(bin_lengths[code], part_length, z0, z1)
            all_parts.append((code, part_length, z0, z1))
        if edge_index % 25000 == 0:
            print(f'[{name}] classified {edge_index:,}/{len(axon_children):,} axon edges', flush=True)
    axon_child_count = np.zeros(len(ids), dtype=np.int32)
    np.add.at(axon_child_count, axon_parents, 1)
    terminal_indices_all = np.flatnonzero((inferred == 3) & (axon_child_count == 0))
    branch_indices_all = np.flatnonzero((inferred == 3) & (axon_child_count >= 2))

    def classified_node_rows(indices: np.ndarray, include_child_count: bool) -> tuple[list[dict], dict[int, list[float]]]:
        rows: list[dict] = []
        positions: dict[int, list[float]] = {code: [] for code in REGION_ORDER}
        for index in indices:
            code = field.point_label(raw_xyz[index])
            if code not in REGION_NAMES:
                continue
            position = float(raw_xyz[index, 2])
            row = {'neuron': name, 'node_id': int(ids[index]), 'html_x': float(xyz[index, 0]), 'html_y': float(xyz[index, 1]), 'html_z': float(xyz[index, 2]), 'tiff_x': float(raw_xyz[index, 0]), 'tiff_y': float(raw_xyz[index, 1]), 'tiff_z': position, 'region': REGION_NAMES[code], 'region_code': int(code)}
            if include_child_count:
                row['axon_child_count'] = int(axon_child_count[index])
            rows.append(row)
            positions[code].append(position)
        return (rows, positions)
    terminal_rows, terminal_positions = classified_node_rows(terminal_indices_all, False)
    branch_rows, branch_positions = classified_node_rows(branch_indices_all, True)
    terminal_counts = {code: len(terminal_positions[code]) for code in REGION_ORDER}
    branch_counts = {code: len(branch_positions[code]) for code in REGION_ORDER}
    for row in terminal_rows:
        index = min(int((row['tiff_z'] - TARGET_LOWER) // BIN_WIDTH), len(bin_starts) - 1)
        row['bin_index'] = int(index)
    for row in branch_rows:
        index = min(int((row['tiff_z'] - TARGET_LOWER) // BIN_WIDTH), len(bin_starts) - 1)
        row['bin_index'] = int(index)
    total_region_length = float(sum(region_lengths.values()))
    total_terminal_count = int(sum(terminal_counts.values()))
    total_branch_count = int(sum(branch_counts.values()))
    region_rows = []
    for code in REGION_ORDER:
        length = float(region_lengths[code])
        mean = region_moment1[code] / length if length else None
        variance = region_moment2[code] / length - mean * mean if length else None
        terminal_values = np.asarray(terminal_positions[code], dtype=np.float64)
        branch_values = np.asarray(branch_positions[code], dtype=np.float64)
        region_rows.append({'neuron': name, 'region': REGION_NAMES[code], 'region_code': code, 'label_voxel_count_ds4': region_volume[code]['label_voxel_count_ds4'], 'region_display_volume': region_volume[code]['region_display_volume'], 'axon_length': length, 'length_fraction': length / total_region_length if total_region_length else None, 'terminal_count': terminal_counts[code], 'terminal_fraction': terminal_counts[code] / total_terminal_count if total_terminal_count else None, 'terminal_density_per_10000': terminal_counts[code] / length * DENSITY_SCALE if length else None, 'branch_point_count': branch_counts[code], 'branch_fraction': branch_counts[code] / total_branch_count if total_branch_count else None, 'branch_density_per_10000': branch_counts[code] / length * DENSITY_SCALE if length else None, 'axon_length_per_million_volume': length / region_volume[code]['region_display_volume'] * VOLUME_DENSITY_SCALE, 'length_weighted_center': mean, 'length_weighted_sd': float(np.sqrt(max(0.0, variance))) if variance is not None else None, 'axon_position_min': float(region_min[code]) if np.isfinite(region_min[code]) else None, 'axon_position_max': float(region_max[code]) if np.isfinite(region_max[code]) else None, 'terminal_distribution': distribution_summary(terminal_values), 'branch_distribution': distribution_summary(branch_values)})
    bin_terminal_counts = {code: np.zeros(len(bin_starts), dtype=np.int32) for code in REGION_ORDER}
    bin_branch_counts = {code: np.zeros(len(bin_starts), dtype=np.int32) for code in REGION_ORDER}
    for row in terminal_rows:
        bin_terminal_counts[row['region_code']][row['bin_index']] += 1
    for row in branch_rows:
        bin_branch_counts[row['region_code']][row['bin_index']] += 1
    bin_rows = []
    for code in REGION_ORDER:
        for index, (lower, upper) in enumerate(zip(bin_starts, bin_ends)):
            length = float(bin_lengths[code][index])
            bin_rows.append({'neuron': name, 'region': REGION_NAMES[code], 'region_code': code, 'bin_start': float(lower), 'bin_end': float(upper), 'bin_center': float((lower + upper) / 2.0), 'axon_length': length, 'terminal_count': int(bin_terminal_counts[code][index]), 'branch_point_count': int(bin_branch_counts[code][index])})
    total_moment1 = sum(region_moment1.values())
    total_moment2 = sum(region_moment2.values())
    total_center = total_moment1 / total_region_length if total_region_length else None
    total_variance = total_moment2 / total_region_length - total_center * total_center if total_region_length else None
    terminal_all = np.asarray([row['tiff_z'] for row in terminal_rows], dtype=np.float64)
    branch_all = np.asarray([row['tiff_z'] for row in branch_rows], dtype=np.float64)
    overview = {'neuron': name, 'three_region_axon_length': total_region_length, 'three_region_terminal_count': total_terminal_count, 'three_region_branch_point_count': total_branch_count, 'terminal_density_per_10000': total_terminal_count / total_region_length * DENSITY_SCALE if total_region_length else None, 'branch_density_per_10000': total_branch_count / total_region_length * DENSITY_SCALE if total_region_length else None, 'length_weighted_center': total_center, 'length_weighted_sd': float(np.sqrt(max(0.0, total_variance))) if total_variance is not None else None, 'axon_position_min': float(min((value for value in region_min.values() if np.isfinite(value)))) if total_region_length else None, 'axon_position_max': float(max((value for value in region_max.values() if np.isfinite(value)))) if total_region_length else None, 'terminal_distribution': distribution_summary(terminal_all), 'branch_distribution': distribution_summary(branch_all), 'source_file': str(path), 'sha256': gray_sha256(path), 'node_count': int(len(ids)), 'axon_edge_count': int(len(axon_children)), 'three_region_intersecting_edge_count': int(intersecting_edge_count), 'soma_node_id': int(ids[int(neuron['root_index'])]), 'target_530_6000_axon_length': float(target_length_530_6000), 'three_region_length_coverage': total_region_length / target_length_530_6000 if target_length_530_6000 else None, 'full_axon_terminal_count': int(len(terminal_indices_all)), 'full_axon_branch_point_count': int(len(branch_indices_all))}
    region_length_error = float(sum((row['axon_length'] for row in region_rows)) - total_region_length)
    bin_length_error = float(sum((row['axon_length'] for row in bin_rows)) - total_region_length)
    region_terminal_error = int(sum((row['terminal_count'] for row in region_rows)) - total_terminal_count)
    bin_terminal_error = int(sum((row['terminal_count'] for row in bin_rows)) - total_terminal_count)
    region_branch_error = int(sum((row['branch_point_count'] for row in region_rows)) - total_branch_count)
    bin_branch_error = int(sum((row['branch_point_count'] for row in bin_rows)) - total_branch_count)
    tolerance = max(1e-06, total_region_length * 1e-10)
    if abs(region_length_error) > tolerance or abs(bin_length_error) > tolerance:
        raise RuntimeError(f'Length reconciliation failed for {name}: region={region_length_error}, bins={bin_length_error}')
    if any((value != 0 for value in (region_terminal_error, bin_terminal_error, region_branch_error, bin_branch_error))):
        raise RuntimeError(f'Node reconciliation failed for {name}')
    diagnostics = {'neuron': name, 'region_length_error': region_length_error, 'bin_length_error': bin_length_error, 'region_terminal_error': region_terminal_error, 'bin_terminal_error': bin_terminal_error, 'region_branch_error': region_branch_error, 'bin_branch_error': bin_branch_error, 'length_fraction_sum': float(sum((row['length_fraction'] for row in region_rows if row['length_fraction'] is not None))), 'terminal_fraction_sum': float(sum((row['terminal_fraction'] for row in region_rows if row['terminal_fraction'] is not None))), 'branch_fraction_sum': float(sum((row['branch_fraction'] for row in region_rows if row['branch_fraction'] is not None)))}
    return {'overview': overview, 'regions': region_rows, 'bins': bin_rows, 'terminals': terminal_rows, 'branch_points': branch_rows, 'diagnostics': diagnostics}

def run_gray_analysis() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, default=LABEL_PATH)
    parser.add_argument('--output', type=Path, default=OUTPUT_PATH)
    parser.add_argument('--partition-description', default='Private finalized dorsal/middle/ventral gray-matter labels.')
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    field = LabelField(args.labels)
    voxel_volume_display = float(field.step ** 3 * np.prod(DISPLAY_SCALE_XYZ))
    region_volume = {}
    for code in REGION_ORDER:
        count = int(np.count_nonzero(field.labels == code))
        region_volume[code] = {'region': REGION_NAMES[code], 'label_voxel_count_ds4': count, 'region_display_volume': count * voxel_volume_display}
    results = [analyze_gray_neuron(name, SWC_DIR / filename, field, region_volume) for name, filename in TARGETS]
    payload = {'parameters': {'three_region_label_path': str(args.labels), 'partition_description': args.partition_description, 'valid_tiff_z_range': [float(field.z[0]), float(field.z[-1])], 'label_grid_step_tiff_xyz': float(field.step), 'coordinate_mapping': {'tiff_x': 'SWC/HTML X / 7.1', 'tiff_y': 'SWC/HTML Z / 6.0', 'tiff_z': 'SWC/HTML Y / 7.1'}, 'length_coordinate_system': 'SWC/HTML display XYZ', 'classification': 'exact traversal of the nearest-neighbor ds4 categorical label volume; portions with code 0 are ignored', 'region_codes': {str(code): REGION_NAMES[code] for code in REGION_ORDER}, 'proportion_denominator': 'sum within dorsal + middle + ventral only', 'terminal_definition': 'inferred axon node with zero inferred axon children after soma rerooting', 'branch_point_definition': 'inferred axon node with at least two inferred axon children after soma rerooting', 'bin_width': BIN_WIDTH, 'density_scale': DENSITY_SCALE, 'volume_density_scale': VOLUME_DENSITY_SCALE}, 'region_volume': [region_volume[code] | {'region_code': code} for code in REGION_ORDER], 'overview_rows': [item['overview'] for item in results], 'region_rows': [row for item in results for row in item['regions']], 'bin_rows': [row for item in results for row in item['bins']], 'terminal_rows': [row for item in results for row in item['terminals']], 'branch_point_rows': [row for item in results for row in item['branch_points']], 'diagnostics': [item['diagnostics'] for item in results]}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'output': str(args.output), 'region_volume': payload['region_volume'], 'overview': payload['overview_rows'], 'region_rows': len(payload['region_rows']), 'bin_rows': len(payload['bin_rows']), 'terminal_rows': len(payload['terminal_rows']), 'branch_point_rows': len(payload['branch_point_rows']), 'diagnostics': payload['diagnostics']}, ensure_ascii=False, indent=2))
SEGMENTS = [{'code': 1, 'name': 'Cervical', 'lower': 4500.0, 'upper': 6000.0}, {'code': 2, 'name': 'Thoracic', 'lower': 2750.0, 'upper': 4500.0}, {'code': 3, 'name': 'Lumbar', 'lower': 1300.0, 'upper': 2750.0}, {'code': 4, 'name': 'Sacral', 'lower': 530.0, 'upper': 1300.0}]
SEGMENT_BY_CODE = {item['code']: item for item in SEGMENTS}
BOUNDARIES = (1300.0, 2750.0, 4500.0)

def segment_code_for_z(z: float) -> int | None:
    if 4500.0 <= z <= 6000.0:
        return 1
    if 2750.0 <= z < 4500.0:
        return 2
    if 1300.0 <= z < 2750.0:
        return 3
    if 530.0 <= z < 1300.0:
        return 4
    return None

def split_by_segments(part_length: float, z0: float, z1: float) -> list[tuple[int, float, float, float]]:
    if part_length <= 0:
        return []
    delta = z1 - z0
    if abs(delta) < 1e-12:
        code = segment_code_for_z(z0)
        return [] if code is None else [(code, part_length, z0, z1)]
    breakpoints = [0.0, 1.0]
    for boundary in BOUNDARIES:
        t = (boundary - z0) / delta
        if 1e-12 < t < 1.0 - 1e-12:
            breakpoints.append(float(t))
    values = sorted(set((round(value, 14) for value in breakpoints)))
    pieces = []
    for t0, t1 in zip(values[:-1], values[1:]):
        middle_z = z0 + delta * (t0 + t1) / 2.0
        code = segment_code_for_z(middle_z)
        if code is None:
            continue
        pieces.append((code, part_length * (t1 - t0), z0 + delta * t0, z0 + delta * t1))
    return pieces

def empty_accumulator() -> dict:
    return {'length': 0.0, 'moment1': 0.0, 'moment2': 0.0, 'min': np.inf, 'max': -np.inf, 'terminal_positions': [], 'branch_positions': []}

def summarize_cell(neuron_name: str, region_code: int, segment_code: int, accumulator: dict, volume: dict, total_length: float, total_terminals: int, total_branches: int) -> dict:
    length = float(accumulator['length'])
    center = accumulator['moment1'] / length if length else None
    variance = accumulator['moment2'] / length - center * center if length else None
    terminals = np.asarray(accumulator['terminal_positions'], dtype=np.float64)
    branches = np.asarray(accumulator['branch_positions'], dtype=np.float64)
    segment = SEGMENT_BY_CODE[segment_code]
    terminal_count = int(len(terminals))
    branch_count = int(len(branches))
    return {'neuron': neuron_name, 'region': REGION_NAMES[region_code], 'region_code': region_code, 'segment': segment['name'], 'segment_code': segment_code, 'segment_lower': segment['lower'], 'segment_upper': segment['upper'], 'label_voxel_count_ds4': volume['label_voxel_count_ds4'], 'region_segment_display_volume': volume['region_segment_display_volume'], 'axon_length': length, 'length_fraction': length / total_length if total_length else None, 'terminal_count': terminal_count, 'terminal_fraction': terminal_count / total_terminals if total_terminals else None, 'branch_point_count': branch_count, 'branch_fraction': branch_count / total_branches if total_branches else None, 'terminal_density_per_10000': terminal_count / length * DENSITY_SCALE if length else None, 'branch_density_per_10000': branch_count / length * DENSITY_SCALE if length else None, 'axon_length_per_million_volume': length / volume['region_segment_display_volume'] * VOLUME_DENSITY_SCALE if volume['region_segment_display_volume'] else None, 'length_weighted_center': center, 'length_weighted_sd': float(np.sqrt(max(0.0, variance))) if variance is not None else None, 'axon_position_min': float(accumulator['min']) if np.isfinite(accumulator['min']) else None, 'axon_position_max': float(accumulator['max']) if np.isfinite(accumulator['max']) else None, 'terminal_distribution': distribution_summary(terminals), 'branch_distribution': distribution_summary(branches)}

def analyze_gray_segment_neuron(name: str, path: Path, field: LabelField, volumes: dict[tuple[int, int], dict]) -> dict:
    neuron = infer_neurites(path)
    ids = neuron['ids']
    xyz = neuron['xyz']
    parent = neuron['parent_index']
    inferred = neuron['inferred_type']
    raw_xyz = np.column_stack([xyz[:, 0] / 7.1, xyz[:, 2] / 6.0, xyz[:, 1] / LONGITUDINAL_SCALE])
    axon_children = np.flatnonzero((parent >= 0) & (inferred == 3))
    axon_parents = parent[axon_children]
    display_lengths = np.linalg.norm(xyz[axon_children] - xyz[axon_parents], axis=1)
    target_length = length_in_target_range(display_lengths, raw_xyz[axon_parents, 2], raw_xyz[axon_children, 2])
    accumulators = {(region, segment['code']): empty_accumulator() for region in REGION_ORDER for segment in SEGMENTS}
    intersecting_edge_count = 0
    for edge_index, (child, parent_index, display_length) in enumerate(zip(axon_children, axon_parents, display_lengths), start=1):
        region_parts = field.segment_parts(raw_xyz[parent_index], raw_xyz[child], float(display_length))
        contributed = False
        for region_code, region_length, z0, z1 in region_parts:
            for segment_code, piece_length, piece_z0, piece_z1 in split_by_segments(region_length, z0, z1):
                acc = accumulators[region_code, segment_code]
                acc['length'] += piece_length
                acc['moment1'] += piece_length * (piece_z0 + piece_z1) / 2.0
                acc['moment2'] += piece_length * (piece_z0 * piece_z0 + piece_z0 * piece_z1 + piece_z1 * piece_z1) / 3.0
                acc['min'] = min(acc['min'], piece_z0, piece_z1)
                acc['max'] = max(acc['max'], piece_z0, piece_z1)
                contributed = True
        if contributed:
            intersecting_edge_count += 1
        if edge_index % 25000 == 0:
            print(f'[{name}] classified {edge_index:,}/{len(axon_children):,} axon edges', flush=True)
    axon_child_count = np.zeros(len(ids), dtype=np.int32)
    np.add.at(axon_child_count, axon_parents, 1)
    terminal_indices = np.flatnonzero((inferred == 3) & (axon_child_count == 0))
    branch_indices = np.flatnonzero((inferred == 3) & (axon_child_count >= 2))

    def classify_nodes(indices: np.ndarray, include_child_count: bool) -> list[dict]:
        rows = []
        for index in indices:
            region_code = field.point_label(raw_xyz[index])
            segment_code = segment_code_for_z(float(raw_xyz[index, 2]))
            if region_code not in REGION_NAMES or segment_code is None:
                continue
            position = float(raw_xyz[index, 2])
            acc = accumulators[region_code, segment_code]
            target = 'branch_positions' if include_child_count else 'terminal_positions'
            acc[target].append(position)
            row = {'neuron': name, 'node_id': int(ids[index]), 'html_x': float(xyz[index, 0]), 'html_y': float(xyz[index, 1]), 'html_z': float(xyz[index, 2]), 'tiff_x': float(raw_xyz[index, 0]), 'tiff_y': float(raw_xyz[index, 1]), 'tiff_z': position, 'region': REGION_NAMES[region_code], 'region_code': int(region_code), 'segment': SEGMENT_BY_CODE[segment_code]['name'], 'segment_code': segment_code}
            if include_child_count:
                row['axon_child_count'] = int(axon_child_count[index])
            rows.append(row)
        return rows
    terminal_rows = classify_nodes(terminal_indices, False)
    branch_rows = classify_nodes(branch_indices, True)
    total_length = float(sum((acc['length'] for acc in accumulators.values())))
    total_terminals = len(terminal_rows)
    total_branches = len(branch_rows)
    cell_rows = [summarize_cell(name, region_code, segment['code'], accumulators[region_code, segment['code']], volumes[region_code, segment['code']], total_length, total_terminals, total_branches) for region_code in REGION_ORDER for segment in SEGMENTS]
    all_terminal_positions = np.asarray([row['tiff_z'] for row in terminal_rows], dtype=np.float64)
    all_branch_positions = np.asarray([row['tiff_z'] for row in branch_rows], dtype=np.float64)
    total_moment1 = sum((acc['moment1'] for acc in accumulators.values()))
    total_moment2 = sum((acc['moment2'] for acc in accumulators.values()))
    total_center = total_moment1 / total_length if total_length else None
    total_variance = total_moment2 / total_length - total_center * total_center if total_length else None
    finite_min = [acc['min'] for acc in accumulators.values() if np.isfinite(acc['min'])]
    finite_max = [acc['max'] for acc in accumulators.values() if np.isfinite(acc['max'])]
    overview = {'neuron': name, 'twelve_cell_axon_length': total_length, 'twelve_cell_terminal_count': total_terminals, 'twelve_cell_branch_point_count': total_branches, 'terminal_density_per_10000': total_terminals / total_length * DENSITY_SCALE if total_length else None, 'branch_density_per_10000': total_branches / total_length * DENSITY_SCALE if total_length else None, 'length_weighted_center': total_center, 'length_weighted_sd': float(np.sqrt(max(0.0, total_variance))) if total_variance is not None else None, 'axon_position_min': float(min(finite_min)) if finite_min else None, 'axon_position_max': float(max(finite_max)) if finite_max else None, 'terminal_distribution': distribution_summary(all_terminal_positions), 'branch_distribution': distribution_summary(all_branch_positions), 'source_file': str(path), 'sha256': gray_sha256(path), 'node_count': int(len(ids)), 'axon_edge_count': int(len(axon_children)), 'twelve_cell_intersecting_edge_count': int(intersecting_edge_count), 'target_530_6000_axon_length': float(target_length), 'twelve_cell_length_coverage': total_length / target_length if target_length else None}
    diagnostics = {'neuron': name, 'cell_length_error': float(sum((row['axon_length'] for row in cell_rows)) - total_length), 'cell_terminal_error': int(sum((row['terminal_count'] for row in cell_rows)) - total_terminals), 'cell_branch_error': int(sum((row['branch_point_count'] for row in cell_rows)) - total_branches), 'length_fraction_sum': float(sum((row['length_fraction'] or 0 for row in cell_rows))), 'terminal_fraction_sum': float(sum((row['terminal_fraction'] or 0 for row in cell_rows))), 'branch_fraction_sum': float(sum((row['branch_fraction'] or 0 for row in cell_rows)))}
    if abs(diagnostics['cell_length_error']) > max(1e-06, total_length * 1e-10):
        raise RuntimeError(f"Length reconciliation failed for {name}: {diagnostics['cell_length_error']}")
    if diagnostics['cell_terminal_error'] or diagnostics['cell_branch_error']:
        raise RuntimeError(f'Node reconciliation failed for {name}: {diagnostics}')
    return {'overview': overview, 'cells': cell_rows, 'terminals': terminal_rows, 'branch_points': branch_rows, 'diagnostics': diagnostics}

def run_gray_segment_analysis() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version-name', required=True)
    parser.add_argument('--partition-description', required=True)
    parser.add_argument('--reference-analysis', type=Path)
    args = parser.parse_args()
    field = LabelField(args.labels)
    voxel_volume_display = float(field.step ** 3 * np.prod(DISPLAY_SCALE_XYZ))
    volumes = {}
    for region_code in REGION_ORDER:
        for segment in SEGMENTS:
            z_selection = np.asarray([segment_code_for_z(float(z)) == segment['code'] for z in field.z])
            count = int(np.count_nonzero(field.labels[z_selection] == region_code))
            volumes[region_code, segment['code']] = {'region': REGION_NAMES[region_code], 'region_code': region_code, 'segment': segment['name'], 'segment_code': segment['code'], 'label_voxel_count_ds4': count, 'region_segment_display_volume': count * voxel_volume_display}
    results = [analyze_gray_segment_neuron(name, SWC_DIR / filename, field, volumes) for name, filename in TARGETS]
    payload = {'parameters': {'version_name': args.version_name, 'partition_description': args.partition_description, 'three_region_label_path': str(args.labels), 'valid_tiff_z_range': [float(field.z[0]), float(field.z[-1])], 'label_grid_step_tiff_xyz': float(field.step), 'segments': SEGMENTS, 'interval_rule': 'Sacral[530,1300), Lumbar[1300,2750), Thoracic[2750,4500), Cervical[4500,6000]', 'coordinate_mapping': {'tiff_x': 'SWC/HTML X / 7.1', 'tiff_y': 'SWC/HTML Z / 6.0', 'tiff_z': 'SWC/HTML Y / 7.1'}, 'classification': 'exact traversal of ds4 categorical labels, then exact split at segment boundaries; code 0 ignored', 'proportion_denominator': 'sum across all 12 region-by-segment cells within the same neuron', 'density_scale': DENSITY_SCALE, 'volume_density_scale': VOLUME_DENSITY_SCALE}, 'cell_volume_rows': [volumes[region, segment['code']] for region in REGION_ORDER for segment in SEGMENTS], 'overview_rows': [item['overview'] for item in results], 'cell_rows': [row for item in results for row in item['cells']], 'terminal_rows': [row for item in results for row in item['terminals']], 'branch_point_rows': [row for item in results for row in item['branch_points']], 'diagnostics': [item['diagnostics'] for item in results]}
    if args.reference_analysis:
        reference = json.loads(args.reference_analysis.read_text(encoding='utf-8'))
        expected = {row['neuron']: row for row in reference['overview_rows']}
        for overview, diagnostic in zip(payload['overview_rows'], payload['diagnostics']):
            prior = expected[overview['neuron']]
            diagnostic['reference_length_error'] = overview['twelve_cell_axon_length'] - prior['three_region_axon_length']
            diagnostic['reference_terminal_error'] = overview['twelve_cell_terminal_count'] - prior['three_region_terminal_count']
            diagnostic['reference_branch_error'] = overview['twelve_cell_branch_point_count'] - prior['three_region_branch_point_count']
            if abs(diagnostic['reference_length_error']) > max(1e-06, overview['twelve_cell_axon_length'] * 1e-10):
                raise RuntimeError(f'Reference length mismatch: {diagnostic}')
            if diagnostic['reference_terminal_error'] or diagnostic['reference_branch_error']:
                raise RuntimeError(f'Reference node mismatch: {diagnostic}')
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'version': args.version_name, 'cell_rows': len(payload['cell_rows']), 'terminal_rows': len(payload['terminal_rows']), 'branch_point_rows': len(payload['branch_point_rows']), 'diagnostics': payload['diagnostics']}, ensure_ascii=False, indent=2), flush=True)
