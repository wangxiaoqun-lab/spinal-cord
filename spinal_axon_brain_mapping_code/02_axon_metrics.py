from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
SWC_DIR = Path('.')
OUTPUT_DIR = Path('.')
LONGITUDINAL_SCALE = 7.1
TARGETS = [('strong_central_3', 'strong_central_3.swc'), ('stong-4_central', 'stong-4_central_done.swc'), ('tree-10_dorsal', 'tree-10_dorsal_done.swc'), ('tree-11_dorsal', 'tree-11_dorsal_done.swc')]
REGIONS = [('Sacral', 530.0, 1300.0, '[530, 1300)'), ('Lumbar', 1300.0, 2750.0, '[1300, 2750)'), ('Thoracic', 2750.0, 4500.0, '[2750, 4500)'), ('Cervical', 4500.0, 6000.0, '[4500, 6000]')]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()

def infer_neurites(path: Path) -> dict:
    data = np.loadtxt(path, comments='#', dtype=np.float64)
    ids = data[:, 0].astype(np.int64)
    swc_types = data[:, 1].astype(np.int16)
    xyz = data[:, 2:5].astype(np.float64)
    radius = data[:, 5].astype(np.float64)
    parent_ids = data[:, 6].astype(np.int64)
    id_to_index = {int(node_id): i for i, node_id in enumerate(ids)}
    original_parent = np.array([id_to_index.get(int(parent_id), -1) for parent_id in parent_ids], dtype=np.int64)
    roots = np.flatnonzero(original_parent < 0)
    soma_candidates = np.flatnonzero(swc_types == 1)
    declared_soma = int(soma_candidates[0] if soma_candidates.size else roots[0])
    thickest = int(np.argmax(radius))
    soma_gap = float(np.linalg.norm(xyz[declared_soma] - xyz[thickest]))
    if radius[thickest] > 1.5 * radius[declared_soma] and soma_gap > 10.0 * radius[thickest]:
        root_index = thickest
        soma_source = 'thickest_remote_region'
    else:
        root_index = declared_soma
        soma_source = 'swc_type_1'
    neighbors: list[list[int]] = [[] for _ in range(len(ids))]
    for child, parent in enumerate(original_parent):
        if parent >= 0:
            neighbors[child].append(int(parent))
            neighbors[int(parent)].append(child)
    soma_radius = max(float(radius[root_index]), float(np.percentile(radius, 99.9)))
    soma_envelope_radius = max(30.0, 6.0 * soma_radius)
    distance_to_soma = np.linalg.norm(xyz - xyz[root_index], axis=1)
    soma_candidate = distance_to_soma <= soma_envelope_radius
    soma_mask = np.zeros(len(ids), dtype=bool)
    soma_mask[root_index] = True
    flood = [root_index]
    for node in flood:
        for neighbor in neighbors[node]:
            if soma_candidate[neighbor] and (not soma_mask[neighbor]):
                soma_mask[neighbor] = True
                flood.append(neighbor)
    parent_index = np.full(len(ids), -2, dtype=np.int64)
    parent_index[soma_mask] = -1
    order = list(np.flatnonzero(soma_mask))
    for node in order:
        for neighbor in neighbors[node]:
            if parent_index[neighbor] == -2:
                parent_index[neighbor] = node
                order.append(neighbor)
    if np.any(parent_index == -2):
        raise ValueError(f'SWC contains nodes disconnected from inferred soma: {path}')
    children: list[list[int]] = [[] for _ in range(len(ids))]
    for child, parent in enumerate(parent_index):
        if parent >= 0:
            children[int(parent)].append(child)
    starts = [child for soma_node in np.flatnonzero(soma_mask) for child in children[int(soma_node)] if not soma_mask[child]]
    edge_length = np.zeros(len(ids), dtype=np.float64)
    valid = parent_index >= 0
    edge_length[valid] = np.linalg.norm(xyz[valid] - xyz[parent_index[valid]], axis=1)
    child_count = np.fromiter((len(items) for items in children), dtype=np.int16)
    branches = []
    for branch_id, start in enumerate(starts, 1):
        stack = [start]
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(children[node])
        idx = np.asarray(nodes, dtype=np.int64)
        branches.append({'branch_id': branch_id, 'nodes': idx, 'length': float(edge_length[idx].sum()), 'reach': float(np.linalg.norm(xyz[idx] - xyz[root_index], axis=1).max()), 'median_radius': float(np.median(radius[idx])), 'forks': int(np.count_nonzero(child_count[idx] > 1))})
    long_range = np.zeros(len(branches), dtype=bool)
    if len(branches) > 1:
        lengths = np.array([branch['length'] for branch in branches])
        reaches = np.array([branch['reach'] for branch in branches])
        radii = np.array([branch['median_radius'] for branch in branches])
        forks = np.array([branch['forks'] for branch in branches], dtype=float)

        def robust_z(values: np.ndarray) -> np.ndarray:
            median = np.median(values)
            scale = np.median(np.abs(values - median))
            if scale < 1e-09:
                scale = np.std(values) + 1e-09
            return (values - median) / (1.4826 * scale + 1e-09)
        scores = 0.45 * robust_z(np.log1p(lengths)) + 0.4 * robust_z(np.log1p(reaches)) - 0.35 * robust_z(np.log1p(radii)) - 0.15 * robust_z(np.log1p(forks / np.maximum(lengths, 1.0)))
        eligible = (lengths >= 0.01 * lengths.max()) | (reaches >= 0.03 * reaches.max())
        long_range = eligible & (lengths >= 0.35 * lengths.max()) & (reaches >= 0.35 * reaches.max())
        if not np.any(long_range):
            long_range[int(np.argmax(np.where(eligible, scores, -np.inf)))] = True
        for index, branch in enumerate(branches):
            branch['score'] = float(scores[index])
            branch['is_axon'] = bool(long_range[index])
    elif branches:
        branches[0]['score'] = 0.0
        branches[0]['is_axon'] = False
    inferred_type = np.full(len(ids), 2, dtype=np.int8)
    for branch in branches:
        if branch.get('is_axon', False):
            inferred_type[branch['nodes']] = 3
    inferred_type[soma_mask] = 1
    return {'ids': ids, 'xyz': xyz, 'radius': radius, 'parent_index': parent_index, 'inferred_type': inferred_type, 'root_index': root_index, 'soma_mask': soma_mask, 'soma_source': soma_source, 'soma_envelope_radius': soma_envelope_radius, 'branches': branches}

def clipped_lengths(lengths: np.ndarray, longitudinal_start: np.ndarray, longitudinal_end: np.ndarray, lower: float, upper: float) -> np.ndarray:
    delta = longitudinal_end - longitudinal_start
    fractions = np.zeros_like(lengths)
    constant = np.abs(delta) < 1e-12
    fractions[constant & (longitudinal_start >= lower) & (longitudinal_start <= upper)] = 1.0
    moving = ~constant
    t0 = np.zeros_like(lengths)
    t1 = np.zeros_like(lengths)
    t0[moving] = (lower - longitudinal_start[moving]) / delta[moving]
    t1[moving] = (upper - longitudinal_start[moving]) / delta[moving]
    enter = np.maximum(0.0, np.minimum(t0, t1))
    leave = np.minimum(1.0, np.maximum(t0, t1))
    fractions[moving] = np.maximum(0.0, leave[moving] - enter[moving])
    return lengths * fractions

def region_for_position(position: float) -> str | None:
    if 530.0 <= position < 1300.0:
        return 'Sacral'
    if 1300.0 <= position < 2750.0:
        return 'Lumbar'
    if 2750.0 <= position < 4500.0:
        return 'Thoracic'
    if 4500.0 <= position <= 6000.0:
        return 'Cervical'
    return None

def analyze_neuron(name: str, path: Path) -> dict:
    neuron = infer_neurites(path)
    ids = neuron['ids']
    xyz = neuron['xyz']
    parent = neuron['parent_index']
    inferred_type = neuron['inferred_type']
    axon_children = np.flatnonzero((parent >= 0) & (inferred_type == 3))
    axon_parents = parent[axon_children]
    vectors = xyz[axon_children] - xyz[axon_parents]
    lengths = np.linalg.norm(vectors, axis=1)
    s0 = xyz[axon_parents, 1] / LONGITUDINAL_SCALE
    s1 = xyz[axon_children, 1] / LONGITUDINAL_SCALE
    total_full_length = float(lengths.sum())
    total_target_length = float(clipped_lengths(lengths, s0, s1, 530.0, 6000.0).sum())
    axon_child_count = np.zeros(len(ids), dtype=np.int32)
    np.add.at(axon_child_count, axon_parents, 1)
    terminal_indices = np.flatnonzero((inferred_type == 3) & (axon_child_count == 0))
    terminal_positions = xyz[terminal_indices, 1] / LONGITUDINAL_SCALE
    terminal_in_target = [int(index) for index, position in zip(terminal_indices, terminal_positions) if region_for_position(float(position)) is not None]
    terminal_details = []
    for index in terminal_in_target:
        position = float(xyz[index, 1] / LONGITUDINAL_SCALE)
        terminal_details.append({'neuron': name, 'node_id': int(ids[index]), 'html_x': float(xyz[index, 0]), 'html_y': float(xyz[index, 1]), 'html_z': float(xyz[index, 2]), 'longitudinal_position': position, 'region': region_for_position(position)})
    region_rows = []
    for region_name, lower, upper, interval_label in REGIONS:
        length = float(clipped_lengths(lengths, s0, s1, lower, upper).sum())
        terminal_count = sum((item['region'] == region_name for item in terminal_details))
        region_rows.append({'neuron': name, 'region': region_name, 'interval': interval_label, 'lower': lower, 'upper': upper, 'axon_length': length, 'terminal_count': int(terminal_count)})
    region_length_sum = float(sum((row['axon_length'] for row in region_rows)))
    region_terminal_sum = int(sum((row['terminal_count'] for row in region_rows)))
    length_error = region_length_sum - total_target_length
    if abs(length_error) > max(1e-06, total_target_length * 1e-10):
        raise RuntimeError(f'Region length reconciliation failed for {name}: {length_error}')
    if region_terminal_sum != len(terminal_details):
        raise RuntimeError(f'Terminal reconciliation failed for {name}')
    root_index = int(neuron['root_index'])
    diagnostics = {'neuron': name, 'source_file': str(path), 'sha256': sha256(path), 'node_count': int(len(ids)), 'axon_edge_count': int(len(axon_children)), 'full_axon_length': total_full_length, 'target_axon_length': total_target_length, 'full_axon_terminal_count': int(len(terminal_indices)), 'target_axon_terminal_count': int(len(terminal_details)), 'soma_node_id': int(ids[root_index]), 'soma_html_x': float(xyz[root_index, 0]), 'soma_html_y': float(xyz[root_index, 1]), 'soma_html_z': float(xyz[root_index, 2]), 'soma_source': neuron['soma_source'], 'soma_envelope_nodes': int(neuron['soma_mask'].sum()), 'soma_envelope_radius': float(neuron['soma_envelope_radius']), 'primary_branch_count': int(len(neuron['branches'])), 'axon_primary_branch_count': int(sum((branch.get('is_axon', False) for branch in neuron['branches']))), 'length_reconciliation_error': length_error, 'terminal_reconciliation_error': region_terminal_sum - len(terminal_details)}
    return {'regions': region_rows, 'terminals': terminal_details, 'diagnostics': diagnostics}

def run_segment_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [analyze_neuron(name, SWC_DIR / filename) for name, filename in TARGETS]
    payload = {'parameters': {'longitudinal_coordinate': 'SWC Y / 7.1', 'length_coordinate_system': 'raw SWC XYZ = embedded HTML XYZ', 'normalization_range': '[530, 6000]', 'terminal_definition': 'inferred axon node with no inferred axon child after soma rerooting', 'regions': [{'name': name, 'lower': lower, 'upper': upper, 'interval': interval} for name, lower, upper, interval in REGIONS]}, 'region_rows': [row for result in results for row in result['regions']], 'terminal_rows': [row for result in results for row in result['terminals']], 'diagnostics': [result['diagnostics'] for result in results]}
    output_path = OUTPUT_DIR / 'axon_segment_analysis.json'
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output_path), 'region_rows': len(payload['region_rows']), 'terminal_rows': len(payload['terminal_rows']), 'summary': [{'neuron': item['neuron'], 'target_length': item['target_axon_length'], 'target_terminals': item['target_axon_terminal_count']} for item in payload['diagnostics']]}, ensure_ascii=False, indent=2))
BIN_WIDTH = 100.0
TARGET_LOWER = 530.0
TARGET_UPPER = 6000.0
DENSITY_SCALE = 10000.0
OUTPUT_PATH = OUTPUT_DIR / 'axon_extended_analysis.json'

def clipped_portions(lengths: np.ndarray, s0: np.ndarray, s1: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delta = s1 - s0
    enter = np.zeros_like(lengths)
    leave = np.ones_like(lengths)
    moving = np.abs(delta) >= 1e-12
    t0 = np.zeros_like(lengths)
    t1 = np.zeros_like(lengths)
    t0[moving] = (lower - s0[moving]) / delta[moving]
    t1[moving] = (upper - s0[moving]) / delta[moving]
    enter[moving] = np.maximum(0.0, np.minimum(t0[moving], t1[moving]))
    leave[moving] = np.minimum(1.0, np.maximum(t0[moving], t1[moving]))
    constant_inside = ~moving & (s0 >= lower) & (s0 <= upper)
    enter[~moving] = 0.0
    leave[~moving] = constant_inside[~moving].astype(np.float64)
    fraction = np.maximum(0.0, leave - enter)
    clipped = lengths * fraction
    start_position = s0 + delta * enter
    end_position = s0 + delta * leave
    return (clipped, enter, leave, start_position, end_position)

def in_interval(values: np.ndarray, lower: float, upper: float, is_last: bool=False) -> np.ndarray:
    if is_last:
        return (values >= lower) & (values <= upper)
    return (values >= lower) & (values < upper)

def distribution_summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {'count': 0, 'mean': None, 'sd': None, 'min': None, 'p10': None, 'median': None, 'p90': None, 'max': None}
    return {'count': int(values.size), 'mean': float(np.mean(values)), 'sd': float(np.std(values)), 'min': float(np.min(values)), 'p10': float(np.percentile(values, 10)), 'median': float(np.percentile(values, 50)), 'p90': float(np.percentile(values, 90)), 'max': float(np.max(values))}

def analyze_extended(name: str, path: Path) -> dict:
    neuron = infer_neurites(path)
    ids = neuron['ids']
    xyz = neuron['xyz']
    parent = neuron['parent_index']
    inferred = neuron['inferred_type']
    axon_children = np.flatnonzero((parent >= 0) & (inferred == 3))
    axon_parents = parent[axon_children]
    lengths = np.linalg.norm(xyz[axon_children] - xyz[axon_parents], axis=1)
    s0 = xyz[axon_parents, 1] / LONGITUDINAL_SCALE
    s1 = xyz[axon_children, 1] / LONGITUDINAL_SCALE
    target_lengths, _, _, target_start, target_end = clipped_portions(lengths, s0, s1, TARGET_LOWER, TARGET_UPPER)
    target_mask = target_lengths > 0
    total_length = float(np.sum(target_lengths))
    axon_child_count = np.zeros(len(ids), dtype=np.int32)
    np.add.at(axon_child_count, axon_parents, 1)
    terminal_indices = np.flatnonzero((inferred == 3) & (axon_child_count == 0))
    branch_indices = np.flatnonzero((inferred == 3) & (axon_child_count >= 2))
    terminal_positions_all = xyz[terminal_indices, 1] / LONGITUDINAL_SCALE
    branch_positions_all = xyz[branch_indices, 1] / LONGITUDINAL_SCALE
    terminal_keep = (terminal_positions_all >= TARGET_LOWER) & (terminal_positions_all <= TARGET_UPPER)
    branch_keep = (branch_positions_all >= TARGET_LOWER) & (branch_positions_all <= TARGET_UPPER)
    terminal_indices = terminal_indices[terminal_keep]
    branch_indices = branch_indices[branch_keep]
    terminal_positions = terminal_positions_all[terminal_keep]
    branch_positions = branch_positions_all[branch_keep]
    terminal_rows = []
    for index, position in zip(terminal_indices, terminal_positions):
        terminal_rows.append({'neuron': name, 'node_id': int(ids[index]), 'html_x': float(xyz[index, 0]), 'html_y': float(xyz[index, 1]), 'html_z': float(xyz[index, 2]), 'longitudinal_position': float(position), 'region': region_for_position(float(position))})
    branch_rows = []
    for index, position in zip(branch_indices, branch_positions):
        branch_rows.append({'neuron': name, 'node_id': int(ids[index]), 'axon_child_count': int(axon_child_count[index]), 'html_x': float(xyz[index, 0]), 'html_y': float(xyz[index, 1]), 'html_z': float(xyz[index, 2]), 'longitudinal_position': float(position), 'region': region_for_position(float(position))})
    region_rows = []
    for region_name, lower, upper, interval in REGIONS:
        region_lengths, _, _, _, _ = clipped_portions(lengths, s0, s1, lower, upper)
        region_length = float(np.sum(region_lengths))
        terminal_count = int(np.count_nonzero(in_interval(terminal_positions, lower, upper, upper == TARGET_UPPER)))
        branch_count = int(np.count_nonzero(in_interval(branch_positions, lower, upper, upper == TARGET_UPPER)))
        span = float(upper - lower)
        region_rows.append({'neuron': name, 'region': region_name, 'interval': interval, 'lower': float(lower), 'upper': float(upper), 'span': span, 'axon_length': region_length, 'terminal_count': terminal_count, 'branch_point_count': branch_count, 'terminal_density_per_10000': terminal_count / region_length * DENSITY_SCALE if region_length else None, 'branch_density_per_10000': branch_count / region_length * DENSITY_SCALE if region_length else None, 'occupancy_intensity': region_length / span})
    bin_rows = []
    bin_starts = np.arange(TARGET_LOWER, TARGET_UPPER, BIN_WIDTH)
    for lower in bin_starts:
        upper = min(lower + BIN_WIDTH, TARGET_UPPER)
        is_last = upper == TARGET_UPPER
        bin_lengths, _, _, _, _ = clipped_portions(lengths, s0, s1, float(lower), float(upper))
        bin_length = float(np.sum(bin_lengths))
        terminal_count = int(np.count_nonzero(in_interval(terminal_positions, lower, upper, is_last)))
        branch_count = int(np.count_nonzero(in_interval(branch_positions, lower, upper, is_last)))
        bin_rows.append({'neuron': name, 'bin_start': float(lower), 'bin_end': float(upper), 'bin_center': float((lower + upper) / 2.0), 'axon_length': bin_length, 'terminal_count': terminal_count, 'branch_point_count': branch_count})
    clipped_lengths = target_lengths[target_mask]
    clipped_starts = target_start[target_mask]
    clipped_ends = target_end[target_mask]
    midpoints = (clipped_starts + clipped_ends) / 2.0
    mean_position = float(np.sum(clipped_lengths * midpoints) / total_length)
    mean_square = float(np.sum(clipped_lengths * (clipped_starts ** 2 + clipped_starts * clipped_ends + clipped_ends ** 2) / 3.0) / total_length)
    sd_position = float(np.sqrt(max(0.0, mean_square - mean_position ** 2)))
    overview = {'neuron': name, 'target_axon_length': total_length, 'target_terminal_count': int(len(terminal_indices)), 'target_branch_point_count': int(len(branch_indices)), 'terminal_density_per_10000': len(terminal_indices) / total_length * DENSITY_SCALE, 'branch_density_per_10000': len(branch_indices) / total_length * DENSITY_SCALE, 'length_weighted_center': mean_position, 'length_weighted_sd': sd_position, 'axon_position_min': float(min(np.min(clipped_starts), np.min(clipped_ends))), 'axon_position_max': float(max(np.max(clipped_starts), np.max(clipped_ends))), 'terminal_distribution': distribution_summary(terminal_positions), 'branch_distribution': distribution_summary(branch_positions), 'source_file': str(path), 'sha256': sha256(path), 'node_count': int(len(ids)), 'axon_edge_count': int(len(axon_children)), 'soma_node_id': int(ids[int(neuron['root_index'])])}
    region_length_error = float(sum((row['axon_length'] for row in region_rows)) - total_length)
    bin_length_error = float(sum((row['axon_length'] for row in bin_rows)) - total_length)
    if abs(region_length_error) > max(1e-06, total_length * 1e-10):
        raise RuntimeError(f'Region length reconciliation failed for {name}: {region_length_error}')
    if abs(bin_length_error) > max(1e-06, total_length * 1e-10):
        raise RuntimeError(f'Bin length reconciliation failed for {name}: {bin_length_error}')
    if sum((row['terminal_count'] for row in region_rows)) != len(terminal_indices):
        raise RuntimeError(f'Region terminal reconciliation failed for {name}')
    if sum((row['branch_point_count'] for row in region_rows)) != len(branch_indices):
        raise RuntimeError(f'Region branch reconciliation failed for {name}')
    diagnostics = {'neuron': name, 'region_length_error': region_length_error, 'bin_length_error': bin_length_error, 'region_terminal_error': sum((row['terminal_count'] for row in region_rows)) - len(terminal_indices), 'bin_terminal_error': sum((row['terminal_count'] for row in bin_rows)) - len(terminal_indices), 'region_branch_error': sum((row['branch_point_count'] for row in region_rows)) - len(branch_indices), 'bin_branch_error': sum((row['branch_point_count'] for row in bin_rows)) - len(branch_indices)}
    return {'overview': overview, 'regions': region_rows, 'bins': bin_rows, 'terminals': terminal_rows, 'branch_points': branch_rows, 'diagnostics': diagnostics}

def run_extended_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [analyze_extended(name, SWC_DIR / filename) for name, filename in TARGETS]
    payload = {'parameters': {'target_range': [TARGET_LOWER, TARGET_UPPER], 'longitudinal_coordinate': 'SWC Y / 7.1', 'length_coordinate_system': 'raw SWC XYZ = embedded HTML XYZ', 'bin_width': BIN_WIDTH, 'density_scale': DENSITY_SCALE, 'terminal_definition': 'inferred axon node with zero inferred axon children after soma rerooting', 'branch_point_definition': 'inferred axon node with at least two inferred axon children after soma rerooting'}, 'overview_rows': [item['overview'] for item in results], 'region_rows': [row for item in results for row in item['regions']], 'bin_rows': [row for item in results for row in item['bins']], 'terminal_rows': [row for item in results for row in item['terminals']], 'branch_point_rows': [row for item in results for row in item['branch_points']], 'diagnostics': [item['diagnostics'] for item in results]}
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT_PATH), 'overview': [{'neuron': row['neuron'], 'length': row['target_axon_length'], 'terminals': row['target_terminal_count'], 'branch_points': row['target_branch_point_count'], 'center': row['length_weighted_center']} for row in payload['overview_rows']], 'region_rows': len(payload['region_rows']), 'bin_rows': len(payload['bin_rows']), 'terminal_rows': len(payload['terminal_rows']), 'branch_point_rows': len(payload['branch_point_rows'])}, ensure_ascii=False, indent=2))
