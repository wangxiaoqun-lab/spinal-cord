


import argparse
import os
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import plotly.graph_objects as go
import plotly.io as pio

from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes





def laplacian_smooth(verts, faces, iters=8, lam=0.30):
    
    verts = verts.copy()
    n = verts.shape[0]

    neigh = [[] for _ in range(n)]
    for a, b, c in faces:
        neigh[a].extend([b, c])
        neigh[b].extend([a, c])
        neigh[c].extend([a, b])

    neigh = [np.array(sorted(set(ns)), dtype=int) for ns in neigh]

    for _ in range(int(iters)):
        delta = np.zeros_like(verts)
        for i, nb in enumerate(neigh):
            if nb.size == 0:
                continue
            delta[i] = verts[nb].mean(axis=0) - verts[i]
        verts += float(lam) * delta

    return verts


def camera_vector_from_viewpoint(elev_deg, azim_deg):
    
    elev = np.radians(float(elev_deg))
    azim = np.radians(float(azim_deg))

    v = np.array([
        np.sin(azim) * np.cos(elev),
        np.cos(azim) * np.cos(elev),
        np.sin(elev)
    ], dtype=float)

    return v / (np.linalg.norm(v) + 1e-12)


def _normalize_rows(mat, eps=1e-12):
    mat = np.asarray(mat, dtype=float)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norm + eps)


def _face_normals(verts, faces):
    
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)

    tri = verts[faces]
    v1 = tri[:, 1] - tri[:, 0]
    v2 = tri[:, 2] - tri[:, 0]
    normals = np.cross(v1, v2)

    return _normalize_rows(normals)


def _reorder_xyz_points(points_xyz, plot_axis_order="XYZ"):
    
    plot_axis_order = str(plot_axis_order).upper()
    if sorted(plot_axis_order) != ["X", "Y", "Z"] or len(plot_axis_order) != 3:
        raise ValueError("plot_axis_order must be a permutation of 'XYZ', e.g. 'XYZ' or 'XZY'.")

    idx_map = {"X": 0, "Y": 1, "Z": 2}
    order = [idx_map[a] for a in plot_axis_order]

    points_xyz = np.asarray(points_xyz, dtype=float)
    return points_xyz[:, order]


def _reorder_xyz_segments(segments_xyz, plot_axis_order="XYZ"):
    
    plot_axis_order = str(plot_axis_order).upper()
    if sorted(plot_axis_order) != ["X", "Y", "Z"] or len(plot_axis_order) != 3:
        raise ValueError("plot_axis_order must be a permutation of 'XYZ', e.g. 'XYZ' or 'XZY'.")

    idx_map = {"X": 0, "Y": 1, "Z": 2}
    order = [idx_map[a] for a in plot_axis_order]

    segments_xyz = np.asarray(segments_xyz, dtype=float)
    return segments_xyz[:, :, order]


def _cap_line_arrays(xs, ys, zs, max_segments=None):
    
    if max_segments is None:
        return xs, ys, zs

    max_len = int(max_segments) * 3

    if len(xs) > max_len:
        xs = xs[:max_len]
        ys = ys[:max_len]
        zs = zs[:max_len]

    return xs, ys, zs





def get_tiff_zyx_shape_fast(tiff_path, tiff_axes="ZYX", verbose=True):
    
    tiff_axes = str(tiff_axes).upper()
    if set(tiff_axes) != set("ZYX") or len(tiff_axes) != 3:
        raise ValueError("tiff_axes must be a permutation of 'ZYX', for example 'ZYX' or 'YXZ'.")

    with tifffile.TiffFile(tiff_path) as tif:
        n_pages = len(tif.pages)
        first_shape = tif.pages[0].shape
        first_dtype = tif.pages[0].dtype

        if n_pages > 1 and len(first_shape) == 2:
            z = n_pages
            y, x = first_shape
            shape_zyx = (z, y, x)

            if verbose:
                print(f"[shape] {tiff_path}")
                print(f"        pages={n_pages}, first_page_shape={first_shape}, dtype={first_dtype}")
                print(f"        interpreted ZYX={shape_zyx}")
                print(f"        interpreted XYZ={(x, y, z)}")

            return shape_zyx

    arr = tifffile.memmap(tiff_path)

    if arr.ndim != 3:
        raise ValueError(f"Cannot interpret TIFF as 3D stack: {tiff_path}, memmap shape={arr.shape}")

    order = [tiff_axes.index("Z"), tiff_axes.index("Y"), tiff_axes.index("X")]
    shape_zyx = tuple(np.asarray(arr.shape)[order].astype(int).tolist())

    if verbose:
        print(f"[shape] {tiff_path}")
        print(f"        memmap shape={arr.shape}, axes={tiff_axes}")
        print(f"        interpreted ZYX={shape_zyx}")
        print(f"        interpreted XYZ={(shape_zyx[2], shape_zyx[1], shape_zyx[0])}")

    return shape_zyx


def check_tiff_shapes_same(tiff_specs, tiff_axes="ZYX", strict=True, verbose=True):
    
    if len(tiff_specs) == 0:
        raise ValueError("tiff_specs is empty.")

    records = []

    for spec in tiff_specs:
        path = spec["path"]
        name = spec.get("name", os.path.basename(path))

        shape_zyx = get_tiff_zyx_shape_fast(
            tiff_path=path,
            tiff_axes=tiff_axes,
            verbose=verbose
        )

        records.append({
            "name": name,
            "path": path,
            "shape_zyx": shape_zyx
        })

    ref_shape = records[0]["shape_zyx"]
    all_same = all(item["shape_zyx"] == ref_shape for item in records)

    if verbose:
        print("\n========== TIFF shape check ==========")
        for item in records:
            flag = "OK" if item["shape_zyx"] == ref_shape else "MISMATCH"
            print(f"{flag:8s} {item['name']}: ZYX={item['shape_zyx']}")
        print("reference ZYX:", ref_shape)
        print("all same:", all_same)
        print("======================================\n")

    if strict and not all_same:
        msg = ["TIFF shapes are not identical:"]
        for item in records:
            msg.append(f"  {item['name']}: {item['shape_zyx']}")
        raise ValueError("\n".join(msg))

    return all_same, ref_shape, records


def read_tiff_foreground_as_mask(
    tiff_path,
    tiff_axes="ZYX",
    threshold=0,
    crop_zyx=None,
    mesh_downsample=(4, 4, 4),
    verbose=True
):
    
    tiff_axes = str(tiff_axes).upper()
    if set(tiff_axes) != set("ZYX") or len(tiff_axes) != 3:
        raise ValueError("tiff_axes must be a permutation of 'ZYX', for example 'ZYX' or 'YXZ'.")

    dz, dy, dx = [int(v) for v in mesh_downsample]

    if verbose:
        print(f"[TIFF] path: {tiff_path}")
        print("[TIFF] trying tifffile.memmap ...")

    arr = None

    try:
        arr_try = tifffile.memmap(tiff_path)

        if verbose:
            print("[TIFF] memmap successful")
            print("[TIFF] memmap shape:", arr_try.shape)
            print("[TIFF] memmap dtype:", arr_try.dtype)

        if arr_try.ndim == 3:
            arr = arr_try

        elif arr_try.ndim == 2:
            if verbose:
                print("[TIFF] memmap returned a 2D page, likely a multi-page TIFF.")
                print("[TIFF] will read pages manually.")
            arr = None

        else:
            raise ValueError(f"Unsupported memmap ndim: {arr_try.ndim}, shape={arr_try.shape}")

    except Exception as e:
        if verbose:
            print("[TIFF] memmap failed, will use TiffFile pages.")
            print("[TIFF] reason:", repr(e))
        arr = None

    
    if arr is not None:
        if verbose:
            print("[TIFF] using 3D memmap array")
            print("[TIFF] raw shape:", arr.shape)
            print("[TIFF] raw axes assumed:", tiff_axes)

        order = [tiff_axes.index("Z"), tiff_axes.index("Y"), tiff_axes.index("X")]
        arr_zyx = np.transpose(arr, order)

        Z, Y, X = arr_zyx.shape
        full_shape_zyx = (Z, Y, X)

        if crop_zyx is None:
            z0, z1 = 0, Z
            y0, y1 = 0, Y
            x0, x1 = 0, X
        else:
            (z0, z1), (y0, y1), (x0, x1) = crop_zyx
            z0 = max(0, int(z0)); z1 = min(Z, int(z1))
            y0 = max(0, int(y0)); y1 = min(Y, int(y1))
            x0 = max(0, int(x0)); x1 = min(X, int(x1))

        sub = arr_zyx[z0:z1:dz, y0:y1:dy, x0:x1:dx]
        mask_small = np.asarray(sub) > threshold

        offset_zyx = np.array([z0, y0, x0], dtype=float)
        spacing_zyx = np.array([dz, dy, dx], dtype=float)

        if verbose:
            print("[TIFF] interpreted ZYX shape:", full_shape_zyx)
            print(f"[TIFF] crop ZYX: z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1}")
            print(f"[TIFF] downsample ZYX: dz={dz}, dy={dy}, dx={dx}")
            print("[TIFF] mask_small shape:", mask_small.shape)
            print("[TIFF] mask_small True voxels:", int(mask_small.sum()))

        if not mask_small.any():
            raise ValueError("No foreground voxels found after threshold/crop/downsample.")

        return mask_small, offset_zyx, spacing_zyx, full_shape_zyx

    
    if verbose:
        print("[TIFF] opening with tifffile.TiffFile for page-wise reading ...")

    with tifffile.TiffFile(tiff_path) as tif:
        n_pages = len(tif.pages)
        first_shape = tif.pages[0].shape
        first_dtype = tif.pages[0].dtype

        if verbose:
            print("[TIFF] number of pages:", n_pages)
            print("[TIFF] first page shape:", first_shape)
            print("[TIFF] first page dtype:", first_dtype)

        if n_pages <= 1:
            raise ValueError(
                f"TIFF has only {n_pages} page and memmap returned 2D shape={first_shape}. "
                "This is not a 3D stack."
            )

        if len(first_shape) != 2:
            raise ValueError(f"Expected each TIFF page to be 2D, but first page shape={first_shape}")

        Z = n_pages
        Y, X = first_shape
        full_shape_zyx = (Z, Y, X)

        if tiff_axes != "ZYX" and verbose:
            print("⚠️ Multi-page TIFF is treated as ZYX because pages are Z slices.")

        if crop_zyx is None:
            z0, z1 = 0, Z
            y0, y1 = 0, Y
            x0, x1 = 0, X
        else:
            (z0, z1), (y0, y1), (x0, x1) = crop_zyx
            z0 = max(0, int(z0)); z1 = min(Z, int(z1))
            y0 = max(0, int(y0)); y1 = min(Y, int(y1))
            x0 = max(0, int(x0)); x1 = min(X, int(x1))

        z_indices = list(range(z0, z1, dz))

        if verbose:
            print("[TIFF] interpreted as multi-page ZYX")
            print("[TIFF] full ZYX shape:", full_shape_zyx)
            print(f"[TIFF] crop ZYX: z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1}")
            print(f"[TIFF] downsample ZYX: dz={dz}, dy={dy}, dx={dx}")
            print("[TIFF] reading downsampled pages:", len(z_indices))

        mask_list = []

        for zi in z_indices:
            page = tif.pages[zi].asarray()
            sub = page[y0:y1:dy, x0:x1:dx]
            mask_list.append(sub > threshold)

        mask_small = np.stack(mask_list, axis=0)

        offset_zyx = np.array([z0, y0, x0], dtype=float)
        spacing_zyx = np.array([dz, dy, dx], dtype=float)

        if verbose:
            print("[TIFF] mask_small shape:", mask_small.shape)
            print("[TIFF] mask_small True voxels:", int(mask_small.sum()))
            print("[TIFF] offset_zyx:", offset_zyx)
            print("[TIFF] spacing_zyx:", spacing_zyx)

        if not mask_small.any():
            raise ValueError("No foreground voxels found after threshold/crop/downsample.")

        return mask_small, offset_zyx, spacing_zyx, full_shape_zyx





def build_surface_mesh_from_mask(
    mask_small,
    offset_zyx,
    spacing_zyx,
    sigma=1.0,
    level=0.5,
    pad_small=2,
    smooth_iters=8,
    smooth_lam=0.30,
    display_scale_zyx=(1.0, 1.0, 1.0),
    verbose=True
):
    
    mask_small = np.asarray(mask_small, dtype=bool)
    spacing_zyx = np.asarray(spacing_zyx, dtype=float)
    offset_zyx = np.asarray(offset_zyx, dtype=float)
    display_scale_zyx = np.asarray(display_scale_zyx, dtype=float)

    if pad_small is not None and int(pad_small) > 0:
        pad_small = int(pad_small)
        mask_pad = np.pad(
            mask_small,
            pad_width=pad_small,
            mode="constant",
            constant_values=False
        )
        offset_zyx_eff = offset_zyx - pad_small * spacing_zyx
    else:
        mask_pad = mask_small
        offset_zyx_eff = offset_zyx.copy()

    spacing_display_zyx = spacing_zyx * display_scale_zyx
    offset_display_zyx = offset_zyx_eff * display_scale_zyx

    if verbose:
        print("[mesh] smoothing sigma:", sigma)
        print("[mesh] marching_cubes level:", level)
        print("[mesh] spacing_zyx:", spacing_zyx)
        print("[mesh] display_scale_zyx:", display_scale_zyx)
        print("[mesh] spacing_display_zyx:", spacing_display_zyx)
        print("[mesh] offset_display_zyx:", offset_display_zyx)

    vol = gaussian_filter(mask_pad.astype(np.float32), sigma=float(sigma))

    verts_zyx, faces, _, _ = marching_cubes(
        vol,
        level=float(level),
        spacing=tuple(spacing_display_zyx)
    )

    verts_zyx = verts_zyx + offset_display_zyx[None, :]

    verts_xyz = verts_zyx[:, [2, 1, 0]].astype(float)
    faces = faces.astype(np.int32)

    if smooth_iters is not None and int(smooth_iters) > 0:
        verts_xyz = laplacian_smooth(
            verts_xyz,
            faces,
            iters=int(smooth_iters),
            lam=float(smooth_lam)
        )

    if verbose:
        print("[mesh] verts:", verts_xyz.shape)
        print("[mesh] faces:", faces.shape)
        print("[mesh] verts XYZ min:", verts_xyz.min(axis=0))
        print("[mesh] verts XYZ max:", verts_xyz.max(axis=0))

    return verts_xyz, faces


def add_surface_to_plotly(
    fig,
    verts,
    faces,
    color="#E5E5E5",
    opacity=0.25,
    name="surface",
    showlegend=False,
):
    
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=float(opacity),
        name=name,
        hoverinfo="skip",
        showscale=False,
        showlegend=showlegend,
    ))

    return fig





def add_feature_lines_for_mesh_original_style(
    fig,
    verts,
    faces,
    cam_vec,
    silhouette=True,
    crease=True,
    crease_deg=50.0,
    line_color="black",
    line_width=1.0,
    max_line_segments=300_000,
    name="feature_lines",
    showlegend=False,
    verbose=False,
):
    
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)

    face_norms = _face_normals(verts, faces)
    crease_th_cos = np.cos(np.deg2rad(float(crease_deg)))

    edge2faces = defaultdict(list)

    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge2faces[(u, v)].append(fi)

    cam_vec = np.asarray(cam_vec, dtype=float)
    cam_vec = cam_vec / (np.linalg.norm(cam_vec) + 1e-12)

    facing = (face_norms @ cam_vec) > 0.0

    xs, ys, zs = [], [], []
    n_draw = 0

    for (u, v), fids in edge2faces.items():
        draw = False

        if len(fids) == 1:
            if silhouette:
                draw = True

        elif len(fids) >= 2:
            f1, f2 = fids[0], fids[1]

            if silhouette and (facing[f1] ^ facing[f2]):
                draw = True

            elif crease:
                dot_val = np.dot(face_norms[f1], face_norms[f2])
                if dot_val < crease_th_cos:
                    draw = True

        if draw:
            p1 = verts[u]
            p2 = verts[v]

            xs += [p1[0], p2[0], None]
            ys += [p1[1], p2[1], None]
            zs += [p1[2], p2[2], None]

            n_draw += 1

    xs, ys, zs = _cap_line_arrays(
        xs,
        ys,
        zs,
        max_segments=max_line_segments
    )

    if verbose:
        print(
            f"[feature] {name}: "
            f"selected_edges={n_draw}, "
            f"plotted_segments<= {max_line_segments}"
        )

    if len(xs) > 0:
        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(
                width=float(line_width),
                color=line_color
            ),
            name=name,
            hoverinfo="skip",
            showlegend=showlegend,
        ))

    return fig





def read_swc_csv_to_segments(
    swc_csv_path,
    swc_coord_scale=(1.0, 1.0, 1.0),
    swc_coord_offset=(0.0, 0.0, 0.0),
    swc_radius_scale=1.0,
    flip_swc_z=False,
    full_shape_zyx=None,
    display_scale_zyx=(1.0, 1.0, 1.0),
    verbose=True
):
    
    suffix = os.path.splitext(str(swc_csv_path))[1].lower()

    if suffix == ".swc":
        raw_cols = [
            "current_node", "neuron_type", "swc_x", "swc_y", "swc_z",
            "radius", "parent_node"
        ]
        df_raw = pd.read_csv(
            swc_csv_path,
            sep=r"\s+",
            comment="#",
            header=None,
            names=raw_cols,
            usecols=range(7),
            engine="c",
        )
        df = pd.DataFrame({
            "current_node": df_raw["current_node"],
            "neuron_type": df_raw["neuron_type"],
            "x-coordinate": df_raw["swc_x"],
            "y-coordinate": df_raw["swc_z"],
            "z-coordinate": df_raw["swc_y"],
            "radius": df_raw["radius"],
            "parent_node": df_raw["parent_node"],
        })
    else:
        df = pd.read_csv(swc_csv_path)

    required_cols = [
        "current_node",
        "parent_node",
        "x-coordinate",
        "y-coordinate",
        "z-coordinate",
        "radius"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    coords_xyz = df[[
        "x-coordinate",
        "y-coordinate",
        "z-coordinate"
    ]].to_numpy(dtype=float)

    swc_coord_scale = np.asarray(swc_coord_scale, dtype=float)
    swc_coord_offset = np.asarray(swc_coord_offset, dtype=float)

    coords_xyz = coords_xyz * swc_coord_scale[None, :] + swc_coord_offset[None, :]

    if flip_swc_z:
        if full_shape_zyx is None:
            raise ValueError("flip_swc_z=True requires full_shape_zyx.")
        Z = full_shape_zyx[0] * display_scale_zyx[0]
        coords_xyz[:, 2] = Z - coords_xyz[:, 2]

    radius = df["radius"].to_numpy(dtype=float) * float(swc_radius_scale)

    node_ids = df["current_node"].to_numpy(dtype=int)
    parent_ids = df["parent_node"].to_numpy(dtype=int)

    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}

    segments = []
    seg_radius = []

    for i, parent in enumerate(parent_ids):
        parent = int(parent)

        if parent < 0:
            continue

        if parent not in id_to_idx:
            continue

        j = id_to_idx[parent]

        p_parent = coords_xyz[j]
        p_child = coords_xyz[i]

        segments.append(np.stack([p_parent, p_child], axis=0))
        seg_radius.append((radius[i] + radius[j]) / 2.0)

    if len(segments) == 0:
        raise ValueError(f"No valid parent-child segments found in SWC CSV: {swc_csv_path}")

    segments = np.stack(segments, axis=0)
    seg_radius = np.asarray(seg_radius, dtype=float)

    if verbose:
        print(f"[SWC] path: {swc_csv_path}")
        print("[SWC] nodes:", df.shape[0])
        print("[SWC] segments:", segments.shape[0])
        print("[SWC] coord scale:", swc_coord_scale)
        print("[SWC] coord offset:", swc_coord_offset)
        print("[SWC] radius scale:", swc_radius_scale)
        print("[SWC] xyz min:", coords_xyz.min(axis=0))
        print("[SWC] xyz max:", coords_xyz.max(axis=0))
        print("[SWC] radius min/max:", float(radius.min()), float(radius.max()))

        if full_shape_zyx is not None:
            Z, Y, X = full_shape_zyx
            display_scale_zyx = np.asarray(display_scale_zyx, dtype=float)

            display_xyz_max = (
                X * display_scale_zyx[2],
                Y * display_scale_zyx[1],
                Z * display_scale_zyx[0],
            )

            print("[check] TIFF ZYX shape:", full_shape_zyx)
            print("[check] display XYZ max:", display_xyz_max)

    return df, coords_xyz, segments, seg_radius


def infer_swc_axon_dendrite(
    df,
    coords_xyz,
    soma_radius_ratio=1.5,
    soma_distance_factor=10.0,
    soma_envelope_factor=6.0,
    soma_envelope_min=30.0,
):
    
    coords_xyz = np.asarray(coords_xyz, dtype=float)
    node_ids = df["current_node"].to_numpy(dtype=int)
    parent_ids = df["parent_node"].to_numpy(dtype=int)
    radius = df["radius"].to_numpy(dtype=float)
    if "neuron_type" in df.columns:
        swc_types = df["neuron_type"].to_numpy(dtype=int)
    else:
        swc_types = np.zeros(len(df), dtype=int)

    id_to_idx = {int(node_id): i for i, node_id in enumerate(node_ids)}
    original_parent = np.array(
        [id_to_idx.get(int(parent_id), -1) for parent_id in parent_ids],
        dtype=np.int64,
    )
    roots = np.flatnonzero(original_parent < 0)
    soma_candidates = np.flatnonzero(swc_types == 1)
    declared_soma = int(
        soma_candidates[0]
        if soma_candidates.size
        else (roots[0] if roots.size else 0)
    )
    thickest = int(np.argmax(radius))
    soma_gap = float(np.linalg.norm(coords_xyz[declared_soma] - coords_xyz[thickest]))
    if (
        radius[thickest] > float(soma_radius_ratio) * radius[declared_soma]
        and soma_gap > float(soma_distance_factor) * radius[thickest]
    ):
        soma_center = thickest
        soma_source = "thickest_remote_region"
    else:
        soma_center = declared_soma
        soma_source = "swc_type_1"

    neighbors = [[] for _ in range(len(node_ids))]
    for child, parent in enumerate(original_parent):
        if parent >= 0:
            neighbors[child].append(int(parent))
            neighbors[int(parent)].append(child)

    soma_scale = max(
        float(radius[soma_center]),
        float(np.percentile(radius, 99.9)),
    )
    envelope_radius = max(
        float(soma_envelope_min),
        float(soma_envelope_factor) * soma_scale,
    )
    distance_to_soma = np.linalg.norm(coords_xyz - coords_xyz[soma_center], axis=1)
    soma_candidate = distance_to_soma <= envelope_radius
    soma_mask = np.zeros(len(node_ids), dtype=bool)
    soma_mask[soma_center] = True
    flood = [soma_center]
    for node in flood:
        for neighbor in neighbors[node]:
            if soma_candidate[neighbor] and not soma_mask[neighbor]:
                soma_mask[neighbor] = True
                flood.append(neighbor)

    rerooted_parent = np.full(len(node_ids), -2, dtype=np.int64)
    rerooted_parent[soma_mask] = -1
    order = list(np.flatnonzero(soma_mask))
    for node in order:
        for neighbor in neighbors[node]:
            if rerooted_parent[neighbor] == -2:
                rerooted_parent[neighbor] = node
                order.append(neighbor)

    children = [[] for _ in range(len(node_ids))]
    for child, parent in enumerate(rerooted_parent):
        if parent >= 0:
            children[int(parent)].append(child)
    starts = [
        child
        for soma_node in np.flatnonzero(soma_mask)
        for child in children[int(soma_node)]
        if not soma_mask[child]
    ]

    edge_length = np.zeros(len(node_ids), dtype=float)
    valid = rerooted_parent >= 0
    edge_length[valid] = np.linalg.norm(
        coords_xyz[valid] - coords_xyz[rerooted_parent[valid]], axis=1
    )
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
        branches.append({
            "branch_id": branch_id,
            "nodes": idx,
            "length": float(edge_length[idx].sum()),
            "reach": float(
                np.linalg.norm(
                    coords_xyz[idx] - coords_xyz[soma_center], axis=1
                ).max()
            ),
            "median_radius": float(np.median(radius[idx])),
            "forks": int(np.count_nonzero(child_count[idx] > 1)),
        })

    node_labels = np.full(len(node_ids), 2, dtype=np.int8)
    if len(branches) > 1:
        lengths = np.array([b["length"] for b in branches])
        reaches = np.array([b["reach"] for b in branches])
        radii = np.array([b["median_radius"] for b in branches])
        forks = np.array([b["forks"] for b in branches], dtype=float)

        def robust_z(values):
            median = np.median(values)
            scale = np.median(np.abs(values - median))
            if scale < 1e-9:
                scale = np.std(values) + 1e-9
            return (values - median) / (1.4826 * scale + 1e-9)

        scores = (
            0.45 * robust_z(np.log1p(lengths))
            + 0.40 * robust_z(np.log1p(reaches))
            - 0.35 * robust_z(np.log1p(radii))
            - 0.15 * robust_z(np.log1p(forks / np.maximum(lengths, 1.0)))
        )
        eligible = (
            (lengths >= 0.01 * lengths.max())
            | (reaches >= 0.03 * reaches.max())
        )
        long_range = (
            eligible
            & (lengths >= 0.35 * lengths.max())
            & (reaches >= 0.35 * reaches.max())
        )
        if not np.any(long_range):
            winner = int(np.argmax(np.where(eligible, scores, -np.inf)))
            long_range[winner] = True
        for index, branch in enumerate(branches):
            branch["score"] = float(scores[index])
            branch["label"] = "axon" if long_range[index] else "dendrite"
            node_labels[branch["nodes"]] = 3 if long_range[index] else 2
    elif branches:
        branches[0]["score"] = 0.0
        branches[0]["label"] = "dendrite"
        node_labels[branches[0]["nodes"]] = 2
    node_labels[soma_mask] = 1

    segment_labels = []
    for child, parent_id in enumerate(parent_ids):
        parent = id_to_idx.get(int(parent_id), -1)
        if parent < 0:
            continue
        if soma_mask[child] or soma_mask[parent]:
            segment_labels.append(1)
        elif node_labels[child] == 3 or node_labels[parent] == 3:
            segment_labels.append(3)
        else:
            segment_labels.append(2)

    diagnostics = {
        "soma_center_index": soma_center,
        "soma_center_xyz": coords_xyz[soma_center].copy(),
        "soma_source": soma_source,
        "soma_envelope_nodes": int(soma_mask.sum()),
        "soma_envelope_radius": float(envelope_radius),
        "branches": branches,
    }
    return node_labels, np.asarray(segment_labels, dtype=np.int8), diagnostics


def label_swc_segments_from_type(df):
    
    node_ids = df["current_node"].to_numpy(dtype=int)
    parent_ids = df["parent_node"].to_numpy(dtype=int)
    if "neuron_type" not in df.columns:
        return np.zeros(np.count_nonzero(parent_ids >= 0), dtype=np.int8)
    swc_types = df["neuron_type"].to_numpy(dtype=int)
    id_to_idx = {int(node_id): i for i, node_id in enumerate(node_ids)}
    labels = []
    for child, parent_id in enumerate(parent_ids):
        parent = id_to_idx.get(int(parent_id), -1)
        if parent < 0:
            continue
        edge_types = {int(swc_types[child]), int(swc_types[parent])}
        if 1 in edge_types:
            labels.append(1)
        elif 2 in edge_types:
            labels.append(3)
        elif 3 in edge_types or 4 in edge_types:
            labels.append(2)
        else:
            labels.append(0)
    return np.asarray(labels, dtype=np.int8)


def add_swc_lines_to_plotly(
    fig,
    segments_xyz,
    segment_radius=None,
    color="#008CFF",
    uniform_line_width=4.0,
    use_radius_width=False,
    radius_linewidth_scale=2.0,
    min_line_width=1.0,
    max_line_width=8.0,
    n_radius_bins=8,
    name_prefix="SWC",
    showlegend=False,
    meta=None,
):
    
    segments_xyz = np.asarray(segments_xyz, dtype=float)

    if segment_radius is None or (not use_radius_width):
        xs, ys, zs = [], [], []
        last_point = None

        for seg in segments_xyz:
            p0, p1 = seg

            
            if last_point is not None and np.allclose(p0, last_point, rtol=0, atol=1e-10):
                xs.append(p1[0])
                ys.append(p1[1])
                zs.append(p1[2])
            else:
                if len(xs) > 0:
                    xs.append(None); ys.append(None); zs.append(None)
                xs += [p0[0], p1[0]]
                ys += [p0[1], p1[1]]
                zs += [p0[2], p1[2]]

            last_point = p1

        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=float(uniform_line_width)),
            name=name_prefix,
            hoverinfo="skip",
            showlegend=showlegend,
            meta=meta,
        ))

        return fig

    segment_radius = np.asarray(segment_radius, dtype=float)

    widths = segment_radius * float(radius_linewidth_scale)
    widths = np.clip(widths, float(min_line_width), float(max_line_width))

    if n_radius_bins is None or int(n_radius_bins) <= 1:
        w = float(np.median(widths))

        xs, ys, zs = [], [], []
        for seg in segments_xyz:
            p0, p1 = seg
            xs += [p0[0], p1[0], None]
            ys += [p0[1], p1[1], None]
            zs += [p0[2], p1[2], None]

        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=w),
            name=name_prefix,
            hoverinfo="skip",
            showlegend=showlegend,
            meta=meta,
        ))

        return fig

    n_radius_bins = int(n_radius_bins)

    q = np.linspace(0, 1, n_radius_bins + 1)
    edges = np.quantile(widths, q)
    edges = np.unique(edges)

    if len(edges) <= 2:
        bin_ids = np.zeros_like(widths, dtype=int)
    else:
        bin_ids = np.digitize(widths, edges[1:-1], right=True)

    for b in sorted(np.unique(bin_ids)):
        sel = bin_ids == b
        segs = segments_xyz[sel]

        xs, ys, zs = [], [], []
        for seg in segs:
            p0, p1 = seg
            xs += [p0[0], p1[0], None]
            ys += [p0[1], p1[1], None]
            zs += [p0[2], p1[2], None]

        w = float(np.median(widths[sel]))

        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=w),
            name=f"{name_prefix}_radius_bin_{b}",
            hoverinfo="skip",
            showlegend=showlegend,
            meta=meta,
        ))

    return fig





def get_plot_crop_box_from_crop_zyx(
    crop_zyx,
    ref_shape_zyx,
    display_scale_zyx=(1.0, 1.0, 1.0),
    plot_axis_order="XYZ",
):
    
    Z, Y, X = ref_shape_zyx
    sz, sy, sx = [float(v) for v in display_scale_zyx]

    if crop_zyx is None:
        z0, z1 = 0, Z
        y0, y1 = 0, Y
        x0, x1 = 0, X
    else:
        (z0, z1), (y0, y1), (x0, x1) = crop_zyx

    xyz_min = np.array([x0 * sx, y0 * sy, z0 * sz], dtype=float)
    xyz_max = np.array([x1 * sx, y1 * sy, z1 * sz], dtype=float)

    plot_axis_order = str(plot_axis_order).upper()
    idx_map = {"X": 0, "Y": 1, "Z": 2}
    order = [idx_map[a] for a in plot_axis_order]

    box_min_plot = xyz_min[order]
    box_max_plot = xyz_max[order]

    return box_min_plot, box_max_plot


def normalize_crop_zyx(crop_zyx, shape_zyx):
    
    shape_zyx = tuple(int(v) for v in shape_zyx)
    if crop_zyx is None:
        return tuple((0, size) for size in shape_zyx)
    if len(crop_zyx) != 3:
        raise ValueError("crop_zyx must contain exactly three (start, end) ranges.")
    normalized = []
    for axis, (bounds, size) in enumerate(zip(crop_zyx, shape_zyx)):
        if len(bounds) != 2:
            raise ValueError(f"crop_zyx axis {axis} must be a (start, end) pair.")
        start = max(0, int(bounds[0]))
        end = min(size, int(bounds[1]))
        if end <= start:
            raise ValueError(
                f"crop_zyx axis {axis} is empty after clamping: "
                f"({start}, {end}) for size {size}."
            )
        normalized.append((start, end))
    return tuple(normalized)


def intersect_crops_zyx(global_crop_zyx, local_crop_zyx, shape_zyx):
    
    global_crop = normalize_crop_zyx(global_crop_zyx, shape_zyx)
    if local_crop_zyx is None:
        return global_crop
    local_crop = normalize_crop_zyx(local_crop_zyx, shape_zyx)
    intersection = []
    for axis, (global_bounds, local_bounds) in enumerate(
        zip(global_crop, local_crop)
    ):
        start = max(global_bounds[0], local_bounds[0])
        end = min(global_bounds[1], local_bounds[1])
        if end <= start:
            raise ValueError(
                f"Global and local crop do not overlap on ZYX axis {axis}: "
                f"{global_bounds} vs {local_bounds}."
            )
        intersection.append((start, end))
    return tuple(intersection)


def clip_segment_to_aabb_3d(p0, p1, box_min, box_max, eps=1e-12):
    
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)

    d = p1 - p0
    tmin, tmax = 0.0, 1.0

    for i in range(3):
        if abs(d[i]) < eps:
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return None
        else:
            t1 = (box_min[i] - p0[i]) / d[i]
            t2 = (box_max[i] - p0[i]) / d[i]

            t_enter = min(t1, t2)
            t_exit = max(t1, t2)

            tmin = max(tmin, t_enter)
            tmax = min(tmax, t_exit)

            if tmin > tmax:
                return None

    q0 = p0 + tmin * d
    q1 = p1 + tmax * d

    return np.stack([q0, q1], axis=0)


def clip_segments_to_box(segments_xyz, box_min, box_max):
    
    segments_xyz = np.asarray(segments_xyz, dtype=float)

    kept = []
    for seg in segments_xyz:
        clipped = clip_segment_to_aabb_3d(
            seg[0], seg[1],
            box_min=box_min,
            box_max=box_max
        )
        if clipped is not None:
            kept.append(clipped)

    if len(kept) == 0:
        return np.empty((0, 2, 3), dtype=float)

    return np.stack(kept, axis=0)


def normalize_oblique_crop(oblique_crop):
    
    if oblique_crop is None or oblique_crop is False:
        return None
    if not isinstance(oblique_crop, dict):
        raise TypeError("oblique_crop must be None or a dict.")
    if not bool(oblique_crop.get("enabled", True)):
        return None

    normal = np.asarray(oblique_crop.get("normal_plot", ()), dtype=float)
    slab_range = np.asarray(oblique_crop.get("range", ()), dtype=float)
    if normal.shape != (3,):
        raise ValueError("oblique_crop['normal_plot'] must contain exactly 3 values.")
    if slab_range.shape != (2,):
        raise ValueError("oblique_crop['range'] must contain exactly 2 values.")
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(slab_range)):
        raise ValueError("oblique_crop values must all be finite.")

    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        raise ValueError("oblique_crop['normal_plot'] cannot be the zero vector.")
    normal = normal / normal_norm
    d_low, d_high = sorted(float(v) / normal_norm for v in slab_range)
    return {
        "enabled": True,
        "normal_plot": normal,
        "range": (d_low, d_high),
    }


def normalize_oblique_planes(
    oblique_planes=None,
    oblique_crop=None,
    display_scale_zyx=(1.0, 1.0, 1.0),
    plot_axis_order="XYZ",
):
    
    raw_planes = oblique_planes
    if raw_planes is None:
        slab = normalize_oblique_crop(oblique_crop)
        if slab is None:
            return []
        normal = tuple(float(v) for v in slab["normal_plot"])
        d_low, d_high = slab["range"]
        raw_planes = [
            {"enabled": True, "normal_plot": normal, "d": d_low, "keep": "ge"},
            {"enabled": True, "normal_plot": normal, "d": d_high, "keep": "le"},
        ]
    if not isinstance(raw_planes, (list, tuple)):
        raise TypeError("oblique_planes must be None, a list, or a tuple.")

    normalized = []
    for index, plane in enumerate(raw_planes):
        if not isinstance(plane, dict):
            raise TypeError(f"oblique_planes[{index}] must be a dict.")
        if not bool(plane.get("enabled", True)):
            continue
        if "normal_voxel" in plane:
            normal_voxel = np.asarray(plane.get("normal_voxel", ()), dtype=float)
            if normal_voxel.shape != (3,) or not np.all(np.isfinite(normal_voxel)):
                raise ValueError(
                    f"oblique_planes[{index}]['normal_voxel'] must contain 3 finite XYZ values."
                )
            sz, sy, sx = [float(v) for v in display_scale_zyx]
            scale_xyz = np.asarray([sx, sy, sz], dtype=float)
            order = [{"X": 0, "Y": 1, "Z": 2}[axis] for axis in str(plot_axis_order).upper()]
            normal = (normal_voxel / scale_xyz)[order]
        else:
            normal = np.asarray(plane.get("normal_plot", ()), dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError(
                f"oblique_planes[{index}]['normal_plot'] must contain 3 finite values."
            )
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            raise ValueError(f"oblique_planes[{index}] has a zero normal vector.")
        d = float(plane.get("d"))
        if not np.isfinite(d):
            raise ValueError(f"oblique_planes[{index}]['d'] must be finite.")
        keep = str(plane.get("keep", "ge")).lower()
        if keep not in {"ge", "le"}:
            raise ValueError(f"oblique_planes[{index}]['keep'] must be 'ge' or 'le'.")
        normalized.append({
            "enabled": True,
            "normal_plot": normal / length,
            "d": d / length,
            "keep": keep,
        })
    return normalized


def apply_oblique_planes_to_mask(
    mask_small,
    offset_zyx,
    spacing_zyx,
    display_scale_zyx,
    plot_axis_order,
    oblique_planes,
    verbose=True,
):
    
    planes = normalize_oblique_planes(oblique_planes=oblique_planes)
    mask = np.asarray(mask_small, dtype=bool)
    if not planes:
        return mask

    offset_zyx = np.asarray(offset_zyx, dtype=float)
    spacing_zyx = np.asarray(spacing_zyx, dtype=float)
    display_scale_zyx = np.asarray(display_scale_zyx, dtype=float)
    idx_map = {"X": 0, "Y": 1, "Z": 2}
    order = [idx_map[a] for a in str(plot_axis_order).upper()]

    z_count, y_count, x_count = mask.shape
    sz, sy, sx = display_scale_zyx
    z_values = (offset_zyx[0] + np.arange(z_count) * spacing_zyx[0]) * sz
    y_values = (offset_zyx[1] + np.arange(y_count) * spacing_zyx[1]) * sy
    x_values = (offset_zyx[2] + np.arange(x_count) * spacing_zyx[2]) * sx
    before = int(np.count_nonzero(mask))

    for plane_index, plane in enumerate(planes):
        normal_xyz = np.zeros(3, dtype=float)
        for plot_idx, xyz_idx in enumerate(order):
            normal_xyz[xyz_idx] = plane["normal_plot"][plot_idx]
        d_xy = normal_xyz[0] * x_values[None, :] + normal_xyz[1] * y_values[:, None]
        for z_idx, z_value in enumerate(z_values):
            values = d_xy + normal_xyz[2] * z_value
            if plane["keep"] == "ge":
                mask[z_idx] &= values >= plane["d"]
            else:
                mask[z_idx] &= values <= plane["d"]
        if verbose:
            print(
                f"[oblique plane {plane_index + 1}] normal_plot={plane['normal_plot']}, "
                f"d={plane['d']}, keep={plane['keep']}"
            )

    after = int(np.count_nonzero(mask))
    if verbose:
        print(f"[oblique planes] foreground voxels: {before} -> {after}")
    if after == 0:
        raise ValueError("oblique_planes removed all foreground voxels.")
    return mask


def clip_segments_to_oblique_planes(segments_plot, oblique_planes, eps=1e-12):
    
    planes = normalize_oblique_planes(oblique_planes=oblique_planes)
    segments = np.asarray(segments_plot, dtype=float)
    if not planes:
        return segments
    if segments.size == 0:
        return np.empty((0, 2, 3), dtype=float)

    kept = []
    for p0, p1 in segments:
        t_enter, t_exit = 0.0, 1.0
        rejected = False
        for plane in planes:
            value0 = float(np.dot(plane["normal_plot"], p0) - plane["d"])
            value1 = float(np.dot(plane["normal_plot"], p1) - plane["d"])
            if plane["keep"] == "le":
                value0, value1 = -value0, -value1
            inside0 = value0 >= -eps
            inside1 = value1 >= -eps
            if inside0 and inside1:
                continue
            if not inside0 and not inside1:
                rejected = True
                break
            t_cross = value0 / (value0 - value1)
            if inside0:
                t_exit = min(t_exit, t_cross)
            else:
                t_enter = max(t_enter, t_cross)
            if t_enter > t_exit + eps:
                rejected = True
                break
        if not rejected:
            direction = p1 - p0
            kept.append(np.stack([
                p0 + t_enter * direction,
                p0 + t_exit * direction,
            ], axis=0))
    if not kept:
        return np.empty((0, 2, 3), dtype=float)
    return np.stack(kept, axis=0)


def apply_oblique_crop_to_mask(
    mask_small,
    offset_zyx,
    spacing_zyx,
    display_scale_zyx,
    plot_axis_order,
    oblique_crop,
    verbose=True,
):
    
    config = normalize_oblique_crop(oblique_crop)
    mask = np.asarray(mask_small, dtype=bool)
    if config is None:
        return mask

    offset_zyx = np.asarray(offset_zyx, dtype=float)
    spacing_zyx = np.asarray(spacing_zyx, dtype=float)
    display_scale_zyx = np.asarray(display_scale_zyx, dtype=float)
    order_name = str(plot_axis_order).upper()
    idx_map = {"X": 0, "Y": 1, "Z": 2}
    order = [idx_map[a] for a in order_name]

    normal_plot = config["normal_plot"]
    normal_xyz = np.zeros(3, dtype=float)
    for plot_idx, xyz_idx in enumerate(order):
        normal_xyz[xyz_idx] = normal_plot[plot_idx]
    d_low, d_high = config["range"]

    z_count, y_count, x_count = mask.shape
    sz, sy, sx = display_scale_zyx
    z_values = (offset_zyx[0] + np.arange(z_count) * spacing_zyx[0]) * sz
    y_values = (offset_zyx[1] + np.arange(y_count) * spacing_zyx[1]) * sy
    x_values = (offset_zyx[2] + np.arange(x_count) * spacing_zyx[2]) * sx
    d_xy = normal_xyz[0] * x_values[None, :] + normal_xyz[1] * y_values[:, None]

    before = int(np.count_nonzero(mask))
    for z_idx, z_value in enumerate(z_values):
        d_values = d_xy + normal_xyz[2] * z_value
        mask[z_idx] &= (d_values >= d_low) & (d_values <= d_high)
    after = int(np.count_nonzero(mask))

    if verbose:
        print("[oblique crop] normal_plot:", normal_plot)
        print("[oblique crop] range:", (d_low, d_high))
        print(f"[oblique crop] foreground voxels: {before} -> {after}")
    if after == 0:
        raise ValueError("oblique_crop removed all foreground voxels.")
    return mask


def clip_segments_to_oblique_slab(segments_plot, oblique_crop, eps=1e-12):
    
    config = normalize_oblique_crop(oblique_crop)
    segments = np.asarray(segments_plot, dtype=float)
    if config is None:
        return segments
    if segments.size == 0:
        return np.empty((0, 2, 3), dtype=float)

    normal = config["normal_plot"]
    d_low, d_high = config["range"]
    kept = []
    for p0, p1 in segments:
        d0 = float(np.dot(normal, p0))
        d1 = float(np.dot(normal, p1))
        delta = d1 - d0
        if abs(delta) < eps:
            if d_low - eps <= d0 <= d_high + eps:
                kept.append(np.stack([p0, p1], axis=0))
            continue
        t0 = (d_low - d0) / delta
        t1 = (d_high - d0) / delta
        t_enter = max(0.0, min(t0, t1))
        t_exit = min(1.0, max(t0, t1))
        if t_enter <= t_exit + eps:
            direction = p1 - p0
            kept.append(np.stack([
                p0 + t_enter * direction,
                p0 + t_exit * direction,
            ], axis=0))
    if not kept:
        return np.empty((0, 2, 3), dtype=float)
    return np.stack(kept, axis=0)


def add_multiple_swc_csvs_to_plotly(
    fig,
    swc_specs,
    ref_shape_zyx,
    display_scale_zyx=(1.0, 1.0, 1.0),
    plot_axis_order="XYZ",
    use_swc_for_limits=True,

    clip_swc_to_crop=True,
    crop_zyx=None,
    oblique_crop=None,
    oblique_planes=None,

    default_swc_coord_scale=(1.0, 1.0, 1.0),
    default_swc_coord_offset=(0.0, 0.0, 0.0),
    default_swc_radius_scale=1.0,
    default_flip_swc_z=False,
    default_swc_color="#008CFF",

    default_use_radius_width=False,
    default_uniform_line_width=2.0,
    default_radius_linewidth_scale=2.0,
    default_min_line_width=1.0,
    default_max_line_width=8.0,
    default_n_radius_bins=8,
    default_distinguish_neurites=False,
    default_axon_color="#FF9F43",
    default_dendrite_color="#FF5F70",
    default_soma_color="#FF5F70",
    default_color_mode=None,

    showlegend=False,
    verbose=True,
):
    
    all_swc_points_plot = []
    swc_records = []

    if swc_specs is None or len(swc_specs) == 0:
        return fig, all_swc_points_plot, swc_records

    if clip_swc_to_crop:
        box_min_plot, box_max_plot = get_plot_crop_box_from_crop_zyx(
            crop_zyx=crop_zyx,
            ref_shape_zyx=ref_shape_zyx,
            display_scale_zyx=display_scale_zyx,
            plot_axis_order=plot_axis_order,
        )
        if verbose:
            print("[SWC clip] box_min_plot:", box_min_plot)
            print("[SWC clip] box_max_plot:", box_max_plot)
    else:
        box_min_plot, box_max_plot = None, None

    for i, spec in enumerate(swc_specs):
        swc_path = spec["path"]
        swc_name = spec.get("name", f"SWC_{i}")

        swc_color = spec.get("color", default_swc_color)

        swc_coord_scale = spec.get("coord_scale", default_swc_coord_scale)
        swc_coord_offset = spec.get("coord_offset", default_swc_coord_offset)
        swc_radius_scale = spec.get("radius_scale", default_swc_radius_scale)
        flip_swc_z = spec.get("flip_z", default_flip_swc_z)

        use_radius_width = spec.get("use_radius_width", default_use_radius_width)
        uniform_line_width = spec.get("line_width", default_uniform_line_width)

        radius_linewidth_scale = spec.get("radius_linewidth_scale", default_radius_linewidth_scale)
        min_line_width = spec.get("min_line_width", default_min_line_width)
        max_line_width = spec.get("max_line_width", default_max_line_width)
        n_radius_bins = spec.get("n_radius_bins", default_n_radius_bins)
        legacy_split = spec.get(
            "distinguish_neurites", default_distinguish_neurites
        )
        color_mode = spec.get("color_mode", default_color_mode)
        if color_mode is None:
            color_mode = "infer_neurites" if legacy_split else "single"
        color_mode = str(color_mode).lower()
        if color_mode not in {"single", "swc_type", "infer_neurites"}:
            raise ValueError(
                f"Unsupported SWC color_mode={color_mode!r} for {swc_name}. "
                "Use 'single', 'swc_type', or 'infer_neurites'."
            )
        distinguish_neurites = color_mode != "single"
        axon_color = spec.get("axon_color", default_axon_color)
        dendrite_color = spec.get("dendrite_color", default_dendrite_color)
        soma_color = spec.get("soma_color", default_soma_color)

        if verbose:
            print("\n" + "-" * 70)
            print(f"[SWC multi] processing {i + 1}/{len(swc_specs)}: {swc_name}")
            print(f"[SWC multi] path: {swc_path}")
            print(f"[SWC multi] color: {swc_color}")
            print(f"[SWC multi] line_width: {uniform_line_width}")
            print("-" * 70)

        df_swc, coords_xyz, segments_xyz, segment_radius = read_swc_csv_to_segments(
            swc_csv_path=swc_path,
            swc_coord_scale=swc_coord_scale,
            swc_coord_offset=swc_coord_offset,
            swc_radius_scale=swc_radius_scale,
            flip_swc_z=flip_swc_z,
            full_shape_zyx=ref_shape_zyx,
            display_scale_zyx=display_scale_zyx,
            verbose=verbose
        )
        if color_mode == "infer_neurites":
            _, segment_labels, neurite_diagnostics = infer_swc_axon_dendrite(
                df=df_swc,
                coords_xyz=coords_xyz,
            )
            if segment_labels.shape[0] != segments_xyz.shape[0]:
                raise RuntimeError(
                    f"Neurite label/segment mismatch for {swc_name}: "
                    f"{segment_labels.shape[0]} != {segments_xyz.shape[0]}"
                )
        elif color_mode == "swc_type":
            segment_labels = label_swc_segments_from_type(df_swc)
            neurite_diagnostics = {"soma_source": "swc_type"}
        else:
            segment_labels = None
            neurite_diagnostics = None

        coords_plot = _reorder_xyz_points(
            coords_xyz,
            plot_axis_order=plot_axis_order
        )

        segments_plot = _reorder_xyz_segments(
            segments_xyz,
            plot_axis_order=plot_axis_order
        )

        if clip_swc_to_crop:
            segments_plot_to_draw = clip_segments_to_box(
                segments_xyz=segments_plot,
                box_min=box_min_plot,
                box_max=box_max_plot
            )

            if verbose:
                print(f"[SWC clip] original segments: {segments_plot.shape[0]}")
                print(f"[SWC clip] kept/clipped segments: {segments_plot_to_draw.shape[0]}")
        else:
            segments_plot_to_draw = segments_plot

        active_oblique_planes = normalize_oblique_planes(
            oblique_planes=oblique_planes,
            oblique_crop=oblique_crop,
        )
        if active_oblique_planes:
            before_oblique = segments_plot_to_draw.shape[0]
            segments_plot_to_draw = clip_segments_to_oblique_planes(
                segments_plot=segments_plot_to_draw,
                oblique_planes=active_oblique_planes,
            )
            if verbose:
                print(
                    f"[SWC oblique crop] segments: "
                    f"{before_oblique} -> {segments_plot_to_draw.shape[0]}"
                )

        if use_swc_for_limits and segments_plot_to_draw.shape[0] > 0:
            all_swc_points_plot.append(segments_plot_to_draw.reshape(-1, 3))

        if segments_plot_to_draw.shape[0] == 0:
            if verbose:
                print(f"[SWC clip] {swc_name}: no segment remains after clipping.")
            swc_records.append({
                "name": swc_name,
                "path": swc_path,
                "df": df_swc,
                "coords_xyz_original": coords_xyz,
                "coords_plot": coords_plot,
                "segments_xyz_original": segments_xyz,
                "segments_plot": segments_plot,
                "segments_plot_drawn": segments_plot_to_draw,
                "color": swc_color,
                "line_width": uniform_line_width,
            })
            continue

        if distinguish_neurites:
            
            
            
            for group_mask, label_name, label_color in (
                (segment_labels != 3, "dendrite+soma", dendrite_color),
                (segment_labels == 3, "axon", axon_color),
            ):
                group_segments = segments_plot[group_mask]
                if clip_swc_to_crop and group_segments.shape[0] > 0:
                    group_segments = clip_segments_to_box(
                        segments_xyz=group_segments,
                        box_min=box_min_plot,
                        box_max=box_max_plot,
                    )
                if active_oblique_planes and group_segments.shape[0] > 0:
                    group_segments = clip_segments_to_oblique_planes(
                        segments_plot=group_segments,
                        oblique_planes=active_oblique_planes,
                    )
                if group_segments.shape[0] == 0:
                    continue
                fig = add_swc_lines_to_plotly(
                    fig=fig,
                    segments_xyz=group_segments,
                    segment_radius=None,
                    color=label_color,
                    uniform_line_width=uniform_line_width,
                    use_radius_width=False,
                    radius_linewidth_scale=radius_linewidth_scale,
                    min_line_width=min_line_width,
                    max_line_width=max_line_width,
                    n_radius_bins=n_radius_bins,
                    name_prefix=f"{swc_name} · {label_name}",
                    showlegend=showlegend,
                    meta={
                        "codex_neurite_group": swc_name,
                        "codex_neurite_role": label_name,
                        "codex_color_mode": color_mode,
                    },
                )
        else:
            fig = add_swc_lines_to_plotly(
                fig=fig,
                segments_xyz=segments_plot_to_draw,
                segment_radius=None,
                color=swc_color,
                uniform_line_width=uniform_line_width,
                use_radius_width=False,
                radius_linewidth_scale=radius_linewidth_scale,
                min_line_width=min_line_width,
                max_line_width=max_line_width,
                n_radius_bins=n_radius_bins,
                name_prefix=swc_name,
                showlegend=showlegend,
            )

        swc_records.append({
            "name": swc_name,
            "path": swc_path,
            "df": df_swc,
            "coords_xyz_original": coords_xyz,
            "coords_plot": coords_plot,
            "segments_xyz_original": segments_xyz,
            "segments_plot": segments_plot,
            "segments_plot_drawn": segments_plot_to_draw,
            "color": swc_color,
            "line_width": uniform_line_width,
            "distinguish_neurites": bool(distinguish_neurites),
            "color_mode": color_mode,
            "axon_color": axon_color,
            "dendrite_color": dendrite_color,
            "soma_color": soma_color,
            "neurite_diagnostics": neurite_diagnostics,
        })

    return fig, all_swc_points_plot, swc_records





def plot_multi_foreground_tiffs_with_swc_plotly(
    tiff_specs,
    swc_csv_path=None,
    swc_specs=None,
    html_path=None,

    tiff_axes="ZYX",
    threshold=0,
    crop_zyx=None,
    oblique_crop=None,
    oblique_planes=None,
    mesh_downsample=(4, 4, 4),
    strict_same_shape=True,

    show_surface=True,
    default_surface_color="#E5E5E5",
    default_surface_opacity=0.25,
    mesh_sigma=1.5,
    mesh_level=0.5,
    mesh_pad_small=2,
    smooth_iters=15,
    smooth_lam=0.40,
    display_scale_zyx=(7.1, 6.0, 7.1),

    show_feature_lines=False,
    feature_silhouette=True,
    feature_crease=True,
    feature_crease_deg=50.0,
    feature_line_width=1.0,
    feature_line_color="black",
    feature_max_line_segments=300_000,
    feature_angle_deg=None,

    show_swc=True,
    swc_coord_scale=(1.0, 1.0, 1.0),
    swc_coord_offset=(0.0, 0.0, 0.0),
    swc_radius_scale=1.0,
    flip_swc_z=False,
    swc_color="#008CFF",

    use_radius_width=False,
    uniform_line_width=4.0,
    radius_linewidth_scale=2.0,
    min_line_width=1.0,
    max_line_width=8.0,
    n_radius_bins=8,
    distinguish_neurites=False,
    axon_color="#FF9F43",
    dendrite_color="#FF5F70",
    soma_color="#FF5F70",
    swc_color_mode=None,

    clip_swc_to_crop=True,
    swc_crop_zyx=None,

    viewpoint=(0, 90),
    camera_distance_factor=2.0,
    camera_projection_type="orthographic",
    crop_size_xyz=None,
    use_swc_for_limits=True,

    plot_axis_order="XYZ",

    background_color="white",
    aspectmode="data",
    title=None,
    showlegend=False,

    save_html=False,
    include_plotlyjs=True,
    verbose=True,
):
    
    if len(tiff_specs) == 0:
        raise ValueError("tiff_specs is empty.")

    if html_path is None:
        html_path = os.path.join(os.getcwd(), "multi_tiff_shells_swc.html")

    out_dir = os.path.dirname(html_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_axis_order = str(plot_axis_order).upper()
    if sorted(plot_axis_order) != ["X", "Y", "Z"] or len(plot_axis_order) != 3:
        raise ValueError("plot_axis_order must be a permutation of 'XYZ', e.g. 'XYZ' or 'XZY'.")

    camera_projection_type = str(camera_projection_type).lower()
    if camera_projection_type not in {"perspective", "orthographic"}:
        raise ValueError("camera_projection_type must be 'perspective' or 'orthographic'.")

    if feature_angle_deg is not None:
        feature_crease_deg = float(feature_angle_deg)

    all_same, ref_shape_zyx, shape_records = check_tiff_shapes_same(
        tiff_specs=tiff_specs,
        tiff_axes=tiff_axes,
        strict=strict_same_shape,
        verbose=verbose
    )

    display_scale_zyx = np.asarray(display_scale_zyx, dtype=float)
    oblique_planes = normalize_oblique_planes(
        oblique_planes=oblique_planes,
        oblique_crop=oblique_crop,
        display_scale_zyx=display_scale_zyx,
        plot_axis_order=plot_axis_order,
    )

    if verbose:
        Z, Y, X = ref_shape_zyx
        print("[multi] reference ZYX shape:", ref_shape_zyx)
        print("[multi] display_scale_zyx:", display_scale_zyx)
        print(
            "[multi] original displayed XYZ max approximately:",
            (
                X * display_scale_zyx[2],
                Y * display_scale_zyx[1],
                Z * display_scale_zyx[0],
            )
        )
        print("[multi] plot_axis_order:", plot_axis_order)

    fig = go.Figure()
    all_points_plot = []
    mesh_records = []
    tiff_crop_records = []

    cam_vec = camera_vector_from_viewpoint(viewpoint[0], viewpoint[1])

    for i, spec in enumerate(tiff_specs):
        path = spec["path"]
        name = spec.get("name", f"surface_{i}")
        color = spec.get("color", default_surface_color)
        opacity = spec.get("opacity", default_surface_opacity)
        this_threshold = spec.get("threshold", threshold)
        this_crop_zyx = intersect_crops_zyx(
            global_crop_zyx=crop_zyx,
            local_crop_zyx=spec.get("crop_zyx"),
            shape_zyx=ref_shape_zyx,
        )
        tiff_crop_records.append(this_crop_zyx)

        if verbose:
            print("\n" + "=" * 70)
            print(f"[multi] processing {i + 1}/{len(tiff_specs)}: {name}")
            print(f"[multi] path: {path}")
            print(f"[multi] effective crop_zyx: {this_crop_zyx}")
            print("=" * 70)

        mask_small, offset_zyx, spacing_zyx, full_shape_zyx = read_tiff_foreground_as_mask(
            tiff_path=path,
            tiff_axes=tiff_axes,
            threshold=this_threshold,
            crop_zyx=this_crop_zyx,
            mesh_downsample=mesh_downsample,
            verbose=verbose
        )
        mask_small = apply_oblique_planes_to_mask(
            mask_small=mask_small,
            offset_zyx=offset_zyx,
            spacing_zyx=spacing_zyx,
            display_scale_zyx=display_scale_zyx,
            plot_axis_order=plot_axis_order,
            oblique_planes=oblique_planes,
            verbose=verbose,
        )

        if strict_same_shape and full_shape_zyx != ref_shape_zyx:
            raise ValueError(
                f"Shape mismatch after reading {name}: {full_shape_zyx} != reference {ref_shape_zyx}"
            )

        verts_xyz, faces = build_surface_mesh_from_mask(
            mask_small=mask_small,
            offset_zyx=offset_zyx,
            spacing_zyx=spacing_zyx,
            sigma=mesh_sigma,
            level=mesh_level,
            pad_small=mesh_pad_small,
            smooth_iters=smooth_iters,
            smooth_lam=smooth_lam,
            display_scale_zyx=display_scale_zyx,
            verbose=verbose
        )

        verts_plot = _reorder_xyz_points(
            verts_xyz,
            plot_axis_order=plot_axis_order
        )

        mesh_records.append({
            "name": name,
            "path": path,
            "shape_zyx": full_shape_zyx,
            "crop_zyx": this_crop_zyx,
            "verts_xyz_original": verts_xyz,
            "verts_plot": verts_plot,
            "faces": faces,
            "color": color,
            "opacity": opacity,
        })

        all_points_plot.append(verts_plot)

        if show_surface:
            fig = add_surface_to_plotly(
                fig=fig,
                verts=verts_plot,
                faces=faces,
                color=color,
                opacity=opacity,
                name=name,
                showlegend=showlegend,
            )

        if show_feature_lines:
            fig = add_feature_lines_for_mesh_original_style(
                fig=fig,
                verts=verts_plot,
                faces=faces,
                cam_vec=cam_vec,
                silhouette=feature_silhouette,
                crease=feature_crease,
                crease_deg=feature_crease_deg,
                line_color=feature_line_color,
                line_width=feature_line_width,
                max_line_segments=feature_max_line_segments,
                name=f"{name}_feature_lines",
                showlegend=False,
                verbose=verbose,
            )

    swc_records = []
    effective_swc_crop_zyx = intersect_crops_zyx(
        global_crop_zyx=crop_zyx,
        local_crop_zyx=swc_crop_zyx,
        shape_zyx=ref_shape_zyx,
    )

    if show_swc:
        if swc_specs is None and swc_csv_path is not None:
            swc_specs = [
                {
                    "path": swc_csv_path,
                    "name": "SWC",
                    "color": swc_color,
                    "coord_scale": swc_coord_scale,
                    "coord_offset": swc_coord_offset,
                    "radius_scale": swc_radius_scale,
                    "flip_z": flip_swc_z,
                    "use_radius_width": use_radius_width,
                    "line_width": uniform_line_width,
                    "radius_linewidth_scale": radius_linewidth_scale,
                    "min_line_width": min_line_width,
                    "max_line_width": max_line_width,
                    "n_radius_bins": n_radius_bins,
                }
            ]

        fig, swc_points_for_limits, swc_records = add_multiple_swc_csvs_to_plotly(
            fig=fig,
            swc_specs=swc_specs,
            ref_shape_zyx=ref_shape_zyx,
            display_scale_zyx=display_scale_zyx,
            plot_axis_order=plot_axis_order,
            use_swc_for_limits=use_swc_for_limits,

            clip_swc_to_crop=clip_swc_to_crop,
            crop_zyx=effective_swc_crop_zyx,
            oblique_planes=oblique_planes,

            default_swc_coord_scale=swc_coord_scale,
            default_swc_coord_offset=swc_coord_offset,
            default_swc_radius_scale=swc_radius_scale,
            default_flip_swc_z=flip_swc_z,
            default_swc_color=swc_color,

            default_use_radius_width=use_radius_width,
            default_uniform_line_width=uniform_line_width,
            default_radius_linewidth_scale=radius_linewidth_scale,
            default_min_line_width=min_line_width,
            default_max_line_width=max_line_width,
            default_n_radius_bins=n_radius_bins,
            default_distinguish_neurites=distinguish_neurites,
            default_axon_color=axon_color,
            default_dendrite_color=dendrite_color,
            default_soma_color=soma_color,
            default_color_mode=swc_color_mode,

            showlegend=showlegend,
            verbose=verbose,
        )

        if use_swc_for_limits and len(swc_points_for_limits) > 0:
            all_points_plot.extend(swc_points_for_limits)

    if len(all_points_plot) == 0:
        raise ValueError("Nothing to plot after cropping.")

    all_points_plot = np.vstack(all_points_plot)

    xmid = np.median(all_points_plot[:, 0])
    ymid = np.median(all_points_plot[:, 1])
    zmid = np.median(all_points_plot[:, 2])

    if crop_size_xyz is not None:
        dx, dy, dz = crop_size_xyz
        xlim = (xmid - dx, xmid + dx)
        ylim = (ymid - dy, ymid + dy)
        zlim = (zmid - dz, zmid + dz)
    else:
        margin = 5
        xlim = (all_points_plot[:, 0].min() - margin, all_points_plot[:, 0].max() + margin)
        ylim = (all_points_plot[:, 1].min() - margin, all_points_plot[:, 1].max() + margin)
        zlim = (all_points_plot[:, 2].min() - margin, all_points_plot[:, 2].max() + margin)

    if verbose:
        print("\n[plot] plot_axis_order:", plot_axis_order)
        print("[plot] xlim:", xlim)
        print("[plot] ylim:", ylim)
        print("[plot] zlim:", zlim)

    eye = dict(
        x=float(-cam_vec[0] * camera_distance_factor),
        y=float(-cam_vec[1] * camera_distance_factor),
        z=float(-cam_vec[2] * camera_distance_factor),
    )

    display_crop_zyx = tuple(
        (
            min(crop[axis][0] for crop in tiff_crop_records),
            max(crop[axis][1] for crop in tiff_crop_records),
        )
        for axis in range(3)
    )

    fig.update_layout(
        title=title,
        showlegend=bool(showlegend),
        meta={
            "codex_volume": {
                "shape_zyx": [int(v) for v in ref_shape_zyx],
                "global_crop_zyx": [
                    [int(a), int(b)]
                    for a, b in normalize_crop_zyx(crop_zyx, ref_shape_zyx)
                ],
                "crop_zyx": [
                    [int(display_crop_zyx[i][0]), int(display_crop_zyx[i][1])]
                    for i in range(3)
                ],
                "tiff_crops_zyx": [
                    [[int(a), int(b)] for a, b in crop]
                    for crop in tiff_crop_records
                ],
                "swc_crop_zyx": [
                    [int(a), int(b)] for a, b in effective_swc_crop_zyx
                ],
                "display_scale_zyx": [float(v) for v in display_scale_zyx],
                "plot_axis_order": plot_axis_order,
            }
        },
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        dragmode="orbit",
        scene=dict(
            bgcolor=background_color,
            dragmode="orbit",
            xaxis=dict(visible=False, range=[xlim[0], xlim[1]]),
            yaxis=dict(visible=False, range=[ylim[0], ylim[1]]),
            zaxis=dict(visible=False, range=[zlim[0], zlim[1]]),
            camera=dict(
                eye=eye,

                
                up=dict(x=0, y=1, z=0),

                
                projection=dict(type=camera_projection_type)
            ),
            aspectmode=aspectmode
        ),
        margin=dict(l=0, r=0, t=40 if title else 0, b=0)
    )

    if save_html:
        fig.write_html(html_path, include_plotlyjs=include_plotlyjs)
        if verbose:
            print(f"\n✅ Saved interactive HTML: {html_path}")

    return fig, mesh_records, swc_records





import os
import json
import math
import plotly.io as pio


def save_plotly_html_with_overlays(
    fig,
    html_path,

    
    
    
    scalebar_um=1000,
    scalebar_label=None,
    scalebar_bar_color="black",
    scalebar_text_color=None,
    scalebar_thickness_px=7,
    scalebar_font_size_px=16,
    scalebar_right_px=28,
    scalebar_bottom_px=28,
    scalebar_resizable=True,
    scalebar_size_options=(250, 500, 1000, 2000, 3000, 10000),

    
    
    
    show_orientation_widget=True,

    
    orientation_axes_plot=None,

    
    orientation_label_pairs=None,

    
    orientation_colors=None,

    orientation_size_px=150,
    orientation_left_px=18,
    orientation_top_px=18,
    orientation_radius_px=42,
    orientation_font_size_px=13,
    orientation_line_width_px=3,
    orientation_resizable=True,
    orientation_min_scale=0.50,
    orientation_max_scale=2.50,
    orientation_resize_sensitivity=0.0015,

    
    
    
    
    
    orientation_rotation_deg=(0.0, 0.0, 0.0),
    orientation_rotation_order=("x", "y", "z"),

    orientation_show_circle=True,
    orientation_circle_color="rgba(255,255,255,0.80)",
    orientation_circle_stroke="rgba(0,0,0,0.25)",
    orientation_circle_stroke_width=1.0,

    
    
    
    show_viewpoint_text=True,
    viewpoint_right_px=24,
    viewpoint_top_px=24,
    viewpoint_font_size_px=14,
    viewpoint_text_color="black",
    viewpoint_bg_color="rgba(255,255,255,0.75)",
    viewpoint_box_width_px=290,

    show_control_panel=True,

    include_plotlyjs=True,
):
    

    
    
    
    if scalebar_label is None:
        scalebar_label = f"{int(scalebar_um)} μm"

    if scalebar_text_color is None:
        scalebar_text_color = scalebar_bar_color

    scalebar_size_options = sorted(
        {float(v) for v in scalebar_size_options if float(v) > 0}
        | {float(scalebar_um)}
    )
    if len(scalebar_size_options) == 0:
        raise ValueError("scalebar_size_options must contain positive values.")

    if orientation_axes_plot is None:
        orientation_axes_plot = {
            "RL": "x",
            "AP": "y",
            "DV": "z",
        }

    if orientation_label_pairs is None:
        orientation_label_pairs = {
            "RL": ("R", "L"),
            "AP": ("A", "P"),
            "DV": ("D", "V"),
        }

    if orientation_colors is None:
        orientation_colors = {
            "RL": "#000000",
            "AP": "#000000",
            "DV": "#000000",
        }

    orientation_axes_plot = {
        str(k).upper(): str(v).lower()
        for k, v in orientation_axes_plot.items()
    }

    orientation_label_pairs = {
        str(k).upper(): v
        for k, v in orientation_label_pairs.items()
    }

    orientation_colors = {
        str(k).upper(): v
        for k, v in orientation_colors.items()
    }

    for k, v in orientation_axes_plot.items():
        if k not in {"RL", "AP", "DV"}:
            raise ValueError("orientation_axes_plot keys must be 'RL', 'AP', and/or 'DV'.")
        if v not in {"x", "y", "z"}:
            raise ValueError("orientation_axes_plot values must be 'x', 'y', or 'z'.")

    orientation_rotation_deg = tuple(float(v) for v in orientation_rotation_deg)
    if len(orientation_rotation_deg) != 3:
        raise ValueError(
            "orientation_rotation_deg must be a tuple/list of 3 numbers: "
            "(rx, ry, rz)."
        )

    orientation_rotation_order = tuple(str(v).lower() for v in orientation_rotation_order)
    for ax in orientation_rotation_order:
        if ax not in {"x", "y", "z"}:
            raise ValueError(
                "orientation_rotation_order must only contain 'x', 'y', and 'z'."
            )

    orientation_min_scale = float(orientation_min_scale)
    orientation_max_scale = float(orientation_max_scale)
    orientation_resize_sensitivity = float(orientation_resize_sensitivity)
    if not (0 < orientation_min_scale <= 1.0 <= orientation_max_scale):
        raise ValueError(
            "Require 0 < orientation_min_scale <= 1 <= orientation_max_scale."
        )
    if orientation_resize_sensitivity <= 0:
        raise ValueError("orientation_resize_sensitivity must be positive.")

    scene = fig.layout.scene

    if scene.xaxis.range is None or scene.yaxis.range is None or scene.zaxis.range is None:
        raise ValueError(
            "fig.layout.scene.xaxis/yaxis/zaxis.range is missing. "
            "Please make sure the figure already has fixed scene ranges."
        )

    x_range = [float(scene.xaxis.range[0]), float(scene.xaxis.range[1])]
    y_range = [float(scene.yaxis.range[0]), float(scene.yaxis.range[1])]
    z_range = [float(scene.zaxis.range[0]), float(scene.zaxis.range[1])]

    if scene.camera is not None and scene.camera.eye is not None:
        eye = {
            "x": float(scene.camera.eye.x),
            "y": float(scene.camera.eye.y),
            "z": float(scene.camera.eye.z),
        }
    else:
        eye = {"x": 1.25, "y": 1.25, "z": 1.25}

    if scene.camera is not None and scene.camera.up is not None:
        up = {
            "x": float(scene.camera.up.x),
            "y": float(scene.camera.up.y),
            "z": float(scene.camera.up.z),
        }
    else:
        up = {"x": 0.0, "y": 1.0, "z": 0.0}

    initial_eye_norm = math.sqrt(
        eye["x"] ** 2 +
        eye["y"] ** 2 +
        eye["z"] ** 2
    )

    
    fig_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        default_width="100%",
        default_height="100%",
        config={
            "responsive": True,
            "displayModeBar": True,
        }
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Plotly 3D with synchronized scalebar</title>
<style>
html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
}}

#plot-wrapper {{
    position: relative;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
}}

#plot-wrapper .plotly-graph-div,
#plot-wrapper .js-plotly-plot,
#plot-wrapper .plot-container,
#plot-wrapper .svg-container {{
    position: absolute !important;
    left: 0 !important;
    top: 0 !important;
    width: 100% !important;
    height: 100% !important;
}}

#plot-overlay-svg {{
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    z-index: 1000;
    pointer-events: none;
    user-select: none;
}}
</style>
</head>
<body>

<div id="plot-wrapper">
    {fig_html}
    <svg id="plot-overlay-svg"></svg>
</div>

<script>
(function() {{
    const wrapper = document.getElementById("plot-wrapper");
    const gd = wrapper.querySelector(".plotly-graph-div");
    const svg = document.getElementById("plot-overlay-svg");

    if (!gd || !svg) {{
        console.error("Plotly graph div or overlay SVG not found.");
        return;
    }}

    const scalebarUm = {float(scalebar_um)};
    const scalebarLabel = {json.dumps(scalebar_label)};
    let scalebarBarColor = {json.dumps(scalebar_bar_color)};
    let scalebarTextColor = {json.dumps(scalebar_text_color)};
    const scalebarThicknessPx = {int(round(float(scalebar_thickness_px)))};
    const scalebarFontSizePx = {int(round(float(scalebar_font_size_px)))};
    const scalebarRightPx = {int(round(float(scalebar_right_px)))};
    const scalebarBottomPx = {int(round(float(scalebar_bottom_px)))};
    const scalebarResizable = {json.dumps(bool(scalebar_resizable))};
    const scalebarSizeOptions = {json.dumps(scalebar_size_options)};
    let currentScalebarIndex = Math.max(
        0,
        scalebarSizeOptions.indexOf(scalebarUm)
    );
    let currentScalebarUm = scalebarSizeOptions[currentScalebarIndex];
    const initialScalebarIndex = currentScalebarIndex;
    window.codexScalebarUnit = window.codexScalebarUnit || "um";

    const axisRanges = {{
        x: {json.dumps(x_range)},
        y: {json.dumps(y_range)},
        z: {json.dumps(z_range)}
    }};

    let overlayFrame = null;
    let overlayFramesRemaining = 0;
    let scalebarBasePixelLength = null;
    let scalebarInitialAspectRatio = null;
    const scalebarDragOffset = {{x: 0, y: 0}};
    const orientationDragOffset = {{x: 0, y: 0, scale: 1.0, originX: 0, originY: 0}};
    const viewpointDragOffset = {{x: 0, y: 0, scale: 1.0, originX: 0, originY: 0}};

    const initialEye = {json.dumps(eye)};
    const initialUp = {json.dumps(up)};
    const initialEyeNorm = {initial_eye_norm};

    const showOrientationWidget = {json.dumps(bool(show_orientation_widget))};
    const orientationAxesPlot = {json.dumps(orientation_axes_plot)};
    const orientationLabelPairs = {json.dumps(orientation_label_pairs)};
    const orientationColors = {json.dumps(orientation_colors)};

    const orientationSizePx = {int(round(float(orientation_size_px)))};
    const orientationLeftPx = {int(round(float(orientation_left_px)))};
    const orientationTopPx = {int(round(float(orientation_top_px)))};
    const orientationRadiusPx = {float(orientation_radius_px)};
    const orientationFontSizePx = {int(round(float(orientation_font_size_px)))};
    const orientationLineWidthPx = {float(orientation_line_width_px)};
    const orientationResizable = {json.dumps(bool(orientation_resizable))};
    const orientationMinScale = {orientation_min_scale};
    const orientationMaxScale = {orientation_max_scale};
    const orientationResizeSensitivity = {orientation_resize_sensitivity};

    const orientationRotationDegXYZ = {{
        x: {float(orientation_rotation_deg[0])},
        y: {float(orientation_rotation_deg[1])},
        z: {float(orientation_rotation_deg[2])}
    }};
    const orientationRotationOrder = {json.dumps(list(orientation_rotation_order))};

    const orientationShowCircle = {json.dumps(bool(orientation_show_circle))};
    let orientationCircleColor = {json.dumps(orientation_circle_color)};
    let orientationCircleStroke = {json.dumps(orientation_circle_stroke)};
    const orientationCircleStrokeWidth = {float(orientation_circle_stroke_width)};

    const showViewpointText = {json.dumps(bool(show_viewpoint_text))};
    const viewpointRightPx = {int(round(float(viewpoint_right_px)))};
    const viewpointTopPx = {int(round(float(viewpoint_top_px)))};
    const viewpointFontSizePx = {int(round(float(viewpoint_font_size_px)))};
    let viewpointTextColor = {json.dumps(viewpoint_text_color)};
    let viewpointBgColor = {json.dumps(viewpoint_bg_color)};
    const viewpointBoxWidthPx = {int(round(float(viewpoint_box_width_px)))};
    const useHtmlViewpointPanel = {json.dumps(bool(show_control_panel))};

    window.codexOverlayVisibility = Object.assign({{
        scalebar: true,
        orientation: showOrientationWidget,
        viewpoint: showViewpointText
    }}, window.codexOverlayVisibility || {{}});
    window.codexOrientationLinked = true;
    window.codexOrientationLockedBasis = null;
    window.codexOrientationEditMode = "move";
    window.codexOrientationManualAngles = {{RL: 0, AP: 0, DV: 0}};
    window.codexOrientationManualRotation = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ];

    function norm(v) {{
        return Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }}

    function normalize(v) {{
        const n = norm(v);
        if (n < 1e-12) return [0, 0, 0];
        return [v[0]/n, v[1]/n, v[2]/n];
    }}

    function dot(a, b) {{
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }}

    function cross(a, b) {{
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ];
    }}

    function makeSvgElem(tag, attrs) {{
        const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
        for (const k in attrs) {{
            el.setAttribute(k, attrs[k]);
        }}
        return el;
    }}

    function applyDragTransform(group, state) {{
        let transform = "translate(" + state.x + " " + state.y + ")";
        if (state.scale !== undefined) {{
            transform += (
                " translate(" + state.originX + " " + state.originY + ")" +
                " scale(" + state.scale + ")" +
                " translate(" + (-state.originX) + " " + (-state.originY) + ")"
            );
        }}
        group.setAttribute("transform", transform);
    }}

    function attachOverlayDrag(group, state) {{
        let dragging = false;
        let startClientX = 0;
        let startClientY = 0;
        let startOffsetX = 0;
        let startOffsetY = 0;

        group.style.pointerEvents = "all";
        group.style.cursor = "move";
        group.style.touchAction = "none";
        applyDragTransform(group, state);

        group.addEventListener("pointerdown", function(event) {{
            if (event.button !== 0) return;
            event.preventDefault();
            event.stopPropagation();
            dragging = true;
            startClientX = event.clientX;
            startClientY = event.clientY;
            startOffsetX = state.x;
            startOffsetY = state.y;
            if (group.setPointerCapture) {{
                group.setPointerCapture(event.pointerId);
            }}
        }});

        group.addEventListener("pointermove", function(event) {{
            if (!dragging) return;
            event.preventDefault();
            event.stopPropagation();
            state.x = startOffsetX + event.clientX - startClientX;
            state.y = startOffsetY + event.clientY - startClientY;
            applyDragTransform(group, state);
        }});

        const finishDrag = function(event) {{
            if (!dragging) return;
            event.preventDefault();
            event.stopPropagation();
            dragging = false;
            if (group.releasePointerCapture) {{
                try {{
                    group.releasePointerCapture(event.pointerId);
                }} catch (err) {{
                }}
            }}
        }};

        group.addEventListener("pointerup", finishDrag);
        group.addEventListener("pointercancel", finishDrag);
    }}

    function clearSvg() {{
        while (svg.firstChild) {{
            svg.removeChild(svg.firstChild);
        }}
    }}

    function getPlotSize() {{
        const rect = wrapper.getBoundingClientRect();
        return {{
            width: Math.max(1, rect.width),
            height: Math.max(1, rect.height)
        }};
    }}

    function forcePlotlyResize() {{
        if (window.Plotly && gd) {{
            try {{
                Plotly.Plots.resize(gd);
            }} catch (err) {{
            }}
        }}
    }}

    function getCurrentCamera() {{
        const fl = gd._fullLayout || {{}};
        const scene = fl.scene || {{}};
        const cam = scene.camera || {{}};
        const internalCamera = scene._scene && scene._scene.camera;

        const internalEye = internalCamera && internalCamera.eye;
        const internalUp = internalCamera && internalCamera.up;
        const internalCenter = internalCamera && internalCamera.center;
        const e = (internalEye && internalEye.length >= 3) ? {{
            x: internalEye[0], y: internalEye[1], z: internalEye[2]
        }} : (cam.eye || initialEye);
        const u = (internalUp && internalUp.length >= 3) ? {{
            x: internalUp[0], y: internalUp[1], z: internalUp[2]
        }} : (cam.up || initialUp);
        const c = (internalCenter && internalCenter.length >= 3) ? {{
            x: internalCenter[0], y: internalCenter[1], z: internalCenter[2]
        }} : (cam.center || {{x: 0, y: 0, z: 0}});

        return {{
            eye: {{
                x: (e.x !== undefined) ? e.x : initialEye.x,
                y: (e.y !== undefined) ? e.y : initialEye.y,
                z: (e.z !== undefined) ? e.z : initialEye.z
            }},
            up: {{
                x: (u.x !== undefined) ? u.x : initialUp.x,
                y: (u.y !== undefined) ? u.y : initialUp.y,
                z: (u.z !== undefined) ? u.z : initialUp.z
            }},
            center: {{
                x: (c.x !== undefined) ? c.x : 0,
                y: (c.y !== undefined) ? c.y : 0,
                z: (c.z !== undefined) ? c.z : 0
            }}
        }};
    }}

    function getInternalAspectRatio() {{
        const fl = gd._fullLayout || {{}};
        const scene = fl.scene || {{}};
        const scene3d = scene._scene;
        const glplot = scene3d && scene3d.glplot;
        let ratio = null;

        if (glplot && typeof glplot.getAspectratio === "function") {{
            ratio = glplot.getAspectratio();
        }} else if (scene.aspectratio) {{
            ratio = scene.aspectratio;
        }}

        if (
            ratio &&
            isFinite(ratio.x) && isFinite(ratio.y) && isFinite(ratio.z) &&
            ratio.x > 1e-12 && ratio.y > 1e-12 && ratio.z > 1e-12
        ) {{
            return {{x: ratio.x, y: ratio.y, z: ratio.z}};
        }}
        return null;
    }}

    function getCameraBasis() {{
        const cam = getCurrentCamera();

        const eye = normalize([
            cam.eye.x - cam.center.x,
            cam.eye.y - cam.center.y,
            cam.eye.z - cam.center.z
        ]);
        let up = normalize([cam.up.x, cam.up.y, cam.up.z]);

        let right = normalize(cross(up, eye));
        if (norm(right) < 1e-12) {{
            right = [1, 0, 0];
        }}

        let screenUp = normalize(cross(eye, right));
        if (norm(screenUp) < 1e-12) {{
            screenUp = [0, 1, 0];
        }}

        return {{
            eye: eye,
            up: up,
            right: right,
            screenUp: screenUp
        }};
    }}

    function getSceneProjector() {{
        const fl = gd._fullLayout || {{}};
        const sceneObj = fl.scene || {{}};

        if (sceneObj._scene && typeof sceneObj._scene.project === "function") {{
            return sceneObj._scene.project.bind(sceneObj._scene);
        }}

        return null;
    }}

    function computeScalebarPixelLengthByProjection() {{
        const projector = getSceneProjector();
        if (!projector) {{
            return null;
        }}

        const basis = getCameraBasis();

        const center = [
            (axisRanges.x[0] + axisRanges.x[1]) / 2.0,
            (axisRanges.y[0] + axisRanges.y[1]) / 2.0,
            (axisRanges.z[0] + axisRanges.z[1]) / 2.0
        ];

        const p0 = center;
        const p1 = [
            center[0] + basis.right[0] * scalebarUm,
            center[1] + basis.right[1] * scalebarUm,
            center[2] + basis.right[2] * scalebarUm
        ];

        let q0, q1;
        try {{
            q0 = projector(p0);
            q1 = projector(p1);
        }} catch (err) {{
            return null;
        }}

        if (!q0 || !q1 || q0.length < 2 || q1.length < 2) {{
            return null;
        }}

        const dx = q1[0] - q0[0];
        const dy = q1[1] - q0[1];
        const len = Math.sqrt(dx*dx + dy*dy);

        if (!isFinite(len) || len <= 0) {{
            return null;
        }}

        return len;
    }}

    function computeScalebarPixelLengthFallback() {{
        const size = getPlotSize();

        const spanX = Math.abs(axisRanges.x[1] - axisRanges.x[0]);
        const spanY = Math.abs(axisRanges.y[1] - axisRanges.y[0]);
        const spanZ = Math.abs(axisRanges.z[1] - axisRanges.z[0]);
        const meanSpan = (spanX + spanY + spanZ) / 3.0;

        const scenePixels = Math.min(size.width, size.height);
        const basePx = (scalebarUm / Math.max(meanSpan, 1e-9)) * scenePixels;

        const cam = getCurrentCamera();
        const eyeNow = cam.eye;
        const currentEyeNorm = Math.sqrt(
            eyeNow.x*eyeNow.x + eyeNow.y*eyeNow.y + eyeNow.z*eyeNow.z
        );

        const zoomFactor = initialEyeNorm / Math.max(currentEyeNorm, 1e-9);

        return basePx * zoomFactor;
    }}

    function computeScalebarPixelLength() {{
        const currentRatio = getInternalAspectRatio();
        const sizeFactor = currentScalebarUm / scalebarUm;

        if (
            scalebarInitialAspectRatio === null &&
            currentRatio !== null
        ) {{
            scalebarInitialAspectRatio = currentRatio;
        }}

        if (scalebarBasePixelLength === null) {{
            let baseLength = computeScalebarPixelLengthByProjection();
            if (baseLength === null) {{
                baseLength = computeScalebarPixelLengthFallback();
            }}
            scalebarBasePixelLength = Math.max(20, baseLength);

        }}

        if (
            currentRatio !== null &&
            scalebarInitialAspectRatio !== null
        ) {{
            const zoomFactor = (
                currentRatio.x / scalebarInitialAspectRatio.x +
                currentRatio.y / scalebarInitialAspectRatio.y +
                currentRatio.z / scalebarInitialAspectRatio.z
            ) / 3.0;
            return Math.max(
                20,
                scalebarBasePixelLength * zoomFactor * sizeFactor
            );
        }}

        return Math.max(20, scalebarBasePixelLength * sizeFactor);
    }}

    function formatScalebarLabel(valueUm) {{
        const unit = window.codexScalebarUnit === "mm" ? "mm" : "um";
        if (unit === "mm") {{
            const valueMm = valueUm / 1000.0;
            const shown = Number.isInteger(valueMm) ? String(valueMm) : String(Number(valueMm.toFixed(3)));
            return shown + " mm";
        }}
        const shown = Number.isInteger(valueUm) ? String(valueUm) : String(Number(valueUm.toFixed(2)));
        return shown + " \u00B5m";
    }}

    function drawScalebar() {{
        const size = getPlotSize();
        const barLen = computeScalebarPixelLength();

        const x2 = size.width - scalebarRightPx;
        const x1 = x2 - barLen;
        const y = size.height - scalebarBottomPx;

        const group = makeSvgElem("g", {{id: "draggable-scalebar"}});
        const hitBox = makeSvgElem("rect", {{
            x: x1 - 12,
            y: y - 18,
            width: barLen + 24,
            height: scalebarFontSizePx + 44,
            fill: "rgba(0,0,0,0)",
            "pointer-events": "all"
        }});
        group.appendChild(hitBox);

        const line = makeSvgElem("line", {{
            x1: x1,
            y1: y,
            x2: x2,
            y2: y,
            stroke: scalebarBarColor,
            "stroke-width": scalebarThicknessPx,
            "stroke-linecap": "butt"
        }});
        group.appendChild(line);

        const text = makeSvgElem("text", {{
            x: (x1 + x2) / 2.0,
            y: y + scalebarFontSizePx + 8,
            fill: scalebarTextColor,
            "font-size": scalebarFontSizePx,
            "font-family": "Arial, Helvetica, sans-serif",
            "text-anchor": "middle",
            "dominant-baseline": "middle",
            "font-weight": "normal"
        }});
        const dynamicLabel = scalebarResizable ? formatScalebarLabel(currentScalebarUm) : scalebarLabel;
        text.appendChild(document.createTextNode(dynamicLabel));
        group.appendChild(text);

        attachOverlayDrag(group, scalebarDragOffset);

        if (scalebarResizable) {{
            group.addEventListener("wheel", function(event) {{
                event.preventDefault();
                event.stopPropagation();
                const step = event.deltaY < 0 ? 1 : -1;
                currentScalebarIndex = Math.max(
                    0,
                    Math.min(
                        scalebarSizeOptions.length - 1,
                        currentScalebarIndex + step
                    )
                );
                currentScalebarUm = scalebarSizeOptions[currentScalebarIndex];
                scheduleOverlayUpdate();
                window.dispatchEvent(new Event("codex-overlay-state-change"));
            }}, {{passive: false}});

            group.addEventListener("dblclick", function(event) {{
                event.preventDefault();
                event.stopPropagation();
                currentScalebarIndex = initialScalebarIndex;
                currentScalebarUm = scalebarSizeOptions[currentScalebarIndex];
                scheduleOverlayUpdate();
                window.dispatchEvent(new Event("codex-overlay-state-change"));
            }});
        }}

        svg.appendChild(group);
    }}

    function getAxisVector(axisName) {{
        if (axisName === "x") return [1, 0, 0];
        if (axisName === "y") return [0, 1, 0];
        if (axisName === "z") return [0, 0, 1];
        return [0, 0, 0];
    }}

    function rotateVectorAroundAxis(v, axis, deg) {{
        const theta = deg * Math.PI / 180.0;
        const k = normalize(axis);

        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);

        const kv = dot(k, v);
        const kCrossV = cross(k, v);

        return [
            v[0] * cosT + kCrossV[0] * sinT + k[0] * kv * (1.0 - cosT),
            v[1] * cosT + kCrossV[1] * sinT + k[1] * kv * (1.0 - cosT),
            v[2] * cosT + kCrossV[2] * sinT + k[2] * kv * (1.0 - cosT)
        ];
    }}

    function applyOrientationRotationXYZ(v) {{
        orientationRotationOrder.forEach(function(rotAxisName) {{
            const deg = orientationRotationDegXYZ[rotAxisName] || 0.0;

            if (Math.abs(deg) < 1e-12) {{
                return;
            }}

            const rotAxisVec = getAxisVector(rotAxisName);

            v = rotateVectorAroundAxis(
                v,
                rotAxisVec,
                deg
            );
        }});

        return v;
    }}

    function applyManualOrientationRotation(v) {{
        const m = window.codexOrientationManualRotation || [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ];
        return [
            v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
            v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
            v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2]
        ];
    }}

    function cloneManualOrientationRotation() {{
        const m = window.codexOrientationManualRotation || [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ];
        return [
            m[0].slice(),
            m[1].slice(),
            m[2].slice()
        ];
    }}

    function rotateManualOrientationRotation(baseMatrix, axis, deg) {{
        return [
            rotateVectorAroundAxis(baseMatrix[0], axis, deg),
            rotateVectorAroundAxis(baseMatrix[1], axis, deg),
            rotateVectorAroundAxis(baseMatrix[2], axis, deg)
        ];
    }}

    function getAnatomicalAxisVector(anatKey) {{
        const axisName = orientationAxesPlot[anatKey];
        let v = getAxisVector(axisName);
        v = applyOrientationRotationXYZ(v);
        return v;
    }}

    function buildManualOrientationRotationFromAngles(angles) {{
        angles = angles || {{RL: 0, AP: 0, DV: 0}};
        let m = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ];
        ["RL", "AP", "DV"].forEach(function(anatKey) {{
            const deg = Number(angles[anatKey]) || 0;
            if (Math.abs(deg) < 1e-12) return;
            const axis = getAnatomicalAxisVector(anatKey);
            m = rotateManualOrientationRotation(m, axis, deg);
        }});
        return m;
    }}

    function normalizeManualAngleDeg(deg) {{
        let v = Number(deg) || 0;
        v = ((v + 180) % 360 + 360) % 360 - 180;
        return v;
    }}

    function normalizeManualAngles(angles) {{
        angles = angles || {{}};
        return {{
            RL: normalizeManualAngleDeg(angles.RL),
            AP: normalizeManualAngleDeg(angles.AP),
            DV: normalizeManualAngleDeg(angles.DV)
        }};
    }}

    function manualRotationMatrixScore(a, b) {{
        let score = 0;
        for (let i = 0; i < 3; i++) {{
            for (let j = 0; j < 3; j++) {{
                const d = a[i][j] - b[i][j];
                score += d * d;
            }}
        }}
        return score;
    }}

    function solveManualAnglesFromRotation(targetMatrix, seedAngles) {{
        const keys = ["RL", "AP", "DV"];
        let best = normalizeManualAngles(seedAngles || window.codexOrientationManualAngles);
        let bestScore = manualRotationMatrixScore(buildManualOrientationRotationFromAngles(best), targetMatrix);
        const steps = [12, 4, 1, 0.25, 0.05];
        steps.forEach(function(step) {{
            let improved = true;
            let guard = 0;
            while (improved && guard < 8) {{
                improved = false;
                guard += 1;
                keys.forEach(function(key) {{
                    [-step, step].forEach(function(delta) {{
                        const trial = {{RL: best.RL, AP: best.AP, DV: best.DV}};
                        trial[key] = normalizeManualAngleDeg(trial[key] + delta);
                        const score = manualRotationMatrixScore(buildManualOrientationRotationFromAngles(trial), targetMatrix);
                        if (score + 1e-12 < bestScore) {{
                            best = trial;
                            bestScore = score;
                            improved = true;
                        }}
                    }});
                }});
            }}
        }});
        return best;
    }}

    function rebuildManualOrientationRotationFromAngles() {{
        window.codexOrientationManualAngles = normalizeManualAngles(window.codexOrientationManualAngles);
        window.codexOrientationManualRotation = buildManualOrientationRotationFromAngles(window.codexOrientationManualAngles);
    }}

    window.codexApplyOrientationManualAngles = function(angles) {{
        window.codexOrientationManualAngles = Object.assign(
            {{RL: 0, AP: 0, DV: 0}},
            angles || {{}}
        );
        rebuildManualOrientationRotationFromAngles();
        scheduleOverlayUpdate();
    }};

    window.codexGetOrientationAxisVector = function(anatKey) {{
        return getRotatedOrientationVector(String(anatKey || "RL").toUpperCase());
    }};

    function copyOverlayOffset(target, value) {{
        if (!value || typeof value !== "object") return;
        ["x", "y", "scale"].forEach(function(key) {{
            const v = Number(value[key]);
            if (Number.isFinite(v)) target[key] = v;
        }});
    }}

    function nearestScalebarIndex(valueUm) {{
        let best = currentScalebarIndex;
        let bestDist = Infinity;
        scalebarSizeOptions.forEach(function(v, idx) {{
            const d = Math.abs(Number(v) - Number(valueUm));
            if (d < bestDist) {{ bestDist = d; best = idx; }}
        }});
        return best;
    }}

    window.codexSetScalebarUnit = function(unit) {{
        window.codexScalebarUnit = unit === "mm" ? "mm" : "um";
        scheduleOverlayUpdate();
        window.dispatchEvent(new Event("codex-overlay-state-change"));
    }};

    window.codexSetOverlayTheme = function(foreground, background) {{
        const fg = String(foreground || "#000000");
        const bg = String(background || "#ffffff");
        scalebarBarColor = fg;
        scalebarTextColor = fg;
        Object.keys(orientationColors).forEach(function(key) {{
            orientationColors[key] = fg;
        }});
        orientationCircleColor = bg === "#000000" ? "rgba(0,0,0,0.58)" : "rgba(255,255,255,0.72)";
        orientationCircleStroke = fg;
        viewpointTextColor = fg;
        viewpointBgColor = bg === "#000000" ? "rgba(0,0,0,0.72)" : "rgba(255,255,255,0.82)";
        window.codexOverlayForeground = fg;
        scheduleOverlayUpdate();
        trackOverlayForFrames(3);
    }};

    window.codexGetOverlayState = function() {{
        return {{
            scalebar: {{
                um: currentScalebarUm,
                index: currentScalebarIndex,
                unit: window.codexScalebarUnit === "mm" ? "mm" : "um",
                offset: {{x: scalebarDragOffset.x, y: scalebarDragOffset.y}}
            }},
            orientation: {{
                offset: {{
                    x: orientationDragOffset.x,
                    y: orientationDragOffset.y,
                    scale: orientationDragOffset.scale
                }}
            }}
        }};
    }};

    window.codexApplyOverlayState = function(state) {{
        if (!state || typeof state !== "object") return;
        if (state.scalebar) {{
            if (state.scalebar.unit) window.codexScalebarUnit = state.scalebar.unit === "mm" ? "mm" : "um";
            if (Number.isFinite(Number(state.scalebar.um))) {{
                currentScalebarIndex = nearestScalebarIndex(Number(state.scalebar.um));
                currentScalebarUm = scalebarSizeOptions[currentScalebarIndex];
            }} else if (Number.isFinite(Number(state.scalebar.index))) {{
                currentScalebarIndex = Math.max(0, Math.min(scalebarSizeOptions.length - 1, Math.round(Number(state.scalebar.index))));
                currentScalebarUm = scalebarSizeOptions[currentScalebarIndex];
            }}
            copyOverlayOffset(scalebarDragOffset, state.scalebar.offset);
        }}
        if (state.orientation) {{
            copyOverlayOffset(orientationDragOffset, state.orientation.offset);
            orientationDragOffset.scale = Math.max(orientationMinScale, Math.min(orientationMaxScale, orientationDragOffset.scale || 1));
        }}
        scheduleOverlayUpdate();
    }};

    function getRotatedOrientationVector(anatKey) {{
        const axisName = orientationAxesPlot[anatKey];

        let v = getAxisVector(axisName);
        v = applyOrientationRotationXYZ(v);
        v = applyManualOrientationRotation(v);

        return v;
    }}

    function projectVectorToScreen(v, rightVec, upVec) {{
        return [
            dot(v, rightVec),
            dot(v, upVec)
        ];
    }}

    function appendSharpArrow(group, x0, y0, xTip, yTip, color) {{
        const dx = xTip - x0;
        const dy = yTip - y0;
        const length = Math.sqrt(dx*dx + dy*dy);
        if (!isFinite(length) || length < 1e-6) return;

        const ux = dx / length;
        const uy = dy / length;
        const headLength = Math.min(11, Math.max(5, length * 0.38));
        const headHalfWidth = headLength * 0.46;
        const xBase = xTip - ux * headLength;
        const yBase = yTip - uy * headLength;
        const px = -uy;
        const py = ux;

        group.appendChild(makeSvgElem("line", {{
            x1: x0,
            y1: y0,
            x2: xBase,
            y2: yBase,
            stroke: color,
            "stroke-width": orientationLineWidthPx,
            "stroke-linecap": "butt"
        }}));

        group.appendChild(makeSvgElem("polygon", {{
            points:
                xTip + "," + yTip + " " +
                (xBase + px * headHalfWidth) + "," +
                (yBase + py * headHalfWidth) + " " +
                (xBase - px * headHalfWidth) + "," +
                (yBase - py * headHalfWidth),
            fill: color,
            stroke: "none"
        }}));
    }}

    function cloneBasis(basis) {{
        return {{
            eye: basis.eye.slice(),
            up: basis.up.slice(),
            right: basis.right.slice(),
            screenUp: basis.screenUp.slice()
        }};
    }}

    function rotateBasisAroundAxis(basis, axis, deg) {{
        return {{
            eye: rotateVectorAroundAxis(basis.eye, axis, deg),
            up: rotateVectorAroundAxis(basis.up, axis, deg),
            right: rotateVectorAroundAxis(basis.right, axis, deg),
            screenUp: rotateVectorAroundAxis(basis.screenUp, axis, deg)
        }};
    }}

    let savedPlotlyPointerEvents = null;
    function setPlotlyInteractionBlocked(blocked) {{
        if (!gd) return;
        if (blocked) {{
            if (savedPlotlyPointerEvents === null) {{
                savedPlotlyPointerEvents = gd.style.pointerEvents || "";
            }}
            gd.style.pointerEvents = "none";
        }} else if (savedPlotlyPointerEvents !== null) {{
            gd.style.pointerEvents = savedPlotlyPointerEvents;
            savedPlotlyPointerEvents = null;
        }}
    }}

    function stopForPlotly(event) {{
        if (!event) return;
        event.preventDefault();
        event.stopPropagation();
        if (typeof event.stopImmediatePropagation === "function") {{
            event.stopImmediatePropagation();
        }}
    }}

    function attachOrientationRotate(group) {{
        let rotating = false;
        let startClientX = 0;
        let startClientY = 0;
        let startBasis = null;
        let startManualRotation = null;
        let startManualAngles = null;

        group.style.pointerEvents = "all";
        group.style.cursor = "grab";
        group.style.touchAction = "none";
        applyDragTransform(group, orientationDragOffset);

        const moveRotate = function(event) {{
            if (!rotating || !startBasis) return;
            stopForPlotly(event);

            const dx = event.clientX - startClientX;
            const dy = event.clientY - startClientY;
            const sensitivity = 0.45;

            let m = startManualRotation;
            m = rotateManualOrientationRotation(m, startBasis.screenUp, dx * sensitivity);
            m = rotateManualOrientationRotation(m, startBasis.right, -dy * sensitivity);
            window.codexOrientationManualRotation = m;
            const nextAngles = solveManualAnglesFromRotation(
                m,
                window.codexOrientationManualAngles || startManualAngles || {{RL: 0, AP: 0, DV: 0}}
            );
            window.codexOrientationManualAngles = nextAngles;
            window.dispatchEvent(new CustomEvent("codex-orientation-calibration-change", {{detail: nextAngles}}));
            scheduleOverlayUpdate();
        }};

        const finishRotate = function(event) {{
            if (!rotating) return;
            stopForPlotly(event);
            rotating = false;
            group.style.cursor = "grab";
            setPlotlyInteractionBlocked(false);
            window.removeEventListener("pointermove", moveRotate, true);
            window.removeEventListener("pointerup", finishRotate, true);
            window.removeEventListener("pointercancel", finishRotate, true);
            window.removeEventListener("mousemove", stopForPlotly, true);
            window.removeEventListener("mouseup", finishRotate, true);
            if (group.releasePointerCapture) {{
                try {{
                    group.releasePointerCapture(event.pointerId);
                }} catch (err) {{
                }}
            }}
        }};

        group.addEventListener("pointerdown", function(event) {{
            if (event.button !== 0) return;
            stopForPlotly(event);
            rotating = true;
            startClientX = event.clientX;
            startClientY = event.clientY;
            startBasis = cloneBasis(window.codexOrientationLockedBasis || getCameraBasis());
            startManualRotation = cloneManualOrientationRotation();
            startManualAngles = Object.assign({{RL: 0, AP: 0, DV: 0}}, window.codexOrientationManualAngles || {{}});
            group.style.cursor = "grabbing";
            setPlotlyInteractionBlocked(true);
            window.addEventListener("pointermove", moveRotate, {{capture: true, passive: false}});
            window.addEventListener("pointerup", finishRotate, {{capture: true, passive: false}});
            window.addEventListener("pointercancel", finishRotate, {{capture: true, passive: false}});
            window.addEventListener("mousemove", stopForPlotly, {{capture: true, passive: false}});
            window.addEventListener("mouseup", finishRotate, {{capture: true, passive: false}});
            if (group.setPointerCapture) {{
                group.setPointerCapture(event.pointerId);
            }}
        }});

        group.addEventListener("mousedown", stopForPlotly, {{capture: true}});
    }}

    function drawOrientationWidget() {{
        if (!showOrientationWidget) {{
            return;
        }}

        const basis = (
            window.codexOrientationLinked === false &&
            window.codexOrientationLockedBasis
        ) ? window.codexOrientationLockedBasis : getCameraBasis();

        const cx = orientationLeftPx + orientationSizePx / 2.0;
        const cy = orientationTopPx + orientationSizePx / 2.0;
        orientationDragOffset.originX = cx;
        orientationDragOffset.originY = cy;

        const group = makeSvgElem("g", {{id: "draggable-orientation"}});
        group.appendChild(makeSvgElem("rect", {{
            x: orientationLeftPx,
            y: orientationTopPx,
            width: orientationSizePx,
            height: orientationSizePx,
            fill: "rgba(0,0,0,0)",
            "pointer-events": "all"
        }}));

        if (orientationShowCircle) {{
            group.appendChild(makeSvgElem("circle", {{
                cx: cx,
                cy: cy,
                r: orientationRadiusPx + 18,
                fill: orientationCircleColor,
                stroke: orientationCircleStroke,
                "stroke-width": orientationCircleStrokeWidth
            }}));
        }}

        group.appendChild(makeSvgElem("circle", {{
            cx: cx,
            cy: cy,
            r: 2.6,
            fill: "black"
        }}));

        const axisOrder = ["RL", "AP", "DV"];

        axisOrder.forEach(function(anatKey) {{
            const axisName = orientationAxesPlot[anatKey];
            const labels = orientationLabelPairs[anatKey];
            const color = orientationColors[anatKey] || "black";

            if (!axisName || !labels) return;

            const v = getRotatedOrientationVector(anatKey);
            const p2 = projectVectorToScreen(v, basis.right, basis.screenUp);

            const px = p2[0];
            const py = -p2[1];

            const xPos = cx + px * orientationRadiusPx;
            const yPos = cy + py * orientationRadiusPx;
            const xNeg = cx - px * orientationRadiusPx;
            const yNeg = cy - py * orientationRadiusPx;

            appendSharpArrow(group, cx, cy, xPos, yPos, color);
            appendSharpArrow(group, cx, cy, xNeg, yNeg, color);

            const labelOffset = 12;

            const posTx = cx + px * (orientationRadiusPx + labelOffset);
            const posTy = cy + py * (orientationRadiusPx + labelOffset);

            const negTx = cx - px * (orientationRadiusPx + labelOffset);
            const negTy = cy - py * (orientationRadiusPx + labelOffset);

            const textPos = makeSvgElem("text", {{
                x: posTx,
                y: posTy,
                fill: color,
                "font-size": orientationFontSizePx,
                "font-family": "Arial, Helvetica, sans-serif",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "font-weight": "bold"
            }});
            textPos.appendChild(document.createTextNode(labels[0]));
            group.appendChild(textPos);

            const textNeg = makeSvgElem("text", {{
                x: negTx,
                y: negTy,
                fill: color,
                "font-size": orientationFontSizePx,
                "font-family": "Arial, Helvetica, sans-serif",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "font-weight": "bold"
            }});
            textNeg.appendChild(document.createTextNode(labels[1]));
            group.appendChild(textNeg);
        }});

        if (
            window.codexOrientationEditMode === "rotate" &&
            window.codexOrientationLinked === false
        ) {{
            attachOrientationRotate(group);
        }} else {{
            attachOverlayDrag(group, orientationDragOffset);
        }}

        if (orientationResizable) {{
            group.addEventListener("wheel", function(event) {{
                event.preventDefault();
                event.stopPropagation();
                const factor = Math.exp(
                    -event.deltaY * orientationResizeSensitivity
                );
                orientationDragOffset.scale = Math.max(
                    orientationMinScale,
                    Math.min(
                        orientationMaxScale,
                        orientationDragOffset.scale * factor
                    )
                );
                applyDragTransform(group, orientationDragOffset);
            }}, {{passive: false}});

            group.addEventListener("dblclick", function(event) {{
                event.preventDefault();
                event.stopPropagation();
                orientationDragOffset.scale = 1.0;
                applyDragTransform(group, orientationDragOffset);
            }});
        }}

        svg.appendChild(group);
    }}

    function getCurrentViewpointAngles() {{
        const cam = getCurrentCamera();

        const v = normalize([
            cam.center.x - cam.eye.x,
            cam.center.y - cam.eye.y,
            cam.center.z - cam.eye.z
        ]);

        const elevRad = Math.asin(Math.max(-1, Math.min(1, v[2])));
        const azimRad = Math.atan2(v[0], v[1]);

        function projectToViewPlane(vec) {{
            const k = dot(vec, v);
            return normalize([
                vec[0] - v[0] * k,
                vec[1] - v[1] * k,
                vec[2] - v[2] * k
            ]);
        }}

        const worldUpCandidates = [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ];
        let refUp = [0, 0, 0];
        for (let i = 0; i < worldUpCandidates.length; i++) {{
            refUp = projectToViewPlane(worldUpCandidates[i]);
            if (norm(refUp) >= 1e-12) {{
                break;
            }}
        }}

        let cameraUp = projectToViewPlane([cam.up.x, cam.up.y, cam.up.z]);
        if (norm(cameraUp) < 1e-12) {{
            cameraUp = refUp;
        }}

        const rollRad = Math.atan2(
            dot(cross(refUp, cameraUp), v),
            dot(refUp, cameraUp)
        );

        const elevDeg = elevRad * 180.0 / Math.PI;
        const azimDeg = azimRad * 180.0 / Math.PI;
        const rollDeg = rollRad * 180.0 / Math.PI;

        return {{
            elev: elevDeg,
            azim: azimDeg,
            roll: rollDeg
        }};
    }}

    function drawViewpointText() {{
        if (
            useHtmlViewpointPanel ||
            !showViewpointText ||
            window.codexOverlayVisibility.viewpoint === false
        ) {{
            return;
        }}

        const size = getPlotSize();
        const vp = getCurrentViewpointAngles();

        const textStr =
            "viewpoint = (" +
            vp.elev.toFixed(1) +
            ", " +
            vp.azim.toFixed(1) +
            ", " +
            vp.roll.toFixed(1) +
            ")";

        const padX = 10;
        const padY = 6;
        const boxW = viewpointBoxWidthPx;
        const boxH = viewpointFontSizePx + padY * 2;

        const x = size.width - viewpointRightPx - boxW;
        const y = viewpointTopPx;

        const group = makeSvgElem("g", {{
            id: "draggable-viewpoint",
            "pointer-events": "all"
        }});
        viewpointDragOffset.originX = x + boxW / 2;
        viewpointDragOffset.originY = y + boxH / 2;

        const hitbox = makeSvgElem("rect", {{
            x: x - 6,
            y: y - 6,
            width: boxW + 12,
            height: boxH + 12,
            fill: "transparent",
            stroke: "none",
            "pointer-events": "all"
        }});
        group.appendChild(hitbox);

        const rect = makeSvgElem("rect", {{
            x: x,
            y: y,
            width: boxW,
            height: boxH,
            rx: 5,
            ry: 5,
            fill: viewpointBgColor,
            stroke: "rgba(0,0,0,0.15)",
            "stroke-width": 1
        }});
        group.appendChild(rect);

        const text = makeSvgElem("text", {{
            x: x + padX,
            y: y + boxH / 2,
            fill: viewpointTextColor,
            "font-size": viewpointFontSizePx,
            "font-family": "Arial, Helvetica, sans-serif",
            "dominant-baseline": "middle",
            "text-anchor": "start"
        }});
        text.appendChild(document.createTextNode(textStr));
        group.appendChild(text);

        attachOverlayDrag(group, viewpointDragOffset);

        group.addEventListener("wheel", function(event) {{
            event.preventDefault();
            event.stopPropagation();
            const factor = Math.exp(-event.deltaY * 0.0015);
            viewpointDragOffset.scale = Math.max(
                0.60,
                Math.min(2.50, viewpointDragOffset.scale * factor)
            );
            applyDragTransform(group, viewpointDragOffset);
        }}, {{passive: false}});

        group.addEventListener("dblclick", function(event) {{
            event.preventDefault();
            event.stopPropagation();
            viewpointDragOffset.scale = 1.0;
            applyDragTransform(group, viewpointDragOffset);
        }});

        svg.appendChild(group);
    }}

    function updateOverlay() {{
        const size = getPlotSize();
        svg.setAttribute("width", size.width);
        svg.setAttribute("height", size.height);
        svg.setAttribute("viewBox", "0 0 " + size.width + " " + size.height);

        clearSvg();
        if (window.codexOverlayVisibility.scalebar !== false) {{
            drawScalebar();
        }}
        if (window.codexOverlayVisibility.orientation !== false) {{
            drawOrientationWidget();
        }}
        drawViewpointText();
    }}

    function scheduleOverlayUpdate() {{
        overlayFramesRemaining = Math.max(overlayFramesRemaining, 1);
        if (overlayFrame !== null) return;

        const runOverlayFrame = function() {{
            overlayFrame = null;
            updateOverlay();
            overlayFramesRemaining -= 1;

            if (overlayFramesRemaining > 0) {{
                overlayFrame = requestAnimationFrame(runOverlayFrame);
            }}
        }};

        overlayFrame = requestAnimationFrame(runOverlayFrame);
    }}

    function trackOverlayForFrames(frameCount) {{
        overlayFramesRemaining = Math.max(
            overlayFramesRemaining,
            Math.max(1, Number(frameCount) || 1)
        );
        if (overlayFrame === null) {{
            scheduleOverlayUpdate();
        }}
    }}

    if (gd.on) {{
        gd.on("plotly_relayout", function() {{
            trackOverlayForFrames(3);
        }});

        gd.on("plotly_afterplot", function() {{
            trackOverlayForFrames(2);
        }});

    }}

    window.addEventListener("resize", function() {{
        scalebarBasePixelLength = null;
        forcePlotlyResize();
        trackOverlayForFrames(3);
    }});

    window.addEventListener("codex-overlay-visibility", function() {{
        trackOverlayForFrames(2);
    }});

    gd.addEventListener("wheel", function() {{
        trackOverlayForFrames(2);
    }}, {{passive: true}});

    gd.addEventListener("pointermove", function(event) {{
        if (event.buttons !== 0) {{
            trackOverlayForFrames(3);
        }}
    }}, {{passive: true}});

    setTimeout(function() {{
        forcePlotlyResize();
        scheduleOverlayUpdate();
    }}, 0);

    setTimeout(function() {{
        forcePlotlyResize();
        scheduleOverlayUpdate();
    }}, 250);

}})();
</script>

</body>
</html>
"""

    if show_control_panel:
        control_panel_html = r'''
<style id="codex-surface-transform-style">
#codex-surface-transform-panel {
  position: fixed;
  left: 190px;
  top: 14px;
  z-index: 1200;
  width: 440px;
  min-width: 360px;
  min-height: 180px;
  height: calc(100vh - 28px);
  max-width: none;
  max-height: calc(100vh - 28px);
  overflow: auto;
  resize: both;
  padding: 0;
  background: rgba(255,255,255,0.84);
  border: 1px solid rgba(30,40,60,0.20);
  border-radius: 8px;
  box-shadow: 0 6px 18px rgba(20,30,50,0.14);
  font: 12px/1.35 Arial, sans-serif;
  color: #13213a;
  transform-origin: left top;
  user-select: none;
}
#codex-surface-transform-panel .codex-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 7px 10px;
  font-weight: 700;
  cursor: move;
  border-bottom: 1px solid rgba(30,40,60,0.12);
}
#codex-surface-transform-panel .codex-minimize {
  width: 24px;
  height: 24px;
  padding: 0;
  flex: 0 0 24px;
  font-size: 17px;
  line-height: 20px;
}
#codex-surface-transform-panel.codex-collapsed {
  width: 44px !important;
  height: 44px !important;
  min-width: 44px !important;
  min-height: 44px !important;
  max-width: 44px !important;
  max-height: 44px !important;
  overflow: hidden;
  resize: none;
  border-radius: 50%;
  background: rgba(255,255,255,0.92);
}
#codex-surface-transform-panel.codex-collapsed .codex-title {
  width: 44px;
  height: 44px;
  box-sizing: border-box;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border-bottom: 0;
}
#codex-surface-transform-panel.codex-collapsed .codex-title > span { display: none; }
#codex-surface-transform-panel.codex-collapsed .codex-body { display: none; }
#codex-surface-transform-panel.codex-collapsed .codex-minimize {
  display: block;
  width: 44px;
  height: 44px;
  flex-basis: 44px;
  border: 0;
  background: transparent;
  font-size: 22px;
  line-height: 44px;
  pointer-events: none;
}
#codex-surface-transform-panel .codex-body { padding: 8px 10px 10px; }
#codex-surface-transform-panel .codex-section-title {
  margin: 10px 0 5px;
  color: rgba(19,33,58,0.72);
  font-weight: 700;
}
#codex-surface-transform-panel .codex-hint {
  color: rgba(19,33,58,0.72);
  font-weight: 400;
  font-size: 11px;
}
#codex-surface-transform-panel label {
  display: grid;
  grid-template-columns: 68px 1fr 62px;
  gap: 7px;
  align-items: center;
  margin: 6px 0;
}
#codex-surface-transform-panel input[type="range"] { width: 100%; }
#codex-surface-transform-panel input[type="number"],
#codex-surface-transform-panel input[type="text"] {
  width: 62px;
  box-sizing: border-box;
  border: 1px solid rgba(30,40,60,0.24);
  border-radius: 4px;
  padding: 2px 4px;
  font: inherit;
}
#codex-surface-transform-panel select {
  box-sizing: border-box;
  border: 1px solid rgba(30,40,60,0.24);
  border-radius: 4px;
  padding: 2px 4px;
  font: inherit;
  background: rgba(255,255,255,0.96);
}
#codex-surface-transform-panel input[type="color"] {
  width: 34px;
  height: 24px;
  padding: 0;
  border: 1px solid rgba(30,40,60,0.24);
  border-radius: 4px;
  background: transparent;
}
#codex-surface-transform-panel button {
  border: 1px solid rgba(30,40,60,0.24);
  background: rgba(255,255,255,0.94);
  color: #13213a;
  border-radius: 5px;
  padding: 4px 8px;
  cursor: pointer;
  font: inherit;
}
#codex-surface-transform-panel button:hover { background: rgba(245,248,255,0.98); }
#codex-surface-transform-panel .codex-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 8px;
}
#codex-surface-transform-panel .codex-check-row {
  display: flex;
  flex-wrap: wrap;
  gap: 7px 10px;
  align-items: center;
  margin: 6px 0;
}
#codex-surface-transform-panel .codex-check-row label {
  display: inline-flex;
  grid-template-columns: none;
  gap: 4px;
  margin: 0;
}
#codex-surface-transform-panel .codex-trace-row {
  display: grid;
  grid-template-columns: minmax(72px,1fr) 38px minmax(72px,1fr) 62px 24px;
  gap: 6px;
  align-items: center;
  margin: 6px 0;
}
#codex-surface-transform-panel .codex-trace-row input[type="number"] {
  width: 62px;
  text-align: right;
}
#codex-surface-transform-panel .codex-trace-name {
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
#codex-surface-transform-panel .codex-angle-row {
  display: grid;
  grid-template-columns: 48px 1fr;
  gap: 6px;
  align-items: center;
  margin: 6px 0;
}
#codex-surface-transform-panel .codex-angle-row label {
  display: none;
}
#codex-surface-transform-panel .codex-config-status {
  min-height: 14px;
}
#codex-surface-transform-panel .codex-file-input {
  display: none;
}
#codex-surface-transform-panel .codex-axis-rotate-row {
  display: grid;
  grid-template-columns: 44px 1fr 64px;
  gap: 6px;
  align-items: center;
  margin: 6px 0;
}
#codex-surface-transform-panel .codex-axis-rotate-row input[type="number"] {
  width: 64px;
}
#codex-surface-transform-panel .codex-calibration-row {
  display: grid;
  grid-template-columns: 30px 1fr 54px;
  gap: 6px;
  align-items: center;
  margin: 5px 0;
}
#codex-surface-transform-panel .codex-calibration-row input[type="number"] {
  width: 54px;
}
#codex-surface-transform-panel .codex-light-row {
  grid-template-columns: 54px minmax(0,1fr) 54px;
}
#codex-surface-transform-panel .codex-light-row > span {
  white-space: nowrap;
}
#codex-surface-transform-panel .codex-export-row {
  display: grid;
  grid-template-columns: 66px minmax(0,1fr) 42px;
  gap: 6px;
  align-items: center;
  margin: 5px 0;
}
#codex-surface-transform-panel .codex-export-row input[type="number"] {
  width: 100%;
  min-width: 0;
  text-align: right;
}
#codex-surface-transform-panel .codex-background-row {
  display: grid;
  grid-template-columns: 66px minmax(0,1fr);
  gap: 6px;
  align-items: center;
  margin: 5px 0;
}
#codex-surface-transform-panel .codex-background-row select { width: 100%; }
#codex-surface-transform-panel .codex-outline-row {
  display: grid;
  grid-template-columns: 66px 36px minmax(0,1fr) 58px;
  gap: 6px;
  align-items: center;
  margin: 5px 0;
}
#codex-surface-transform-panel .codex-outline-row input[type="number"] {
  width: 58px;
  text-align: right;
}
#codex-surface-transform-panel .codex-export-status {
  min-height: 16px;
  user-select: text;
}
#codex-surface-transform-panel .codex-slice-row {
  display: grid;
  grid-template-columns: 24px minmax(0,1fr) 72px;
  gap: 6px;
  align-items: center;
  margin: 4px 0;
}
#codex-surface-transform-panel .codex-slice-row input[type="number"] {
  width: 72px;
  text-align: right;
}
#codex-apply-oblique-display-cut,
#codex-restore-oblique-display-cut,
#codex-oblique-cut-status {
  display: none !important;
}
#codex-surface-transform-panel .codex-oblique-row {
  display: grid;
  grid-template-columns: 54px repeat(3, minmax(0, 1fr));
  gap: 6px;
  align-items: center;
  margin: 5px 0;
}
#codex-surface-transform-panel .codex-oblique-row.two-values {
  grid-template-columns: 54px minmax(0, 1fr) minmax(0, 1fr);
}
#codex-surface-transform-panel .codex-oblique-row input[type="number"] {
  width: 100%;
  min-width: 0;
  text-align: right;
}
#codex-surface-transform-panel .codex-oblique-params {
  padding: 6px;
  border: 1px solid rgba(30,40,60,0.16);
  border-radius: 5px;
  background: rgba(248,250,255,0.90);
  color: #13213a;
  font: 10px/1.45 Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word;
  user-select: text;
}
#codex-surface-transform-panel .codex-note {
  color: rgba(19,33,58,0.66);
  font-size: 11px;
  margin: 4px 0;
}
#codex-surface-transform-panel .codex-current-view {
  font: 11px/1.45 Consolas, monospace;
  color: rgba(19,33,58,0.78);
  margin: 4px 0 6px;
  white-space: nowrap;
}
#codex-viewpoint-display {
  position: fixed;
  right: 24px;
  top: 24px;
  z-index: 1190;
  padding: 7px 10px;
  background: rgba(255,255,255,0.84);
  border: 1px solid rgba(30,40,60,0.20);
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(20,30,50,0.12);
  color: #13213a;
  font: 12px/1.4 Consolas, monospace;
  pointer-events: auto;
  cursor: move;
  user-select: none;
  touch-action: none;
  white-space: nowrap;
  transform-origin: left top;
}
</style>
<div id="codex-viewpoint-display">viewpoint = (...)</div>
<div id="codex-surface-transform-panel">
  <div class="codex-title" id="codex-surface-panel-title">
    <span>Display controls</span>
    <span class="codex-hint">Drag / minimize</span>
    <button id="codex-panel-minimize" class="codex-minimize" type="button" title="Minimize control panel" aria-label="Minimize control panel">−</button>
  </div>
  <div class="codex-body">
    <div class="codex-section-title">Overlays</div>
    <div class="codex-check-row">
      <label><input id="codex-show-orientation" type="checkbox" checked>Orientation arrows</label>
      <label><input id="codex-show-scalebar" type="checkbox" checked>Scale bar</label>
      <label><input id="codex-show-viewpoint" type="checkbox" checked>View angles</label>
      <label><input id="codex-link-orientation" type="checkbox" checked>Synchronize arrows</label>
    </div>
    <div class="codex-section-title">Arrow calibration</div>
    <div class="codex-note">Adjusts the relationship between the orientation arrows and the model orientation; the calibration is retained when synchronization is re-enabled.</div>
    <div class="codex-check-row">
      <label><input id="codex-orientation-mode-move" name="codex-orientation-mode" type="radio" value="move" checked>Move arrows</label>
      <label><input id="codex-orientation-mode-rotate" name="codex-orientation-mode" type="radio" value="rotate">Rotate arrows</label>
    </div>
    <div class="codex-calibration-row"><span>RL</span><input id="codex-orient-cal-rl-range" type="range" min="-180" max="180" step="1" value="0"><input id="codex-orient-cal-rl" type="number" step="0.1" value="0"></div>
    <div class="codex-calibration-row"><span>AP</span><input id="codex-orient-cal-ap-range" type="range" min="-180" max="180" step="1" value="0"><input id="codex-orient-cal-ap" type="number" step="0.1" value="0"></div>
    <div class="codex-calibration-row"><span>DV</span><input id="codex-orient-cal-dv-range" type="range" min="-180" max="180" step="1" value="0"><input id="codex-orient-cal-dv" type="number" step="0.1" value="0"></div>
    <div class="codex-actions">
      <button id="codex-reset-orientation-cal" type="button">Reset arrow calibration</button>
    </div>

    <div class="codex-section-title">View</div>
    <div id="codex-view-current" class="codex-current-view">viewpoint = (...)</div>
    <div class="codex-angle-row"><span>elev</span><input id="codex-view-elev" type="number" step="0.1" value="0"><label><input id="codex-lock-elev" type="checkbox">Lock</label></div>
    <div class="codex-angle-row"><span>azim</span><input id="codex-view-azim" type="number" step="0.1" value="90"><label><input id="codex-lock-azim" type="checkbox">Lock</label></div>
    <div class="codex-angle-row"><span>roll</span><input id="codex-view-roll" type="number" step="0.1" value="0"><label><input id="codex-lock-roll" type="checkbox">Lock</label></div>
    <div class="codex-actions">
      <button id="codex-apply-view" type="button">Apply view</button>
      <button id="codex-read-view" type="button">Read current view</button>
    </div>

    <div class="codex-section-title">Rotate about arrow axis</div>
    <div class="codex-note">Select an orientation-arrow axis and rotate the model view about it by the specified step.</div>
    <div class="codex-axis-rotate-row">
      <span>Axis</span>
      <select id="codex-axis-rotate-axis">
        <option value="RL">RL</option>
        <option value="AP">AP</option>
        <option value="DV">DV</option>
      </select>
      <input id="codex-axis-rotate-step" type="number" step="0.1" value="5">
    </div>
    <div class="codex-actions">
      <button id="codex-axis-rotate-neg" type="button">− Rotate</button>
      <button id="codex-axis-rotate-pos" type="button">+ Rotate</button>
    </div>

    <div class="codex-section-title">Background</div>
    <div class="codex-background-row">
      <span>Color</span>
      <select id="codex-background-theme">
        <option value="white">White background</option>
        <option value="black">Black background</option>
      </select>
    </div>

    <div class="codex-section-title">Shells</div>
    <div id="codex-shell-controls" class="codex-note">Detecting mesh3d traces...</div>

    <div class="codex-section-title">Surface lighting</div>
    <div class="codex-check-row"><label><input id="codex-lighting-enabled" type="checkbox">Enable custom lighting</label></div>
    <div class="codex-note">By default, each shell retains the lighting from the original file. The uniform settings below apply only when enabled.</div>
    <div class="codex-calibration-row codex-light-row"><span>Azimuth</span><input id="codex-light-azimuth-range" type="range" min="-180" max="180" step="1" value="90"><input id="codex-light-azimuth" type="number" min="-180" max="180" step="1" value="90"></div>
    <div class="codex-calibration-row codex-light-row"><span>Elevation</span><input id="codex-light-elevation-range" type="range" min="-89" max="89" step="1" value="45"><input id="codex-light-elevation" type="number" min="-89" max="89" step="1" value="45"></div>
    <div class="codex-calibration-row codex-light-row"><span>Ambient</span><input id="codex-light-ambient-range" type="range" min="0" max="1" step="0.01" value="0.65"><input id="codex-light-ambient" type="number" min="0" max="1" step="0.01" value="0.65"></div>
    <div class="codex-calibration-row codex-light-row"><span>Diffuse</span><input id="codex-light-diffuse-range" type="range" min="0" max="1" step="0.01" value="0.75"><input id="codex-light-diffuse" type="number" min="0" max="1" step="0.01" value="0.75"></div>
    <div class="codex-calibration-row codex-light-row"><span>Specular</span><input id="codex-light-specular-range" type="range" min="0" max="1" step="0.01" value="0.50"><input id="codex-light-specular" type="number" min="0" max="1" step="0.01" value="0.50"></div>
    <div class="codex-calibration-row codex-light-row"><span>Roughness</span><input id="codex-light-roughness-range" type="range" min="0" max="1" step="0.01" value="0.90"><input id="codex-light-roughness" type="number" min="0" max="1" step="0.01" value="0.90"></div>
    <div class="codex-actions"><button id="codex-restore-lighting" type="button">Restore original lighting</button></div>

    <div class="codex-section-title">Transparency sorting</div>
    <div class="codex-check-row"><label><input id="codex-transparency-sort" type="checkbox">Optimize transparent triangle order after rotation stops</label></div>
    <div class="codex-actions"><button id="codex-transparency-sort-now" type="button">Sort now</button></div>
    <div id="codex-transparency-sort-status" class="codex-note">Disabled by default. Sorting large models may take some time.</div>

    <div class="codex-section-title">Lines</div>
    <div id="codex-line-controls" class="codex-note">Detecting scatter3d lines...</div>

    <div class="codex-section-title">Auto-cropped export</div>
    <div class="codex-note">Automatically detects the model aspect ratio from the current projection and removes blank margins. Long-side length in centimeters and DPI jointly determine the raster model's pixel dimensions.</div>
    <div class="codex-export-row"><span>Long side</span><input id="codex-export-long-cm" type="number" min="1" max="100" step="0.1" value="20"><span>cm</span></div>
    <div class="codex-export-row"><span>DPI</span><input id="codex-export-dpi" type="number" min="72" max="1200" step="1" value="300"><span>dpi</span></div>
    <div class="codex-export-row"><span>Margin</span><input id="codex-export-margin-pct" type="number" min="0" max="30" step="0.5" value="5"><span>%</span></div>
    <div class="codex-check-row">
      <label><input id="codex-export-orientation" type="checkbox" checked>Orientation arrows</label>
      <label><input id="codex-export-scalebar" type="checkbox" checked>Scale bar</label>
      <label><input id="codex-export-allow-oversize" type="checkbox">Attempt oversized export</label>
    </div>
    <div class="codex-section-title">Export outline</div>
    <div class="codex-check-row"><label><input id="codex-export-outline" type="checkbox">Add outermost outline</label></div>
    <div class="codex-outline-row"><span>Color / width</span><input id="codex-export-outline-color" type="color" value="#000000"><input id="codex-export-outline-width-range" type="range" min="1" max="40" step="1" value="4"><input id="codex-export-outline-width" type="number" min="1" max="40" step="1" value="4"></div>
    <div class="codex-note">Calculated only during export and does not affect model rotation. Width is measured in output pixels.</div>
    <div class="codex-note">Include in filename</div>
    <div class="codex-check-row">
      <label><input id="codex-filename-view" type="checkbox" checked>Current angles</label>
      <label><input id="codex-filename-background" type="checkbox" checked>Background</label>
      <label><input id="codex-filename-pixels" type="checkbox" checked>Pixel dimensions</label>
      <label><input id="codex-filename-dpi" type="checkbox" checked>DPI</label>
    </div>
    <div class="codex-note">Oversized mode removes the output limits of 8192 pixels per side and 50 million total pixels. If the intermediate WebGL canvas is too large, the model is first rendered completely at a safe size, then scaled proportionally to the target dimensions to avoid clipping and distortion.</div>
    <div id="codex-export-status" class="codex-note codex-export-status"></div>
    <div class="codex-actions">
      <button id="codex-export-png" type="button">Composite PNG</button>
      <button id="codex-export-pdf" type="button">Layered PDF</button>
      <button id="codex-export-svg" type="button">Editable SVG</button>
    </div>

    <div class="codex-section-title">Axis-aligned clipping</div>
    <div class="codex-note">This is browser-side display clipping: it changes the 3D axis ranges but does not regenerate a smaller mesh.</div>
    <div id="codex-slice-volume-note" class="codex-note"></div>
    <div id="codex-slice-controls"></div>
    <div class="codex-actions">
      <button id="codex-apply-slice" type="button">Apply once</button>      <button id="codex-reset-slice" type="button">Restore full range</button>
    </div>

    <div class="codex-section-title">Oblique clipping preview</div>
    <div class="codex-note">Each plane is controlled independently; the final result retains the intersection of the specified sides of all planes. Parameters use the current HTML x/y/z display coordinates and can be copied directly into Python.</div>
    <div class="codex-oblique-row two-values">
      <span>Current plane</span>
      <select id="codex-oblique-plane-select"></select>
      <select id="codex-oblique-keep" title="Which side of the plane to retain">
        <option value="ge">Retain n·p ≥ d</option>
        <option value="le">Retain n·p ≤ d</option>
      </select>
    </div>
    <div class="codex-oblique-row">
      <span>Normal vector</span>
      <input id="codex-oblique-nx" type="number" step="0.01" value="0" title="HTML x component">
      <input id="codex-oblique-ny" type="number" step="0.01" value="1" title="HTML y component">
      <input id="codex-oblique-nz" type="number" step="0.01" value="1" title="HTML z component">
    </div>
    <div class="codex-oblique-row two-values">
      <span>Drag forward/backward</span>
      <input id="codex-oblique-d-range" type="range" min="0" max="1" step="0.01" value="0.5">
      <input id="codex-oblique-d" type="number" step="0.01" value="0.5" title="Plane position d">
    </div>
    <div class="codex-oblique-row two-values">
      <span>Opacity</span>
      <input id="codex-oblique-opacity-range" type="range" min="0.05" max="0.90" step="0.01" value="0.55" title="Higher values are clearer">
      <input id="codex-oblique-opacity" type="number" min="0.05" max="0.90" step="0.01" value="0.55" title="Higher values are clearer">
    </div>
    <div class="codex-oblique-row two-values">
      <span>Rotate about axis</span>
      <select id="codex-oblique-rotate-axis">
        <option value="AP">AP</option><option value="DV">DV</option><option value="RL">RL</option>
      </select>
      <input id="codex-oblique-rotate-step" type="number" step="0.1" value="5" title="Rotation step (degrees)">
    </div>
    <div class="codex-actions">
      <button id="codex-oblique-rotate-neg" type="button">− Rotate</button>
      <button id="codex-oblique-rotate-pos" type="button">+ Rotate</button>
      <button id="codex-oblique-move-neg" type="button">Backward</button>
      <button id="codex-oblique-move-pos" type="button">Forward</button>
      <input id="codex-oblique-move-step" type="number" step="0.01" value="50" title="Forward/backward movement step">
    </div>
    <div class="codex-check-row">
      <label><input id="codex-show-oblique-preview" type="checkbox">Show preview plane</label>
    </div>
    <div class="codex-actions">
      <button id="codex-add-oblique-plane" type="button">Add plane</button>
      <button id="codex-delete-oblique-plane" type="button">Delete current plane</button>
      <button id="codex-reset-oblique-preview" type="button">Reset planes</button>
    </div>
    <div class="codex-actions">
      <button id="codex-apply-oblique-display-cut" type="button">Apply display clipping</button>
      <button id="codex-restore-oblique-display-cut" type="button">Restore unclipped view</button>
    </div>
    <div id="codex-oblique-cut-status" class="codex-note">Display clipping does not generate new closed cut surfaces. For final results, regenerate the model using the Python parameters below.</div>
    <div id="codex-oblique-status" class="codex-note"></div>
    <div id="codex-oblique-params" class="codex-oblique-params"></div>
  </div>
</div>
<script>
(function() {
  const wrapper = document.getElementById("plot-wrapper");
  const gd = wrapper && wrapper.querySelector(".plotly-graph-div");
  const panel = document.getElementById("codex-surface-transform-panel");
  const viewpointPanel = document.getElementById("codex-viewpoint-display");
  const title = document.getElementById("codex-surface-panel-title");
  const minBtn = document.getElementById("codex-panel-minimize");
  if (!gd || !panel || !window.Plotly) return;

  function byId(id) { return document.getElementById(id); }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function num(id, fallback) {
    const v = parseFloat(byId(id).value);
    return Number.isFinite(v) ? v : fallback;
  }
  let backgroundTheme = (
    String((gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.bgcolor) || "").toLowerCase() === "black" ||
    String((gd._fullLayout && gd._fullLayout.paper_bgcolor) || "").toLowerCase() === "black"
  ) ? "black" : "white";
  function overlayColor() {
    return backgroundTheme === "black" ? "#ffffff" : "#000000";
  }
  function isFeatureLineTrace(trace) {
    const name = String(trace && trace.name || "");
    return /feature[_\s-]*lines?/i.test(name);
  }
  function applyBackground(theme) {
    backgroundTheme = theme === "black" ? "black" : "white";
    const background = backgroundTheme === "black" ? "#000000" : "#ffffff";
    const foreground = overlayColor();
    window.codexExportBackground = backgroundTheme;
    window.codexOverlayForeground = foreground;
    document.documentElement.style.background = background;
    document.body.style.background = background;
    wrapper.style.background = background;
    if (viewpointPanel) {
      viewpointPanel.style.color = foreground;
      viewpointPanel.style.background = backgroundTheme === "black" ? "rgba(0,0,0,0.72)" : "rgba(255,255,255,0.82)";
      viewpointPanel.style.borderColor = foreground;
    }
    const outlineColor = byId("codex-export-outline-color");
    if (outlineColor && outlineColor.dataset.userEdited !== "true") outlineColor.value = foreground;
    if (typeof window.codexSetOverlayTheme === "function") {
      window.codexSetOverlayTheme(foreground, background);
    }
    const featureLineIndices = [];
    (gd.data || []).forEach(function(trace, index) {
      if (isFeatureLineTrace(trace)) featureLineIndices.push(index);
    });
    const lineUpdate = featureLineIndices.length ?
      Plotly.restyle(gd, {"line.color": foreground}, featureLineIndices) : Promise.resolve();
    return Promise.all([
      Plotly.relayout(gd, {
        "paper_bgcolor": background,
        "plot_bgcolor": background,
        "font.color": foreground,
        "title.font.color": foreground,
        "scene.bgcolor": background
      }),
      lineUpdate
    ]).then(function() {
      if (featureLineIndices.length) buildTraceControls();
      if (window.Plotly && Plotly.Plots) Plotly.Plots.resize(gd);
    });
  }

  function cloneLightingValue(value) {
    return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
  }
  const originalMeshLighting = [];
  (gd.data || []).forEach(function(trace, index) {
    if (trace && trace.type === "mesh3d") {
      originalMeshLighting.push({
        index: index,
        lighting: cloneLightingValue(trace.lighting),
        lightposition: cloneLightingValue(trace.lightposition)
      });
    }
  });
  function originalMeshIndices() {
    return originalMeshLighting.map(function(item) { return item.index; });
  }
  function restoreOriginalSurfaceLighting() {
    if (!gd || !window.Plotly) return;
    originalMeshLighting.forEach(function(original) {
      Plotly.restyle(gd, {
        lighting: original.lighting === undefined ? null : cloneLightingValue(original.lighting),
        lightposition: original.lightposition === undefined ? null : cloneLightingValue(original.lightposition)
      }, [original.index]);
    });
    byId("codex-lighting-enabled").checked = false;
  }
  function applySurfaceLighting() {
    if (!byId("codex-lighting-enabled").checked) {
      restoreOriginalSurfaceLighting();
      return;
    }
    const indices = originalMeshIndices();
    if (!indices.length) return;
    const azimuth = num("codex-light-azimuth", 90) * Math.PI / 180;
    const elevation = clamp(num("codex-light-elevation", 45), -89, 89) * Math.PI / 180;
    const horizontal = Math.cos(elevation);
    const radius = 141421.356;
    Plotly.restyle(gd, {
      "lighting.ambient": clamp(num("codex-light-ambient", 0.65), 0, 1),
      "lighting.diffuse": clamp(num("codex-light-diffuse", 0.75), 0, 1),
      "lighting.specular": clamp(num("codex-light-specular", 0.50), 0, 1),
      "lighting.roughness": clamp(num("codex-light-roughness", 0.90), 0, 1),
      "lightposition.x": radius * horizontal * Math.sin(azimuth),
      "lightposition.y": radius * Math.sin(elevation),
      "lightposition.z": radius * horizontal * Math.cos(azimuth)
    }, indices);
  }

  const transparencySortSources = [];
  const decodedPlotlyArrayCache = new WeakMap();
  function decodePlotlyNumericArray(value) {
    if (Array.isArray(value) || ArrayBuffer.isView(value)) return value;
    if (!value || typeof value !== "object" || typeof value.bdata !== "string") return null;
    if (decodedPlotlyArrayCache.has(value)) return decodedPlotlyArrayCache.get(value);
    const constructors = {i1:Int8Array,u1:Uint8Array,i2:Int16Array,u2:Uint16Array,i4:Int32Array,u4:Uint32Array,f4:Float32Array,f8:Float64Array};
    const Constructor = constructors[String(value.dtype || "").toLowerCase()];
    if (!Constructor) return null;
    const binary = atob(value.bdata);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
    const length = Math.floor(bytes.byteLength / Constructor.BYTES_PER_ELEMENT);
    const decoded = new Constructor(bytes.buffer, 0, length);
    decodedPlotlyArrayCache.set(value, decoded);
    return decoded;
  }
  function plotlyTraceNumericArray(traceIndex, key, preferFullData) {
    const rawTrace = gd.data && gd.data[traceIndex];
    const fullTrace = gd._fullData && gd._fullData[traceIndex];
    const candidates = preferFullData ? [fullTrace && fullTrace[key], rawTrace && rawTrace[key]] : [rawTrace && rawTrace[key], fullTrace && fullTrace[key]];
    for (let index = 0; index < candidates.length; index++) {
      const decoded = decodePlotlyNumericArray(candidates[index]);
      if (decoded && typeof decoded.length === "number") return decoded;
    }
    return null;
  }
  function refreshTransparencySortSources() {
    if (transparencySortSources.length) return;
    originalMeshIndices().forEach(function(index) {
      const i = plotlyTraceNumericArray(index, "i", true);
      const j = plotlyTraceNumericArray(index, "j", true);
      const k = plotlyTraceNumericArray(index, "k", true);
      if (i && j && k && i.length && i.length === j.length && i.length === k.length) {
        transparencySortSources.push({index:index, i:i, j:j, k:k});
      }
    });
  }
  let transparencySortTimer = null;
  let transparencySortRunning = false;
  let transparencySortAgain = false;
  function setTransparencySortStatus(message, isError) {
    const status = byId("codex-transparency-sort-status");
    if (!status) return;
    status.textContent = message;
    status.style.color = isError ? "#b2182b" : "";
  }
  function axisDepthScale(axis, aspectValue) {
    const range = axis && axis.range;
    const span = range && range.length >= 2 ? Math.abs(Number(range[1]) - Number(range[0])) : 1;
    return (Number(aspectValue) || 1) / (Number.isFinite(span) && span > 1e-12 ? span : 1);
  }
  function currentTransparencyDepthVector() {
    const camera = getCurrentCamera();
    const eye = {x:camera.eye.x-camera.center.x, y:camera.eye.y-camera.center.y, z:camera.eye.z-camera.center.z};
    const scene = gd && gd._fullLayout && gd._fullLayout.scene;
    const aspect = (scene && scene.aspectratio) || {x:1,y:1,z:1};
    return {
      x: eye.x * axisDepthScale(scene && scene.xaxis, aspect.x),
      y: eye.y * axisDepthScale(scene && scene.yaxis, aspect.y),
      z: eye.z * axisDepthScale(scene && scene.zaxis, aspect.z)
    };
  }
  function newIndexArrayLike(source, length) {
    return ArrayBuffer.isView(source) && source.constructor ? new source.constructor(length) : new Array(length);
  }
  function sortedTriangleArrays(source, trace, vector) {
    const count = source.i.length;
    const depths = new Float32Array(count);
    const xs = trace.x || [], ys = trace.y || [], zs = trace.z || [];
    let minimumDepth = Infinity, maximumDepth = -Infinity;
    for (let face = 0; face < count; face++) {
      const a=Number(source.i[face]), b=Number(source.j[face]), c=Number(source.k[face]);
      let depth=((Number(xs[a])+Number(xs[b])+Number(xs[c]))*vector.x+(Number(ys[a])+Number(ys[b])+Number(ys[c]))*vector.y+(Number(zs[a])+Number(zs[b])+Number(zs[c]))*vector.z)/3;
      if (!Number.isFinite(depth)) depth=0;
      depths[face]=depth;
      if (depth<minimumDepth) minimumDepth=depth;
      if (depth>maximumDepth) maximumDepth=depth;
    }
    if (!(maximumDepth > minimumDepth)) return {i:source.i,j:source.j,k:source.k};
    let bucketCount=16384;
    while (bucketCount<131072 && bucketCount<count/8) bucketCount*=2;
    const counts=new Uint32Array(bucketCount);
    const bucketScale=(bucketCount-1)/(maximumDepth-minimumDepth);
    for (let face=0;face<count;face++) counts[Math.max(0,Math.min(bucketCount-1,Math.floor((depths[face]-minimumDepth)*bucketScale)))]++;
    const cursor=new Uint32Array(bucketCount);
    let runningOffset=0;
    for (let bucket=0;bucket<bucketCount;bucket++){cursor[bucket]=runningOffset;runningOffset+=counts[bucket];}
    const sortedI=newIndexArrayLike(source.i,count), sortedJ=newIndexArrayLike(source.j,count), sortedK=newIndexArrayLike(source.k,count);
    for (let face=0;face<count;face++) {
      const bucket=Math.max(0,Math.min(bucketCount-1,Math.floor((depths[face]-minimumDepth)*bucketScale)));
      const target=cursor[bucket]++;
      sortedI[target]=source.i[face]; sortedJ[target]=source.j[face]; sortedK[target]=source.k[face];
    }
    return {i:sortedI,j:sortedJ,k:sortedK};
  }
  async function sortTransparencyForCurrentView() {
    if (!byId("codex-transparency-sort").checked) return;
    if (transparencySortRunning) { transparencySortAgain=true; return; }
    refreshTransparencySortSources();
    if (!transparencySortSources.length) { setTransparencySortStatus("No sortable transparent meshes were found.",true); return; }
    transparencySortRunning=true; transparencySortAgain=false;
    try {
      const vector=currentTransparencyDepthVector();
      let processed=0;
      for (let sourceIndex=0;sourceIndex<transparencySortSources.length;sourceIndex++) {
        const source=transparencySortSources[sourceIndex];
        const trace=(gd.data&&gd.data[source.index])||(gd._fullData&&gd._fullData[source.index])||{};
        const opacity=Number(trace.opacity);
        if (trace.visible===false||(Number.isFinite(opacity)&&opacity>=0.999)) continue;
        setTransparencySortStatus("Sorting transparent shell "+(sourceIndex+1)+"/"+transparencySortSources.length+" for the current view...",false);
        const sortTrace={x:plotlyTraceNumericArray(source.index,"x",false),y:plotlyTraceNumericArray(source.index,"y",false),z:plotlyTraceNumericArray(source.index,"z",false)};
        if (!sortTrace.x||!sortTrace.y||!sortTrace.z) throw new Error("Unable to read the vertex coordinates of shell "+(sourceIndex+1)+".");
        const sorted=sortedTriangleArrays(source,sortTrace,vector);
        await Plotly.restyle(gd,{i:[sorted.i],j:[sorted.j],k:[sorted.k]},[source.index]);
        processed++;
        await new Promise(function(resolve){setTimeout(resolve,0);});
      }
      setTransparencySortStatus(processed?"Transparency sorting was updated for the current view.":"The current shell is opaque or hidden; sorting is unnecessary.",false);
    } catch(error) {
      console.error(error);
      setTransparencySortStatus("Transparency sorting failed: "+(error&&error.message?error.message:String(error)),true);
    } finally {
      transparencySortRunning=false;
      if (transparencySortAgain) scheduleTransparencySort(80);
    }
  }
  function scheduleTransparencySort(delay) {
    if (!byId("codex-transparency-sort").checked) return;
    if (transparencySortTimer!==null) clearTimeout(transparencySortTimer);
    setTransparencySortStatus("The view is changing; sorting will restart automatically when movement stops...",false);
    transparencySortTimer=setTimeout(function(){transparencySortTimer=null;sortTransparencyForCurrentView();},Math.max(0,Number(delay)||0));
  }
  async function restoreOriginalTransparencyOrder() {
    if (transparencySortTimer!==null) {clearTimeout(transparencySortTimer);transparencySortTimer=null;}
    refreshTransparencySortSources();
    for (let sourceIndex=0;sourceIndex<transparencySortSources.length;sourceIndex++) {
      const source=transparencySortSources[sourceIndex];
      await Plotly.restyle(gd,{i:[source.i],j:[source.j],k:[source.k]},[source.index]);
    }
    setTransparencySortStatus("The original triangle order has been restored.",false);
  }
  function exportSettings() {
    const longCm = clamp(num("codex-export-long-cm", 20), 1, 100);
    const dpi = clamp(Math.round(num("codex-export-dpi", 300)), 72, 1200);
    const marginPct = clamp(num("codex-export-margin-pct", 5), 0, 30);
    return {
      longCm: longCm,
      dpi: dpi,
      marginPct: marginPct,
      longPx: Math.max(1, Math.round(longCm / 2.54 * dpi)),
      includeOrientation: !!byId("codex-export-orientation").checked,
      includeScalebar: !!byId("codex-export-scalebar").checked,
      includeOutline: !!byId("codex-export-outline").checked,
      outlineColor: byId("codex-export-outline-color").value,
      outlineWidth: clamp(Math.round(num("codex-export-outline-width", 4)), 1, 40),
      allowOversize: !!byId("codex-export-allow-oversize").checked,
      filenameView: !!byId("codex-filename-view").checked,
      filenameBackground: !!byId("codex-filename-background").checked,
      filenamePixels: !!byId("codex-filename-pixels").checked,
      filenameDpi: !!byId("codex-filename-dpi").checked
    };
  }
  function updateExportStatus(message, isError) {
    const status = byId("codex-export-status");
    if (!status) return;
    if (message) {
      status.textContent = message;
      status.style.color = isError ? "#b2182b" : "";
      return;
    }
    const settings = exportSettings();
    status.textContent = (
      "Target long side: " + settings.longPx +
      " px. The short side will be calculated automatically from the current model projection during export." +
      (settings.allowOversize ? " Oversized mode is enabled." : "")
    );
    status.style.color = settings.allowOversize ? "#a14f00" : "";
  }
  function crc32(bytes) {
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) {
      crc ^= bytes[i];
      for (let bit = 0; bit < 8; bit++) {
        crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
      }
    }
    return (crc ^ 0xffffffff) >>> 0;
  }
  function writeUint32BE(target, offset, value) {
    target[offset] = (value >>> 24) & 255;
    target[offset + 1] = (value >>> 16) & 255;
    target[offset + 2] = (value >>> 8) & 255;
    target[offset + 3] = value & 255;
  }
  function pngBlobWithDpi(dataUrl, dpi) {
    const base64 = dataUrl.substring(dataUrl.indexOf(",") + 1);
    const binary = atob(base64);
    const original = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) original[i] = binary.charCodeAt(i);

    const pixelsPerMeter = Math.round(dpi / 0.0254);
    const chunk = new Uint8Array(21);
    writeUint32BE(chunk, 0, 9);
    chunk[4] = 112; chunk[5] = 72; chunk[6] = 89; chunk[7] = 115;
    writeUint32BE(chunk, 8, pixelsPerMeter);
    writeUint32BE(chunk, 12, pixelsPerMeter);
    chunk[16] = 1;
    writeUint32BE(chunk, 17, crc32(chunk.subarray(4, 17)));

    const insertAt = 33;
    const output = new Uint8Array(original.length + chunk.length);
    output.set(original.subarray(0, insertAt), 0);
    output.set(chunk, insertAt);
    output.set(original.subarray(insertAt), insertAt + chunk.length);
    return new Blob([output], {type: "image/png"});
  }
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function() { URL.revokeObjectURL(url); }, 30000);
  }
  function exportBaseName() {
    try {
      return decodeURIComponent(
        (window.location.pathname.split("/").pop() || "plot")
          .replace(/\.html?$/i, "")
      );
    } catch (error) {
      return "plot";
    }
  }
  function filenameNumber(value) {
    const rounded = Math.abs(Number(value)) < 0.0005 ? 0 : Number(Number(value).toFixed(1));
    return String(rounded).replace(/-/g, "m").replace(/\./g, "p");
  }
  function exportFilename(assets, kind, extension) {
    const settings = assets.settings;
    const parts = [exportBaseName()];
    if (settings.filenameView) {
      const view = currentViewAngles();
      parts.push("elev" + filenameNumber(view.elev));
      parts.push("azim" + filenameNumber(view.azim));
      parts.push("roll" + filenameNumber(view.roll));
    }
    if (settings.filenameBackground) parts.push(backgroundTheme === "black" ? "bgBlack" : "bgWhite");
    if (settings.filenamePixels) parts.push(assets.widthPx + "x" + assets.heightPx);
    if (settings.filenameDpi) parts.push(settings.dpi + "dpi");
    parts.push(kind);
    return parts.join("_") + "." + extension;
  }
  function imageFromUrl(url) {
    return new Promise(function(resolve, reject) {
      const image = new Image();
      image.onload = function() { resolve(image); };
      image.onerror = reject;
      image.src = url;
    });
  }
  function canvasToBlob(canvas, type, quality) {
    return new Promise(function(resolve, reject) {
      canvas.toBlob(function(blob) {
        if (blob) resolve(blob);
        else reject(new Error("Canvas conversion returned no data."));
      }, type, quality);
    });
  }
  function colorToRgba(color) {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    const ctx = canvas.getContext("2d", {willReadFrequently: true});
    ctx.clearRect(0, 0, 1, 1);
    ctx.fillStyle = color || "rgba(0,0,0,0)";
    ctx.fillRect(0, 0, 1, 1);
    return Array.from(ctx.getImageData(0, 0, 1, 1).data);
  }
  function plotBackgroundRgba() {
    const full = gd._fullLayout || {};
    const scene = full.scene || {};
    return colorToRgba(
      scene.bgcolor || full.paper_bgcolor || full.plot_bgcolor || "white"
    );
  }
  function findModelBounds(canvas, background) {
    const ctx = canvas.getContext("2d", {willReadFrequently: true});
    const width = canvas.width;
    const height = canvas.height;
    const data = ctx.getImageData(0, 0, width, height).data;
    let minX = width, minY = height, maxX = -1, maxY = -1;
    const tolerance = 1;
    for (let y = 0; y < height; y++) {
      let offset = y * width * 4;
      for (let x = 0; x < width; x++, offset += 4) {
        const alpha = data[offset + 3];
        if (alpha < 2 && background[3] < 2) continue;
        const delta = Math.max(
          Math.abs(data[offset] - background[0]),
          Math.abs(data[offset + 1] - background[1]),
          Math.abs(data[offset + 2] - background[2]),
          Math.abs(alpha - background[3])
        );
        if (delta > tolerance) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < minX || maxY < minY) {
      return {x: 0, y: 0, width: width, height: height};
    }
    return {
      x: minX,
      y: minY,
      width: Math.max(1, maxX - minX + 1),
      height: Math.max(1, maxY - minY + 1)
    };
  }
  function addExportOuterOutline(canvas, background, color, widthPx) {
    const ctx = canvas.getContext("2d", {willReadFrequently: true});
    if (!ctx) throw new Error("The browser cannot read the output canvas, so the outer outline cannot be calculated.");
    const width = canvas.width;
    const height = canvas.height;
    const pixels = width * height;
    const image = ctx.getImageData(0, 0, width, height);
    const data = image.data;
    const distance = new Uint16Array(pixels);
    const infinity = 65535;
    const straight = 3;
    const diagonal = 4;
    const tolerance = 1;
    for (let index=0,offset=0;index<pixels;index++,offset+=4) {
      const alpha=data[offset+3];
      const delta=Math.max(Math.abs(data[offset]-background[0]),Math.abs(data[offset+1]-background[1]),Math.abs(data[offset+2]-background[2]),Math.abs(alpha-background[3]));
      distance[index]=delta>tolerance?0:infinity;
    }
    for (let y=0;y<height;y++) {
      const row=y*width;
      for (let x=0;x<width;x++) {
        const index=row+x;
        if (distance[index]===0) continue;
        let best=distance[index];
        if (x>0) best=Math.min(best,distance[index-1]+straight);
        if (y>0) {
          best=Math.min(best,distance[index-width]+straight);
          if (x>0) best=Math.min(best,distance[index-width-1]+diagonal);
          if (x+1<width) best=Math.min(best,distance[index-width+1]+diagonal);
        }
        distance[index]=best;
      }
    }
    for (let y=height-1;y>=0;y--) {
      const row=y*width;
      for (let x=width-1;x>=0;x--) {
        const index=row+x;
        if (distance[index]===0) continue;
        let best=distance[index];
        if (x+1<width) best=Math.min(best,distance[index+1]+straight);
        if (y+1<height) {
          best=Math.min(best,distance[index+width]+straight);
          if (x>0) best=Math.min(best,distance[index+width-1]+diagonal);
          if (x+1<width) best=Math.min(best,distance[index+width+1]+diagonal);
        }
        distance[index]=best;
      }
    }
    const outline=colorToRgba(color);
    const limit=Math.max(1,Math.round(widthPx))*straight;
    const outlineAlpha=outline[3]/255;
    for (let index=0,offset=0;index<pixels;index++,offset+=4) {
      if (distance[index]<=0||distance[index]>limit) continue;
      data[offset]=Math.round(outline[0]*outlineAlpha+data[offset]*(1-outlineAlpha));
      data[offset+1]=Math.round(outline[1]*outlineAlpha+data[offset+1]*(1-outlineAlpha));
      data[offset+2]=Math.round(outline[2]*outlineAlpha+data[offset+2]*(1-outlineAlpha));
      data[offset+3]=Math.max(data[offset+3],outline[3]);
    }
    ctx.putImageData(image,0,0);
  }
  function expandedBounds(bounds, marginPct, width, height) {
    const margin = Math.max(
      4,
      Math.min(width, height) * 0.005,
      Math.max(bounds.width, bounds.height) * marginPct / 100
    );
    const x0 = clamp(bounds.x - margin, 0, width);
    const y0 = clamp(bounds.y - margin, 0, height);
    const x1 = clamp(bounds.x + bounds.width + margin, 0, width);
    const y1 = clamp(bounds.y + bounds.height + margin, 0, height);
    return {
      x: x0,
      y: y0,
      width: Math.max(1, x1 - x0),
      height: Math.max(1, y1 - y0)
    };
  }
  function overlayPlacement(groupId, corner, crop) {
    const group = document.getElementById(groupId);
    if (!group) return null;
    const wrapperRect = wrapper.getBoundingClientRect();
    const rect = group.getBoundingClientRect();
    const current = {
      left: rect.left - wrapperRect.left,
      top: rect.top - wrapperRect.top,
      right: rect.right - wrapperRect.left,
      bottom: rect.bottom - wrapperRect.top,
      width: rect.width,
      height: rect.height
    };
    const inset = Math.max(5, Math.min(crop.width, crop.height) * 0.025);
    let targetLeft = current.left;
    let targetTop = current.top;
    if (corner === "top-left") {
      targetLeft = crop.x + inset;
      targetTop = crop.y + inset;
    } else {
      targetLeft = crop.x + crop.width - inset - current.width;
      targetTop = crop.y + crop.height - inset - current.height;
    }
    return {
      kind: corner === "top-left" ? "orientation" : "scalebar",
      group: group,
      dx: targetLeft - current.left,
      dy: targetTop - current.top
    };
  }
  function overlayPlacements(settings, crop) {
    const placements = [];
    if (settings.includeOrientation) {
      const orientation = overlayPlacement(
        "draggable-orientation", "top-left", crop
      );
      if (orientation) placements.push(orientation);
    }
    if (settings.includeScalebar) {
      const scalebar = overlayPlacement(
        "draggable-scalebar", "bottom-right", crop
      );
      if (scalebar) placements.push(scalebar);
    }
    return placements;
  }
  function placementSvgMarkup(placement) {
    const clone = placement.group.cloneNode(true);
    clone.removeAttribute("id");
    clone.querySelectorAll("[id]").forEach(function(element) {
      element.removeAttribute("id");
    });
    return (
      '<g data-export-part="' + placement.kind + '" transform="translate(' +
      placement.dx + " " + placement.dy + ')">' +
      clone.outerHTML + "</g>"
    );
  }
  function overlaySvgDocument(assets, onlyKind) {
    const scaleX = assets.widthPx / assets.crop.width;
    const scaleY = assets.heightPx / assets.crop.height;
    const tx = -assets.crop.x * scaleX;
    const ty = -assets.crop.y * scaleY;
    const parts = assets.placements
      .filter(function(item) { return !onlyKind || item.kind === onlyKind; })
      .map(placementSvgMarkup)
      .join("");
    return (
      '<svg xmlns="http://www.w3.org/2000/svg" width="' + assets.widthPx +
      '" height="' + assets.heightPx + '" viewBox="0 0 ' +
      assets.widthPx + " " + assets.heightPx + '">' +
      '<g transform="matrix(' + scaleX + " 0 0 " + scaleY + " " +
      tx + " " + ty + ')">' + parts + "</g></svg>"
    );
  }
  async function drawSvgOnCanvas(canvas, svgText) {
    if (!svgText || svgText.indexOf("data-export-part") < 0) return;
    const blob = new Blob([svgText], {type: "image/svg+xml;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    try {
      const image = await imageFromUrl(url);
      canvas.getContext("2d").drawImage(image, 0, 0);
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  let cachedWebGlMaxDimension = null;
  function webGlMaxDimension() {
    if (cachedWebGlMaxDimension !== null) {
      return cachedWebGlMaxDimension;
    }
    let detected = 8192;
    try {
      const canvas = document.createElement("canvas");
      const gl = (
        canvas.getContext("webgl2", {failIfMajorPerformanceCaveat: false}) ||
        canvas.getContext("webgl", {failIfMajorPerformanceCaveat: false}) ||
        canvas.getContext("experimental-webgl")
      );
      if (gl) {
        const textureLimit = Number(gl.getParameter(gl.MAX_TEXTURE_SIZE));
        const renderbufferLimit = Number(
          gl.getParameter(gl.MAX_RENDERBUFFER_SIZE)
        );
        const limits = [textureLimit, renderbufferLimit].filter(function(v) {
          return Number.isFinite(v) && v > 0;
        });
        if (limits.length) detected = Math.min.apply(null, limits);
        const loseContext = gl.getExtension("WEBGL_lose_context");
        if (loseContext) loseContext.loseContext();
      }
    } catch (error) {
      console.warn("Unable to query WebGL export limit.", error);
    }
    cachedWebGlMaxDimension = Math.max(2048, Math.floor(detected));
    return cachedWebGlMaxDimension;
  }
  function exceedsSafeExportLimit(width, height) {
    const safeDimension = 8192;
    const safePixels = 50000000;
    return (
      width > safeDimension ||
      height > safeDimension ||
      width * height > safePixels
    );
  }
  function validateFinalDimensions(width, height, settings) {
    const exceedsSafeLimit = exceedsSafeExportLimit(width, height);
    if (exceedsSafeLimit && !settings.allowOversize) {
      throw new Error(
        "The output requires " + width + " × " + height +
        " px, exceeding the default safety limit (8192 pixels per side and 50 million total pixels). " +
        "Reduce the long-side length in centimeters or DPI, or select 'Attempt oversized export'."
      );
    }
    if (
      settings.allowOversize &&
      (width > 32767 || height > 32767)
    ) {
      throw new Error(
        "The output requires " + width + " × " + height +
        " px, exceeding the usual browser limit of 32767 px per side for a 2D canvas."
      );
    }
    return exceedsSafeLimit;
  }
  function reliableWebGlScale(sourceWidth, sourceHeight) {
    const actualDimension = webGlMaxDimension();
    const reliableDimension = Math.min(8192, actualDimension);
    const dimensionScale = Math.min(
      reliableDimension / sourceWidth,
      reliableDimension / sourceHeight
    );
    const pixelScale = Math.sqrt(
      50000000 / Math.max(1, sourceWidth * sourceHeight)
    );
    return Math.max(0.01, Math.min(dimensionScale, pixelScale));
  }
  async function prepareExportAssets() {
    const settings = exportSettings();
    const rect = wrapper.getBoundingClientRect();
    const sourceWidth = Math.max(1, Math.round(rect.width));
    const sourceHeight = Math.max(1, Math.round(rect.height));
    const previewScale = Math.min(
      1,
      1400 / Math.max(sourceWidth, sourceHeight)
    );
    const previewWidth = Math.max(1, Math.round(sourceWidth * previewScale));
    const previewHeight = Math.max(1, Math.round(sourceHeight * previewScale));
    updateExportStatus("Detecting the current projection boundary...", false);
    const previewUrl = await Plotly.toImage(gd, {
      format: "png",
      width: previewWidth,
      height: previewHeight,
      scale: 1
    });
    const previewImage = await imageFromUrl(previewUrl);
    const previewCanvas = document.createElement("canvas");
    previewCanvas.width = previewWidth;
    previewCanvas.height = previewHeight;
    previewCanvas.getContext("2d").drawImage(previewImage, 0, 0);
    const previewBounds = findModelBounds(
      previewCanvas, plotBackgroundRgba()
    );
    const modelBounds = {
      x: previewBounds.x / previewScale,
      y: previewBounds.y / previewScale,
      width: previewBounds.width / previewScale,
      height: previewBounds.height / previewScale
    };
    const crop = expandedBounds(
      modelBounds,
      settings.marginPct,
      sourceWidth,
      sourceHeight
    );
    const pixelScale = settings.longPx / Math.max(crop.width, crop.height);
    const widthPx = Math.max(1, Math.round(crop.width * pixelScale));
    const heightPx = Math.max(1, Math.round(crop.height * pixelScale));
    const oversizedOutput = validateFinalDimensions(
      widthPx, heightPx, settings
    );
    const requestedCaptureWidth = Math.max(
      1, Math.round(sourceWidth * pixelScale)
    );
    const requestedCaptureHeight = Math.max(
      1, Math.round(sourceHeight * pixelScale)
    );
    const oversizedCapture = exceedsSafeExportLimit(
      requestedCaptureWidth, requestedCaptureHeight
    );
    if (oversizedCapture && !settings.allowOversize) {
      throw new Error(
        "The intermediate canvas required for automatic cropping is " +
        requestedCaptureWidth + " × " + requestedCaptureHeight +
        " px, exceeding the default safety limit. Increase the margin, reduce the long-side length in centimeters or DPI, " +
        "or select 'Attempt oversized export'."
      );
    }
    let renderPixelScale = pixelScale;
    let stableUpscale = false;
    if (oversizedCapture && settings.allowOversize) {
      renderPixelScale = Math.min(
        pixelScale,
        reliableWebGlScale(sourceWidth, sourceHeight)
      );
      stableUpscale = renderPixelScale < pixelScale * 0.999;
    }
    const captureWidth = Math.max(
      1, Math.round(sourceWidth * renderPixelScale)
    );
    const captureHeight = Math.max(
      1, Math.round(sourceHeight * renderPixelScale)
    );
    updateExportStatus(
      (
        "Rendering " + widthPx + " × " + heightPx + " px..." +
        (
          stableUpscale ?
          " WebGL will first render the complete model at " + captureWidth + " × " +
          captureHeight + " px, then scale it proportionally to the output dimensions." :
          oversizedOutput ?
          " The browser may become temporarily unresponsive in oversized mode." :
          ""
        )
      ),
      oversizedOutput || stableUpscale
    );
    const fullUrl = await Plotly.toImage(gd, {
      format: "png",
      width: captureWidth,
      height: captureHeight,
      scale: 1
    });
    const fullImage = await imageFromUrl(fullUrl);
    const actualCaptureWidth = (
      fullImage.naturalWidth || fullImage.width || captureWidth
    );
    const actualCaptureHeight = (
      fullImage.naturalHeight || fullImage.height || captureHeight
    );
    const requestedAspect = captureWidth / captureHeight;
    const actualAspect = actualCaptureWidth / actualCaptureHeight;
    if (
      !Number.isFinite(actualAspect) ||
      Math.abs(actualAspect / requestedAspect - 1) > 0.002
    ) {
      throw new Error(
        "The browser returned " + actualCaptureWidth + " × " +
        actualCaptureHeight + " px, whose aspect ratio does not match the requested canvas. " +
        "Export was stopped to avoid creating a stretched or clipped image."
      );
    }
    if (
      actualCaptureWidth < captureWidth * 0.995 ||
      actualCaptureHeight < captureHeight * 0.995
    ) {
      stableUpscale = true;
    }
    const modelCanvas = document.createElement("canvas");
    modelCanvas.width = widthPx;
    modelCanvas.height = heightPx;
    const modelContext = modelCanvas.getContext("2d");
    if (!modelContext) {
      throw new Error(
        "The browser cannot create a " + widthPx + " × " + heightPx +
        " 2D output canvas. Reduce the long-side length in centimeters or DPI."
      );
    }
    modelContext.imageSmoothingEnabled = true;
    modelContext.imageSmoothingQuality = "high";
    const sx = actualCaptureWidth / sourceWidth;
    const sy = actualCaptureHeight / sourceHeight;
    modelContext.drawImage(
      fullImage,
      crop.x * sx,
      crop.y * sy,
      crop.width * sx,
      crop.height * sy,
      0,
      0,
      widthPx,
      heightPx
    );
    if (settings.includeOutline) {
      updateExportStatus("Calculating the export outline...", false);
      addExportOuterOutline(
        modelCanvas,
        plotBackgroundRgba(),
        settings.outlineColor,
        settings.outlineWidth
      );
    }
    const widthCm = (
      widthPx >= heightPx ?
      settings.longCm :
      settings.longCm * widthPx / heightPx
    );
    const heightCm = (
      heightPx >= widthPx ?
      settings.longCm :
      settings.longCm * heightPx / widthPx
    );
    return {
      settings: settings,
      crop: crop,
      widthPx: widthPx,
      heightPx: heightPx,
      widthCm: widthCm,
      heightCm: heightCm,
      modelCanvas: modelCanvas,
      placements: overlayPlacements(settings, crop),
      background: plotBackgroundRgba(),
      renderWidth: actualCaptureWidth,
      renderHeight: actualCaptureHeight,
      stableUpscale: stableUpscale
    };
  }
  function setExportButtonsDisabled(disabled) {
    ["codex-export-png", "codex-export-pdf", "codex-export-svg"]
      .forEach(function(id) {
        const button = byId(id);
        if (button) button.disabled = disabled;
      });
  }
  function waitForOverlayRedraw() {
    return new Promise(function(resolve) {
      requestAnimationFrame(function() {
        requestAnimationFrame(resolve);
      });
    });
  }
  async function withPreparedExport(task) {
    setExportButtonsDisabled(true);
    const requested = exportSettings();
    window.codexOverlayVisibility = window.codexOverlayVisibility || {};
    const previousOverlayVisibility = {
      orientation: window.codexOverlayVisibility.orientation,
      scalebar: window.codexOverlayVisibility.scalebar
    };
    try {
      if (requested.includeOrientation) {
        window.codexOverlayVisibility.orientation = true;
      }
      if (requested.includeScalebar) {
        window.codexOverlayVisibility.scalebar = true;
      }
      window.dispatchEvent(new Event("codex-overlay-visibility"));
      await waitForOverlayRedraw();
      const assets = await prepareExportAssets();
      await task(assets);
      updateExportStatus(
        "Complete: " + assets.widthPx + " × " + assets.heightPx +
        " px，" + assets.settings.dpi + " DPI。" +
        (
          assets.stableUpscale ?
          " The model was scaled proportionally from a complete " + assets.renderWidth + " × " +
          assets.renderHeight + " px WebGL image, preserving its extent and proportions but adding no true detail beyond that render resolution." :
          ""
        ),
        false
      );
    } catch (error) {
      console.error(error);
      updateExportStatus(
        "Export failed: " + (error && error.message ? error.message : String(error)),
        true
      );
    } finally {
      window.codexOverlayVisibility.orientation =
        previousOverlayVisibility.orientation;
      window.codexOverlayVisibility.scalebar =
        previousOverlayVisibility.scalebar;
      window.dispatchEvent(new Event("codex-overlay-visibility"));
      setExportButtonsDisabled(false);
    }
  }
  async function exportCompositePng() {
    return withPreparedExport(async function(assets) {
      const canvas = document.createElement("canvas");
      canvas.width = assets.widthPx;
      canvas.height = assets.heightPx;
      canvas.getContext("2d").drawImage(assets.modelCanvas, 0, 0);
      await drawSvgOnCanvas(canvas, overlaySvgDocument(assets));
      const dataUrl = canvas.toDataURL("image/png");
      const blob = pngBlobWithDpi(dataUrl, assets.settings.dpi);
      downloadBlob(
        blob,
        exportFilename(assets, "composite", "png")
      );
    });
  }
  function editableSvgText(assets) {
    const modelUrl = assets.modelCanvas.toDataURL("image/png");
    const scaleX = assets.widthPx / assets.crop.width;
    const scaleY = assets.heightPx / assets.crop.height;
    const tx = -assets.crop.x * scaleX;
    const ty = -assets.crop.y * scaleY;
    function layer(kind, label) {
      const content = assets.placements
        .filter(function(item) { return item.kind === kind; })
        .map(placementSvgMarkup)
        .join("");
      return (
        '<g id="layer-' + kind +
        '" inkscape:groupmode="layer" inkscape:label="' + label + '">' +
        '<g transform="matrix(' + scaleX + " 0 0 " + scaleY + " " +
        tx + " " + ty + ')">' + content + "</g></g>"
      );
    }
    return (
      '<?xml version="1.0" encoding="UTF-8"?>\n' +
      '<svg xmlns="http://www.w3.org/2000/svg" ' +
      'xmlns:xlink="http://www.w3.org/1999/xlink" ' +
      'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ' +
      'width="' + assets.widthCm + 'cm" height="' + assets.heightCm +
      'cm" viewBox="0 0 ' + assets.widthPx + " " + assets.heightPx + '">' +
      '<metadata>Raster model at ' + assets.settings.dpi +
      ' DPI; orientation and scalebar remain vector layers.</metadata>' +
      '<g id="layer-model" inkscape:groupmode="layer" inkscape:label="Model">' +
      '<image id="model-image" x="0" y="0" width="' + assets.widthPx +
      '" height="' + assets.heightPx + '" xlink:href="' + modelUrl + '"/>' +
      "</g>" +
      layer("orientation", "Orientation") +
      layer("scalebar", "Scalebar") +
      "</svg>"
    );
  }
  async function exportEditableSvg() {
    return withPreparedExport(async function(assets) {
      const blob = new Blob(
        [editableSvgText(assets)],
        {type: "image/svg+xml;charset=utf-8"}
      );
      downloadBlob(blob, exportFilename(assets, "editable", "svg"));
    });
  }
  function pdfNum(value) {
    return Number(value.toFixed(4)).toString();
  }
  function pdfText(value) {
    let output = "";
    String(value).split("").forEach(function(char) {
      const code = char.charCodeAt(0);
      if (char === "\\" || char === "(" || char === ")") {
        output += "\\" + char;
      } else if (code === 181) {
        output += "\\265";
      } else if (code >= 32 && code <= 126) {
        output += char;
      } else {
        output += "?";
      }
    });
    return output;
  }
  function blendedPdfColor(value, background) {
    const rgba = colorToRgba(value || "black");
    const alpha = rgba[3] / 255;
    return [
      (rgba[0] * alpha + background[0] * (1 - alpha)) / 255,
      (rgba[1] * alpha + background[1] * (1 - alpha)) / 255,
      (rgba[2] * alpha + background[2] * (1 - alpha)) / 255,
      alpha
    ];
  }
  function transformSvgPoint(matrix, x, y, dx, dy) {
    return {
      x: matrix.a * x + matrix.c * y + matrix.e + dx,
      y: matrix.b * x + matrix.d * y + matrix.f + dy
    };
  }
  function svgMatrixScale(matrix) {
    return (
      Math.hypot(matrix.a, matrix.b) +
      Math.hypot(matrix.c, matrix.d)
    ) / 2;
  }
  function overlayPdfCommands(assets, kind, pageWidth, pageHeight) {
    const placement = assets.placements.find(function(item) {
      return item.kind === kind;
    });
    if (!placement) return "";
    const scaleX = pageWidth / assets.crop.width;
    const scaleY = pageHeight / assets.crop.height;
    function map(point) {
      return {
        x: (point.x - assets.crop.x) * scaleX,
        y: pageHeight - (point.y - assets.crop.y) * scaleY
      };
    }
    function fillStroke(element, allowFill, allowStroke) {
      const fillValue = element.getAttribute("fill");
      const strokeValue = element.getAttribute("stroke");
      const fill = (
        allowFill && fillValue && fillValue !== "none" ?
        blendedPdfColor(fillValue, assets.background) : null
      );
      const stroke = (
        allowStroke && strokeValue && strokeValue !== "none" ?
        blendedPdfColor(strokeValue, assets.background) : null
      );
      return {fill: fill, stroke: stroke};
    }
    function colorCommand(color, stroke) {
      if (!color || color[3] <= 0) return "";
      return (
        pdfNum(color[0]) + " " + pdfNum(color[1]) + " " +
        pdfNum(color[2]) + (stroke ? " RG\n" : " rg\n")
      );
    }
    let commands = "";
    placement.group.querySelectorAll("line,polygon,circle,rect,text")
      .forEach(function(element) {
        const matrix = element.getCTM();
        if (!matrix) return;
        const matrixScale = svgMatrixScale(matrix);
        const tag = element.tagName.toLowerCase();
        if (tag === "line") {
          const style = fillStroke(element, false, true);
          if (!style.stroke || style.stroke[3] <= 0) return;
          const p1 = map(transformSvgPoint(
            matrix,
            parseFloat(element.getAttribute("x1") || "0"),
            parseFloat(element.getAttribute("y1") || "0"),
            placement.dx,
            placement.dy
          ));
          const p2 = map(transformSvgPoint(
            matrix,
            parseFloat(element.getAttribute("x2") || "0"),
            parseFloat(element.getAttribute("y2") || "0"),
            placement.dx,
            placement.dy
          ));
          const width = (
            parseFloat(element.getAttribute("stroke-width") || "1") *
            matrixScale * (scaleX + scaleY) / 2
          );
          commands += colorCommand(style.stroke, true);
          commands += pdfNum(width) + " w\n";
          commands += (
            pdfNum(p1.x) + " " + pdfNum(p1.y) + " m " +
            pdfNum(p2.x) + " " + pdfNum(p2.y) + " l S\n"
          );
        } else if (tag === "polygon") {
          const style = fillStroke(element, true, true);
          const raw = (element.getAttribute("points") || "")
            .trim().split(/\s+/).map(function(pair) {
              const values = pair.split(",").map(Number);
              return transformSvgPoint(
                matrix, values[0], values[1], placement.dx, placement.dy
              );
            }).filter(function(point) {
              return Number.isFinite(point.x) && Number.isFinite(point.y);
            });
          if (!raw.length) return;
          const points = raw.map(map);
          commands += colorCommand(style.fill, false);
          commands += colorCommand(style.stroke, true);
          commands += pdfNum(points[0].x) + " " + pdfNum(points[0].y) + " m\n";
          points.slice(1).forEach(function(point) {
            commands += pdfNum(point.x) + " " + pdfNum(point.y) + " l\n";
          });
          commands += "h " + (
            style.fill && style.stroke ? "B" :
            style.fill ? "f" : "S"
          ) + "\n";
        } else if (tag === "circle") {
          const style = fillStroke(element, true, true);
          if (
            (!style.fill || style.fill[3] <= 0) &&
            (!style.stroke || style.stroke[3] <= 0)
          ) return;
          const center = map(transformSvgPoint(
            matrix,
            parseFloat(element.getAttribute("cx") || "0"),
            parseFloat(element.getAttribute("cy") || "0"),
            placement.dx,
            placement.dy
          ));
          const radius = (
            parseFloat(element.getAttribute("r") || "0") *
            matrixScale * (scaleX + scaleY) / 2
          );
          const k = radius * 0.5522847498;
          commands += colorCommand(style.fill, false);
          commands += colorCommand(style.stroke, true);
          if (style.stroke) {
            const width = (
              parseFloat(element.getAttribute("stroke-width") || "1") *
              matrixScale * (scaleX + scaleY) / 2
            );
            commands += pdfNum(width) + " w\n";
          }
          commands += pdfNum(center.x + radius) + " " + pdfNum(center.y) + " m\n";
          commands += (
            pdfNum(center.x + radius) + " " + pdfNum(center.y + k) + " " +
            pdfNum(center.x + k) + " " + pdfNum(center.y + radius) + " " +
            pdfNum(center.x) + " " + pdfNum(center.y + radius) + " c\n"
          );
          commands += (
            pdfNum(center.x - k) + " " + pdfNum(center.y + radius) + " " +
            pdfNum(center.x - radius) + " " + pdfNum(center.y + k) + " " +
            pdfNum(center.x - radius) + " " + pdfNum(center.y) + " c\n"
          );
          commands += (
            pdfNum(center.x - radius) + " " + pdfNum(center.y - k) + " " +
            pdfNum(center.x - k) + " " + pdfNum(center.y - radius) + " " +
            pdfNum(center.x) + " " + pdfNum(center.y - radius) + " c\n"
          );
          commands += (
            pdfNum(center.x + k) + " " + pdfNum(center.y - radius) + " " +
            pdfNum(center.x + radius) + " " + pdfNum(center.y - k) + " " +
            pdfNum(center.x + radius) + " " + pdfNum(center.y) + " c\nh " +
            (style.fill && style.stroke ? "B" : style.fill ? "f" : "S") + "\n"
          );
        } else if (tag === "rect") {
          const style = fillStroke(element, true, true);
          if (
            (!style.fill || style.fill[3] <= 0) &&
            (!style.stroke || style.stroke[3] <= 0)
          ) return;
          const p0 = map(transformSvgPoint(
            matrix,
            parseFloat(element.getAttribute("x") || "0"),
            parseFloat(element.getAttribute("y") || "0"),
            placement.dx,
            placement.dy
          ));
          const p1 = map(transformSvgPoint(
            matrix,
            parseFloat(element.getAttribute("x") || "0") +
              parseFloat(element.getAttribute("width") || "0"),
            parseFloat(element.getAttribute("y") || "0") +
              parseFloat(element.getAttribute("height") || "0"),
            placement.dx,
            placement.dy
          ));
          commands += colorCommand(style.fill, false);
          commands += colorCommand(style.stroke, true);
          commands += (
            pdfNum(Math.min(p0.x, p1.x)) + " " +
            pdfNum(Math.min(p0.y, p1.y)) + " " +
            pdfNum(Math.abs(p1.x - p0.x)) + " " +
            pdfNum(Math.abs(p1.y - p0.y)) + " re " +
            (style.fill && style.stroke ? "B" : style.fill ? "f" : "S") + "\n"
          );
        } else if (tag === "text") {
          const style = fillStroke(element, true, false);
          if (!style.fill || style.fill[3] <= 0) return;
          const point = map(transformSvgPoint(
            matrix,
            parseFloat(element.getAttribute("x") || "0"),
            parseFloat(element.getAttribute("y") || "0"),
            placement.dx,
            placement.dy
          ));
          const value = element.textContent || "";
          const fontSize = (
            parseFloat(element.getAttribute("font-size") || "12") *
            matrixScale * (scaleX + scaleY) / 2
          );
          const bold = element.getAttribute("font-weight") === "bold";
          const centered = element.getAttribute("text-anchor") === "middle";
          const textWidth = value.length * fontSize * (bold ? 0.60 : 0.54);
          const x = point.x - (centered ? textWidth / 2 : 0);
          const y = point.y - fontSize * 0.34;
          commands += colorCommand(style.fill, false);
          commands += (
            "BT /" + (bold ? "F2" : "F1") + " " + pdfNum(fontSize) +
            " Tf " + pdfNum(x) + " " + pdfNum(y) +
            " Td (" + pdfText(value) + ") Tj ET\n"
          );
        }
      });
    return commands;
  }
  function asciiBytes(text) {
    return new TextEncoder().encode(text);
  }
  function concatBytes(parts) {
    const length = parts.reduce(function(total, part) {
      return total + part.length;
    }, 0);
    const output = new Uint8Array(length);
    let offset = 0;
    parts.forEach(function(part) {
      output.set(part, offset);
      offset += part.length;
    });
    return output;
  }
  async function pdfImageData(canvas) {
    if (typeof CompressionStream !== "undefined") {
      const rgba = canvas.getContext("2d", {willReadFrequently: true})
        .getImageData(0, 0, canvas.width, canvas.height).data;
      const rgb = new Uint8Array(canvas.width * canvas.height * 3);
      for (let source = 0, target = 0; source < rgba.length; source += 4) {
        const alpha = rgba[source + 3] / 255;
        rgb[target++] = Math.round(
          rgba[source] * alpha + 255 * (1 - alpha)
        );
        rgb[target++] = Math.round(
          rgba[source + 1] * alpha + 255 * (1 - alpha)
        );
        rgb[target++] = Math.round(
          rgba[source + 2] * alpha + 255 * (1 - alpha)
        );
      }
      const compressed = await new Response(
        new Blob([rgb]).stream().pipeThrough(new CompressionStream("deflate"))
      ).arrayBuffer();
      return {
        bytes: new Uint8Array(compressed),
        filter: "/Filter /FlateDecode"
      };
    }
    const jpeg = await canvasToBlob(canvas, "image/jpeg", 0.98);
    return {
      bytes: new Uint8Array(await jpeg.arrayBuffer()),
      filter: "/Filter /DCTDecode"
    };
  }
  async function layeredPdfBlob(assets) {
    const longPoints = assets.settings.longCm / 2.54 * 72;
    const pageWidth = (
      assets.widthPx >= assets.heightPx ?
      longPoints :
      longPoints * assets.widthPx / assets.heightPx
    );
    const pageHeight = (
      assets.heightPx >= assets.widthPx ?
      longPoints :
      longPoints * assets.heightPx / assets.widthPx
    );
    const orientationCommands = overlayPdfCommands(
      assets, "orientation", pageWidth, pageHeight
    );
    const scalebarCommands = overlayPdfCommands(
      assets, "scalebar", pageWidth, pageHeight
    );
    const content = (
      "/OC /LModel BDC\nq\n" +
      pdfNum(pageWidth) + " 0 0 " + pdfNum(pageHeight) +
      " 0 0 cm\n/Model Do\nQ\nEMC\n" +
      "/OC /LOrientation BDC\n" + orientationCommands + "EMC\n" +
      "/OC /LScalebar BDC\n" + scalebarCommands + "EMC\n"
    );
    const image = await pdfImageData(assets.modelCanvas);
    const objects = new Array(11);
    objects[1] = asciiBytes(
      "<< /Type /Catalog /Pages 2 0 R " +
      "/OCProperties << /OCGs [7 0 R 8 0 R 9 0 R] " +
      "/D << /Order [7 0 R 8 0 R 9 0 R] /ON [7 0 R 8 0 R 9 0 R] >> >> >>"
    );
    objects[2] = asciiBytes("<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    objects[3] = asciiBytes(
      "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " +
      pdfNum(pageWidth) + " " + pdfNum(pageHeight) + "] " +
      "/Resources << /XObject << /Model 5 0 R >> " +
      "/Font << /F1 6 0 R /F2 10 0 R >> " +
      "/Properties << /LModel 7 0 R /LOrientation 8 0 R /LScalebar 9 0 R >> >> " +
      "/Contents 4 0 R >>"
    );
    const contentBytes = asciiBytes(content);
    objects[4] = concatBytes([
      asciiBytes("<< /Length " + contentBytes.length + " >>\nstream\n"),
      contentBytes,
      asciiBytes("endstream")
    ]);
    objects[5] = concatBytes([
      asciiBytes(
        "<< /Type /XObject /Subtype /Image /Width " + assets.widthPx +
        " /Height " + assets.heightPx +
        " /ColorSpace /DeviceRGB /BitsPerComponent 8 " +
        image.filter + " /Length " + image.bytes.length + " >>\nstream\n"
      ),
      image.bytes,
      asciiBytes("\nendstream")
    ]);
    objects[6] = asciiBytes(
      "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    );
    objects[7] = asciiBytes("<< /Type /OCG /Name (Model) >>");
    objects[8] = asciiBytes("<< /Type /OCG /Name (Orientation) >>");
    objects[9] = asciiBytes("<< /Type /OCG /Name (Scalebar) >>");
    objects[10] = asciiBytes(
      "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    );
    const parts = [asciiBytes("%PDF-1.7\n%1234\n")];
    const offsets = new Array(objects.length).fill(0);
    let offset = parts[0].length;
    for (let index = 1; index < objects.length; index++) {
      offsets[index] = offset;
      const objectBytes = concatBytes([
        asciiBytes(index + " 0 obj\n"),
        objects[index],
        asciiBytes("\nendobj\n")
      ]);
      parts.push(objectBytes);
      offset += objectBytes.length;
    }
    const xrefOffset = offset;
    let xref = "xref\n0 " + objects.length + "\n";
    xref += "0000000000 65535 f \n";
    for (let index = 1; index < objects.length; index++) {
      xref += String(offsets[index]).padStart(10, "0") + " 00000 n \n";
    }
    xref += (
      "trailer\n<< /Size " + objects.length + " /Root 1 0 R >>\n" +
      "startxref\n" + xrefOffset + "\n%%EOF\n"
    );
    parts.push(asciiBytes(xref));
    return new Blob(parts, {type: "application/pdf"});
  }
  async function exportLayeredPdf() {
    return withPreparedExport(async function(assets) {
      const blob = await layeredPdfBlob(assets);
      downloadBlob(blob, exportFilename(assets, "layered", "pdf"));
    });
  }
  function bindControlPair(rangeId, numberId, minimum, maximum, fallback, onCommit) {
    const range = byId(rangeId);
    const number = byId(numberId);
    range.addEventListener("input", function() {
      number.value = range.value;
    });
    number.addEventListener("input", function() {
      const value = parseFloat(number.value);
      range.value = String(clamp(Number.isFinite(value) ? value : fallback, minimum, maximum));
    });
    range.addEventListener("change", function() {
      if (onCommit) onCommit();
    });
    number.addEventListener("change", function() {
      if (onCommit) onCommit();
    });
  }
  bindControlPair("codex-export-outline-width-range", "codex-export-outline-width", 1, 40, 4, updateExportStatus);
  [
    ["codex-light-azimuth-range","codex-light-azimuth",-180,180,90],
    ["codex-light-elevation-range","codex-light-elevation",-89,89,45],
    ["codex-light-ambient-range","codex-light-ambient",0,1,0.65],
    ["codex-light-diffuse-range","codex-light-diffuse",0,1,0.75],
    ["codex-light-specular-range","codex-light-specular",0,1,0.50],
    ["codex-light-roughness-range","codex-light-roughness",0,1,0.90]
  ].forEach(function(pair) {
    bindControlPair(pair[0],pair[1],pair[2],pair[3],pair[4],function() {
      byId("codex-lighting-enabled").checked = true;
      applySurfaceLighting();
    });
  });
  byId("codex-lighting-enabled").addEventListener("change", applySurfaceLighting);
  byId("codex-restore-lighting").addEventListener("click", restoreOriginalSurfaceLighting);
  byId("codex-transparency-sort").addEventListener("change", function() {
    if (this.checked) scheduleTransparencySort(0);
    else restoreOriginalTransparencyOrder();
  });
  byId("codex-transparency-sort-now").addEventListener("click", function() {
    byId("codex-transparency-sort").checked = true;
    scheduleTransparencySort(0);
  });
  const outlineColorInput = byId("codex-export-outline-color");
  outlineColorInput.addEventListener("input", function() {
    outlineColorInput.dataset.userEdited = "true";
  });
  [
    "codex-export-long-cm",
    "codex-export-dpi",
    "codex-export-margin-pct",
    "codex-export-orientation",
    "codex-export-scalebar",
    "codex-export-outline",
    "codex-export-outline-color",
    "codex-export-outline-width-range",
    "codex-export-outline-width",
    "codex-export-allow-oversize",
    "codex-filename-view",
    "codex-filename-background",
    "codex-filename-pixels",
    "codex-filename-dpi"
  ].forEach(function(id) {
    const input = byId(id);
    if (input) input.addEventListener("input", function() {
      updateExportStatus();
    });
    if (input) input.addEventListener("change", function() {
      updateExportStatus();
    });
  });
  byId("codex-export-png").addEventListener("click", exportCompositePng);
  byId("codex-export-pdf").addEventListener("click", exportLayeredPdf);
  byId("codex-export-svg").addEventListener("click", exportEditableSvg);
  updateExportStatus();
  function norm(v) { return Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }
  function normalize(v) {
    const n = norm(v);
    return n < 1e-12 ? [0,0,0] : [v[0]/n, v[1]/n, v[2]/n];
  }
  function dot(a,b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
  function cross(a,b) {
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]];
  }
  function projectToPlane(vec, normal) {
    const k = dot(vec, normal);
    return normalize([vec[0]-normal[0]*k, vec[1]-normal[1]*k, vec[2]-normal[2]*k]);
  }
  function rotateAroundAxis(vec, axis, rad) {
    const c = Math.cos(rad), s = Math.sin(rad);
    const kv = dot(axis, vec);
    const cr = cross(axis, vec);
    return [
      vec[0]*c + cr[0]*s + axis[0]*kv*(1-c),
      vec[1]*c + cr[1]*s + axis[1]*kv*(1-c),
      vec[2]*c + cr[2]*s + axis[2]*kv*(1-c)
    ];
  }
  function getCurrentCamera() {
    const fl = gd._fullLayout || {};
    const scene = fl.scene || {};
    const cam = scene.camera || {};
    const internal = scene._scene && scene._scene.camera;
    const ie = internal && internal.eye;
    const iu = internal && internal.up;
    const ic = internal && internal.center;
    const e = (ie && ie.length >= 3) ? {x:ie[0], y:ie[1], z:ie[2]} : (cam.eye || {x:1.25,y:1.25,z:1.25});
    const u = (iu && iu.length >= 3) ? {x:iu[0], y:iu[1], z:iu[2]} : (cam.up || {x:0,y:1,z:0});
    const c = (ic && ic.length >= 3) ? {x:ic[0], y:ic[1], z:ic[2]} : (cam.center || {x:0,y:0,z:0});
    return {eye:e, up:u, center:c};
  }
  function getCameraBasis() {
    const cam = getCurrentCamera();
    const eye = normalize([cam.eye.x - cam.center.x, cam.eye.y - cam.center.y, cam.eye.z - cam.center.z]);
    let up = normalize([cam.up.x, cam.up.y, cam.up.z]);
    let right = normalize(cross(up, eye));
    if (norm(right) < 1e-12) right = [1, 0, 0];
    let screenUp = normalize(cross(eye, right));
    if (norm(screenUp) < 1e-12) screenUp = [0, 1, 0];
    return {eye: eye.slice(), up: up.slice(), right: right.slice(), screenUp: screenUp.slice()};
  }
  function cloneManualOrientationRotation() {
    const m = window.codexOrientationManualRotation || [[1,0,0], [0,1,0], [0,0,1]];
    return [m[0].slice(), m[1].slice(), m[2].slice()];
  }
  function rebaseManualOrientationToCurrentCamera() {
    const oldBasis = window.codexOrientationLockedBasis;
    if (!oldBasis) return;
    const newBasis = getCameraBasis();
    const oldM = cloneManualOrientationRotation();
    window.codexOrientationManualRotation = oldM.map(function(v) {
      const cRight = dot(v, oldBasis.right);
      const cUp = dot(v, oldBasis.screenUp);
      const cEye = dot(v, oldBasis.eye);
      return [
        cRight * newBasis.right[0] + cUp * newBasis.screenUp[0] + cEye * newBasis.eye[0],
        cRight * newBasis.right[1] + cUp * newBasis.screenUp[1] + cEye * newBasis.eye[1],
        cRight * newBasis.right[2] + cUp * newBasis.screenUp[2] + cEye * newBasis.eye[2]
      ];
    });
  }
  function currentViewAngles() {
    const cam = getCurrentCamera();
    const v = normalize([cam.center.x - cam.eye.x, cam.center.y - cam.eye.y, cam.center.z - cam.eye.z]);
    const elev = Math.asin(clamp(v[2], -1, 1)) * 180 / Math.PI;
    const azim = Math.atan2(v[0], v[1]) * 180 / Math.PI;
    const candidates = [[0,0,1], [0,1,0], [1,0,0]];
    let refUp = [0,0,0];
    for (let i = 0; i < candidates.length; i++) {
      refUp = projectToPlane(candidates[i], v);
      if (norm(refUp) >= 1e-12) break;
    }
    let camUp = projectToPlane([cam.up.x, cam.up.y, cam.up.z], v);
    if (norm(camUp) < 1e-12) camUp = refUp;
    const roll = Math.atan2(dot(cross(refUp, camUp), v), dot(refUp, camUp)) * 180 / Math.PI;
    return {elev:elev, azim:azim, roll:roll};
  }
  function updateViewDisplay(syncInputs) {
    const a = currentViewAngles();
    const text = "viewpoint = (" + a.elev.toFixed(1) + ", " + a.azim.toFixed(1) + ", " + a.roll.toFixed(1) + ")";
    byId("codex-view-current").textContent = text;
    if (viewpointPanel) viewpointPanel.textContent = text;
    if (syncInputs) {
      byId("codex-view-elev").value = a.elev.toFixed(1);
      byId("codex-view-azim").value = a.azim.toFixed(1);
      byId("codex-view-roll").value = a.roll.toFixed(1);
    }
  }
  function updateAngleLocks() {
    ["elev","azim","roll"].forEach(function(k) {
      byId("codex-view-" + k).disabled = byId("codex-lock-" + k).checked;
    });
  }
  function applyView() {
    const current = currentViewAngles();
    const elevDeg = byId("codex-lock-elev").checked ? current.elev : num("codex-view-elev", current.elev);
    const azimDeg = byId("codex-lock-azim").checked ? current.azim : num("codex-view-azim", current.azim);
    const rollDeg = byId("codex-lock-roll").checked ? current.roll : num("codex-view-roll", current.roll);
    const cam = getCurrentCamera();
    const dir = [cam.eye.x - cam.center.x, cam.eye.y - cam.center.y, cam.eye.z - cam.center.z];
    const radius = Math.max(1e-12, norm(dir));
    const elev = clamp(elevDeg, -89.999, 89.999) * Math.PI / 180;
    const azim = azimDeg * Math.PI / 180;
    const v = normalize([Math.cos(elev) * Math.sin(azim), Math.cos(elev) * Math.cos(azim), Math.sin(elev)]);
    const eye = {
      x: cam.center.x - v[0] * radius,
      y: cam.center.y - v[1] * radius,
      z: cam.center.z - v[2] * radius
    };
    const candidates = [[0,0,1], [0,1,0], [1,0,0]];
    let refUp = [0,0,0];
    for (let i = 0; i < candidates.length; i++) {
      refUp = projectToPlane(candidates[i], v);
      if (norm(refUp) >= 1e-12) break;
    }
    const upVec = normalize(rotateAroundAxis(refUp, v, rollDeg * Math.PI / 180));
    const update = {
      "scene.camera.eye": eye,
      "scene.camera.up": {x: upVec[0], y: upVec[1], z: upVec[2]},
      "scene.camera.center": cam.center
    };
    const p = Plotly.relayout(gd, update);
    const done = function() { updateViewDisplay(true); };
    if (p && p.then) p.then(done); else done();
  }
  function rotateViewAroundOrientationAxis(sign) {
    if (!window.Plotly || !gd) return;
    const axisSelect = byId("codex-axis-rotate-axis");
    const stepInput = byId("codex-axis-rotate-step");
    const anatKey = axisSelect ? axisSelect.value : "RL";
    const stepDegRaw = parseFloat(stepInput && stepInput.value);
    const stepDeg = Number.isFinite(stepDegRaw) ? stepDegRaw : 5;
    const axisGetter = window.codexGetOrientationAxisVector;
    if (typeof axisGetter !== "function") return;

    const axis = normalize(axisGetter(anatKey));
    if (norm(axis) < 1e-12) return;

    const cam = getCurrentCamera();
    const dir = [
      cam.eye.x - cam.center.x,
      cam.eye.y - cam.center.y,
      cam.eye.z - cam.center.z
    ];
    const up = [cam.up.x, cam.up.y, cam.up.z];
    const angleRad = (Number(sign) || 1) * stepDeg * Math.PI / 180;
    const dirNew = rotateAroundAxis(dir, axis, angleRad);
    const upNew = normalize(rotateAroundAxis(up, axis, angleRad));
    const eye = {
      x: cam.center.x + dirNew[0],
      y: cam.center.y + dirNew[1],
      z: cam.center.z + dirNew[2]
    };
    const update = {
      "scene.camera.eye": eye,
      "scene.camera.up": {x: upNew[0], y: upNew[1], z: upNew[2]},
      "scene.camera.center": cam.center
    };
    const p = Plotly.relayout(gd, update);
    const done = function() {
      updateViewDisplay(true);
      window.dispatchEvent(new Event("codex-overlay-visibility"));
    };
    if (p && p.then) p.then(done); else done();
  }
  function colorToHex(value, fallback) {
    if (!value) return fallback;
    const s = String(value).trim();
    if (/^#[0-9a-f]{6}$/i.test(s)) return s;
    if (/^#[0-9a-f]{3}$/i.test(s)) return "#" + s[1]+s[1]+s[2]+s[2]+s[3]+s[3];
    const m = s.match(/rgba?\(([^)]+)\)/i);
    if (m) {
      const parts = m[1].split(",").map(function(v) { return parseFloat(v); });
      if (parts.length >= 3 && parts.slice(0,3).every(Number.isFinite)) {
        return "#" + parts.slice(0,3).map(function(v) {
          return clamp(Math.round(v), 0, 255).toString(16).padStart(2, "0");
        }).join("");
      }
    }
    return fallback;
  }
  function makeTraceRow(kind, idx, label, color, value, min, max, step, visibleInitial, applyFn) {
    const row = document.createElement("div");
    row.className = "codex-trace-row";
    const name = document.createElement("div");
    name.className = "codex-trace-name";
    name.textContent = label || (kind + " " + idx);
    name.title = name.textContent;
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = color;
    let committedColor = colorInput.value;
    const range = document.createElement("input");
    range.type = "range";
    range.min = min; range.max = max; range.step = step; range.value = value;
    const number = document.createElement("input");
    number.type = "number";
    number.min = min; number.max = max; number.step = step; number.value = value;
    const visible = document.createElement("input");
    visible.type = "checkbox";
    visible.checked = visibleInitial !== false && visibleInitial !== "legendonly";
    const apply = function() {
      number.value = range.value;
      applyFn(idx, committedColor, parseFloat(range.value), visible.checked);
    };
    range.addEventListener("input", function() { number.value = range.value; });
    range.addEventListener("change", apply);
    number.addEventListener("input", function() { range.value = number.value; });
    number.addEventListener("keydown", function(event) {
      if (event.key === "Enter") {
        event.preventDefault();
        range.value = number.value;
        apply();
      }
    });
    number.addEventListener("blur", function() {
      range.value = number.value;
      apply();
    });
    colorInput.addEventListener("change", function() {
      committedColor = colorInput.value;
      apply();
    });
    visible.addEventListener("change", apply);
    row.appendChild(name);
    row.appendChild(colorInput);
    row.appendChild(range);
    row.appendChild(number);
    row.appendChild(visible);
    return row;
  }
  function makeNeuriteGroupControls(groupName, items) {
    const card = document.createElement("div");
    card.style.cssText = "margin:8px 0;padding:7px;border:1px solid rgba(30,40,60,.18);border-radius:6px;background:rgba(248,250,255,.90)";
    const header = document.createElement("div");
    header.style.cssText = "display:grid;grid-template-columns:minmax(100px,1fr) 92px;gap:7px;align-items:center;margin-bottom:5px";
    const title = document.createElement("div");
    title.textContent = groupName;
    title.title = groupName;
    title.style.cssText = "font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    const mode = document.createElement("select");
    mode.innerHTML = '<option value="separate">Separate controls</option><option value="linked">Linked controls</option>';
    header.appendChild(title);
    header.appendChild(mode);
    card.appendChild(header);

    const ordered = items.slice().sort(function(a, b) {
      const ar = String((a.trace.meta || {}).codex_neurite_role || "");
      const br = String((b.trace.meta || {}).codex_neurite_role || "");
      return ar.indexOf("dendrite") === 0 ? -1 : (br.indexOf("dendrite") === 0 ? 1 : 0);
    });
    const controls = [];
    ordered.forEach(function(item) {
      const tr = item.trace;
      const role = String((tr.meta || {}).codex_neurite_role || "line");
      const row = document.createElement("div");
      row.className = "codex-trace-row";
      const name = document.createElement("div");
      name.className = "codex-trace-name";
      name.textContent = role.indexOf("axon") === 0 ? "Axon" : "Dendrites + soma";
      const color = document.createElement("input");
      color.type = "color";
      color.value = colorToHex(tr.line && tr.line.color, "#008cff");
      const range = document.createElement("input");
      range.type = "range";
      range.min = "0.1"; range.max = "20"; range.step = "0.1";
      range.value = Number.isFinite(+(tr.line && tr.line.width)) ? +(tr.line && tr.line.width) : 2;
      const number = document.createElement("input");
      number.type = "number";
      number.min = "0.1"; number.max = "20"; number.step = "0.1";
      number.value = range.value;
      const visible = document.createElement("input");
      visible.type = "checkbox";
      visible.checked = (
        tr.visible !== false
        && tr.visible !== "legendonly"
        && !(Number.isFinite(+tr.opacity) && +tr.opacity <= 0)
      );
      row.append(name, color, range, number, visible);
      card.appendChild(row);
      controls.push({item:item, color:color, range:range, number:number, visible:visible});
    });

    function apply(source) {
      const color = source.color.value;
      const width = Number(source.range.value);
      const shown = source.visible.checked;
      source.number.value = source.range.value;
      if (mode.value === "linked") {
        controls.forEach(function(control) {
          control.color.value = color;
          control.range.value = String(width);
          control.number.value = String(width);
          control.visible.checked = shown;
        });
        Plotly.restyle(
          gd,
          {
            "line.color": color,
            "line.width": width,
            opacity: shown ? 1 : 0,
            visible: true
          },
          controls.map(control => control.item.idx)
        );
      } else {
        Plotly.restyle(
          gd,
          {
            "line.color": color,
            "line.width": width,
            opacity: shown ? 1 : 0,
            visible: true
          },
          [source.item.idx]
        );
      }
    }
    controls.forEach(function(control) {
      control.range.addEventListener("input", function() {
        control.number.value = control.range.value;
      });
      control.range.addEventListener("change", function() { apply(control); });
      control.number.addEventListener("change", function() {
        control.range.value = control.number.value;
        apply(control);
      });
      control.color.addEventListener("change", function() { apply(control); });
      control.visible.addEventListener("change", function() { apply(control); });
    });
    mode.addEventListener("change", function() {
      if (mode.value === "linked" && controls.length) apply(controls[0]);
    });
    return card;
  }
  function buildTraceControls() {
    const shellBox = byId("codex-shell-controls");
    const lineBox = byId("codex-line-controls");
    shellBox.textContent = "";
    lineBox.textContent = "";
    shellBox.className = "";
    lineBox.className = "";
    const traces = gd.data || [];
    const shells = [];
    const lines = [];
    const neuriteGroups = {};
    traces.forEach(function(tr, idx) {
      const type = (tr.type || "").toLowerCase();
      const mode = String(tr.mode || "");
      if (type === "mesh3d") shells.push({trace:tr, idx:idx});
      if (type === "scatter3d" && mode.indexOf("lines") >= 0) {
        const meta = tr.meta || {};
        const groupName = meta.codex_neurite_group;
        if (groupName) {
          if (!neuriteGroups[groupName]) neuriteGroups[groupName] = [];
          neuriteGroups[groupName].push({trace:tr, idx:idx});
        } else {
          lines.push({trace:tr, idx:idx});
        }
      }
    });
    if (shells.length === 0) shellBox.textContent = "No mesh3d shells were detected.";
    shells.forEach(function(item, n) {
      const tr = item.trace;
      const color = colorToHex(tr.color, "#d9d9d9");
      const op = Number.isFinite(+tr.opacity) ? +tr.opacity : 0.35;
      const row = makeTraceRow("shell", item.idx, tr.name || ("shell " + (n+1)), color, op, 0, 1, 0.01, tr.visible, function(idx, c, v, shown) {
        Plotly.restyle(gd, {color: c, opacity: v, visible: shown}, [idx]);
        scheduleTransparencySort(180);
      });
      shellBox.appendChild(row);
    });
    const neuriteNames = Object.keys(neuriteGroups);
    if (lines.length === 0 && neuriteNames.length === 0) {
      lineBox.textContent = "No scatter3d lines were detected.";
    }
    neuriteNames.forEach(function(groupName) {
      lineBox.appendChild(
        makeNeuriteGroupControls(groupName, neuriteGroups[groupName])
      );
    });
    lines.forEach(function(item, n) {
      const tr = item.trace;
      const color = colorToHex(tr.line && tr.line.color, "#008cff");
      const width = Number.isFinite(+(tr.line && tr.line.width)) ? +(tr.line && tr.line.width) : 2;
      const row = makeTraceRow("line", item.idx, tr.name || ("line " + (n+1)), color, width, 0.1, 20, 0.1, tr.visible, function(idx, c, v, shown) {
        Plotly.restyle(
          gd,
          {
            "line.color": c,
            "line.width": v,
            opacity: shown ? 1 : 0,
            visible: true
          },
          [idx]
        );
      });
      lineBox.appendChild(row);
    });
  }
  window.codexBuildTraceControls = buildTraceControls;
  function getInitialRanges() {
    const scene = (gd._fullLayout && gd._fullLayout.scene) || {};
    function axisRange(axis) {
      const r = scene[axis + "axis"] && scene[axis + "axis"].range;
      return (r && r.length === 2) ? [Number(r[0]), Number(r[1])] : [0, 1];
    }
    return {x:axisRange("x"), y:axisRange("y"), z:axisRange("z")};
  }
  const fullRanges = getInitialRanges();
  const codexVolumeMeta = __CODEX_VOLUME_META__;
  const codexShapeZYX = codexVolumeMeta.shape_zyx || [1,1,1];
  const codexCropZYX = codexVolumeMeta.crop_zyx || [[0,codexShapeZYX[0]],[0,codexShapeZYX[1]],[0,codexShapeZYX[2]]];
  const codexScaleZYX = codexVolumeMeta.display_scale_zyx || [1,1,1];
  const codexPlotAxisOrder = String(codexVolumeMeta.plot_axis_order || "XYZ").toUpperCase();
  const tiffFullRanges = {
    x:[0,Number(codexShapeZYX[2])],
    y:[0,Number(codexShapeZYX[1])],
    z:[0,Number(codexShapeZYX[0])]
  };
  function tiffRangesToPlotRanges(tiffRanges) {
    const scaledOriginal = {
      X:[tiffRanges.x[0]*Number(codexScaleZYX[2]),tiffRanges.x[1]*Number(codexScaleZYX[2])],
      Y:[tiffRanges.y[0]*Number(codexScaleZYX[1]),tiffRanges.y[1]*Number(codexScaleZYX[1])],
      Z:[tiffRanges.z[0]*Number(codexScaleZYX[0]),tiffRanges.z[1]*Number(codexScaleZYX[0])]
    };
    return {
      x:scaledOriginal[codexPlotAxisOrder[0]].slice(),
      y:scaledOriginal[codexPlotAxisOrder[1]].slice(),
      z:scaledOriginal[codexPlotAxisOrder[2]].slice()
    };
  }
  function voxelPointToPlot(point) {
    const original={X:point[0]*Number(codexScaleZYX[2]),Y:point[1]*Number(codexScaleZYX[1]),Z:point[2]*Number(codexScaleZYX[0])};
    return [original[codexPlotAxisOrder[0]],original[codexPlotAxisOrder[1]],original[codexPlotAxisOrder[2]]];
  }
  function plotVectorToVoxel(vector) {
    const original={X:0,Y:0,Z:0};
    original[codexPlotAxisOrder[0]]=vector[0];
    original[codexPlotAxisOrder[1]]=vector[1];
    original[codexPlotAxisOrder[2]]=vector[2];
    return [original.X/Number(codexScaleZYX[2]),original.Y/Number(codexScaleZYX[1]),original.Z/Number(codexScaleZYX[0])];
  }
  function voxelPlaneToPlot(plane) {
    const scaledOriginal={
      X:plane.normal[0]/Number(codexScaleZYX[2]),
      Y:plane.normal[1]/Number(codexScaleZYX[1]),
      Z:plane.normal[2]/Number(codexScaleZYX[0])
    };
    return {
      normal:[scaledOriginal[codexPlotAxisOrder[0]],scaledOriginal[codexPlotAxisOrder[1]],scaledOriginal[codexPlotAxisOrder[2]]],
      d:plane.d,keep:plane.keep,opacity:plane.opacity
    };
  }
  function makeSliceAspectRatioFromRanges(ranges) {
    const sx = Math.max(1e-9, Math.abs(ranges.x[1] - ranges.x[0]));
    const sy = Math.max(1e-9, Math.abs(ranges.y[1] - ranges.y[0]));
    const sz = Math.max(1e-9, Math.abs(ranges.z[1] - ranges.z[0]));
    const m = Math.max(sx, sy, sz);
    return {x: sx / m, y: sy / m, z: sz / m};
  }
  function setSliceUi(axis, range) {
    byId("codex-slice-" + axis + "-min-range").value = range[0];
    byId("codex-slice-" + axis + "-min").value = String(Math.round(Number(range[0])));
    byId("codex-slice-" + axis + "-max-range").value = range[1];
    byId("codex-slice-" + axis + "-max").value = String(Math.round(Number(range[1])));
  }
  function readSliceRange(axis) {
    const full = tiffFullRanges[axis];
    let lo = num("codex-slice-" + axis + "-min", full[0]);
    let hi = num("codex-slice-" + axis + "-max", full[1]);
    lo = clamp(lo, full[0], full[1]);
    hi = clamp(hi, full[0], full[1]);
    if (lo > hi) { const t = lo; lo = hi; hi = t; }
    return [lo, hi];
  }
  function applySlice(ranges) {
    const voxelRanges = ranges || {x:readSliceRange("x"), y:readSliceRange("y"), z:readSliceRange("z")};
    const r = tiffRangesToPlotRanges(voxelRanges);
    Plotly.relayout(gd, {
      "scene.xaxis.range": r.x,
      "scene.yaxis.range": r.y,
      "scene.zaxis.range": r.z,
      "scene.aspectmode": "manual",
      "scene.aspectratio": makeSliceAspectRatioFromRanges(r)
    });
  }
  function buildSliceControls() {
    const box = byId("codex-slice-controls");
    box.textContent = "";
    ["x","y","z"].forEach(function(axis) {
      const full = tiffFullRanges[axis];
      ["min","max"].forEach(function(side) {
        const row = document.createElement("div");
        row.className = "codex-slice-row";
        const lab = document.createElement("span");
        lab.textContent = axis.toUpperCase() + (side === "min" ? " low" : " high");
        const range = document.createElement("input");
        range.type = "range";
        range.id = "codex-slice-" + axis + "-" + side + "-range";
        range.min = full[0]; range.max = full[1];
        range.step = 1;
        const number = document.createElement("input");
        number.type = "number";
        number.id = "codex-slice-" + axis + "-" + side;
        number.step = 1;
        number.value = String(side === "min" ? full[0] : full[1]);
        range.value = number.value;
        range.addEventListener("input", function() { number.value = String(Math.round(Number(range.value))); });
        range.addEventListener("change", function() { applySlice(); });
        number.addEventListener("input", function() { range.value = number.value; });
        number.addEventListener("keydown", function(event) {
          if (event.key === "Enter") {
            event.preventDefault();
            range.value = number.value;
            applySlice();
          }
        });
        number.addEventListener("blur", function() {
          range.value = number.value;
          applySlice();
        });
        row.appendChild(lab); row.appendChild(range); row.appendChild(number);
        box.appendChild(row);
      });
    });
    const note=byId("codex-slice-volume-note");
    if (note) note.textContent="TIFF shape (Z,Y,X) = ("+codexShapeZYX.join(", ")+ "); currently loaded crop_zyx = ("+codexCropZYX.map(function(r){return "("+r[0]+", "+r[1]+")";}).join(", ")+"). Panel inputs use TIFF X/Y/Z voxel coordinates.";
  }
  const obliquePreviewNames = [
    "__codex_oblique_preview_low__",
    "__codex_oblique_preview_high__"
  ];
  function readObliqueConfig() {
    const normal = [
      num("codex-oblique-nx", 0),
      num("codex-oblique-ny", 1),
      num("codex-oblique-nz", 1)
    ];
    let low = num("codex-oblique-low", 0);
    let high = num("codex-oblique-high", 1);
    if (low > high) { const t = low; low = high; high = t; }
    return {normal:normal, low:low, high:high};
  }
  function obliqueProjectionBounds(normal) {
    const xs = tiffFullRanges.x, ys = tiffFullRanges.y, zs = tiffFullRanges.z;
    const values = [];
    xs.forEach(function(x) { ys.forEach(function(y) { zs.forEach(function(z) {
      values.push(normal[0]*x + normal[1]*y + normal[2]*z);
    }); }); });
    return [Math.min.apply(null, values), Math.max.apply(null, values)];
  }
  function setDefaultObliqueRange() {
    let normal = [
      num("codex-oblique-nx", 0),
      num("codex-oblique-ny", 1),
      num("codex-oblique-nz", 1)
    ];
    if (norm(normal) < 1e-12) normal = [0, 1, 1];
    const bounds = obliqueProjectionBounds(normal);
    const span = bounds[1] - bounds[0];
    byId("codex-oblique-low").value = (bounds[0] + span * 0.40).toFixed(2);
    byId("codex-oblique-high").value = (bounds[0] + span * 0.60).toFixed(2);
  }
  function planeBoxPolygon(normal, d) {
    const xs = fullRanges.x, ys = fullRanges.y, zs = fullRanges.z;
    const corners = [
      [xs[0],ys[0],zs[0]], [xs[1],ys[0],zs[0]],
      [xs[0],ys[1],zs[0]], [xs[1],ys[1],zs[0]],
      [xs[0],ys[0],zs[1]], [xs[1],ys[0],zs[1]],
      [xs[0],ys[1],zs[1]], [xs[1],ys[1],zs[1]]
    ];
    const edges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
    const points = [];
    function addUnique(point) {
      const duplicate = points.some(function(q) {
        const dx=point[0]-q[0], dy=point[1]-q[1], dz=point[2]-q[2];
        return dx*dx + dy*dy + dz*dz < 1e-12;
      });
      if (!duplicate) points.push(point);
    }
    edges.forEach(function(edge) {
      const a=corners[edge[0]], b=corners[edge[1]];
      const da=dot(normal,a)-d, db=dot(normal,b)-d;
      if (Math.abs(da) < 1e-10) addUnique(a.slice());
      if (Math.abs(db) < 1e-10) addUnique(b.slice());
      if (da*db < 0) {
        const t=da/(da-db);
        addUnique([a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), a[2]+t*(b[2]-a[2])]);
      }
    });
    if (points.length < 3) return [];
    const center=[0,0,0];
    points.forEach(function(p) { center[0]+=p[0]; center[1]+=p[1]; center[2]+=p[2]; });
    center[0]/=points.length; center[1]/=points.length; center[2]/=points.length;
    const n=normalize(normal);
    const helper=Math.abs(n[0]) < 0.8 ? [1,0,0] : [0,1,0];
    const u=normalize(cross(n,helper));
    const v=normalize(cross(n,u));
    points.sort(function(a,b) {
      const ar=[a[0]-center[0],a[1]-center[1],a[2]-center[2]];
      const br=[b[0]-center[0],b[1]-center[1],b[2]-center[2]];
      return Math.atan2(dot(ar,v),dot(ar,u))-Math.atan2(dot(br,v),dot(br,u));
    });
    return points;
  }
  function previewTrace(points, name, color, opacity) {
    const i=[], j=[], k=[];
    for (let idx=1; idx<points.length-1; idx++) { i.push(0); j.push(idx); k.push(idx+1); }
    return {
      type:"mesh3d", x:points.map(p=>p[0]), y:points.map(p=>p[1]), z:points.map(p=>p[2]),
      i:i, j:j, k:k, color:color, opacity:Number.isFinite(opacity)?opacity:0.18, name:name,
      hoverinfo:"skip", showscale:false, showlegend:false
    };
  }
  function previewOutlineTrace(points,name,color) {
    const closed=points.concat([points[0]]);
    return {
      type:"scatter3d",mode:"lines",x:closed.map(p=>p[0]),y:closed.map(p=>p[1]),z:closed.map(p=>p[2]),
      line:{color:color,width:4},name:name,hoverinfo:"skip",showlegend:false
    };
  }
  function removeObliquePreview() {
    const indices=[];
    (gd.data || []).forEach(function(trace, idx) {
      if (obliquePreviewNames.indexOf(trace.name) >= 0) indices.push(idx);
    });
    if (indices.length) Plotly.deleteTraces(gd, indices.sort(function(a,b){return b-a;}));
  }
  function formatObliqueParams(config) {
    function f(value) { return Number(value).toFixed(6).replace(/0+$/, "").replace(/\.$/, ""); }
    return "oblique_crop = {\n" +
      "    \"enabled\": True,\n" +
      "    \"normal_plot\": (" + config.normal.map(f).join(", ") + "),\n" +
      "    \"range\": (" + f(config.low) + ", " + f(config.high) + "),\n" +
      "}";
  }
  function updateObliquePreview() {
    removeObliquePreview();
    const config=readObliqueConfig();
    byId("codex-oblique-params").textContent=formatObliqueParams(config);
    if (norm(config.normal) < 1e-12) {
      byId("codex-oblique-status").textContent="The normal vector cannot be all zeros.";
      return;
    }
    const bounds=obliqueProjectionBounds(config.normal);
    const lowPoints=planeBoxPolygon(config.normal,config.low);
    const highPoints=planeBoxPolygon(config.normal,config.high);
    if (byId("codex-show-oblique-preview").checked) {
      const traces=[];
      if (lowPoints.length >= 3) traces.push(previewTrace(lowPoints,obliquePreviewNames[0],"#2f80ed"));
      if (highPoints.length >= 3) traces.push(previewTrace(highPoints,obliquePreviewNames[1],"#eb5757"));
      if (traces.length) Plotly.addTraces(gd,traces);
    }
    byId("codex-oblique-status").textContent =
      "The model d range for the current normal vector is approximately [" + bounds[0].toFixed(2) + ", " + bounds[1].toFixed(2) + "]; retain the region between the two planes.";
  }
  let codexObliquePlanes = [];
  let codexSelectedObliquePlane = 0;
  const codexObliqueColors = ["#2f80ed", "#eb5757", "#27ae60", "#9b51e0", "#f2994a", "#00a6c6"];
  function codexPlaneName(index) { return "__codex_oblique_plane_" + index + "__"; }
  function removeAllObliquePlaneTraces() {
    const indices=[];
    (gd.data || []).forEach(function(trace, idx) {
      if (String(trace.name || "").indexOf("__codex_oblique_plane_") === 0) indices.push(idx);
    });
    if (indices.length) Plotly.deleteTraces(gd, indices.sort(function(a,b){return b-a;}));
  }
  function currentObliquePlane() { return codexObliquePlanes[codexSelectedObliquePlane] || null; }
  function planeBoundsForNormal(normal) { return obliqueProjectionBounds(normal); }
  function refreshObliquePlaneSelect() {
    const select=byId("codex-oblique-plane-select");
    select.textContent="";
    codexObliquePlanes.forEach(function(_,index) {
      const option=document.createElement("option");
      const color=codexObliqueColors[index%codexObliqueColors.length];
      option.value=String(index); option.textContent="P"+(index+1)+" · "+color.toUpperCase();
      option.style.color=color;
      select.appendChild(option);
    });
    select.value=String(codexSelectedObliquePlane);
  }
  function saveCurrentObliquePlane() {
    const plane=currentObliquePlane();
    if (!plane) return;
    plane.normal=[num("codex-oblique-nx",0),num("codex-oblique-ny",1),num("codex-oblique-nz",1)];
    plane.d=num("codex-oblique-d",plane.d);
    plane.opacity=clamp(num("codex-oblique-opacity",plane.opacity||0.55),0.05,0.90);
    plane.keep=byId("codex-oblique-keep").value === "le" ? "le" : "ge";
  }
  function loadCurrentObliquePlane() {
    const plane=currentObliquePlane();
    if (!plane) return;
    byId("codex-oblique-nx").value=Number(plane.normal[0]).toFixed(6);
    byId("codex-oblique-ny").value=Number(plane.normal[1]).toFixed(6);
    byId("codex-oblique-nz").value=Number(plane.normal[2]).toFixed(6);
    byId("codex-oblique-d").value=Number(plane.d).toFixed(2);
    byId("codex-oblique-opacity").value=Number(plane.opacity||0.55).toFixed(2);
    byId("codex-oblique-opacity-range").value=Number(plane.opacity||0.55).toFixed(2);
    byId("codex-oblique-keep").value=plane.keep;
    const bounds=planeBoundsForNormal(plane.normal);
    const slider=byId("codex-oblique-d-range");
    slider.min=bounds[0]; slider.max=bounds[1];
    slider.step=Math.max((bounds[1]-bounds[0])/2000,0.01);
    slider.value=plane.d;
  }
  function formatObliquePlanesParams() {
    function f(value) { return Number(value).toFixed(6).replace(/0+$/,"").replace(/\.$/,""); }
    const lines=["oblique_planes = ["];
    codexObliquePlanes.forEach(function(plane) {
      lines.push("    {\"enabled\": True, \"normal_voxel\": (" + plane.normal.map(f).join(", ") + "), \"d\": " + f(plane.d) + ", \"keep\": \"" + plane.keep + "\"},");
    });
    lines.push("]");
    return lines.join("\n");
  }
  function obliqueDirectionTraces(points,plane,index,color) {
    const center=[0,0,0];
    points.forEach(function(p){center[0]+=p[0];center[1]+=p[1];center[2]+=p[2];});
    center[0]/=points.length;center[1]/=points.length;center[2]/=points.length;
    let direction=normalize(plane.normal);
    if (plane.keep==="le") direction=direction.map(function(v){return -v;});
    const dx=fullRanges.x[1]-fullRanges.x[0],dy=fullRanges.y[1]-fullRanges.y[0],dz=fullRanges.z[1]-fullRanges.z[0];
    const arrowLength=Math.max(Math.sqrt(dx*dx+dy*dy+dz*dz)*0.05,1);
    const tip=[center[0]+direction[0]*arrowLength,center[1]+direction[1]*arrowLength,center[2]+direction[2]*arrowLength];
    const headLength=arrowLength*0.26;
    const baseCenter=[tip[0]-direction[0]*headLength,tip[1]-direction[1]*headLength,tip[2]-direction[2]*headLength];
    const helper=Math.abs(direction[0])<0.8?[1,0,0]:[0,1,0];
    const sideA=normalize(cross(direction,helper));
    const sideB=normalize(cross(direction,sideA));
    const headRadius=headLength*0.34;
    const arrowVertices=[tip];
    const arrowSides=6;
    for(let sideIndex=0;sideIndex<arrowSides;sideIndex++){
      const angle=2*Math.PI*sideIndex/arrowSides;
      arrowVertices.push([
        baseCenter[0]+headRadius*(sideA[0]*Math.cos(angle)+sideB[0]*Math.sin(angle)),
        baseCenter[1]+headRadius*(sideA[1]*Math.cos(angle)+sideB[1]*Math.sin(angle)),
        baseCenter[2]+headRadius*(sideA[2]*Math.cos(angle)+sideB[2]*Math.sin(angle))
      ]);
    }
    const arrowI=[],arrowJ=[],arrowK=[];
    for(let sideIndex=0;sideIndex<arrowSides;sideIndex++){
      arrowI.push(0);arrowJ.push(1+sideIndex);arrowK.push(1+((sideIndex+1)%arrowSides));
    }
    const rejectTip=[center[0]-direction[0]*arrowLength*0.62,center[1]-direction[1]*arrowLength*0.62,center[2]-direction[2]*arrowLength*0.62];
    const prefix=codexPlaneName(index);
    return [
      {
        type:"scatter3d",mode:"lines",x:[center[0],baseCenter[0]],y:[center[1],baseCenter[1]],z:[center[2],baseCenter[2]],
        line:{color:color,width:5},name:prefix+"direction_line",hoverinfo:"skip",showlegend:false
      },
      {
        type:"mesh3d",x:arrowVertices.map(p=>p[0]),y:arrowVertices.map(p=>p[1]),z:arrowVertices.map(p=>p[2]),
        i:arrowI,j:arrowJ,k:arrowK,color:color,opacity:1,flatshading:true,
        name:prefix+"direction_arrow",hoverinfo:"skip",showscale:false,showlegend:false
      },
      {
        type:"scatter3d",mode:"text",x:[tip[0]],y:[tip[1]],z:[tip[2]],
        text:["P"+(index+1)+" retained side"],textposition:"top center",textfont:{color:color,size:15},
        name:prefix+"direction_label",hoverinfo:"skip",showlegend:false
      },
      {
        type:"scatter3d",mode:"lines",x:[center[0],rejectTip[0]],y:[center[1],rejectTip[1]],z:[center[2],rejectTip[2]],
        line:{color:"#6b7280",width:4,dash:"dash"},name:prefix+"reject_line",hoverinfo:"skip",showlegend:false
      },
      {
        type:"scatter3d",mode:"text",x:[rejectTip[0]],y:[rejectTip[1]],z:[rejectTip[2]],
        text:["P"+(index+1)+" discarded side ×"],textposition:"bottom center",textfont:{color:"#6b7280",size:13},
        name:prefix+"reject_label",hoverinfo:"skip",showlegend:false
      }
    ];
  }
  function updateMultiObliquePreview() {
    saveCurrentObliquePlane();
    removeAllObliquePlaneTraces();
    const traces=[];
    if (byId("codex-show-oblique-preview").checked) {
      codexObliquePlanes.forEach(function(plane,index) {
        if (norm(plane.normal) < 1e-12) return;
        const plotPlane=voxelPlaneToPlot(plane);
        const points=planeBoxPolygon(plotPlane.normal,plotPlane.d);
        if (points.length >= 3) {
          const color=codexObliqueColors[index%codexObliqueColors.length];
          const surfaceOpacity=clamp(Number(plane.opacity)||0.55,0.05,0.90);
          traces.push(previewTrace(points,codexPlaneName(index)+"surface",color,surfaceOpacity));
          traces.push(previewOutlineTrace(points,codexPlaneName(index)+"outline",color));
          obliqueDirectionTraces(points,plotPlane,index,color).forEach(function(trace){traces.push(trace);});
        }
      });
    }
    if (traces.length) Plotly.addTraces(gd,traces);
    byId("codex-oblique-params").textContent=formatObliquePlanesParams();
    const plane=currentObliquePlane();
    if (!plane || norm(plane.normal)<1e-12) {
      byId("codex-oblique-status").textContent="The normal vector of the current plane cannot be all zeros.";
      return;
    }
    const bounds=planeBoundsForNormal(plane.normal);
    const planeColor=codexObliqueColors[codexSelectedObliquePlane%codexObliqueColors.length].toUpperCase();
    byId("codex-oblique-status").textContent="P"+(codexSelectedObliquePlane+1)+" ("+planeColor+") d="+Number(plane.d).toFixed(2)+"; the arrow points to the retained side; approximate drag range ["+bounds[0].toFixed(2)+", "+bounds[1].toFixed(2)+"]; "+(plane.keep==="ge"?"retain n·p ≥ d":"retain n·p ≤ d")+".";
  }
  const codexObliqueOriginalDisplayData = {};
  let codexObliqueOriginalSceneState = null;
  const codexObliqueSceneAnchorName="__codex_oblique_scene_anchor__";
  function ensureObliqueSceneAnchor() {
    const existing=(gd.data||[]).some(function(trace){return trace.name===codexObliqueSceneAnchorName;});
    if (existing) return Promise.resolve();
    const xs=fullRanges.x,ys=fullRanges.y,zs=fullRanges.z;
    const corners=[];
    xs.forEach(function(x){ys.forEach(function(y){zs.forEach(function(z){corners.push([x,y,z]);});});});
    return Plotly.addTraces(gd,{
      type:"scatter3d",mode:"markers",name:codexObliqueSceneAnchorName,
      x:corners.map(function(p){return p[0];}),
      y:corners.map(function(p){return p[1];}),
      z:corners.map(function(p){return p[2];}),
      marker:{size:0.1,color:"rgba(0,0,0,0)"},opacity:0.001,
      hoverinfo:"skip",showlegend:false
    });
  }
  function captureObliqueSceneState() {
    const scene=(gd._fullLayout&&gd._fullLayout.scene)||{};
    const camera=getCurrentCamera();
    function rangeOf(axis) {
      const range=scene[axis+"axis"]&&scene[axis+"axis"].range;
      return range&&range.length===2?[Number(range[0]),Number(range[1])]:fullRanges[axis].slice();
    }
    const ratio=scene.aspectratio||makeSliceAspectRatioFromRanges({x:rangeOf("x"),y:rangeOf("y"),z:rangeOf("z")});
    return {
      ranges:{x:rangeOf("x"),y:rangeOf("y"),z:rangeOf("z")},
      aspectmode:scene.aspectmode||"data",
      aspectratio:{x:Number(ratio.x)||1,y:Number(ratio.y)||1,z:Number(ratio.z)||1},
      camera:{eye:camera.eye,up:camera.up,center:camera.center}
    };
  }
  function relayoutWithLockedObliqueScene(state,restoreMode) {
    const update={
      "scene.xaxis.range":state.ranges.x,
      "scene.yaxis.range":state.ranges.y,
      "scene.zaxis.range":state.ranges.z,
      "scene.xaxis.autorange":false,
      "scene.yaxis.autorange":false,
      "scene.zaxis.autorange":false,
      "scene.aspectmode":restoreMode?state.aspectmode:"manual",
      "scene.aspectratio":state.aspectratio,
      "scene.camera":state.camera,
      "scene.uirevision":"codex-oblique-fixed-scene"
    };
    return Plotly.relayout(gd,update);
  }
  function pointInsideAllObliquePlanes(point) {
    for (let index=0;index<codexObliquePlanes.length;index++) {
      const plane=voxelPlaneToPlot(codexObliquePlanes[index]);
      const value=dot(plane.normal,point)-plane.d;
      if (plane.keep==="ge" ? value < -1e-9 : value > 1e-9) return false;
    }
    return true;
  }
  function clipDisplaySegmentToObliquePlanes(p0,p1) {
    let enter=0,exit=1;
    for (let index=0;index<codexObliquePlanes.length;index++) {
      const plane=voxelPlaneToPlot(codexObliquePlanes[index]);
      let value0=dot(plane.normal,p0)-plane.d;
      let value1=dot(plane.normal,p1)-plane.d;
      if (plane.keep==="le") { value0=-value0; value1=-value1; }
      const inside0=value0>=-1e-9,inside1=value1>=-1e-9;
      if (inside0&&inside1) continue;
      if (!inside0&&!inside1) return null;
      const crossAt=value0/(value0-value1);
      if (inside0) exit=Math.min(exit,crossAt); else enter=Math.max(enter,crossAt);
      if (enter>exit+1e-9) return null;
    }
    const direction=[p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]];
    return [
      [p0[0]+enter*direction[0],p0[1]+enter*direction[1],p0[2]+enter*direction[2]],
      [p0[0]+exit*direction[0],p0[1]+exit*direction[1],p0[2]+exit*direction[2]]
    ];
  }
  const codexClosedCutPrefix="__codex_closed_cut_mesh__";
  function clipMeshByOnePlaneClosed(mesh,plane) {
    const normal=plane.normal,d=plane.d,sign=plane.keep==="le"?-1:1,eps=1e-8;
    const x=mesh.x.slice(),y=mesh.y.slice(),z=mesh.z.slice();
    const outI=[],outJ=[],outK=[],cutSegments=[],edgeCache=new Map();
    function value(index){return sign*(normal[0]*x[index]+normal[1]*y[index]+normal[2]*z[index]-d);}
    function intersection(a,b,va,vb){
      const key=a<b?a+":"+b:b+":"+a;
      if(edgeCache.has(key))return edgeCache.get(key);
      const t=va/(va-vb),idx=x.length;
      x.push(x[a]+t*(x[b]-x[a]));y.push(y[a]+t*(y[b]-y[a]));z.push(z[a]+t*(z[b]-z[a]));
      edgeCache.set(key,idx);return idx;
    }
    function addTriangle(a,b,c){if(a!==b&&b!==c&&c!==a){outI.push(a);outJ.push(b);outK.push(c);}}
    for(let face=0;face<mesh.i.length;face++){
      const input=[mesh.i[face],mesh.j[face],mesh.k[face]],output=[],cuts=[];
      for(let e=0;e<3;e++){
        const a=input[e],b=input[(e+1)%3],va=value(a),vb=value(b),insideA=va>=-eps,insideB=vb>=-eps;
        if(insideA)output.push(a);
        if(insideA!==insideB){const q=intersection(a,b,va,vb);output.push(q);cuts.push(q);}
      }
      if(output.length>=3){for(let q=1;q<output.length-1;q++)addTriangle(output[0],output[q],output[q+1]);}
      const uniqueCuts=cuts.filter(function(v,index,array){return array.indexOf(v)===index;});
      if(uniqueCuts.length===2&&uniqueCuts[0]!==uniqueCuts[1])cutSegments.push(uniqueCuts);
    }
    const adjacency=new Map();
    cutSegments.forEach(function(seg){
      if(!adjacency.has(seg[0]))adjacency.set(seg[0],[]);if(!adjacency.has(seg[1]))adjacency.set(seg[1],[]);
      adjacency.get(seg[0]).push(seg[1]);adjacency.get(seg[1]).push(seg[0]);
    });
    const used=new Set(),loops=[];
    function edgeKey(a,b){return a<b?a+":"+b:b+":"+a;}
    cutSegments.forEach(function(seg){
      if(used.has(edgeKey(seg[0],seg[1])))return;
      const loop=[seg[0]],start=seg[0];let previous=seg[0],current=seg[1];used.add(edgeKey(previous,current));
      for(let guard=0;guard<cutSegments.length+2;guard++){
        loop.push(current);if(current===start)break;
        const candidates=(adjacency.get(current)||[]).filter(function(next){return next!==previous&&!used.has(edgeKey(current,next));});
        if(!candidates.length)break;
        const next=candidates[0];used.add(edgeKey(current,next));previous=current;current=next;
      }
      if(loop.length>=4&&loop[loop.length-1]===start){loop.pop();loops.push(loop);}
    });
    const desired=plane.keep==="ge"?normal.map(function(v){return -v;}):normal.slice();
    loops.forEach(function(loop){
      let cx=0,cy=0,cz=0;loop.forEach(function(idx){cx+=x[idx];cy+=y[idx];cz+=z[idx];});
      cx/=loop.length;cy/=loop.length;cz/=loop.length;const center=x.length;x.push(cx);y.push(cy);z.push(cz);
      let areaNormal=[0,0,0];
      for(let q=0;q<loop.length;q++){
        const a=loop[q],b=loop[(q+1)%loop.length];
        areaNormal[0]+=(y[a]-cy)*(z[b]-cz)-(z[a]-cz)*(y[b]-cy);
        areaNormal[1]+=(z[a]-cz)*(x[b]-cx)-(x[a]-cx)*(z[b]-cz);
        areaNormal[2]+=(x[a]-cx)*(y[b]-cy)-(y[a]-cy)*(x[b]-cx);
      }
      if(dot(areaNormal,desired)<0)loop=loop.slice().reverse();
      for(let q=0;q<loop.length;q++)addTriangle(center,loop[q],loop[(q+1)%loop.length]);
    });
    return {x:x,y:y,z:z,i:outI,j:outJ,k:outK,capLoops:loops.length};
  }
  function buildClosedCutMeshTrace(trace,traceIndex) {
    let mesh={x:Array.from(trace.x),y:Array.from(trace.y),z:Array.from(trace.z),i:Array.from(trace.i),j:Array.from(trace.j),k:Array.from(trace.k)};
    codexObliquePlanes.forEach(function(voxelPlane){mesh=clipMeshByOnePlaneClosed(mesh,voxelPlaneToPlot(voxelPlane));});
    return {
      type:"mesh3d",x:mesh.x,y:mesh.y,z:mesh.z,i:mesh.i,j:mesh.j,k:mesh.k,
      name:codexClosedCutPrefix+traceIndex,color:trace.color,opacity:trace.opacity,
      flatshading:trace.flatshading,lighting:trace.lighting,lightposition:trace.lightposition,
      hoverinfo:"skip",showscale:false,showlegend:false
    };
  }
  function applyObliqueDisplayCut() {
    saveCurrentObliquePlane();
    ensureObliqueSceneAnchor();
    const lockedSceneState=captureObliqueSceneState();
    if (!codexObliqueOriginalSceneState) codexObliqueOriginalSceneState=lockedSceneState;
    const status=byId("codex-oblique-cut-status");
    status.textContent="Applying display clipping. Please wait...";
    setTimeout(function() {
      const promises=[],generatedMeshes=[];
      let meshBefore=0,meshAfter=0,lineBefore=0,lineAfter=0;
      (gd.data||[]).forEach(function(trace,traceIndex) {
        const name=String(trace.name||"");
        if (name===codexObliqueSceneAnchorName || name.indexOf(codexClosedCutPrefix)===0 || name.indexOf("__codex_oblique_plane_")===0 || obliquePreviewNames.indexOf(name)>=0) return;
        const type=String(trace.type||"").toLowerCase();
        if (type==="mesh3d" && trace.i && trace.j && trace.k && trace.x && trace.y && trace.z) {
          if (!codexObliqueOriginalDisplayData[traceIndex]) {
            codexObliqueOriginalDisplayData[traceIndex]={kind:"mesh",x:trace.x,y:trace.y,z:trace.z,i:trace.i,j:trace.j,k:trace.k};
          }
          const original=codexObliqueOriginalDisplayData[traceIndex];
          meshBefore+=original.i.length;
          const generated=buildClosedCutMeshTrace(trace,traceIndex);
          meshAfter+=generated.i.length;generatedMeshes.push(generated);
          promises.push(Plotly.restyle(gd,{visible:false},[traceIndex]));
        } else if (type==="scatter3d" && String(trace.mode||"").indexOf("lines")>=0 && trace.x && trace.y && trace.z) {
          if (!codexObliqueOriginalDisplayData[traceIndex]) {
            codexObliqueOriginalDisplayData[traceIndex]={kind:"line",x:trace.x,y:trace.y,z:trace.z};
          }
          const original=codexObliqueOriginalDisplayData[traceIndex];
          const nextX=[],nextY=[],nextZ=[];
          let previous=null;
          for (let pointIndex=0;pointIndex<original.x.length;pointIndex++) {
            const rawX=original.x[pointIndex],rawY=original.y[pointIndex],rawZ=original.z[pointIndex];
            if (rawX===null || rawX===undefined || rawY===null || rawY===undefined || rawZ===null || rawZ===undefined) {
              previous=null;
              continue;
            }
            const point=[Number(rawX),Number(rawY),Number(rawZ)];
            if (!point.every(Number.isFinite)) { previous=null; continue; }
            if (previous) {
              lineBefore++;
              const clipped=clipDisplaySegmentToObliquePlanes(previous,point);
              if (clipped) {
                nextX.push(clipped[0][0],clipped[1][0],null);
                nextY.push(clipped[0][1],clipped[1][1],null);
                nextZ.push(clipped[0][2],clipped[1][2],null);
                lineAfter++;
              }
            }
            previous=point;
          }
          promises.push(Plotly.restyle(gd,{x:[nextX],y:[nextY],z:[nextZ]},[traceIndex]));
        }
      });
      Promise.all(promises).then(function() {
        if(generatedMeshes.length)return Plotly.addTraces(gd,generatedMeshes);
      }).then(function() {
        return Plotly.redraw(gd);
      }).then(function() {
        return relayoutWithLockedObliqueScene(lockedSceneState,false);
      }).then(function() {
        status.textContent="Display clipping complete: mesh triangles "+meshBefore+" → "+meshAfter+", line segments "+lineBefore+" → "+lineAfter+". Final closed cut surfaces still require regeneration in Python.";
      }).catch(function(error) {
        status.textContent="Display clipping failed: "+String(error&&error.message||error);
      });
    },20);
  }
  function restoreObliqueDisplayCut() {
    const generatedIndices=[];
    (gd.data||[]).forEach(function(trace,index){if(String(trace.name||"").indexOf(codexClosedCutPrefix)===0)generatedIndices.push(index);});
    const removeGenerated=generatedIndices.length?Plotly.deleteTraces(gd,generatedIndices.sort(function(a,b){return b-a;})):Promise.resolve();
    removeGenerated.then(function(){
      const promises=[];
      Object.keys(codexObliqueOriginalDisplayData).forEach(function(key) {
        const traceIndex=Number(key),original=codexObliqueOriginalDisplayData[key];
        if (original.kind==="mesh") promises.push(Plotly.restyle(gd,{visible:true},[traceIndex]));
        else promises.push(Plotly.restyle(gd,{x:[original.x],y:[original.y],z:[original.z]},[traceIndex]));
      });
      return Promise.all(promises);
    }).then(function(){return Plotly.redraw(gd);}).then(function(){
      if (codexObliqueOriginalSceneState) return relayoutWithLockedObliqueScene(codexObliqueOriginalSceneState,true);
    }).then(function(){byId("codex-oblique-cut-status").textContent="The unclipped view and original three-axis proportions have been restored.";});
  }
  function resetMultipleObliquePlanes() {
    const normal=[0,1,1];
    const bounds=planeBoundsForNormal(normal);
    const span=bounds[1]-bounds[0];
    codexObliquePlanes=[
      {normal:normal.slice(),d:bounds[0]+span*0.40,keep:"ge",opacity:0.55},
      {normal:normal.slice(),d:bounds[0]+span*0.60,keep:"le",opacity:0.55}
    ];
    codexSelectedObliquePlane=0;
    refreshObliquePlaneSelect(); loadCurrentObliquePlane(); updateMultiObliquePreview();
  }
  function addObliquePlane() {
    saveCurrentObliquePlane();
    const source=currentObliquePlane();
    const normal=source ? source.normal.slice() : [0,1,1];
    const bounds=planeBoundsForNormal(normal);
    codexObliquePlanes.push({normal:normal,d:(bounds[0]+bounds[1])/2,keep:"ge",opacity:0.55});
    codexSelectedObliquePlane=codexObliquePlanes.length-1;
    refreshObliquePlaneSelect(); loadCurrentObliquePlane(); updateMultiObliquePreview();
  }
  function deleteCurrentObliquePlane() {
    if (codexObliquePlanes.length<=1) {
      byId("codex-oblique-status").textContent="Keep at least one preview plane."; return;
    }
    codexObliquePlanes.splice(codexSelectedObliquePlane,1);
    codexSelectedObliquePlane=Math.min(codexSelectedObliquePlane,codexObliquePlanes.length-1);
    refreshObliquePlaneSelect(); loadCurrentObliquePlane(); updateMultiObliquePreview();
  }
  function rotateCurrentObliqueNormal(sign) {
    saveCurrentObliquePlane();
    const plane=currentObliquePlane(); if (!plane || norm(plane.normal)<1e-12) return;
    const oldNormal=plane.normal.slice();
    const oldNormalNorm=norm(oldNormal);
    const oldPlotPlane=voxelPlaneToPlot(plane);
    const polygon=planeBoxPolygon(oldPlotPlane.normal,oldPlotPlane.d);
    let pivot=[0,0,0];
    if (polygon.length>=3) {
      polygon.forEach(function(point){
        const voxelPoint=plotVectorToVoxel(point);
        pivot[0]+=voxelPoint[0];pivot[1]+=voxelPoint[1];pivot[2]+=voxelPoint[2];
      });
      pivot[0]/=polygon.length;pivot[1]/=polygon.length;pivot[2]/=polygon.length;
    } else {
      const scale=plane.d/(oldNormalNorm*oldNormalNorm);
      pivot=[oldNormal[0]*scale,oldNormal[1]*scale,oldNormal[2]*scale];
    }
    const anat=byId("codex-oblique-rotate-axis").value;
    let axis;
    if (typeof window.codexGetOrientationAxisVector === "function") axis=window.codexGetOrientationAxisVector(anat);
    else {
      const fallback={AP:[0,1,0],DV:[1,0,0],RL:[0,0,1]}; axis=fallback[anat]||fallback.AP;
    }
    axis=normalize(plotVectorToVoxel(axis));
    const step=num("codex-oblique-rotate-step",5)*Math.PI/180*(Number(sign)||1);
    plane.normal=normalize(rotateAroundAxis(plane.normal,axis,step));
    plane.d=dot(plane.normal,pivot);
    loadCurrentObliquePlane(); updateMultiObliquePreview();
  }
  function moveCurrentObliquePlane(sign) {
    saveCurrentObliquePlane();
    const plane=currentObliquePlane(); if (!plane) return;
    plane.d += (Number(sign)||1)*num("codex-oblique-move-step",50);
    loadCurrentObliquePlane(); updateMultiObliquePreview();
  }
  function applyOverlayToggles() {
    window.codexOverlayVisibility = window.codexOverlayVisibility || {};
    window.codexOverlayVisibility.orientation = byId("codex-show-orientation").checked;
    window.codexOverlayVisibility.scalebar = byId("codex-show-scalebar").checked;
    window.codexOverlayVisibility.viewpoint = byId("codex-show-viewpoint").checked;
    if (viewpointPanel) {
      viewpointPanel.style.display = byId("codex-show-viewpoint").checked ? "block" : "none";
    }
    window.dispatchEvent(new Event("codex-overlay-visibility"));
  }
  function applyOrientationLinkToggle() {
    const linked = byId("codex-link-orientation").checked;
    window.codexOrientationLinked = linked;
    if (!linked) {
      if (!window.codexOrientationLockedBasis) {
        window.codexOrientationLockedBasis = getCameraBasis();
      }
    } else {
      rebaseManualOrientationToCurrentCamera();
      window.codexOrientationLockedBasis = null;
      window.codexOrientationEditMode = "move";
      const moveRadio = byId("codex-orientation-mode-move");
      if (moveRadio) moveRadio.checked = true;
    }
    window.dispatchEvent(new Event("codex-overlay-visibility"));
  }
  function applyOrientationEditMode() {
    const selected = panel.querySelector('input[name="codex-orientation-mode"]:checked');
    window.codexOrientationEditMode = selected ? selected.value : "move";
    if (window.codexOrientationEditMode === "rotate") {
      byId("codex-link-orientation").checked = false;
      if (!window.codexOrientationLockedBasis) {
        window.codexOrientationLockedBasis = getCameraBasis();
      }
      window.codexOrientationLinked = false;
    }
    window.dispatchEvent(new Event("codex-overlay-visibility"));
  }
  function setOrientationCalibrationUi(anatKey, value) {
    const key = anatKey.toLowerCase();
    const range = byId("codex-orient-cal-" + key + "-range");
    const number = byId("codex-orient-cal-" + key);
    if (range) range.value = value;
    if (number) number.value = Number(value).toFixed(1);
  }
  function applyOrientationCalibrationFromUi() {
    const read = function(anatKey) {
      const key = anatKey.toLowerCase();
      const input = byId("codex-orient-cal-" + key);
      const value = parseFloat(input && input.value);
      return Number.isFinite(value) ? value : 0;
    };
    window.codexOrientationManualAngles = {
      RL: read("RL"),
      AP: read("AP"),
      DV: read("DV")
    };
    if (typeof window.codexApplyOrientationManualAngles === "function") {
      window.codexApplyOrientationManualAngles(window.codexOrientationManualAngles);
    }
    window.dispatchEvent(new Event("codex-overlay-visibility"));
  }
  function resetOrientationCalibration() {
    window.codexOrientationManualAngles = {RL: 0, AP: 0, DV: 0};
    ["RL", "AP", "DV"].forEach(function(anatKey) {
      setOrientationCalibrationUi(anatKey, 0);
    });
    if (typeof window.codexApplyOrientationManualAngles === "function") {
      window.codexApplyOrientationManualAngles(window.codexOrientationManualAngles);
    }
    window.dispatchEvent(new Event("codex-overlay-visibility"));
  }
  function syncOrientationCalibrationUiFromAngles(angles) {
    const a = Object.assign({RL: 0, AP: 0, DV: 0}, angles || window.codexOrientationManualAngles || {});
    ["RL", "AP", "DV"].forEach(function(anatKey) {
      setOrientationCalibrationUi(anatKey, a[anatKey] || 0);
    });
  }
  window.addEventListener("codex-orientation-calibration-change", function(event) {
    syncOrientationCalibrationUiFromAngles(event.detail || window.codexOrientationManualAngles);
  });
  function initOrientationCalibrationControls() {
    ["RL", "AP", "DV"].forEach(function(anatKey) {
      const key = anatKey.toLowerCase();
      const range = byId("codex-orient-cal-" + key + "-range");
      const number = byId("codex-orient-cal-" + key);
      if (!range || !number) return;
      const sync = function(source) {
        if (source === "range") {
          number.value = Number(range.value).toFixed(2);
        } else {
          range.value = number.value;
        }
        applyOrientationCalibrationFromUi();
      };
      range.addEventListener("input", function() { sync("range"); });
      number.addEventListener("change", function() { sync("number"); });
    });
    const resetButton = byId("codex-reset-orientation-cal");
    if (resetButton) {
      resetButton.addEventListener("click", resetOrientationCalibration);
    }
    resetOrientationCalibration();
  }
  function initViewpointDrag() {
    if (!viewpointPanel) return;
    const pos = {x:0, y:0, scale:1};
    function render() {
      viewpointPanel.style.transform = (
        "translate(" + pos.x + "px," + pos.y + "px) scale(" + pos.scale + ")"
      );
    }
    let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;
    viewpointPanel.addEventListener("pointerdown", function(event) {
      if (event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      dragging = true;
      sx = event.clientX; sy = event.clientY; ox = pos.x; oy = pos.y;
      if (viewpointPanel.setPointerCapture) viewpointPanel.setPointerCapture(event.pointerId);
    });
    viewpointPanel.addEventListener("pointermove", function(event) {
      if (!dragging) return;
      event.preventDefault();
      event.stopPropagation();
      pos.x = ox + event.clientX - sx;
      pos.y = oy + event.clientY - sy;
      render();
    });
    function finish(event) {
      if (!dragging) return;
      event.preventDefault();
      event.stopPropagation();
      dragging = false;
      if (viewpointPanel.releasePointerCapture) {
        try { viewpointPanel.releasePointerCapture(event.pointerId); } catch (err) {}
      }
    }
    viewpointPanel.addEventListener("pointerup", finish);
    viewpointPanel.addEventListener("pointercancel", finish);
    viewpointPanel.addEventListener("wheel", function(event) {
      event.preventDefault();
      event.stopPropagation();
      pos.scale = clamp(pos.scale * Math.exp(-event.deltaY * 0.0015), 0.60, 2.50);
      render();
    }, {passive:false});
    viewpointPanel.addEventListener("dblclick", function(event) {
      event.preventDefault();
      event.stopPropagation();
      pos.scale = 1;
      render();
    });
    render();
  }
  function initPanelDrag() {
    const pos = {x:190, y:14, scale:1};
    const expandedButtonCenterX = 296;
    const expandedButtonCenterY = 19;
    const collapsedButtonCenterX = 22;
    const collapsedButtonCenterY = 22;
    function collapsedAnchorDx() {
      return (expandedButtonCenterX - collapsedButtonCenterX) * pos.scale;
    }
    function collapsedAnchorDy() {
      return (expandedButtonCenterY - collapsedButtonCenterY) * pos.scale;
    }
    function render() {
      panel.style.left = pos.x + "px";
      panel.style.top = pos.y + "px";
      panel.style.transform = "scale(" + pos.scale + ")";
    }
    let dragging = false, moved = false, sx = 0, sy = 0, ox = 0, oy = 0;
    let expandedPanelSize = {width: panel.style.width || "", height: panel.style.height || ""};
    function collapsePanelSize() {
      expandedPanelSize = {width: panel.style.width || "", height: panel.style.height || ""};
      panel.style.width = "44px";
      panel.style.height = "44px";
    }
    function restorePanelSize() {
      panel.style.width = expandedPanelSize.width || "";
      panel.style.height = expandedPanelSize.height || "";
    }
    title.addEventListener("pointerdown", function(event) {
      if (event.button !== 0) return;
      if (event.target && event.target.closest("button")) return;
      event.preventDefault();
      event.stopPropagation();
      dragging = true; moved = false;
      sx = event.clientX; sy = event.clientY; ox = pos.x; oy = pos.y;
      if (title.setPointerCapture) title.setPointerCapture(event.pointerId);
    });
    title.addEventListener("pointermove", function(event) {
      if (!dragging) return;
      event.preventDefault();
      event.stopPropagation();
      if (Math.abs(event.clientX - sx) > 3 || Math.abs(event.clientY - sy) > 3) moved = true;
      pos.x = ox + event.clientX - sx;
      pos.y = oy + event.clientY - sy;
      render();
    });
    function endDrag(event) {
      if (!dragging) return;
      dragging = false;
      if (title.releasePointerCapture) {
        try { title.releasePointerCapture(event.pointerId); } catch (err) {}
      }
    }
    title.addEventListener("pointerup", endDrag);
    title.addEventListener("pointercancel", endDrag);
    title.addEventListener("click", function(event) {
      if (!panel.classList.contains("codex-collapsed") || moved) return;
      event.preventDefault();
      event.stopPropagation();
      pos.x -= collapsedAnchorDx();
      pos.y -= collapsedAnchorDy();
      panel.classList.remove("codex-collapsed");
      minBtn.textContent = "−";
      render();
    });
    minBtn.addEventListener("click", function(event) {
      event.preventDefault();
      event.stopPropagation();
      if (panel.classList.contains("codex-collapsed")) return;
      pos.x += collapsedAnchorDx();
      pos.y += collapsedAnchorDy();
      panel.classList.add("codex-collapsed");
      minBtn.textContent = "☰";
      render();
    });
    panel.addEventListener("pointerdown", function(event) { event.stopPropagation(); });
    panel.addEventListener("wheel", function(event) {
      event.stopPropagation();
      if (event.target && event.target.closest("input,button,.codex-body")) return;
      event.preventDefault();
      pos.scale = clamp(pos.scale * Math.exp(-event.deltaY * 0.0012), 0.75, 1.60);
      render();
    }, {passive:false});
    render();
  }
  byId("codex-show-orientation").checked = window.codexOverlayVisibility.orientation !== false;
  byId("codex-show-scalebar").checked = window.codexOverlayVisibility.scalebar !== false;
  byId("codex-show-viewpoint").checked = window.codexOverlayVisibility.viewpoint !== false;
  byId("codex-link-orientation").checked = window.codexOrientationLinked !== false;
  byId(
    window.codexOrientationEditMode === "rotate" ?
    "codex-orientation-mode-rotate" :
    "codex-orientation-mode-move"
  ).checked = true;

  ["codex-show-orientation","codex-show-scalebar","codex-show-viewpoint"].forEach(function(id) {
    byId(id).addEventListener("change", applyOverlayToggles);
  });
  byId("codex-link-orientation").addEventListener("change", applyOrientationLinkToggle);
  panel.querySelectorAll('input[name="codex-orientation-mode"]').forEach(function(el) {
    el.addEventListener("change", applyOrientationEditMode);
  });
  ["elev","azim","roll"].forEach(function(k) {
    byId("codex-lock-" + k).addEventListener("change", updateAngleLocks);
  });
  byId("codex-apply-view").addEventListener("click", applyView);
  byId("codex-read-view").addEventListener("click", function() { updateViewDisplay(true); });
  byId("codex-axis-rotate-neg").addEventListener("click", function() { rotateViewAroundOrientationAxis(-1); });
  byId("codex-axis-rotate-pos").addEventListener("click", function() { rotateViewAroundOrientationAxis(1); });
  byId("codex-apply-slice").addEventListener("click", function() { applySlice(); });
  byId("codex-reset-slice").addEventListener("click", function() {
    ["x","y","z"].forEach(function(axis) { setSliceUi(axis, tiffFullRanges[axis]); });
    applySlice(tiffFullRanges);
  });
  byId("codex-oblique-plane-select").addEventListener("change", function() {
    saveCurrentObliquePlane();
    codexSelectedObliquePlane=parseInt(this.value,10)||0;
    loadCurrentObliquePlane(); updateMultiObliquePreview();
  });
  byId("codex-oblique-keep").addEventListener("change", updateMultiObliquePreview);
  byId("codex-show-oblique-preview").addEventListener("change", updateMultiObliquePreview);
  byId("codex-add-oblique-plane").addEventListener("click", addObliquePlane);
  byId("codex-delete-oblique-plane").addEventListener("click", deleteCurrentObliquePlane);
  byId("codex-reset-oblique-preview").addEventListener("click", resetMultipleObliquePlanes);
  byId("codex-apply-oblique-display-cut").addEventListener("click", applyObliqueDisplayCut);
  byId("codex-restore-oblique-display-cut").addEventListener("click", restoreObliqueDisplayCut);
  byId("codex-oblique-rotate-neg").addEventListener("click", function(){rotateCurrentObliqueNormal(-1);});
  byId("codex-oblique-rotate-pos").addEventListener("click", function(){rotateCurrentObliqueNormal(1);});
  byId("codex-oblique-move-neg").addEventListener("click", function(){moveCurrentObliquePlane(-1);});
  byId("codex-oblique-move-pos").addEventListener("click", function(){moveCurrentObliquePlane(1);});
  byId("codex-oblique-d-range").addEventListener("input", function(){
    byId("codex-oblique-d").value=Number(this.value).toFixed(2); updateMultiObliquePreview();
  });
  byId("codex-oblique-opacity-range").addEventListener("input", function(){
    byId("codex-oblique-opacity").value=Number(this.value).toFixed(2); updateMultiObliquePreview();
  });
  ["codex-oblique-nx","codex-oblique-ny","codex-oblique-nz","codex-oblique-d","codex-oblique-opacity"].forEach(function(id) {
    byId(id).addEventListener("keydown", function(event) {
      if (event.key === "Enter") { event.preventDefault(); updateMultiObliquePreview(); loadCurrentObliquePlane(); }
    });
    byId(id).addEventListener("change", function(){ updateMultiObliquePreview(); loadCurrentObliquePlane(); });
  });
  if (gd.on) {
    gd.on("plotly_relayout", function() {
      updateViewDisplay(false);
      scheduleTransparencySort(350);
    });
    gd.on("plotly_afterplot", function() { updateViewDisplay(false); });
  }
  window.addEventListener("resize", function() { updateViewDisplay(false); });
  initPanelDrag();
  initViewpointDrag();
  initOrientationCalibrationControls();
  updateAngleLocks();
  updateViewDisplay(true);
  buildTraceControls();
  const backgroundSelect = byId("codex-background-theme");
  backgroundSelect.value = backgroundTheme;
  backgroundSelect.addEventListener("change", function() {
    applyBackground(backgroundSelect.value);
  });
  applyBackground(backgroundTheme);
  buildSliceControls();
  resetMultipleObliquePlanes();
  applyOverlayToggles();
})();
</script>

<script id="codex-config-enhancer-script">
(function() {
  const wrapper = document.getElementById("plot-wrapper");
  const gd = wrapper && wrapper.querySelector(".plotly-graph-div");
  const panel = document.getElementById("codex-surface-transform-panel");
  if (!panel || !gd) return;

  function byId(id) { return document.getElementById(id); }

  function hideViewLocks() {
    ["codex-lock-elev", "codex-lock-azim", "codex-lock-roll"].forEach(function(id) {
      const el = byId(id);
      if (el) el.checked = false;
      if (el && el.parentElement) el.parentElement.style.display = "none";
    });
    panel.querySelectorAll(".codex-angle-row").forEach(function(row) {
      row.style.gridTemplateColumns = "48px 1fr";
    });
  }

  function addScalebarUnitControl() {
    if (!byId("codex-scalebar-unit")) {
      const anchor = byId("codex-link-orientation");
      const row = anchor && anchor.closest(".codex-check-row");
      const section = document.createElement("div");
      section.id = "codex-scale-unit-extra-controls";
      section.innerHTML = '<div class="codex-section-title">\u6bd4\u4f8b\u5c3a</div>' +
        '<div class="codex-axis-rotate-row"><span>\u5355\u4f4d</span><select id="codex-scalebar-unit"><option value="um">\u5fae\u7c73 &micro;m</option><option value="mm">\u6beb\u7c73 mm</option></select><span></span></div>' +
        '<div class="codex-note">\u6bd4\u4f8b\u5c3a\u957f\u5ea6\u4ecd\u53ef\u5728\u6bd4\u4f8b\u5c3a\u4e0a\u6eda\u8f6e\u5207\u6362\uff1b\u8fd9\u91cc\u53ea\u6539\u53d8\u663e\u793a\u5355\u4f4d\u3002</div>';
      if (row && row.parentElement) row.parentElement.insertBefore(section, row.nextSibling);
      else panel.querySelector(".codex-body").insertBefore(section, panel.querySelector(".codex-body").firstChild);
    }

    const unit = byId("codex-scalebar-unit");
    if (!unit || unit.dataset.codexUnitBound) return;
    unit.dataset.codexUnitBound = "1";
    unit.value = window.codexScalebarUnit === "mm" ? "mm" : "um";
    unit.addEventListener("change", function() {
      if (typeof window.codexSetScalebarUnit === "function") {
        window.codexSetScalebarUnit(unit.value);
      } else {
        window.codexScalebarUnit = unit.value === "mm" ? "mm" : "um";
        window.dispatchEvent(new Event("codex-overlay-visibility"));
      }
    });
  }

  hideViewLocks();
  addScalebarUnitControl();
})();
</script>
'''
        layout_meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
        volume_meta = layout_meta.get("codex_volume", {})
        control_panel_html = control_panel_html.replace(
            "__CODEX_VOLUME_META__",
            json.dumps(volume_meta, ensure_ascii=False),
        )
        html = html.replace("</body>", control_panel_html + "\n</body>")

    out_dir = os.path.dirname(html_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


NEURON_STYLES = (
    ("#008CFF", "#0066CC", "#66BFFF"),
    ("#FF7F0E", "#D65F00", "#FFB366"),
    ("#D62728", "#B2182B", "#F08080"),
    ("#00A6C6", "#008AA6", "#72D6E6"),
)


def _existing_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Input file does not exist: {path}")
    return path


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct outer/inner spinal-cord shells and four inferred-neurite "
            "SWCs as a standalone interactive HTML."
        )
    )
    parser.add_argument("--outer-mask", required=True, type=_existing_file,
                        help="Outer spinal-cord binary TIFF stack.")
    parser.add_argument("--inner-mask", required=True, type=_existing_file,
                        help="Inner spinal-cord binary TIFF stack.")
    parser.add_argument(
        "--swc", required=True, type=_existing_file, nargs=4,
        metavar=("SWC1", "SWC2", "SWC3", "SWC4"),
        help="Exactly four SWC reconstruction files, in display order.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("spinal_shell_four_neurons.html"),
        help="Destination HTML path (default: spinal_shell_four_neurons.html).",
    )
    parser.add_argument(
        "--crop-zyx", type=int, nargs=6,
        metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
        default=(0, 6418, 20, 1310, 0, 939),
        help="Shared TIFF/SWC crop in Z0 Z1 Y0 Y1 X0 X1 order.",
    )
    parser.add_argument("--downsample-zyx", type=int, nargs=3, default=(4, 4, 4),
                        metavar=("DZ", "DY", "DX"))
    parser.add_argument("--display-scale-zyx", type=float, nargs=3,
                        default=(7.1, 6.0, 7.1), metavar=("SZ", "SY", "SX"))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--smooth-iters", type=int, default=15)
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _crop_from_flat(values):
    z0, z1, y0, y1, x0, x1 = (int(value) for value in values)
    if not (z0 < z1 and y0 < y1 and x0 < x1):
        raise ValueError("Each --crop-zyx lower bound must be smaller than its upper bound.")
    return ((z0, z1), (y0, y1), (x0, x1))


def _swc_specs(paths, line_width):
    specs = []
    for path, (base_color, axon_color, other_color) in zip(paths, NEURON_STYLES):
        specs.append({
            "path": str(path),
            "name": path.stem,
            "color": base_color,
            "line_width": float(line_width),
            "use_radius_width": False,
            "distinguish_neurites": True,
            "color_mode": "infer_neurites",
            "axon_color": axon_color,
            "dendrite_color": other_color,
            "soma_color": other_color,
        })
    return specs


def build_spinal_shell_swc_html(args):
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    crop_zyx = _crop_from_flat(args.crop_zyx)
    verbose = not args.quiet
    black = args.background == "black"
    foreground = "#FFFFFF" if black else "#000000"
    shell_specs = [
        {
            "path": str(args.outer_mask),
            "name": "outer_shell",
            "color": "rgb(240, 240, 240)",
            "opacity": 0.40,
            "threshold": float(args.threshold),
            "crop_zyx": crop_zyx,
        },
        {
            "path": str(args.inner_mask),
            "name": "inner_shell",
            "color": "rgb(200, 200, 200)",
            "opacity": 0.40,
            "threshold": float(args.threshold),
            "crop_zyx": crop_zyx,
        },
    ]
    swc_specs = _swc_specs(args.swc, args.line_width)

    fig, mesh_records, swc_records = plot_multi_foreground_tiffs_with_swc_plotly(
        tiff_specs=shell_specs,
        swc_specs=swc_specs,
        html_path=str(output),
        tiff_axes="ZYX",
        threshold=float(args.threshold),
        crop_zyx=crop_zyx,
        mesh_downsample=tuple(args.downsample_zyx),
        strict_same_shape=True,
        oblique_planes=None,
        show_surface=True,
        default_surface_color="#D9D9D9",
        default_surface_opacity=0.40,
        mesh_sigma=1.0,
        mesh_level=0.5,
        mesh_pad_small=2,
        smooth_iters=int(args.smooth_iters),
        smooth_lam=0.20,
        display_scale_zyx=tuple(args.display_scale_zyx),
        show_swc=True,
        swc_coord_scale=(1.0, 1.0, 1.0),
        swc_coord_offset=(0.0, 0.0, 0.0),
        swc_radius_scale=1.0,
        flip_swc_z=False,
        swc_color="#008CFF",
        uniform_line_width=float(args.line_width),
        use_radius_width=False,
        distinguish_neurites=True,
        swc_color_mode="infer_neurites",
        axon_color="#FF9F43",
        dendrite_color="#FF5F70",
        soma_color="#FF5F70",
        clip_swc_to_crop=True,
        swc_crop_zyx=None,
        viewpoint=(0.0, 0.0),
        camera_distance_factor=2.0,
        camera_projection_type="orthographic",
        use_swc_for_limits=True,
        plot_axis_order="XZY",
        background_color=args.background,
        aspectmode="data",
        title=None,
        showlegend=False,
        show_feature_lines=True,
        feature_silhouette=True,
        feature_crease=True,
        feature_angle_deg=10.0,
        feature_line_width=0.05,
        feature_line_color=foreground,
        feature_max_line_segments=1_000_000,
        save_html=False,
        include_plotlyjs=True,
        verbose=verbose,
    )

    save_plotly_html_with_overlays(
        fig=fig,
        html_path=str(output),
        scalebar_um=1000,
        scalebar_label="1000 μm",
        scalebar_bar_color=foreground,
        scalebar_text_color=foreground,
        scalebar_thickness_px=7,
        scalebar_font_size_px=16,
        scalebar_right_px=28,
        scalebar_bottom_px=28,
        scalebar_resizable=True,
        scalebar_size_options=(100, 250, 500, 1000, 2000, 3000, 5000, 10000),
        show_orientation_widget=True,
        orientation_axes_plot={"RL": "z", "AP": "y", "DV": "x"},
        orientation_label_pairs={"RL": ("R", "L"), "AP": ("A", "P"), "DV": ("V", "D")},
        orientation_colors={"RL": foreground, "AP": foreground, "DV": foreground},
        orientation_size_px=150,
        orientation_left_px=18,
        orientation_top_px=18,
        orientation_radius_px=42,
        orientation_font_size_px=13,
        orientation_line_width_px=3,
        orientation_resizable=True,
        orientation_min_scale=0.50,
        orientation_max_scale=2.50,
        orientation_resize_sensitivity=0.0015,
        orientation_rotation_deg=(0.0, 30.0, 0.0),
        orientation_rotation_order=("x", "y", "z"),
        orientation_show_circle=False,
        orientation_circle_color="rgba(255,255,255,0.80)",
        orientation_circle_stroke="rgba(0,0,0,0.25)",
        orientation_circle_stroke_width=1.0,
        show_viewpoint_text=True,
        viewpoint_right_px=24,
        viewpoint_top_px=24,
        viewpoint_font_size_px=14,
        viewpoint_text_color=foreground,
        viewpoint_bg_color="rgba(0,0,0,0.72)" if black else "rgba(255,255,255,0.75)",
        viewpoint_box_width_px=290,
        show_control_panel=True,
        include_plotlyjs=True,
    )

    summary = {
        "output": str(output),
        "shells": [record["name"] for record in mesh_records],
        "neurons": [record["name"] for record in swc_records],
        "neurite_inference": "morphology-based soma/axon/dendrite inference",
        "gray_region_data_embedded": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output


def main():
    parser = _build_cli_parser()
    args = parser.parse_args()
    build_spinal_shell_swc_html(args)


if __name__ == "__main__":
    main()
