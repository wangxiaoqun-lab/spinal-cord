

def plot_region_3d_view_with_surface(
    workDir,
    target_region_ids=None,
    atlas_path="path/to/atlas_template.tiff",
    anno_path="path/to/atlas_annotation.tiff",
    cell_path_suffix="cells_transformed_to_Atlas_aligned.npy",
    region_name=None,

    alpha_bg=0.40,
    show_surface=True,
    show_points=True,

    background_color="white",
    main_surface_color="#E5E5E5",           
    pts_col="white",
    aux_color="auto",                                               

    point_size=2.0,

    viewpoint=(10, -50),
    viewpoints=None,                                                      
    crop_size=None,

    hemisphere='full',                                         
    AP_lim=None,                                    
    DV_lim=None,                                    

    scalebar_um=1000,
    voxel_um=25,
    arrow=False,

    save_pdf=False,
    pdf_rasterize_surface=False,                                       
    save_png=True,
    proj='test',
    save_dir="3D.fig",
    save_prefix="region_3d_surface",
    show_figure=True,

    feature_lines=False,
    silhouette=True,
    crease=True,
    crease_deg=50.0,
    line_color='auto',
    line_width=0.3,
    line_alpha=0.25,
    max_line_segments=300000,

    outer_outline=False,
    outer_outline_color='auto',
    outer_outline_width=0.8,
    outer_outline_alpha=0.9,

    point_rearrange="none",                                   
    density_sigma_vox=2.0,
    jitter_base_vox=1.5,
    jitter_alpha=0.7,
    jitter_axes=(0.3, 0.3, 1.0),
    jitter_clip_vox=6.0,
    density_floor=1e-6,
    density_use_region_bbox=True,
    random_seed=0,

    point_groups=None,                          
    points_show_mode="all_on",                           
    points_show_other=True,
    points_other_color=None,
    points_unmatched_name="other",

    verbose=True,
):
    
    import os
    import re
    import numpy as np
    import tifffile
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter
    from collections import defaultdict

    def _parse_color_to_rgb01(c):
        if c is None:
            return None
        s = str(c).strip().lower()

        named = {
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "gray":  (0.5, 0.5, 0.5),
            "grey":  (0.5, 0.5, 0.5),
            "cyan":  (0.0, 1.0, 1.0),
            "skyblue": (135/255, 206/255, 235/255),
            "magenta": (1.0, 0.0, 1.0),
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 0.5, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
        }
        if s in named:
            return named[s]

        if s.startswith("#"):
            hx = s[1:]
            if len(hx) == 3:
                try:
                    return tuple(int(ch * 2, 16) / 255.0 for ch in hx)
                except Exception:
                    return None
            if len(hx) == 6:
                try:
                    return (
                        int(hx[0:2], 16) / 255.0,
                        int(hx[2:4], 16) / 255.0,
                        int(hx[4:6], 16) / 255.0,
                    )
                except Exception:
                    return None
            return None

        m = re.match(r"rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*(?:,\s*([0-9.]+)\s*)?\)$", s)
        if m:
            try:
                r = float(m.group(1)); g = float(m.group(2)); b = float(m.group(3))
                if max(r, g, b) > 1.5:
                    r /= 255.0; g /= 255.0; b /= 255.0
                r = float(np.clip(r, 0, 1))
                g = float(np.clip(g, 0, 1))
                b = float(np.clip(b, 0, 1))
                return (r, g, b)
            except Exception:
                return None
        return None

    def _is_light_background(bg):
        rgb = _parse_color_to_rgb01(bg)
        if rgb is None:
            return False
        r, g, b = rgb
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return lum >= 0.85

    def _resolve_aux_color(bg, aux):
        if aux is None:
            return "white"
        if str(aux).strip().lower() != "auto":
            return aux
        return "black" if _is_light_background(bg) else "white"

    def _auto_aux_if_white(param_color, aux_resolved):
        if param_color is None:
            return None
        s = str(param_color).strip().lower()
        if s in ("auto", "white"):
            return aux_resolved
        return param_color

    AUX = _resolve_aux_color(background_color, aux_color)
    line_color_eff = _auto_aux_if_white(line_color, AUX)
    outer_outline_color_eff = _auto_aux_if_white(outer_outline_color, AUX)

    def _normalize_mpl_color(c):
        if c is None:
            return None
        s = str(c).strip()
        m = re.match(r"rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*(?:,\s*([0-9.]+)\s*)?\)$", s, flags=re.I)
        if m:
            vals = [float(m.group(i)) for i in [1, 2, 3]]
            vals = [min(max(v, 0.0), 255.0) for v in vals]
            return tuple(v / 255.0 for v in vals)
        return c

    main_surface_color_eff = _normalize_mpl_color(main_surface_color)
    pts_col_eff = _normalize_mpl_color(pts_col)
    line_color_eff = _normalize_mpl_color(line_color_eff)
    outer_outline_color_eff = _normalize_mpl_color(outer_outline_color_eff)

    def laplacian_smooth(verts, faces, iters=10, lam=0.3):
        V = verts.copy()
        n = V.shape[0]
        neigh = [[] for _ in range(n)]
        for (a, b, c) in faces:
            neigh[a].extend([b, c])
            neigh[b].extend([a, c])
            neigh[c].extend([a, b])
        neigh = [np.array(sorted(set(ns)), dtype=int) for ns in neigh]
        for _ in range(iters):
            delta = np.zeros_like(V)
            for i, nb in enumerate(neigh):
                if nb.size == 0:
                    continue
                delta[i] = V[nb].mean(axis=0) - V[i]
            V += lam * delta
        return V

    def _build_mesh_from_mask(mask_bool, sigma=1.5, level=0.5, smooth_iters=15, smooth_lam=0.40,
                              bbox_pad=2, bbox_use=True):
        
        mask_bool = np.asarray(mask_bool, dtype=bool)
        if not mask_bool.any():
            raise ValueError('mask_bool has no True voxels')

        if bbox_use:
            idx = np.argwhere(mask_bool)
            z0, y0, x0 = idx.min(axis=0)
            z1, y1, x1 = idx.max(axis=0) + 1
            sub = mask_bool[z0:z1, y0:y1, x0:x1]
            pad = int(bbox_pad) if bbox_pad is not None else 0
            if pad > 0:
                sub = np.pad(sub, pad_width=pad, mode='constant', constant_values=False)
            offset_zyx = np.array([z0 - pad, y0 - pad, x0 - pad], dtype=float)
        else:
            sub = mask_bool
            offset_zyx = np.array([0.0, 0.0, 0.0], dtype=float)

        m = gaussian_filter(sub.astype(float), sigma=float(sigma))
        verts_zyx, faces, _, _ = marching_cubes(m, level=float(level))
        verts_zyx = verts_zyx + offset_zyx[None, :]
        verts_xyz = verts_zyx[:, [2, 1, 0]].astype(float)
        faces = faces.astype(np.int32)
        verts_xyz = laplacian_smooth(verts_xyz, faces, iters=int(smooth_iters), lam=float(smooth_lam))
        return verts_xyz, faces

    def _normalize_rows(M, eps=1e-12):
        M = np.asarray(M, dtype=float)
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / (n + eps)

    def _face_normals(verts, faces):
        tri = verts[faces]
        v1 = tri[:, 1] - tri[:, 0]
        v2 = tri[:, 2] - tri[:, 0]
        n = np.cross(v1, v2)
        return _normalize_rows(n)

    def _collect_edges(faces):
        edge2faces = defaultdict(list)
        for fi, (a, b, c) in enumerate(faces):
            for u, v in ((a, b), (b, c), (c, a)):
                if u > v:
                    u, v = v, u
                edge2faces[(u, v)].append(fi)
        return edge2faces

    def camera_vector_from_viewpoint(elev_deg, azim_deg):
        elev = np.deg2rad(float(elev_deg))
        azim = np.deg2rad(float(azim_deg))
        vec = np.array([
            np.sin(azim) * np.cos(elev),
            np.cos(azim) * np.cos(elev),
            np.sin(elev)
        ], dtype=float)
        n = np.linalg.norm(vec)
        return vec / (n + 1e-12)

    def _draw_feature_lines(
        ax, verts, faces, cam_vec,
        draw_silhouette=True,
        draw_crease=True,
        crease_th_cos=np.cos(np.deg2rad(50.0)),
        color="black",
        lw=0.3,
        alpha=0.25,
        max_segments=None,
    ):
        face_norms = _face_normals(verts, faces)
        facing = (face_norms @ cam_vec) > 0.0
        edge2faces = _collect_edges(faces)
        segs = []

        for (u, v), fids in edge2faces.items():
            if len(fids) == 1:
                if draw_silhouette:
                    segs.append(np.stack([verts[u], verts[v]], axis=0))
                continue

            f1, f2 = fids[0], fids[1]
            n1, n2 = face_norms[f1], face_norms[f2]

            if draw_silhouette and (facing[f1] ^ facing[f2]):
                segs.append(np.stack([verts[u], verts[v]], axis=0))
                continue

            if draw_crease and (np.dot(n1, n2) < crease_th_cos):
                segs.append(np.stack([verts[u], verts[v]], axis=0))

        if len(segs) == 0:
            return

        segs = np.stack(segs, axis=0)
        if max_segments is not None and segs.shape[0] > int(max_segments):
            keep = np.linspace(0, segs.shape[0] - 1, int(max_segments)).astype(int)
            segs = segs[keep]

        lc = Line3DCollection(segs, colors=[color], linewidths=lw, alpha=alpha)
        ax.add_collection3d(lc)

    def _project_points_to_camera(pts_xyz, cam_vec):
        
        cam_vec = np.asarray(cam_vec, dtype=float)
        cam_vec = cam_vec / (np.linalg.norm(cam_vec) + 1e-12)

        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(up, cam_vec)) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

        u = np.cross(up, cam_vec)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(cam_vec, u)
        v = v / (np.linalg.norm(v) + 1e-12)

        uv = np.stack([pts_xyz @ u, pts_xyz @ v], axis=1)
        return uv

    def _convex_hull_order_2d(pts2d):
        
        pts2d = np.asarray(pts2d, dtype=float)
        if pts2d.shape[0] <= 2:
            return np.arange(pts2d.shape[0], dtype=int)

        order = np.lexsort((pts2d[:, 1], pts2d[:, 0]))
        pts = pts2d[order]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for i, p in enumerate(pts):
            while len(lower) >= 2 and cross(pts[lower[-2]], pts[lower[-1]], p) <= 0:
                lower.pop()
            lower.append(i)

        upper = []
        for i, p in reversed(list(enumerate(pts))):
            while len(upper) >= 2 and cross(pts[upper[-2]], pts[upper[-1]], p) <= 0:
                upper.pop()
            upper.append(i)

        hull_local = lower[:-1] + upper[:-1]
        hull_order = order[np.array(hull_local, dtype=int)]
        return hull_order

    def _draw_outer_outline(ax, verts, cam_vec, color="black", lw=0.8, alpha=0.9):
        if verts.shape[0] < 3:
            return
        uv = _project_points_to_camera(verts, cam_vec)
        hull_idx = _convex_hull_order_2d(uv)
        if hull_idx.size < 3:
            return
        ring = verts[hull_idx]
        ring = np.vstack([ring, ring[0:1]])
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, linewidth=lw, alpha=alpha)

    def _density_jitter_points(
        pts_xyz,
        sigma_vox=2.0,
        base_vox=1.5,
        alpha=0.7,
        axes=(0.3, 0.3, 1.0),
        clip_vox=6.0,
        floor=1e-6,
        rng=None,
        use_bbox=True,
        verbose=False,
    ):
        from scipy.ndimage import gaussian_filter
        if rng is None:
            rng = np.random.default_rng(0)

        pts = np.asarray(pts_xyz, dtype=float)
        if pts.shape[0] == 0:
            return pts

        pmin = np.floor(pts.min(axis=0)).astype(int)
        pmax = np.ceil(pts.max(axis=0)).astype(int)
        margin = int(np.ceil(3 * float(sigma_vox) + 2))
        gmin = pmin - margin
        gmax = pmax + margin

        pts_shift = pts - gmin[None, :]
        grid_shape = (gmax - gmin + 1).astype(int)
        if np.any(grid_shape <= 1):
            return pts

        xi = np.clip(np.round(pts_shift[:, 0]).astype(int), 0, grid_shape[0] - 1)
        yi = np.clip(np.round(pts_shift[:, 1]).astype(int), 0, grid_shape[1] - 1)
        zi = np.clip(np.round(pts_shift[:, 2]).astype(int), 0, grid_shape[2] - 1)

        dens = np.zeros(grid_shape, dtype=np.float32)
        np.add.at(dens, (xi, yi, zi), 1.0)
        dens = gaussian_filter(dens, sigma=float(sigma_vox), mode="constant")

        fx = np.clip(pts_shift[:, 0], 0, grid_shape[0] - 1)
        fy = np.clip(pts_shift[:, 1], 0, grid_shape[1] - 1)
        fz = np.clip(pts_shift[:, 2], 0, grid_shape[2] - 1)

        x0 = np.floor(fx).astype(int); x1 = np.clip(x0 + 1, 0, grid_shape[0] - 1)
        y0 = np.floor(fy).astype(int); y1 = np.clip(y0 + 1, 0, grid_shape[1] - 1)
        z0 = np.floor(fz).astype(int); z1 = np.clip(z0 + 1, 0, grid_shape[2] - 1)

        xd = fx - x0; yd = fy - y0; zd = fz - z0

        c000 = dens[x0, y0, z0]; c100 = dens[x1, y0, z0]
        c010 = dens[x0, y1, z0]; c110 = dens[x1, y1, z0]
        c001 = dens[x0, y0, z1]; c101 = dens[x1, y0, z1]
        c011 = dens[x0, y1, z1]; c111 = dens[x1, y1, z1]

        c00 = c000 * (1 - xd) + c100 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        d_local = c0 * (1 - zd) + c1 * zd
        d_local = np.asarray(d_local, dtype=float)

        med = np.median(d_local[d_local > floor]) if np.any(d_local > floor) else 1.0
        scale = np.power(med / (d_local + floor), float(alpha))
        std = float(base_vox) * scale

        axw = np.asarray(axes, dtype=float)
        axw = axw / (np.max(axw) + 1e-12)
        std_xyz = std[:, None] * axw[None, :]

        disp = rng.normal(loc=0.0, scale=std_xyz, size=pts.shape)

        if clip_vox is not None:
            clip_vox = float(clip_vox)
            mag = np.linalg.norm(disp, axis=1)
            too = mag > clip_vox
            if np.any(too):
                disp[too] *= (clip_vox / (mag[too] + 1e-12))[:, None]

        pts_jit = pts + disp

        if verbose:
            print(f"[rearrange] density sigma={sigma_vox}, base={base_vox}, alpha={alpha}, axes={axes}")
            print(f"[rearrange] d_local median={np.median(d_local):.4f}, jitter std median={np.median(std):.4f}")

        return pts_jit

    def _as_set_ids(v):
        if v is None:
            return set()
        if isinstance(v, (list, tuple, set, np.ndarray)):
            return set(int(x) for x in v)
        return {int(v)}

    annotation = tifffile.imread(anno_path)
    cells = np.load(os.path.join(workDir, cell_path_suffix))[:, [2, 1, 0]]             
    Z, Y, X = annotation.shape

    if target_region_ids is None:
        target_region_ids = np.unique(annotation)
        target_region_ids = target_region_ids[target_region_ids > 0].tolist()

    annotation = np.flip(annotation, axis=0)
    cells[:, 0] = Z - cells[:, 0]

    mask = np.isin(annotation, target_region_ids)

    if hemisphere == 'right':
        mask[:, :, int(X / 2):] = False
    elif hemisphere == 'left':
        mask[:, :, :int(X / 2)] = False

    if AP_lim is not None:
        y_min, y_max = AP_lim
        mask[:, :y_min, :] = False
        mask[:, y_max:, :] = False
    else:
        y_min, y_max = None, None

    if DV_lim is not None:
        z_min, z_max = DV_lim
        mask[:z_min, :, :] = False
        mask[z_max:, :, :] = False
    else:
        z_min, z_max = None, None

    if not mask.any():
        print(f"⚠️ No voxels for region {region_name or target_region_ids} after cropping. Nothing to render.")
        return []

    verts, faces = _build_mesh_from_mask(
        mask,
        sigma=1.5,
        level=0.5,
        smooth_iters=15,
        smooth_lam=0.4,
        bbox_pad=2,
        bbox_use=True,
    )

    valid_idx = []
    valid_ids = []
    cells_int = cells.astype(int)
    target_set = set(int(x) for x in target_region_ids)

    for i, (z, y, x) in enumerate(cells_int):
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            rid = int(annotation[z, y, x])
            if rid in target_set:
                valid_idx.append(i)
                valid_ids.append(rid)

    region_cells = cells[valid_idx]                         
    region_cell_ids = np.asarray(valid_ids, int)              

    if region_cells.shape[0] > 0:
        if hemisphere == 'right':
            keep = region_cells[:, 2] < X / 2
            region_cells = region_cells[keep]
            region_cell_ids = region_cell_ids[keep]
        elif hemisphere == 'left':
            keep = region_cells[:, 2] >= X / 2
            region_cells = region_cells[keep]
            region_cell_ids = region_cell_ids[keep]

        if AP_lim is not None:
            keep = (region_cells[:, 1] >= y_min) & (region_cells[:, 1] < y_max)
            region_cells = region_cells[keep]
            region_cell_ids = region_cell_ids[keep]

        if DV_lim is not None:
            keep = (region_cells[:, 0] >= z_min) & (region_cells[:, 0] < z_max)
            region_cells = region_cells[keep]
            region_cell_ids = region_cell_ids[keep]

    if region_cells.shape[0] == 0:
        region_cells = None
        region_cell_ids = None
        if verbose:
            print(f"⚠️ No cells found in region {region_name or target_region_ids} at {workDir}, only plotting background.")

    if region_cells is not None:
        all_points = np.vstack([verts, region_cells[:, [2, 1, 0]]])
    else:
        all_points = verts

    xmid, ymid, zmid = np.median(all_points[:, 0]), np.median(all_points[:, 1]), np.median(all_points[:, 2])
    if crop_size is not None:
        dx, dy, dz = crop_size
        xlim = (xmid - dx, xmid + dx)
        ylim = (ymid - dy, ymid + dy)
        zlim = (zmid - dz, zmid + dz)
    else:
        margin = 5
        xlim = (all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ylim = (all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        zlim = (all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

    if verbose:
        n_cells = 0 if region_cells is None else region_cells.shape[0]
        print(f"[main] verts={verts.shape[0]}, faces={faces.shape[0]}, region_cells={n_cells}")
        print(f"[colors] background_color={background_color}, aux_color={aux_color} -> AUX={AUX}")

    if viewpoints is None:
        viewpoints = [viewpoint]
    else:
        viewpoints = list(viewpoints)
        if len(viewpoints) == 0:
            viewpoints = [viewpoint]

    save_path = os.path.join(workDir, f"{save_dir}/{proj}")
    os.makedirs(save_path, exist_ok=True)
    tag = f"{region_name}_{hemisphere}" if region_name else "_".join(map(str, target_region_ids))

    saved_files = []

    for vp in viewpoints:
        elev, azim = vp
        cam_vec = camera_vector_from_viewpoint(elev, azim)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('ortho')
        ax.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)

        if show_surface:
            mesh = Poly3DCollection(
                verts[faces],
                alpha=float(alpha_bg),
                facecolor=main_surface_color_eff,
                edgecolor='none'
            )
            if bool(pdf_rasterize_surface):
                mesh.set_rasterized(True)
            ax.add_collection3d(mesh)

        if feature_lines:
            crease_th_cos = np.cos(np.deg2rad(float(crease_deg)))
            _draw_feature_lines(
                ax=ax,
                verts=verts,
                faces=faces,
                cam_vec=cam_vec,
                draw_silhouette=bool(silhouette),
                draw_crease=bool(crease),
                crease_th_cos=crease_th_cos,
                color=line_color_eff,
                lw=float(line_width),
                alpha=float(line_alpha),
                max_segments=max_line_segments,
            )

        if outer_outline:
            _draw_outer_outline(
                ax=ax,
                verts=verts,
                cam_vec=cam_vec,
                color=outer_outline_color_eff,
                lw=float(outer_outline_width),
                alpha=float(outer_outline_alpha),
            )

        if show_points and region_cells is not None:
            pts_xyz = region_cells[:, [2, 1, 0]].astype(float)
            region_cell_ids_eff = region_cell_ids.copy()

            if point_rearrange is not None and str(point_rearrange).lower() != "none":
                mode = str(point_rearrange).lower()
                if mode == "density_jitter":
                    rng = np.random.default_rng(random_seed)
                    pts_xyz = _density_jitter_points(
                        pts_xyz,
                        sigma_vox=float(density_sigma_vox),
                        base_vox=float(jitter_base_vox),
                        alpha=float(jitter_alpha),
                        axes=tuple(jitter_axes),
                        clip_vox=float(jitter_clip_vox) if jitter_clip_vox is not None else None,
                        floor=float(density_floor),
                        rng=rng,
                        use_bbox=bool(density_use_region_bbox),
                        verbose=verbose,
                    )
                else:
                    raise ValueError("point_rearrange must be 'none' or 'density_jitter'")

            if point_groups is None:
                ax.scatter(
                    pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2],
                    s=float(point_size), c=pts_col_eff, alpha=1.0, linewidths=0
                )
            else:
                mode = str(points_show_mode).lower().strip()
                if mode not in ("all_on", "all_off"):
                    raise ValueError("points_show_mode must be 'all_on' or 'all_off'")
                default_visible = (mode == "all_on")
                assigned = np.zeros(len(pts_xyz), dtype=bool)

                for g in point_groups:
                    gname = str(g.get("name", "group")).strip()
                    gids = _as_set_ids(g.get("ids", []))
                    gcol = _normalize_mpl_color(g.get("color", pts_col_eff))
                    gshow = bool(g.get("show", default_visible))

                    if len(gids) == 0:
                        continue

                    sel = np.isin(region_cell_ids_eff, list(gids))
                    if not np.any(sel):
                        if verbose:
                            print(f"[points] group '{gname}' has 0 points, skipped")
                        continue

                    assigned |= sel

                    if gshow:
                        ax.scatter(
                            pts_xyz[sel, 0], pts_xyz[sel, 1], pts_xyz[sel, 2],
                            s=float(point_size), c=gcol, alpha=1.0, linewidths=0
                        )

                    if verbose:
                        print(f"[points] group '{gname}': n={int(sel.sum())}, show={gshow}, color={gcol}")

                other_sel = ~assigned
                if np.any(other_sel):
                    other_color = pts_col_eff if (points_other_color is None) else _normalize_mpl_color(points_other_color)
                    other_show = bool(points_show_other) and default_visible
                    if other_show:
                        ax.scatter(
                            pts_xyz[other_sel, 0], pts_xyz[other_sel, 1], pts_xyz[other_sel, 2],
                            s=float(point_size), c=other_color, alpha=1.0, linewidths=0
                        )
                    if verbose:
                        print(f"[points] '{points_unmatched_name}': n={int(other_sel.sum())}, show={other_show}, color={other_color}")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_box_aspect([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((0, 0, 0, 0))
            axis.set_tick_params(colors=(0, 0, 0, 0))
            axis.line.set_color((0, 0, 0, 0))
        ax.grid(False)

        scalebar_vox = float(scalebar_um) / float(voxel_um)
        bar_origin = [
            xlim[1] - 0.10 * (xlim[1] - xlim[0]) - scalebar_vox,
            ylim[0] + 0.05 * (ylim[1] - ylim[0]),
            zlim[0] + 0.05 * (zlim[1] - zlim[0])
        ]
        bar_end = [bar_origin[0] + scalebar_vox, bar_origin[1], bar_origin[2]]
        ax.plot(
            [bar_origin[0], bar_end[0]],
            [bar_origin[1], bar_end[1]],
            [bar_origin[2], bar_end[2]],
            color=AUX, linewidth=3
        )
        ax.text(
            bar_origin[0] + scalebar_vox / 2,
            bar_origin[1] - 0.02 * (ylim[1] - ylim[0]),
            bar_origin[2],
            f"{int(scalebar_um)} μm",
            color=AUX,
            ha='center', va='top', fontsize=10
        )

        if arrow:
            arrow_len = 0.1 * (xlim[1] - xlim[0])
            arrow_head_len = arrow_len * 0.8
            arrow_tail_len = arrow_len * 0.3
            arrow_tail_full = arrow_len * 0.8
            origin = [
                xlim[0] + 0.1 * (xlim[1] - xlim[0]),
                ylim[0] + 0.1 * (ylim[1] - ylim[0]),
                zlim[1] - 0.1 * (zlim[1] - zlim[0])
            ]

            if hemisphere == 'full':
                ax.plot([origin[0], origin[0]], [origin[1], origin[1] + arrow_tail_full], [origin[2], origin[2]], color=AUX, linewidth=1)
                ax.quiver(origin[0], origin[1], origin[2], 0, -arrow_head_len, 0, color=AUX, linewidth=1, arrow_length_ratio=0.2)
                ax.text(origin[0], origin[1] - arrow_head_len - 4, origin[2] - 4, "A", color=AUX, ha='center', va='bottom')
                ax.text(origin[0], origin[1] + arrow_tail_full + 4, origin[2] + 4, "P", color=AUX, ha='center', va='top')
            else:
                ax.plot([origin[0], origin[0]], [origin[1], origin[1] + arrow_tail_len], [origin[2], origin[2]], color=AUX, linewidth=1)
                ax.quiver(origin[0], origin[1], origin[2], 0, -arrow_head_len, 0, color=AUX, linewidth=1, arrow_length_ratio=0.2)
                ax.text(origin[0], origin[1] - arrow_head_len - 4, origin[2] - 4, "A", color=AUX, ha='center', va='bottom')
                ax.text(origin[0], origin[1] + arrow_tail_len + 4, origin[2] + 4, "P", color=AUX, ha='center', va='top')

            if hemisphere == 'full':
                ax.plot([origin[0] - arrow_tail_full, origin[0]], [origin[1], origin[1]], [origin[2], origin[2]], color=AUX, linewidth=1)
                ax.quiver(origin[0], origin[1], origin[2], arrow_head_len, 0, 0, color=AUX, linewidth=1, arrow_length_ratio=0.2)
                ax.text(origin[0] + arrow_head_len + 2, origin[1], origin[2] - 1, "L", color=AUX, ha='left', va='center')
                ax.text(origin[0] - arrow_tail_full - 2, origin[1], origin[2] + 2, "R", color=AUX, ha='right', va='center')
            elif hemisphere == 'left':
                ax.plot([origin[0] - arrow_tail_len, origin[0]], [origin[1], origin[1]], [origin[2], origin[2]], color=AUX, linewidth=1)
                ax.quiver(origin[0], origin[1], origin[2], arrow_head_len, 0, 0, color=AUX, linewidth=1, arrow_length_ratio=0.2)
                ax.text(origin[0] + arrow_head_len + 1, origin[1], origin[2] - 1, "L", color=AUX, ha='left', va='center')
                ax.text(origin[0] - arrow_tail_len - 4, origin[1], origin[2] + 2, "M", color=AUX, ha='right', va='center')
            elif hemisphere == 'right':
                ax.plot([origin[0] - arrow_tail_len, origin[0]], [origin[1], origin[1]], [origin[2], origin[2]], color=AUX, linewidth=1)
                ax.quiver(origin[0], origin[1], origin[2], arrow_head_len, 0, 0, color=AUX, linewidth=1, arrow_length_ratio=0.2)
                ax.text(origin[0] + arrow_head_len + 1, origin[1], origin[2] - 1, "M", color=AUX, ha='left', va='center')
                ax.text(origin[0] - arrow_tail_len - 4, origin[1], origin[2] + 2, "L", color=AUX, ha='right', va='center')

            ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2] - arrow_tail_len, origin[2]], color=AUX, linewidth=1)
            ax.quiver(origin[0], origin[1], origin[2], 0, 0, arrow_head_len, color=AUX, linewidth=1, arrow_length_ratio=0.2)
            ax.text(origin[0], origin[1], origin[2] + arrow_head_len + 1, "D", color=AUX, ha='center', va='bottom')
            ax.text(origin[0], origin[1], origin[2] - arrow_tail_len - 1, "V", color=AUX, ha='center', va='top')

        ax.view_init(elev=elev, azim=azim)
        if region_name:
            ax.set_title(f"3D view: {region_name}", color=AUX)
        plt.tight_layout()

        vp_tag = f"elev{elev}_azim{azim}"
        png_path = os.path.join(save_path, f"{save_prefix}_{tag}_{vp_tag}.png")
        pdf_path = os.path.join(save_path, f"{save_prefix}_{tag}_{vp_tag}.pdf")

        if save_png:
            plt.savefig(
                png_path,
                dpi=300,
                facecolor=background_color,
                bbox_inches='tight',
                pad_inches=0
            )
            saved_files.append(png_path)

        if save_pdf:
            plt.savefig(
                pdf_path,
                dpi=300,
                facecolor=background_color,
                bbox_inches='tight',
                pad_inches=0
            )
            saved_files.append(pdf_path)

        if verbose:
            print(f"✅ saved viewpoint {vp} -> {save_path}")

        if show_figure:
            plt.show()
        else:
            plt.close(fig)

    return saved_files

roi_defs = [
    {"name":"Fiber tracts", "ids":[
        1009,967,840,1016,21,665,538,900,901,93,229,794,832,158,62,
        848,916,336,117,125,917,237,933,948,482,506,658,841,413,949,
        792,932,514,380,697,798,1116,911,991,768,301,484682528,1099,
        618,449,443,530,466,603,737,428,436,940,908,884,824,54,1083,
        611,802,595,349,46,753,673,690,681,484682512,960,728,744,752,
        78,1123,553,326,866,812,850,983,896,1092,484682520,484682524,
        784,190,198,6,924,776,971,956,579,964,986,1108,484682516,1000,
        760,102,863,397,877,1043,1060
    ], "color":"#8000FF", "show": False},

    {"name":"Cerebellum", "ids":[
        512,528,645,951,920,976,984,968,928,1091,936,912,944,957,1073,1041,
        1025,1033,1007,1017,1056,1064,1049,519,589508455,989,91,846
    ], "color":"#C8C8C8", "show": True},

    {"name":"Cerebral cortex", "ids":[
        688,703,780,583,295,303,311,451,942,966,952,319,327,334,131,
        695,315,669,312782628,312782640,312782652,312782648,312782636,
        312782644,312782632,385,593,305,778,721,821,33,394,401,1046,
        1066,441,281,433,402,601,905,1114,649,1074,233,533,41,257,469,
        565,805,501,425,902,377,750,869,393,269,312782574,312782594,312782590,
        312782586,312782578,312782582,312782598,
        409,74,973,613,121,421,573,247,1011,156,678,600,527,243,252,1002,251,
        1005,954,847,735,816,1018,520,990,598,1023,755,959,1027,791,249,696,
        643,456,759,1057,180,638,662,187,148,36,714,746,1125,608,969,680,288,
        731,620,910,527696977,582,484,723,412,448,630,440,488,972,171,84,304,
        132,363,541,289,1127,97,729,234,786,44,827,1054,556,1081,707,184,526157196,
        667,68,526322264,526157192,22,417,312782604,312782616,312782620,312782612,
        312782608,312782624,312782546,312782554,312782558,312782566,312782562,312782570,
        312782550,31,48,588,819,296,810,772,39,919,211,935,1015,927,677,897,1058,849,
        1010,1106,857,95,119,704,675,694,800,699,111,314,163,355,344,120,104,783,996,
        1101,328,831,922,540,692,888,335,368,453,378,873,862,1090,893,1035,806,322,
        337,478,1094,1128,113,510,1030,182305689,182305713,182305701,182305709,
        182305697,182305705,182305693,345,1102,878,950,2,974,657,329,1038,1062,201,
        981,1070,1047,353,889,929,654,702,558,838,361,9,1086,1006,1111,461,670,369,
        450,1026,625,945,577,854,254,879,545,442,330,434,274,610,894,906,965,671,279,
        774,886,542,430,687,622,590,500,993,1085,1021,767,962,656,985,648,844,882,320,
        943,895,977,988,836,1045,427,1089,1080,375,463,382,423,982,726,10703,10704,632,
        19,822,502,484682508,1084,909,918,28,20,139,52,1121,926,727,664,743,526,543,843,
        1037,484682470,589508447,698,566,814,788,961,159,619,1139,268,260,631,639,647,663,655,507,589,597,605,151,188,196,204
    ], "color":"#3E77B5", "show": True},

    {"name":"Hypothalamus", "ids":[
        1097,157,223,332,38,30,118,390,290,576073704,194,356,364,173,797,804,614,470,
        226,141,286,689,763,272,523,72,830,338,126,1109,452,347,263,914,576073699,133,
        467,63,980,331,525,491,606826647,606826655,606826659,606826651,732,210,557,1126,1,515,693,1004,88,946,10671
    ], "color":"#479FB3", "show": True},

    {"name":"Thalamus", "ids":[
        549,856,138,1020,560581551,325,1029,218,239,127,1096,1104,155,1113,1120,255,
        64,262,51,189,930,560581563,599,907,575,958,483,186,571,560581559,15,149,181,
        444,362,366,59,1077,1014,178,27,563807439,321,864,609,406,414,422,637,563807435,
        709,741,733,725,718,629,685,1044,1008,170,496345664,496345668,496345672,475,1079,1072,1088
    ], "color":"#6EC5A4", "show": True},

    {"name":"RVM", "ids":[
        379,222,230,206,1048,1107,136,307,235,955,963,83,1069,978
    ], "color":"#C42C4B", "show": True},

    {"name":"PARN", "ids":[
        852
    ], "color":"#A1D9A4", "show": True},

    {"name":"MDRNd", "ids":[
        1098
    ], "color":"#CCEA9D", "show": True},

    {"name":"VNC", "ids":[
        701,225,209,217,202
    ], "color":"#ECF7A2", "show": True},

    {"name":"VII", "ids":[
        661
    ], "color":"#FEFEBD", "show": True},

    {"name":"DMX", "ids":[
        839
    ], "color":"#FEE899", "show": True},

    {"name":"MY-sen", "ids":[
        386,720,711,1039,207,651,607,96,101,642,429,903,437,445,589508451
    ], "color":"#FDCA78", "show": True},

    {"name":"Pons", "ids":[
        771,987,549009223,534,549009227,621,898,880,574,549009215,1093,280,
        931,549009219,318,599626927,1132,867,123,7,398,114,122,105,612,1117,
        679,358,146,350,147,162,238,604
    ], "color":"#FBA55C", "show": True},

    {"name":"MBmot", "ids":[
        323,616,606826663,607344830,549009211,757,75,231,66,128,214,749,381,58,
        115,294,17,42,26,10,975,1100,706,634,628,531,215,549009203,1061,246,795,614454277,67,587,50,35
    ], "color":"#F57446", "show": True},

    {"name":"MBsta", "ids":[
        348,1052,374,165,100,607344846,607344838,607344842,607344834,607344862,607344850,607344858,607344854,591,872,12,197
    ], "color":"#E1514A", "show": True},
]

saved = plot_region_3d_view_with_surface(
    workDir="path/to/cell_data/",
    cell_path_suffix="AAV_cells_atlas_voxels_25um.npy",
    target_region_ids=None,
    region_name="Whole",

    background_color="white",
    main_surface_color="#F0F0F0",
    pts_col="skyblue",                                                
    aux_color="auto",

    alpha_bg=0.20,
    point_size=4.0,

    feature_lines=True,
    silhouette=True,
    crease=True,
    crease_deg=5.0,
    line_color="auto",
    line_width=0.1,
    line_alpha=0.1,
    max_line_segments=300000,

    outer_outline=False,
    outer_outline_color="auto",
    outer_outline_width=0.6,
    outer_outline_alpha=0.1,

    point_rearrange="density_jitter",
    density_sigma_vox=2.0,
    jitter_base_vox=5.0,
    jitter_alpha=0,
    jitter_axes=(0, 1.0, 0),
    jitter_clip_vox=6.0,
    density_floor=1e-6,
    density_use_region_bbox=True,
    random_seed=0,

    arrow=True,

    point_groups=roi_defs,
    points_show_mode="all_off",
    points_show_other=False,

    viewpoints=[
        (0, 90),             
        (0, 0),             
        (90, 0),               
    ],

    save_png=True,
    save_pdf=True,
    show_figure=False,
    pdf_rasterize_surface=True,
    proj="tube4_static_multi_view",
    save_dir="3D.fig",
    save_prefix="Whole_static"
)

print("Saved files:")
for p in saved:
    print(p)
