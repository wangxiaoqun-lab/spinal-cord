import argparse
import csv
import json
import math
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from tqdm import tqdm
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from skimage.measure import find_contours, label as sk_label
from scipy.ndimage import gaussian_filter, zoom, binary_closing
from scipy import interpolate
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

orientation = "sagittal"

TREE_JSON = "path/to/atlas_index.json"
ANNO_TIFF = f"path/to/{orientation}_annotation.tiff"
OUT_DIR = "output/brain_zscore_heatmap"
REGION_ZSCORE_FILE = Path(__file__).with_name("brain_region_zscores.csv")

CMAP_NAME = "RdYlBu_r"

ZSCORE_VMIN = -1.0
ZSCORE_VMAX = 4.0

PRINT_COLOR_MATCH = True

ENABLE_REPORT = False

WHITELIST_ACRONYMS = None

BLACKLIST_ACRONYMS = set([

])

SKIP_IDS = set()

ALPHA = 0.45
UPSCALE = 3
SMOOTH_SIGMA = 1.2
SPLINE_SMOOTH = 1.0

SLICE_LIST = None
SLICE_RANGE = (110, 231, 1)

FIGSIZE = (8, 8)
DPI = 300

USE_FILL_COLOR = False
DRAW_OUTLINE = True

OUTLINE_COLOR = (0, 0, 0)
OUTLINE_WIDTH = 0.4

SHOW_LABELS = False
LABEL_HALF_MODE = "right"

LABEL_USE_ACRONYM = True
LABEL_FONTSIZE = 4

DEFAULT_UNKNOWN_COLOR = (180, 180, 180)

HIGHLIGHT_REGIONS = [
    {
        "name": "Cerebellum",
        "acronym": "CB",
        "ids": [976, 984, 1091, 957, 1056, 519],
        "color": (200, 200, 200),
        "alpha": 0.6,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "Cerebral cortex",
        "acronym": "CTX",
        "ids": [337, 361, 985],
        "color": (62, 119, 181),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "Hypothalamus",
        "acronym": "HY",
        "ids": [38, 290, 467],
        "color": (71, 159, 179),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "Thalamus",
        "acronym": "TH",
        "ids": [856, 864],
        "color": (110, 197, 164),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "RVM",
        "acronym": "RVM",
        "ids": [379, 222, 230, 206, 1048, 1107, 661, 136, 307, 235, 955, 963, 83, 1069, 978],
        "color": (196, 44, 75),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "PARN",
        "acronym": "PARN",
        "ids": [852],
        "color": (161, 217, 164),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "MDRNd",
        "acronym": "MDRNd",
        "ids": [1098],
        "color": (204, 234, 157),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "VNC",
        "acronym": "VNC",
        "ids": [225, 209, 217, 202],
        "color": (236, 247, 162),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "VII",
        "acronym": "VII",
        "ids": [661],
        "color": (254, 254, 189),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "DMX",
        "acronym": "DMX",
        "ids": [839],
        "color": (254, 232, 153),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "MY-sen",
        "acronym": "MY-sen",
        "ids": [711, 1039, 651, 429],
        "color": (253, 202, 120),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "Pons",
        "acronym": "P",
        "ids": [987, 1132, 358, 146, 350],
        "color": (251, 165, 92),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "MBmot",
        "acronym": "MBmot",
        "ids": [128, 214, 795],
        "color": (245, 116, 70),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
    {
        "name": "MBsta",
        "acronym": "MBsta",
        "ids": [1052],
        "color": (225, 81, 74),
        "alpha": 1.0,
        "draw_outline": False,
        "outline_color": (0, 0, 0),
        "outline_width": 0.6,
    },
]

def normalize_key(x):

    if x is None:
        return None

    if isinstance(x, float) and math.isnan(x):
        return None

    s = str(x).strip()

    if s == "":
        return None

    return s.lower()

def load_region_zscore_table(file_path):

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Z-score table not found: {file_path}")

    with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Brain Region", "Acronym", "Z score-mean"}
        missing = required.difference(reader.fieldnames or [])

        if missing:
            raise ValueError(f"Z-score table is missing columns: {sorted(missing)}")

        rows = []

        for line_number, row in enumerate(reader, start=2):
            region_name = (row.get("Brain Region") or "").strip() or None
            acronym = (row.get("Acronym") or "").strip() or None
            z_text = (row.get("Z score-mean") or "").strip()

            if region_name is None and acronym is None and z_text == "":
                continue
            if z_text == "":
                raise ValueError(f"Missing Z score at CSV line {line_number}")

            rows.append({
                "Brain Region": region_name,
                "Acronym": acronym,
                "Z score-mean": float(z_text),
            })

    if not rows:
        raise ValueError(f"Z-score table is empty: {file_path}")

    return rows

def load_region_color_mapping_from_table(region_zscore_table, cmap_name="RdYlBu", vmin=-1.0, vmax=4.0):

    if vmin >= vmax:
        raise ValueError(f"Invalid color range: vmin={vmin}, vmax={vmax}; vmin must be less than vmax.")

    if not isinstance(region_zscore_table, (list, tuple)) or len(region_zscore_table) == 0:
        raise ValueError("REGION_ZSCORE_TABLE must not be empty.")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cmap_name)

    name_to_color = {}
    acronym_to_color = {}
    name_to_z = {}
    acronym_to_z = {}

    for i, row in enumerate(region_zscore_table):

        if not isinstance(row, dict):
            raise TypeError(f"Item {i} in REGION_ZSCORE_TABLE is not a dict.")

        if "Z score-mean" not in row:
            raise KeyError(f"Item {i} in REGION_ZSCORE_TABLE is missing 'Z score-mean'.")

        region_name = normalize_key(row.get("Brain Region"))
        acronym = normalize_key(row.get("Acronym"))
        z = float(row["Z score-mean"])

        rgba = cmap(norm(z))
        rgb255 = tuple(int(round(c * 255)) for c in rgba[:3])

        if region_name is not None:
            name_to_color[region_name] = rgb255
            name_to_z[region_name] = z

        if acronym is not None:
            acronym_to_color[acronym] = rgb255
            acronym_to_z[acronym] = z

    return {
        "norm": norm,
        "cmap": cmap,
        "name_to_color": name_to_color,
        "acronym_to_color": acronym_to_color,
        "name_to_z": name_to_z,
        "acronym_to_z": acronym_to_z,
        "vmin": vmin,
        "vmax": vmax,
    }

def apply_zscore_colors_to_highlight_regions(highlight_regions, color_info):

    matched = []
    unmatched = []

    name_to_color = color_info["name_to_color"]
    acronym_to_color = color_info["acronym_to_color"]
    name_to_z = color_info["name_to_z"]
    acronym_to_z = color_info["acronym_to_z"]

    for h in highlight_regions:

        name_key = normalize_key(h.get("name"))
        acronym_key = normalize_key(h.get("acronym"))

        color = None
        z = None
        match_mode = None

        if name_key in name_to_color:
            color = name_to_color[name_key]
            z = name_to_z[name_key]
            match_mode = "Brain Region"
        elif acronym_key in acronym_to_color:
            color = acronym_to_color[acronym_key]
            z = acronym_to_z[acronym_key]
            match_mode = "Acronym"

        if color is not None:
            h["color"] = color
            matched.append((h.get("name"), h.get("acronym"), z, color, match_mode))
        else:
            unmatched.append((h.get("name"), h.get("acronym")))

    if PRINT_COLOR_MATCH:
        print("\n" + "=" * 80)
        print("[Z-SCORE COLOR MATCH]")
        for item in matched:
            print(f"matched | name={item[0]!r} | acronym={item[1]!r} | z={item[2]:.6f} | color={item[3]} | by={item[4]}")
        if unmatched:
            print("-" * 80)
            print("[UNMATCHED REGIONS]")
            for item in unmatched:
                print(f"unmatched | name={item[0]!r} | acronym={item[1]!r}")
        print("=" * 80 + "\n")

    return highlight_regions

def load_tree_index(tree_json):

    with open(tree_json, "r") as f:
        tree = json.load(f)

    id2node = {}

    def walk(node):

        sid = node.get("id")

        if sid is not None:
            id2node[int(sid)] = node

        for ch in node.get("children", []):
            walk(ch)

    if "root" in tree:
        walk(tree["root"])
    else:
        for r in tree.get("roots", []):
            walk(r)

    return id2node

def report_slice_header(z, lab):

    if not ENABLE_REPORT:
        return

    unique_ids = np.unique(lab)

    print("\n" + "=" * 70)
    print(f"[SLICE] z = {z}")
    print(f"[SLICE] shape = {lab.shape}")
    print(f"[SLICE] unique id count = {len(unique_ids)}")
    print(f"[SLICE] first 20 ids = {unique_ids[:20].tolist()}")
    print("=" * 70)

def format_slice_tag():

    if SLICE_LIST is not None:

        z = [int(x) for x in SLICE_LIST]

        if len(z) <= 20:
            return "list_" + "_".join(map(str, z))

        return f"list_n{len(z)}_{min(z)}to{max(z)}"

    start, stop, step = SLICE_RANGE

    return f"range_{start}_{stop}_{step}"

def format_output_tag():

    parts = [orientation, "atlas", format_slice_tag()]

    parts.append("fill" if USE_FILL_COLOR else "nofill")
    parts.append("outline" if DRAW_OUTLINE else "nooutline")
    parts.append(CMAP_NAME)

    return "_".join(parts)

def smooth_contour(cnt):

    if len(cnt) < 5:
        return cnt

    x, y = cnt[:, 1], cnt[:, 0]

    try:

        tck, _ = interpolate.splprep([x, y], s=SPLINE_SMOOTH, per=True)

        u_new = np.linspace(0, 1, 300)

        x_new, y_new = interpolate.splev(u_new, tck)

        return np.column_stack([y_new, x_new])

    except:

        return cnt

def largest_component_centroid_half(mask, mode=None):

    if not mask.any():
        return None

    H, W = mask.shape

    submask = mask.copy()

    if mode == "left":
        submask[:, int(W / 2):] = False
    elif mode == "right":
        submask[:, :int(W / 2)] = False
    elif mode == "upper":
        submask[int(H / 2):, :] = False
    elif mode == "lower":
        submask[:int(H / 2), :] = False

    if not submask.any():
        submask = mask

    lab = sk_label(submask)

    if lab.max() == 0:
        return None

    areas = np.bincount(lab.ravel())
    areas[0] = 0

    k = areas.argmax()

    ys, xs = np.where(lab == k)

    return (xs.mean(), ys.mean())

def mask_to_patches(mask, rgb, alpha=0.4, use_fill=True):

    patches = []

    if not mask.any():
        return patches

    mask = np.pad(mask, 2)

    mask = binary_closing(mask, structure=np.ones((3, 3)))

    arr = zoom(mask.astype(float), UPSCALE, order=1)

    arr = gaussian_filter(arr, sigma=SMOOTH_SIGMA * UPSCALE)

    contours = find_contours(arr, 0.5)

    for cnt in contours:

        cnt = cnt / UPSCALE

        cnt[:, 0] -= 2
        cnt[:, 1] -= 2

        cnt = smooth_contour(cnt)

        if len(cnt) < 5:
            continue

        verts = [(c[1], c[0]) for c in cnt]

        verts.append(verts[0])

        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]

        path = MplPath(verts, codes)

        if use_fill:
            face = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha)
        else:
            face = "none"

        patch = PathPatch(
            path,
            facecolor=face,
            edgecolor=(0, 0, 0, 1),
            linewidth=OUTLINE_WIDTH,
            antialiased=True
        )

        patches.append(patch)

    return patches

def build_z_indices(Z):

    if SLICE_LIST is not None:

        return sorted(set(int(x) for x in SLICE_LIST if 0 <= int(x) < Z))

    start, stop, step = SLICE_RANGE

    if start is None:
        start = 0
    if stop is None:
        stop = Z
    if step is None:
        step = 1

    return list(range(start, stop, step))

def main(region_zscore_file=REGION_ZSCORE_FILE):

    region_zscore_table = load_region_zscore_table(region_zscore_file)

    color_info = load_region_color_mapping_from_table(
        region_zscore_table=region_zscore_table,
        cmap_name=CMAP_NAME,
        vmin=ZSCORE_VMIN,
        vmax=ZSCORE_VMAX,
    )
    apply_zscore_colors_to_highlight_regions(HIGHLIGHT_REGIONS, color_info)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    id2node = load_tree_index(TREE_JSON)

    print("Loaded nodes:", len(id2node))

    anno = tiff.imread(ANNO_TIFF)

    if anno.ndim == 2:
        anno = anno[None, ...]

    Z, H, W = anno.shape

    z_indices = build_z_indices(Z)

    pdf_name = f"{format_output_tag()}.pdf"

    pdf_path = out_dir / pdf_name

    with PdfPages(pdf_path) as pdf:

        for z in tqdm(z_indices):

            lab = anno[z]

            report_slice_header(z, lab)

            fig, ax = plt.subplots(figsize=FIGSIZE)

            if HIGHLIGHT_REGIONS:

                for h in HIGHLIGHT_REGIONS:

                    ids_input = [int(x) for x in h["ids"]]

                    if ENABLE_REPORT:

                        print("\n[HIGHLIGHT GROUP]")
                        print("name:", h["name"])
                        print("input_ids:", ids_input)

                    ids_expand = []

                    for sid in ids_input:

                        node = id2node.get(sid)

                        if node and "mapped_ids" in node:
                            ids_expand.extend(node["mapped_ids"])
                        else:
                            ids_expand.append(sid)

                    ids_arr = np.asarray(list(set(ids_expand)), dtype=lab.dtype)

                    mask = np.isin(lab, ids_arr)

                    if ENABLE_REPORT:

                        if mask.any():

                            hit_ids = np.unique(lab[mask]).tolist()

                            print("hit_ids:", hit_ids)
                            print("mask_pixels:", int(mask.sum()))

                        else:

                            print("hit_ids: []")
                            print("mask_pixels: 0")

                    patches = mask_to_patches(mask, h["color"], alpha=h["alpha"])

                    for p in patches:
                        ax.add_patch(p)

            ids_slice = np.unique(lab)

            for sid_raw in ids_slice:

                sid = int(sid_raw)

                if sid in SKIP_IDS:
                    continue

                mask = (lab == sid_raw)

                if not mask.any():
                    continue

                node = id2node.get(sid)

                if node:

                    acr = node.get("acronym", "")
                    rgb = node.get("rgb_triplet") or DEFAULT_UNKNOWN_COLOR

                else:

                    acr = ""
                    rgb = DEFAULT_UNKNOWN_COLOR

                if WHITELIST_ACRONYMS and acr not in WHITELIST_ACRONYMS:
                    continue

                if BLACKLIST_ACRONYMS and acr in BLACKLIST_ACRONYMS:
                    continue

                patches = mask_to_patches(mask, rgb, alpha=ALPHA, use_fill=USE_FILL_COLOR)

                for p in patches:
                    ax.add_patch(p)

                if SHOW_LABELS:

                    c = largest_component_centroid_half(mask, LABEL_HALF_MODE)

                    if c:

                        label = acr if acr else str(sid)

                        ax.text(
                            c[0], c[1], label,
                            fontsize=LABEL_FONTSIZE,
                            ha="center",
                            va="center"
                        )

            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

            ax.set_aspect("equal")

            ax.axis("off")

            ax.set_title(f"{orientation} slice {z}")

            pdf.savefig(fig, dpi=DPI, bbox_inches="tight", pad_inches=0)

            plt.close(fig)

    print("PDF saved:", pdf_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render atlas slices using an external regional Z-score table.")
    parser.add_argument(
        "--zscore-table",
        type=Path,
        default=REGION_ZSCORE_FILE,
        help="CSV containing Brain Region, Acronym, and Z score-mean columns.",
    )
    args = parser.parse_args()
    main(args.zscore_table)
