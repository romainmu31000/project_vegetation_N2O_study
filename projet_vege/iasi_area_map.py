"""
Interactive world map of an IASI L2 file. Drag a box on the map to select an area:
statistics for the selected points appear under the map. No command line needed:
edit the SETTINGS below and press Run.

Install once:
    python -m pip install xarray netcdf4 h5netcdf h5py numpy pandas matplotlib
Optional, real coastlines (otherwise a built-in coarse outline is used):
    python -m pip install cartopy

Keys in the map window
    drag mouse : select an area (the box can be moved / resized afterwards)
    s          : save the selected points to a CSV file
    z          : zoom the map to the selection
    r          : reset to the whole world
"""
import operator
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import RectangleSelector

# =============================== SETTINGS ===================================
FILE = r"C:\Users\romai\Desktop\projet_vege\iasi_ret_L2_n2o_ch4_2020_07_20_v2.02.nc"
OUTPUT_DIR = r"C:\Users\romai\Desktop\projet_vege"

VARIABLE = None          # variable to map; None = auto-pick (see the printed list)
LEVEL = None             # index along an extra dimension (vertical level); None = average
QUALITY_FILTER = None    # e.g. ("quality_flag", "==", 0)   or   ("cost", "<", 2)
INITIAL_BBOX = None      # preselected area (north, west, south, east), e.g. (45, -0.5, 42.2, 4.9)
MAX_DRAWN_POINTS = 300_000   # display limit only; statistics always use all points

DISPLAY = "points"         # "grid"   = mean value per lat/lon cell, semi-transparent (best when dense)
                         # "points" = raw points, semi-transparent
GRID_DEG = 0.1           # cell size in degrees for DISPLAY = "grid" (0.5 = finer, 2 = coarser)
POINT_ALPHA = 0.55       # transparency of the data layer (0 = invisible, 1 = opaque)
CMAP = "turbo"           # colormap of the values

VIEW_EXTENT = (70, 110, 10, 150)   # map window shown: (north, west, south, east) in degrees
                                   # = latitude 10N-70N, longitude 110E-150E
                                   # None = whole world
# ============================================================================

LAT_NAMES = ["latitude", "lat", "pixel_latitude", "lat_deg", "latitudes"]
LON_NAMES = ["longitude", "lon", "pixel_longitude", "lon_deg", "longitudes"]
TIME_NAMES = ["time", "datetime", "date_time", "obs_time", "time_utc"]

# Coarse continent outlines (lon, lat), used when cartopy is not installed
FALLBACK_LAND = [
    [(-168,66),(-162,70),(-156,71),(-141,70),(-128,70),(-115,68),(-95,72),(-85,70),(-82,63),(-93,59),(-94,52),(-82,52),(-79,57),(-77,62),(-70,60),(-62,57),(-56,52),(-60,47),(-66,45),(-70,42),(-74,40),(-76,35),(-81,31),(-80,25),(-83,29),(-90,30),(-97,27),(-97,22),(-94,18),(-90,21),(-87,21),(-88,16),(-84,15),(-83,10),(-79,9),(-80,7),(-84,9),(-88,13),(-95,16),(-105,20),(-106,23),(-112,29),(-115,31),(-110,23),(-112,25),(-117,32),(-121,35),(-124,40),(-124,47),(-128,51),(-135,58),(-142,60),(-150,60),(-158,57),(-165,55),(-158,59),(-165,62),(-166,65)],
    [(-80,8),(-77,8),(-72,12),(-62,10),(-52,5),(-50,0),(-44,-2),(-35,-5),(-39,-14),(-41,-22),(-48,-26),(-53,-34),(-58,-38),(-62,-40),(-65,-47),(-69,-52),(-68,-55),(-73,-50),(-74,-40),(-71,-30),(-70,-18),(-76,-14),(-81,-6),(-80,-2),(-78,2),(-77,5)],
    [(-73,78),(-60,82),(-30,83),(-20,80),(-18,72),(-25,68),(-40,65),(-44,60),(-50,62),(-54,67),(-58,75)],
    [(-17,21),(-13,28),(-9,32),(-6,36),(10,37),(11,33),(20,31),(32,31),(35,28),(43,12),(51,12),(48,5),(40,-3),(40,-15),(35,-24),(32,-29),(27,-34),(19,-35),(15,-27),(12,-17),(13,-8),(9,-1),(9,4),(4,6),(-8,4),(-13,8),(-17,14)],
    [(-9,37),(-9,43),(-2,44),(-4,48),(2,51),(8,54),(9,57),(11,55),(14,54),(21,55),(24,60),(30,60),(22,65),(25,66),(18,63),(17,57),(11,59),(5,62),(14,68),(25,71),(40,68),(44,67),(60,70),(70,73),(80,73),(100,77),(113,74),(130,71),(150,70),(170,70),(180,68),(180,65),(170,60),(163,59),(156,51),(155,58),(142,59),(137,54),(141,48),(135,43),(129,41),(127,35),(122,40),(118,38),(121,31),(122,26),(117,23),(108,21),(106,18),(109,12),(105,9),(100,13),(103,2),(100,7),(98,16),(94,17),(91,23),(87,21),(80,15),(77,8),(73,17),(70,21),(67,25),(57,26),(56,24),(59,22),(52,17),(45,13),(39,22),(35,28),(34,31),(36,36),(30,36),(27,37),(26,40),(23,40),(22,37),(19,40),(18,42),(13,45),(12,44),(16,41),(18,40),(16,38),(12,42),(8,44),(3,43),(0,39),(-2,37),(-5,36)],
    [(114,-22),(122,-18),(130,-12),(136,-12),(137,-16),(142,-11),(146,-19),(153,-26),(151,-34),(147,-38),(141,-38),(135,-34),(130,-32),(124,-34),(115,-34),(114,-27)],
    [(-5,50),(1,51),(2,53),(-2,56),(-2,58),(-6,58),(-5,54),(-3,53),(-5,52)],
    [(-10,52),(-6,52),(-6,55),(-10,54)],
    [(130,31),(135,34),(140,36),(142,40),(142,45),(140,41),(136,37),(131,34)],
    [(44,-25),(48,-25),(50,-15),(49,-12),(44,-17)],
    [(95,5),(104,-2),(106,-6),(101,-3),(96,2)],
    [(109,1),(117,7),(119,1),(116,-4),(110,-3)],
    [(131,-1),(141,-3),(150,-10),(141,-9),(135,-4)],
    [(172,-34),(178,-38),(175,-41),(172,-41)],
    [(-180,-72),(-120,-74),(-60,-64),(-58,-72),(0,-70),(60,-67),(120,-66),(160,-70),(180,-68),(180,-90),(-180,-90)],
]


# ----------------------------------------------------------------------------
# Reading the file
# ----------------------------------------------------------------------------
def open_dataset(path):
    last = None
    for engine in (None, "netcdf4", "h5netcdf"):
        for decode_times in (True, False):
            try:
                return xr.open_dataset(path, engine=engine, decode_times=decode_times)
            except Exception as exc:
                last = exc
    sys.exit(f"Could not open {path}\nLast error: {last}\n\n"
             "If it says a backend is missing, run:\n"
             "    python -m pip install netcdf4 h5netcdf h5py")


def find_var(ds, names, standard_names=(), units=()):
    lower = {k.lower(): k for k in ds.variables}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for k, v in ds.variables.items():
        if v.attrs.get("standard_name", "") in standard_names or v.attrs.get("units", "") in units:
            return k
    return None


def print_info(ds):
    print("\n=== DIMENSIONS ===")
    for d, n in ds.sizes.items():
        print(f"  {d}: {n}")
    print("\n=== VARIABLES (name | dims | units | description) ===")
    for name, v in ds.variables.items():
        print(f"  {name:32s} {str(v.dims):36s} {v.attrs.get('units', ''):10s} {v.attrs.get('long_name', '')}")


def apply_filter(ds, flt):
    ops = {">=": operator.ge, "<=": operator.le, "==": operator.eq,
           "!=": operator.ne, ">": operator.gt, "<": operator.lt}
    name, op, value = flt
    cond = ops[op](ds[name], value)
    if cond.ndim == 1:
        return ds.isel({cond.dims[0]: np.flatnonzero(cond.values)})
    return ds.where(cond)


def auto_variable(ds, geo_dims, skip):
    for k, v in ds.data_vars.items():
        if k in skip or not np.issubdtype(v.dtype, np.floating):
            continue
        if v.ndim > 0 and set(geo_dims) <= set(v.dims):
            return k
    return None


def reduce_extra_dims(var, geo_dims, level):
    extra = [d for d in var.dims if d not in geo_dims]
    if extra and level is not None:
        var = var.isel({extra[0]: level})
        print(f"Using index {level} of dimension '{extra[0]}'")
        extra = extra[1:]
    if extra:
        print(f"Averaging over {extra} (set LEVEL to pick one)")
        var = var.mean(extra, skipna=True)
    return var


def load_points(file_path=None, variable=None, level=LEVEL, quality_filter=QUALITY_FILTER,
                missing_values=()):
    """Load geolocated values; optional arguments let other maps reuse this loader."""
    ds = open_dataset(FILE if file_path is None else file_path)
    print_info(ds)
    latn = find_var(ds, LAT_NAMES, ("latitude",), ("degrees_north", "degree_north"))
    lonn = find_var(ds, LON_NAMES, ("longitude",), ("degrees_east", "degree_east"))
    timen = find_var(ds, TIME_NAMES, ("time",))
    print(f"\nDetected -> latitude: {latn} | longitude: {lonn} | time: {timen}")
    if not (latn and lonn):
        sys.exit("Latitude/longitude not detected: edit LAT_NAMES / LON_NAMES using the list above.")
    geo_dims = sorted(set(ds[latn].dims) | set(ds[lonn].dims))

    if quality_filter:
        ds = apply_filter(ds, quality_filter)
        print(f"After filter: {dict(ds.sizes)}")

    name = variable or VARIABLE or auto_variable(ds, geo_dims, {latn, lonn, timen})
    if name is None or name not in ds.variables:
        sys.exit("No variable selected: set VARIABLE using the list above.")
    print(f"\nVariable mapped: {name}" + ("" if variable or VARIABLE else "  (auto-picked, set VARIABLE to change)"))

    var = ds[name]
    if missing_values:
        var = var.where(~var.isin(missing_values))
    var = reduce_extra_dims(var, geo_dims, level)
    lat, lon = ds[latn], ds[lonn]
    t = None
    if lat.ndim == 1 and lon.ndim == 1 and lat.dims != lon.dims:       # regular grid
        var = var.transpose(lat.dims[0], lon.dims[0])
        lon2, lat2 = np.meshgrid(lon.values, lat.values)
        x, y, c = lon2.ravel(), lat2.ravel(), var.values.ravel()
    else:                                                               # one lat/lon per observation
        if lat.ndim > 1:
            var = var.transpose(*lat.dims)
        x, y, c = lon.values.ravel(), lat.values.ravel(), var.values.ravel()
        if timen and ds[timen].dims == lat.dims:
            t = ds[timen].values.ravel()
    x = ((x + 180) % 360) - 180
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if not ok.any():
        sys.exit("All values are NaN: nothing to show.")
    units = ds[name].attrs.get("units", "")
    return x[ok], y[ok], c[ok], (t[ok] if t is not None else None), name, units


# ----------------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------------
def make_axes():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(12, 7.2))
        ax = fig.add_axes([0.05, 0.16, 0.82, 0.78], projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor="#eef4f9", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#e4dfd0", zorder=0)
        # coastlines and borders are drawn ABOVE the data, with a white halo,
        # so the world map stays readable whatever the point density
        ax.add_feature(cfeature.COASTLINE, edgecolor="white", lw=2.6, zorder=6)
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", lw=0.8, zorder=7)
        ax.add_feature(cfeature.BORDERS, edgecolor="white", lw=1.6, zorder=6)
        ax.add_feature(cfeature.BORDERS, edgecolor="#222222", lw=0.4, zorder=7)
        ax.gridlines(draw_labels=True, lw=0.3, alpha=0.5)
        return fig, ax, ccrs.PlateCarree()
    except Exception as exc:
        print(f"(cartopy not used: {exc.__class__.__name__}; using the coarse built-in outline)")
        fig = plt.figure(figsize=(12, 7.2))
        ax = fig.add_axes([0.05, 0.16, 0.82, 0.78])
        ax.set_facecolor("#eef4f9")
        for poly in FALLBACK_LAND:
            ax.add_patch(Polygon(poly, closed=True, fc="#e4dfd0", ec="none", zorder=1))
            ax.add_patch(Polygon(poly, closed=True, fill=False, ec="white", lw=2.6, zorder=6))
            ax.add_patch(Polygon(poly, closed=True, fill=False, ec="black", lw=0.8, zorder=7))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)
        return fig, ax, None


def main():
    x, y, c, t, name, units = load_points()
    label = f"{name} [{units}]" if units else name
    print(f"\n{len(x):,} valid points loaded.")

    fig, ax, transform = make_axes()
    vmin, vmax = np.percentile(c, [2, 98])
    kw = {"transform": transform} if transform else {}
    if DISPLAY == "grid":
        # mean value per lat/lon cell: dense data never turns into a solid blob
        nx, ny = int(np.ceil(360 / GRID_DEG)), int(np.ceil(180 / GRID_DEG))
        gx = np.clip(np.floor((x + 180) / GRID_DEG).astype(int), 0, nx - 1)
        gy = np.clip(np.floor((y + 90) / GRID_DEG).astype(int), 0, ny - 1)
        flat = gy * nx + gx
        total = np.bincount(flat, weights=c, minlength=nx * ny)
        count = np.bincount(flat, minlength=nx * ny)
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan).reshape(ny, nx)
        lons = -180 + GRID_DEG * np.arange(nx + 1)
        lats = -90 + GRID_DEG * np.arange(ny + 1)
        sc = ax.pcolormesh(lons, lats, np.ma.masked_invalid(mean), cmap=CMAP, vmin=vmin,
                           vmax=vmax, alpha=POINT_ALPHA + 0.15, zorder=3, rasterized=True, **kw)
        print(f"Grid display: {int((count > 0).sum()):,} cells of {GRID_DEG} deg (statistics use all points).")
    else:
        idx = np.arange(len(x))
        if len(x) > MAX_DRAWN_POINTS:
            idx = idx[:: int(np.ceil(len(x) / MAX_DRAWN_POINTS))]
            print(f"Drawing {len(idx):,} of {len(x):,} points (statistics use all).")
        sc = ax.scatter(x[idx], y[idx], c=c[idx], s=4, cmap=CMAP, vmin=vmin, vmax=vmax,
                        alpha=POINT_ALPHA, linewidths=0, rasterized=True, zorder=3, **kw)
    cax = fig.add_axes([0.89, 0.16, 0.017, 0.78])
    fig.colorbar(sc, cax=cax, label=label)
    ax.set_title(f"{label}  -  drag a box to select an area   (s: save CSV | z: zoom | r: reset)")
    info = fig.text(0.05, 0.04, "No area selected.", fontsize=11, family="monospace", va="center")

    state = {"box": None}

    def select(eclick, erelease):
        w, e = sorted([eclick.xdata, erelease.xdata])
        s, n = sorted([eclick.ydata, erelease.ydata])
        update_box(n, w, s, e)

    def update_box(n, w, s, e):
        state["box"] = (n, w, s, e)
        m = (y >= s) & (y <= n) & (x >= w) & (x <= e)
        k = int(m.sum())
        if k:
            v = c[m]
            text = (f"Area N{n:.2f} W{w:.2f} S{s:.2f} E{e:.2f}   |   {k:,} points\n"
                    f"mean {v.mean():.4g}   median {np.median(v):.4g}   std {v.std():.4g}   "
                    f"min {v.min():.4g}   max {v.max():.4g}   {units}")
        else:
            text = f"Area N{n:.2f} W{w:.2f} S{s:.2f} E{e:.2f}   |   0 points in this area"
        info.set_text(text)
        print(text.replace("\n", "\n    "))
        fig.canvas.draw_idle()

    selector = RectangleSelector(ax, select, useblit=True, interactive=True,
                                 minspanx=0.2, minspany=0.2, button=[1],
                                 props=dict(facecolor="#d2461e", edgecolor="#d2461e", alpha=0.2))

    def set_view(w, e, s, n):
        if transform is not None:
            ax.set_extent([w, e, s, n], crs=transform)
        else:
            ax.set_xlim(w, e)
            ax.set_ylim(s, n)
        fig.canvas.draw_idle()

    def on_key(event):
        box = state["box"]
        if event.key == "r":
            set_view(*DEFAULT_VIEW)
        elif event.key == "z" and box:
            n, w, s, e = box
            padx, pady = (e - w) * 0.15 + 0.5, (n - s) * 0.15 + 0.5
            set_view(max(w - padx, -180), min(e + padx, 180), max(s - pady, -90), min(n + pady, 90))
        elif event.key == "s" and box:
            n, w, s, e = box
            m = (y >= s) & (y <= n) & (x >= w) & (x <= e)
            df = pd.DataFrame({"lat": y[m], "lon": x[m], name: c[m]})
            if t is not None:
                df.insert(0, "time", t[m])
            path = os.path.join(OUTPUT_DIR, f"iasi_selection_N{n:.1f}_W{w:.1f}_S{s:.1f}_E{e:.1f}.csv")
            df.to_csv(path, index=False)
            print(f"Saved {len(df):,} points to {path}")
            info.set_text(info.get_text() + f"\nSaved: {path}")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # initial window (west, east, south, north); 'r' returns to it
    DEFAULT_VIEW = ((VIEW_EXTENT[1], VIEW_EXTENT[3], VIEW_EXTENT[2], VIEW_EXTENT[0])
                    if VIEW_EXTENT else (-180, 180, -90, 90))
    set_view(*DEFAULT_VIEW)

    if INITIAL_BBOX:
        n, w, s, e = INITIAL_BBOX
        selector.extents = (w, e, s, n)
        update_box(n, w, s, e)

    plt.show()


if __name__ == "__main__":
    main()