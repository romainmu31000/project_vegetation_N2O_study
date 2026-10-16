"""
Interactive side-by-side maps of Copernicus temperature, humidity, NDVI and IASI N2O.
Edit the SETTINGS below and press Run (no command line).

Layers
    1  Temperature (deg C)   daily mean 2 m air temperature  - ERA5-Land, Copernicus Climate Data Store
    2  Humidity (%)          daily mean relative humidity    - ERA5-Land (computed from dewpoint)
    3  NDVI                  vegetation index                - Copernicus Land Monitoring Service
                             (NetCDF file that YOU download once, see NDVI_FILE)

The world map is drawn behind the data; coastlines and borders are drawn above it,
with a white halo, so they stay visible whatever the values.

Keys in the map window
    1 / 2 / ...: choose which layer to save
    drag mouse : select an area; statistics for ALL layers are shown under the maps
    s          : save the selected cells / points of the active layer to CSV
    z          : zoom to the selection        r : back to the initial window

Install once
    python -m pip install cdsapi xarray netcdf4 h5netcdf h5py numpy pandas matplotlib cartopy
Temperature / humidity need your CDS key in ~/.cdsapirc and the ERA5-Land licence accepted
on the dataset page "ERA5-Land post-processed daily statistics" (Download tab).
"""
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import RectangleSelector

# =============================== SETTINGS ===================================
DATE = "2020-01-01"                    # day shown for temperature / humidity (YYYY-MM-DD)
                                       # ERA5-Land is available with ~5 days of delay
VIEW_EXTENT = (70, 110, 10, 150)       # area downloaded AND shown: (north, west, south, east)
                                       # latitude 10N-70N, longitude 110E-150E

NDVI_FILE = None                       # path of a Copernicus NDVI NetCDF file, e.g.
                                       # r"C:\Users\romai\Desktop\projet_vege\ndvi_300m.nc"
                                       # None = omit NDVI
NDVI_VARIABLE = None                   # None = auto-detect (variable containing "ndvi")
IASI_FILE = None                       # None = find the IASI file matching DATE in OUTPUT_DIR
IASI_LEVEL = None                      # None = average N2O retrieval across all vertical levels
IASI_MISSING_VALUES = (-9999.0,)        # IASI N2O no-data marker, masked before averaging

CACHE_DIR = r"C:\Users\romai\Desktop\projet_vege\cache"
OUTPUT_DIR = r"C:\Users\romai\Desktop\projet_vege"
ALPHA = 0.72                           # transparency of the data layer
MAX_CELLS_PER_AXIS = 1500              # NDVI is averaged down to at most this many cells per axis
MAX_DRAWN_POINTS = 300_000             # IASI display limit; statistics and exports use all points
START_LAYER = 1                        # layer selected for CSV export at startup
# ============================================================================

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
# Data
# ----------------------------------------------------------------------------
def open_dataset(path):
    last = None
    for engine in (None, "netcdf4", "h5netcdf"):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:
            last = exc
    sys.exit(f"Could not open {path}\nLast error: {last}\n"
             "If a backend is missing: python -m pip install netcdf4 h5netcdf h5py")


def open_era5_dataset(path):
    """Open CDS NetCDF responses, which may be ZIP archives of per-variable files."""
    if not zipfile.is_zipfile(path):
        return open_dataset(path)

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(path) as archive:
            members = [member for member in archive.infolist()
                       if not member.is_dir() and member.filename.lower().endswith(".nc")]
            if not members:
                sys.exit(f"No NetCDF files found in CDS response {path}")

            datasets = []
            try:
                for index, member in enumerate(members):
                    member_path = os.path.join(temp_dir, f"{index}.nc")
                    with open(member_path, "wb") as extracted:
                        extracted.write(archive.read(member))
                    datasets.append(open_dataset(member_path))
                return xr.merge(datasets, join="exact", compat="no_conflicts").load()
            finally:
                for dataset in datasets:
                    dataset.close()


def get_era5_layers(date, view):
    """Daily mean temperature (deg C) and relative humidity (%) from ERA5-Land."""
    n, w, s, e = view
    area = [min(n + 0.2, 90), max(w - 0.2, -180), max(s - 0.2, -90), min(e + 0.2, 180)]
    year, month, day = date.split("-")
    os.makedirs(CACHE_DIR, exist_ok=True)
    target = os.path.join(CACHE_DIR, f"era5land_{date}_N{area[0]:.1f}_W{area[1]:.1f}_"
                                     f"S{area[2]:.1f}_E{area[3]:.1f}.nc")
    if not os.path.exists(target):
        try:
            import cdsapi
        except ImportError:
            sys.exit("Missing package: python -m pip install cdsapi")
        print(f"Requesting ERA5-Land for {date} (can take a few minutes)...")
        cdsapi.Client().retrieve(
            "derived-era5-land-daily-statistics",
            {
                "variable": ["2m_temperature", "2m_dewpoint_temperature"],
                "year": year, "month": month, "day": [day],
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "1_hourly",
                "area": area,                      # N, W, S, E
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            target,
        )
    else:
        print(f"Using cached {target}")
    ds = open_era5_dataset(target)
    if "expver" in ds.dims:
        ds = ds.mean("expver")
    for tname in ("valid_time", "time"):
        if tname in ds.dims:
            ds = ds.isel({tname: 0})
    t = ds["t2m"] - 273.15
    td = ds["d2m"] - 273.15
    rh = (100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))).clip(0, 100)
    lat, lon = ds["latitude"].values, ds["longitude"].values
    return (lat, lon,
            t.transpose("latitude", "longitude").values,
            rh.transpose("latitude", "longitude").values)


def get_ndvi_layer(path, view):
    """NDVI from a Copernicus Land (CGLS) NetCDF file, cropped to the view and averaged down."""
    ds = open_dataset(path)
    name = (NDVI_VARIABLE
            or next((k for k in ds.data_vars if "ndvi" in k.lower()), None)
            or next(iter(ds.data_vars), None))
    if name is None:
        sys.exit("No variable found in the NDVI file.")
    print(f"NDVI variable: {name}")
    da = ds[name]
    latn = next((n for n in ("lat", "latitude") if n in da.dims), None)
    lonn = next((n for n in ("lon", "longitude") if n in da.dims), None)
    if not (latn and lonn):
        sys.exit(f"Latitude/longitude dimensions not found in NDVI variable (dims: {da.dims}).")
    for d in da.dims:                                   # keep the first time step etc.
        if d not in (latn, lonn):
            da = da.isel({d: 0})
    n, w, s, e = view
    lat = da[latn].values
    if float(da[lonn].max()) > 180:                     # 0..360 -> -180..180
        da = da.assign_coords({lonn: ((da[lonn] + 180) % 360) - 180}).sortby(lonn)
    da = da.sel({latn: slice(n, s) if lat[0] > lat[-1] else slice(s, n),
                 lonn: slice(w, e)})
    f = int(np.ceil(max(da.sizes[latn], da.sizes[lonn]) / MAX_CELLS_PER_AXIS))
    if f > 1:
        print(f"NDVI: averaging {f}x{f} cells to keep the map responsive")
        da = da.coarsen({latn: f, lonn: f}, boundary="trim").mean()
    da = da.load().sortby(latn)
    da = da.where((da >= -0.2) & (da <= 1.0))           # drop flag / fill values
    da = da.transpose(latn, lonn)
    return da[latn].values, da[lonn].values, da.values


def get_iasi_n2o_layer(date):
    """Load N2O observation points using the shared IASI file loader."""
    path = IASI_FILE or os.path.join(
        OUTPUT_DIR, f"iasi_ret_L2_n2o_ch4_{date.replace('-', '_')}_v2.02.nc")
    if not os.path.isfile(path):
        sys.exit(f"IASI file not found: {path}\n"
                 "Set IASI_FILE or place the matching date's file in OUTPUT_DIR.")

    from iasi_area_map import load_points

    lon, lat, values, _, _, units = load_points(
        file_path=path, variable="n2o_retrieval", level=IASI_LEVEL,
        missing_values=IASI_MISSING_VALUES)
    lo, hi = np.nanpercentile(values, [2, 98])
    level_note = "mean across retrieval levels" if IASI_LEVEL is None else f"level index {IASI_LEVEL}"
    return dict(name="N2O", units=units, lat=lat, lon=lon, z=values,
                cmap="turbo", vmin=lo, vmax=hi,
                note=f"IASI {date}; {level_note}", kind="points")


def build_layers():
    layers = []
    lat, lon, t, rh = get_era5_layers(DATE, VIEW_EXTENT)
    for name, units, z, cmap in (("Temperature", "°C", t, "RdYlBu_r"),
                                 ("Humidity", "%", rh, "YlGnBu")):
        lo, hi = np.nanpercentile(z, [2, 98])
        layers.append(dict(name=name, units=units, lat=lat, lon=lon, z=z, cmap=cmap,
                           vmin=lo, vmax=hi, note=f"ERA5-Land daily mean, {DATE}", kind="grid"))
    if NDVI_FILE:
        la, lo_, z = get_ndvi_layer(NDVI_FILE, VIEW_EXTENT)
        layers.append(dict(name="NDVI", units="", lat=la, lon=lo_, z=z, cmap="YlGn",
                           vmin=0.0, vmax=0.9, note=os.path.basename(NDVI_FILE), kind="grid"))
    else:
        print("NDVI_FILE is not set: omitting the NDVI map.")
    layers.append(get_iasi_n2o_layer(DATE))
    return layers


def box_values(L, n, w, s, e):
    if L["kind"] == "points":
        mask = ((L["lat"] >= s) & (L["lat"] <= n)
                & (L["lon"] >= w) & (L["lon"] <= e))
        return mask, None, L["z"][mask]
    mi = (L["lat"] >= s) & (L["lat"] <= n)
    mj = (L["lon"] >= w) & (L["lon"] <= e)
    return mi, mj, L["z"][np.ix_(mi, mj)]


# ----------------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------------
def make_axes(count):
    fig = plt.figure(figsize=(6.5 * count, 7.2))
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        transform = ccrs.PlateCarree()
        axes = [fig.add_subplot(1, count, i + 1, projection=transform) for i in range(count)]
        for ax in axes:
            ax.set_global()
            ax.add_feature(cfeature.OCEAN, facecolor="#eef4f9", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#e4dfd0", zorder=0)
            ax.add_feature(cfeature.COASTLINE, edgecolor="white", lw=2.6, zorder=6)
            ax.add_feature(cfeature.COASTLINE, edgecolor="black", lw=0.8, zorder=7)
            ax.add_feature(cfeature.BORDERS, edgecolor="white", lw=1.6, zorder=6)
            ax.add_feature(cfeature.BORDERS, edgecolor="#222222", lw=0.4, zorder=7)
            ax.gridlines(draw_labels=True, lw=0.3, alpha=0.5)
        return fig, axes, transform
    except Exception as exc:
        print(f"(cartopy not used: {exc.__class__.__name__}; coarse built-in outline instead)")
        axes = [fig.add_subplot(1, count, i + 1) for i in range(count)]
        for ax in axes:
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
        return fig, axes, None


def main():
    layers = build_layers()
    fig, axes, transform = make_axes(len(layers))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.84, bottom=0.23, wspace=0.3)
    info = fig.text(0.04, 0.025, "Drag a box on any map to select an area.",
                    fontsize=9, family="monospace", va="bottom")
    state = {"i": min(max(START_LAYER, 1), len(layers)) - 1, "box": None}
    kw = {"transform": transform} if transform else {}

    for i, (L, ax) in enumerate(zip(layers, axes)):
        if L["kind"] == "points":
            idx = np.arange(L["z"].size)
            if idx.size > MAX_DRAWN_POINTS:
                idx = idx[::int(np.ceil(idx.size / MAX_DRAWN_POINTS))]
            image = ax.scatter(L["lon"][idx], L["lat"][idx], c=L["z"][idx], s=4,
                               cmap=L["cmap"], vmin=L["vmin"], vmax=L["vmax"],
                               alpha=ALPHA, linewidths=0, rasterized=True, zorder=3, **kw)
        else:
            image = ax.pcolormesh(L["lon"], L["lat"], np.ma.masked_invalid(L["z"]),
                                  cmap=L["cmap"], vmin=L["vmin"], vmax=L["vmax"],
                                  alpha=ALPHA, shading="auto", zorder=3,
                                  rasterized=True, **kw)
        label = f"{L['name']} [{L['units']}]" if L["units"] else L["name"]
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=label)
        ax.set_title(f"{i + 1}. {label}\n{L['note']}", fontsize=10)

    def update_box(i, n, w, s, e):
        state["i"] = i
        state["box"] = (n, w, s, e)
        lines = [f"Area N{n:.2f} W{w:.2f} S{s:.2f} E{e:.2f}"]
        for L in layers:
            _, _, sub = box_values(L, n, w, s, e)
            v = sub[np.isfinite(sub)]
            u = f" {L['units']}" if L["units"] else ""
            if v.size:
                lines.append(f"  {L['name']:<12} n={v.size:>7,}  mean {v.mean():8.3f}  "
                             f"min {v.min():8.3f}  max {v.max():8.3f}{u}")
            else:
                lines.append(f"  {L['name']:<12} no data in this area")
        text = "\n".join(lines)
        info.set_text(text)
        print(text)
        fig.canvas.draw_idle()

    selectors = []
    for i, ax in enumerate(axes):
        def select(eclick, erelease, layer_index=i):
            if eclick.xdata is None or erelease.xdata is None:
                return
            w, e = sorted([eclick.xdata, erelease.xdata])
            s, n = sorted([eclick.ydata, erelease.ydata])
            update_box(layer_index, n, w, s, e)

        selectors.append(RectangleSelector(
            ax, select, useblit=True, interactive=True,
            minspanx=0.05, minspany=0.05, button=[1],
            props=dict(facecolor="#d2461e", edgecolor="#d2461e", alpha=0.2)))

    def set_view(w, e, s, n):
        for ax in axes:
            if transform is not None:
                ax.set_extent([w, e, s, n], crs=transform)
            else:
                ax.set_xlim(w, e)
                ax.set_ylim(s, n)
        fig.canvas.draw_idle()

    north, west, south, east = VIEW_EXTENT
    default_view = (west, east, south, north)
    fig.suptitle("Copernicus + IASI maps | drag: select | number keys: save layer | "
                 "s: save CSV | z: zoom | r: reset", fontsize=12)

    def on_key(event):
        box = state["box"]
        if event.key in tuple(str(i + 1) for i in range(len(layers))):
            state["i"] = int(event.key) - 1
        elif event.key == "r":
            set_view(*default_view)
        elif event.key == "z" and box:
            n, w, s, e = box
            px, py = (e - w) * 0.15 + 0.2, (n - s) * 0.15 + 0.2
            set_view(max(w - px, -180), min(e + px, 180), max(s - py, -90), min(n + py, 90))
        elif event.key == "s" and box:
            n, w, s, e = box
            L = layers[state["i"]]
            first, second, sub = box_values(L, n, w, s, e)
            if L["kind"] == "points":
                df = pd.DataFrame({"lat": L["lat"][first], "lon": L["lon"][first],
                                   L["name"].lower(): sub})
                count_label = "points"
            else:
                la, lo = np.meshgrid(L["lat"][first], L["lon"][second], indexing="ij")
                ok = np.isfinite(sub)
                df = pd.DataFrame({"lat": la[ok], "lon": lo[ok], L["name"].lower(): sub[ok]})
                count_label = "cells"
            path = os.path.join(OUTPUT_DIR, f"{L['name'].lower()}_N{n:.1f}_W{w:.1f}_S{s:.1f}_E{e:.1f}.csv")
            df.to_csv(path, index=False)
            print(f"Saved {len(df):,} {count_label} to {path}")
            info.set_text(info.get_text() + f"\nSaved: {path}")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    set_view(*default_view)
    plt.show()


if __name__ == "__main__":
    main()
