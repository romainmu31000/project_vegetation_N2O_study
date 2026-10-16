"""Retrieve CGLS NDVI composites and create a yearly animation.

Install dependencies with:
    python -m pip install rasterio matplotlib cartopy numpy pillow

Authenticate with a CDSE access token in CDSE_ACCESS_TOKEN, or set CDSE_USERNAME
and enter the password at the hidden prompt. The CDS API key in .cdsapirc is not
a CDSE credential. Set YEAR, VIEW_EXTENT, and OUTPUT_DIR below, then run:
    python vegetation_index.py
"""
import getpass
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =============================== SETTINGS ===================================
YEAR = 2020
VIEW_EXTENT = (70, 110, 10, 150)  # (north, west, south, east)
OUTPUT_DIR = Path(__file__).resolve().parent / "vegetation_data"
MAX_DISPLAY_PIXELS = 1600  # output raster's longest side; source data is 300 m
MOVIE_FPS = 2
# ============================================================================

CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
DATASET_ID = "ndvi_global_300m_10daily_v2"
NDVI_SCALE = 0.0001


def request_json(url, data=None, headers=None):
    request = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "Copernicus rejected the credentials. Check that these are CDSE "
                "credentials/a current CDSE access token, not the CDS API key."
            ) from exc
        raise RuntimeError(f"Copernicus request failed (HTTP {exc.code}).") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to Copernicus Data Space: {exc.reason}") from exc


def get_access_token():
    token = os.environ.get("CDSE_ACCESS_TOKEN")
    if token:
        return token

    username = os.environ.get("CDSE_USERNAME") or input("CDSE username: ")
    password = os.environ.get("CDSE_PASSWORD") or getpass.getpass("CDSE password: ")

    if not username or not password:
        raise RuntimeError("A CDSE access token or CDSE username and password are required.")

    form = urllib.parse.urlencode({
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }).encode()
    payload = request_json(
        TOKEN_URL,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("CDSE did not return an access token.")
    return token


def find_composite(date):
    start = f"{date}T00:00:00.000Z"
    end = f"{date}T23:59:59.999Z"
    filters = (
        "Collection/Name eq 'CLMS' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'datasetIdentifier' "
        f"and att/OData.CSC.StringAttribute/Value eq '{DATASET_ID}') and "
        f"ContentDate/Start le {end} and ContentDate/End ge {start}"
    )
    query = urllib.parse.urlencode({
        "$filter": filters,
        "$orderby": "ContentDate/Start desc",
        "$top": "1",
    })
    results = request_json(f"{CATALOG_URL}?{query}").get("value", [])
    if not results:
        raise RuntimeError(f"No CGLS 300 m NDVI composite covers {date}.")
    return results[0]


def find_year_composites(year):
    start = f"{year}-01-01T00:00:00.000Z"
    end = f"{year + 1}-01-01T00:00:00.000Z"
    filters = (
        "Collection/Name eq 'CLMS' and "
        "contains(Name,'NDVI300') and "
        f"ContentDate/Start ge {start} and ContentDate/Start lt {end}"
    )
    query = urllib.parse.urlencode({
        "$filter": filters,
        "$orderby": "ContentDate/Start asc",
        "$top": "1000",
    })
    products = request_json(f"{CATALOG_URL}?{query}").get("value", [])
    composites = sorted(
        (
            product for product in products
            if product.get("Name", "").endswith("_OLCI_V3.0.1_cog")
        ),
        key=lambda product: product.get("ContentDate", {}).get("Start", ""),
    )
    if not composites:
        raise RuntimeError(
            f"No CGLS 300 m OLCI V3.0.1 NDVI composites are available for {year}."
        )
    return composites


def output_path(output_dir, date, extent, product=None):
    north, west, south, east = extent
    series = ""
    if product:
        series_name = product["Name"].split("_GLOBE_")[-1].removesuffix("_cog")
        series = f"_{series_name}"
    name = (
        f"ndvi_cgls_{date}_N{north:g}_W{west:g}_S{south:g}_E{east:g}"
        f"{series}.tif"
    )
    return output_dir / name


def find_ndvi_file(product):
    product_name = urllib.parse.quote(product["Name"], safe="")
    nodes_url = f"{DOWNLOAD_URL}({product['Id']})/Nodes({product_name})/Nodes"
    nodes = request_json(nodes_url).get("result", [])
    candidates = [
        node for node in nodes
        if node.get("Name", "").startswith("c_gls_NDVI300-NDVI_")
        and node.get("Name", "").endswith(".tiff")
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one NDVI TIFF in product {product['Name']}, found {len(candidates)}."
        )
    return candidates[0]


def retrieve_window(product, extent, access_token, destination):
    north, west, south, east = extent
    ndvi_file = find_ndvi_file(product)
    product_name = urllib.parse.quote(product["Name"], safe="")
    file_name = urllib.parse.quote(ndvi_file["Name"], safe="")
    url = (
        f"{DOWNLOAD_URL}({product['Id']})/Nodes({product_name})/"
        f"Nodes({file_name})/$value"
    )
    headers = f"Authorization: Bearer {access_token}"

    print(f"Retrieving map window from CGLS product {product['Name']}...")
    try:
        with rasterio.Env(GDAL_HTTP_HEADERS=headers):
            with rasterio.open(f"/vsicurl/{url}") as source:
                if source.crs is None:
                    raise RuntimeError("The NDVI product has no coordinate reference system.")

                bounds = transform_bounds(
                    "EPSG:4326", source.crs, west, south, east, north
                )
                window = source.window(*bounds).intersection(
                    Window(0, 0, source.width, source.height)
                )
                window = window.round_offsets().round_lengths()
                if window.width <= 0 or window.height <= 0:
                    raise RuntimeError("The selected extent is outside the NDVI product.")

                width_degrees = east - west
                height_degrees = north - south
                if height_degrees > width_degrees:
                    rows = MAX_DISPLAY_PIXELS
                    columns = max(1, round(rows * width_degrees / height_degrees))
                else:
                    columns = MAX_DISPLAY_PIXELS
                    rows = max(1, round(columns * height_degrees / width_degrees))
                values = source.read(
                    1,
                    window=window,
                    out_shape=(rows, columns),
                    masked=True,
                    resampling=Resampling.average,
                ).astype(np.float32).filled(np.nan)
                source_transform = (
                    source.window_transform(window)
                    * rasterio.Affine.scale(window.width / columns, window.height / rows)
                )

                scale = source.scales[0] if source.scales else 1.0
                if scale == 1.0 and np.issubdtype(np.dtype(source.dtypes[0]), np.integer):
                    scale = NDVI_SCALE

                pixel_width = (east - west) / columns
                pixel_height = (north - south) / rows
                destination_transform = from_origin(
                    west, north, pixel_width, pixel_height
                )
                ndvi = np.full((rows, columns), np.nan, dtype=np.float32)
                reproject(
                    source=values,
                    destination=ndvi,
                    src_transform=source_transform,
                    src_crs=source.crs,
                    src_nodata=np.nan,
                    dst_transform=destination_transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
    except rasterio.errors.RasterioIOError as exc:
        raise RuntimeError(
            "Could not read the CGLS product. Check the CDSE token/account and "
            f"internet connection. Rasterio reported: {exc}"
        ) from exc

    ndvi *= scale
    ndvi[(ndvi < -0.2) | (ndvi > 1.0)] = np.nan
    if not np.isfinite(ndvi).any():
        raise RuntimeError("The retrieved map window contains no valid NDVI data.")

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": columns,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": destination_transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    temporary = destination.with_name(destination.stem + ".partial.tif")
    try:
        with rasterio.open(temporary, "w", **profile) as output:
            output.write(ndvi, 1)
            output.update_tags(
                date=date_from_composite(product),
                product=product["Name"],
                units="NDVI",
                requested_extent=f"N{north:g} W{west:g} S{south:g} E{east:g}",
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return product


def date_from_composite(product):
    return product.get("ContentDate", {}).get("Start", "unknown")[:10]


def retrieve_ndvi(date, extent, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_path(output_dir, date, extent, product)
    if destination.exists():
        print(f"Using cached map window: {destination}")
        return destination

    product = find_composite(date)
    token = get_access_token()
    retrieve_window(product, extent, token, destination)
    print(f"Saved resampled NDVI map window: {destination}")
    return destination


def display_ndvi(path, date, extent):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    north, west, south, east = extent
    with rasterio.open(path) as source:
        values = source.read(1, masked=True)
        product_name = source.tags().get("product", "")
    if not values.count():
        raise RuntimeError(f"No valid NDVI data to display in {path}.")

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent([west, east, south, north], crs=projection)
    ax.add_feature(cfeature.OCEAN, facecolor="#e8f1f7", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#e4dfd0", zorder=0)
    image = ax.imshow(
        values,
        extent=(west, east, south, north),
        origin="upper",
        transform=projection,
        cmap="YlGn",
        vmin=-0.2,
        vmax=1.0,
        interpolation="nearest",
        zorder=2,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="white", linewidth=2.2, zorder=4)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#222222", linewidth=0.55, zorder=5)
    ax.add_feature(cfeature.BORDERS, edgecolor="white", linewidth=1.4, zorder=4)
    ax.add_feature(cfeature.BORDERS, edgecolor="#333333", linewidth=0.4, zorder=5)
    ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.55)
    fig.colorbar(image, ax=ax, shrink=0.78, pad=0.04, label="NDVI")
    composite_note = f"\n{product_name}" if product_name else ""
    ax.set_title(f"Copernicus Global Land Service NDVI for {date}{composite_note}")
    plt.show()


def create_yearly_movie(year, extent, output_dir):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    composites = find_year_composites(year)
    token = get_access_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    for product in composites:
        date = date_from_composite(product)
        destination = output_path(output_dir, date, extent, product)
        if destination.exists():
            print(f"Using cached map window: {destination}")
        else:
            retrieve_window(product, extent, token, destination)
            print(f"Saved resampled NDVI map window: {destination}")
        frame_paths.append(destination)

    north, west, south, east = extent
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent([west, east, south, north], crs=projection)
    ax.add_feature(cfeature.OCEAN, facecolor="#e8f1f7", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#e4dfd0", zorder=0)
    image = ax.imshow(
        np.full((2, 2), np.nan, dtype=np.float32),
        extent=(west, east, south, north),
        origin="upper",
        transform=projection,
        cmap="YlGn",
        vmin=-0.2,
        vmax=1.0,
        interpolation="nearest",
        zorder=2,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="white", linewidth=2.2, zorder=4)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#222222", linewidth=0.55, zorder=5)
    ax.add_feature(cfeature.BORDERS, edgecolor="white", linewidth=1.4, zorder=4)
    ax.add_feature(cfeature.BORDERS, edgecolor="#333333", linewidth=0.4, zorder=5)
    ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.55)
    fig.colorbar(image, ax=ax, shrink=0.78, pad=0.04, label="NDVI")

    def update(frame_index):
        frame_path = frame_paths[frame_index]
        with rasterio.open(frame_path) as source:
            values = source.read(1, masked=True)
        if not values.count():
            raise RuntimeError(f"No valid NDVI data to animate in {frame_path}.")
        image.set_data(values)
        date = date_from_composite(composites[frame_index])
        ax.set_title(f"CGLS NDVI evolution in {year} — composite from {date}")
        return (image,)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_paths),
        interval=1000 // MOVIE_FPS,
        blit=False,
        repeat=True,
    )
    movie_path = output_dir / f"ndvi_evolution_{year}.gif"
    try:
        animation.save(movie_path, writer=PillowWriter(fps=MOVIE_FPS))
    finally:
        plt.close(fig)
    print(f"Saved NDVI animation with {len(frame_paths)} composites: {movie_path}")
    return movie_path


def main():
    north, west, south, east = VIEW_EXTENT
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise SystemExit(
            "VIEW_EXTENT must satisfy -90 <= south < north <= 90 and "
            "-180 <= west < east <= 180."
        )

    try:
        create_yearly_movie(YEAR, VIEW_EXTENT, OUTPUT_DIR)
    except (RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
