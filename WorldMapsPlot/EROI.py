import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
from rasterio.plot import plotting_extent
from rasterio import warp
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick


def setup_configuration():
    """Initialize all configuration parameters"""
    np.seterr(divide='ignore', invalid='ignore')

    return {
        'GTI_file': 'GTI.tif',
        'TEMP_file': 'TEMP.tif',
        'index': 0,  # 0 = 2024, 1 = 2040
        'mode': "SiPV",  # SiPV, OPV
        'temp_coeff': 0.34,  # %/K
        'years': [2024, 2040],
        'save_path': 'EROI_{mode}_{year}.png'
    }


def calculate_technology_parameters(index):
    """Calculate parameters for both SiPV and OPV technologies"""
    # SiPV Parameters
    lifetime_SiPV = 25
    cost_module_SiPV = [0.149816348, 0.047021468]
    cost_bos_SiPV = [0.356692043, 0.109367308]

    CED_module_SiPV = [7.565275996, 4.721121563]
    CED_bos_SiPV = [4.235803116, 2.616081324]
    CED_SiPV = (CED_module_SiPV[index] + CED_bos_SiPV[index]) / 3.6

    # OPV Parameters
    lifetime_OPV = 20
    cost_module_OPV = [1.369844321, 0.032450411]
    cost_bos_OPV = np.array(cost_bos_SiPV) * 0.9

    CED_module_OPV = [3.349163012, 0.666468894]
    CED_bos_OPV = np.array(CED_bos_SiPV) * 0.75
    CED_OPV = (CED_module_OPV[index] + CED_bos_OPV[index]) / 3.6

    return {
        'lifetime_SiPV': lifetime_SiPV,
        'lifetime_OPV': lifetime_OPV,
        'CED_SiPV': CED_SiPV,
        'CED_OPV': CED_OPV
    }


def load_raster_data(config):
    """Load and process temperature and GTI raster data"""
    with rasterio.open(config['TEMP_file']) as src2:
        temp = src2.read(1, masked=True)
        temp_crs = src2.crs
        temp_transform = src2.transform
        temp = np.ma.masked_where(temp > 200, temp)

    with rasterio.open(config['GTI_file']) as src:
        gti = src.read(1, masked=True)
        gti_crs = src.crs
        extent = plotting_extent(src)
        gti_transform = src.transform
        gti = np.ma.masked_where(gti == 0, gti)

        src_crs = CRS.from_user_input(temp_crs)
        reprojected_data = warp.reproject(
            temp,
            src_transform=temp_transform,
            src_crs=src_crs,
            dst_crs=gti_crs,
            dst_transform=gti_transform,
            destination=gti,
            resampling=warp.Resampling.nearest
        )

    reprojected_temp = reprojected_data[0].reshape(gti.shape)
    reprojected_temp = np.ma.masked_where(reprojected_temp > 200, reprojected_temp)

    return gti, extent, reprojected_temp


def calculate_energy_production(gti, temp_eff, tech_params):
    """Calculate lifetime energy production"""
    kwh_lifetime_SiPV = (temp_eff * gti / (1 * 24) * 1 * 24 * 365 *
                         tech_params['lifetime_SiPV'] / 1000)
    kwh_lifetime_OPV = (gti / (1 * 24) * 1 * 24 * 365 *
                        tech_params['lifetime_OPV'] / 1000)
    return kwh_lifetime_SiPV, kwh_lifetime_OPV


def create_visualization(result, extent, config):
    """Create and customize the visualization"""
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world[~world["continent"].isin(["Antarctica", "Seven seas (open ocean)"])]

    fig, ax = plt.subplots(figsize=(25, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    world.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.0)
    world.boundary.plot(ax=ax, color="black", linewidth=0.7)
    ax.set_extent([-180, 180, -60.1, 65], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.0, linestyle=':')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.xlocator = MultipleLocator(base=60)
    gl.ylocator = MultipleLocator(base=15)

    colormap = LinearSegmentedColormap.from_list('custom',
                                                 [(0, 'purple'), (0.5, 'yellow'), (1, 'green')])

    levels = np.linspace(0, 80, 200)
    cf = ax.contourf(np.flipud(result), extend='neither', levels=levels,
                     cmap=colormap, extent=extent, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',
                        aspect=30, fraction=0.05, pad=0.15)
    cbar.set_label(f'EROI - {config["mode"]} in {config["years"][config["index"]]}',
                   rotation=0, labelpad=-100, va='top', fontsize=28, fontfamily='DejaVu Sans')
    cbar.ax.tick_params(labelsize=26)
    cbar.locator = mtick.LinearLocator(numticks=6)
    cbar.formatter = mtick.FuncFormatter(lambda x, pos: '%.0f' % x)
    cbar.update_ticks()

    ax.axis('off')

    plt.savefig(config['save_path'].format(
        mode=config['mode'],
        year=config['years'][config["index"]]
    ))
    plt.close()


def main():
    """Main execution function"""
    # Setup initial configuration
    config = setup_configuration()
    tech_params = calculate_technology_parameters(config['index'])

    # Load and process raster data
    gti, extent, reprojected_temp = load_raster_data(config)
    temp_eff = (1 - config['temp_coeff'] * reprojected_temp / 100)

    # Calculate energy production
    kwh_lifetime_SiPV, kwh_lifetime_OPV = calculate_energy_production(
        gti, temp_eff, tech_params)

    # Calculate EROI
    if config['mode'] == 'SiPV':
        result = kwh_lifetime_SiPV / tech_params['CED_SiPV']
    elif config['mode'] == 'OPV':
        result = kwh_lifetime_OPV / tech_params['CED_OPV']

    print('Min/Max EROI:', np.min(result), np.max(result))
    print('Mean EROI:', np.mean(result))

    # Create visualization
    create_visualization(result, extent, config)


if __name__ == "__main__":
    main()