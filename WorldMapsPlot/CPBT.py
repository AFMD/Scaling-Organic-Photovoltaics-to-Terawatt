import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import rasterio
from rasterio.features import rasterize
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
        'GTI_file': 'GTI_new.tif',
        'TEMP_file': 'TEMP2_new.tif',
        'index': 0,  # 0 = 2024, 1 = 2040
        'mode': "SiPV",  # SiPV, OPV
        'temp_coeff': 0.34,  # %/K
        'years': [2024, 2040],
        'save_path': 'CPBT_{mode}_{year}.png'
    }


def calculate_technology_parameters(index):
    """Calculate parameters for both SiPV and OPV technologies"""
    # SiPV Parameters
    lifetime_SiPV = 25
    cost_module_SiPV = [0.149816348, 0.047021468]
    cost_bos_SiPV = [0.356692043, 0.109367308]

    CED_module_SiPV = [7.565275996, 4.721121563]
    CED_bos_SiPV = [4.235803116, 2.616081324]
    ECI_module_SiPV = [0.0729, 0.0405]
    ECI_bos_SiPV = [0.0729, 0.0405]

    CED_SiPV = (CED_module_SiPV[index] + CED_bos_SiPV[index]) / 3.6
    emissions_SiPV = (CED_module_SiPV[index] * ECI_module_SiPV[index] +
                     CED_bos_SiPV[index] * ECI_bos_SiPV[index])

    # OPV Parameters
    lifetime_OPV = 20
    cost_module_OPV = [1.369844321, 0.032450411]
    cost_bos_OPV = np.array(cost_bos_SiPV) * 0.9

    CED_module_OPV = [3.349163012, 0.666468894]
    CED_bos_OPV = np.array(CED_bos_SiPV) * 0.75
    ECI_module_OPV = [0.036094737, 0.020052632]
    ECI_bos_OPV = [0.0729, 0.0405]

    CED_OPV = (CED_module_OPV[index] + CED_bos_OPV[index]) / 3.6
    emissions_OPV = (CED_module_OPV[index] * ECI_module_OPV[index] +
                    CED_bos_OPV[index] * ECI_bos_OPV[index])

    return {
        'lifetime_SiPV': lifetime_SiPV,
        'lifetime_OPV': lifetime_OPV,
        'CED_SiPV': CED_SiPV,
        'CED_OPV': CED_OPV,
        'emissions_SiPV': emissions_SiPV,
        'emissions_OPV': emissions_OPV
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

    return gti, extent, reprojected_temp, src


def calculate_energy_production(gti, temp_eff, tech_params):
    """Calculate lifetime energy production"""
    kwh_lifetime_SiPV = (temp_eff * gti / (1 * 24) * 1 * 24 * 365 *
                        tech_params['lifetime_SiPV'] / 1000)
    kwh_lifetime_OPV = (gti / (1 * 24) * 1 * 24 * 365 *
                       tech_params['lifetime_OPV'] / 1000)
    return kwh_lifetime_SiPV, kwh_lifetime_OPV


def apply_carbon_intensity(result, gti, tech_params, config, kwh_lifetime_SiPV, kwh_lifetime_OPV):
    """Apply carbon intensity calculations for each country"""
    carbon_intensity = pd.read_csv('carbon-intensity-electricity.csv')
    carbon_intensity_2022 = carbon_intensity[carbon_intensity['Year'] == 2022]

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    rows, cols = result.shape

    country_mapping = {
        'ASEAN (Ember)': None,  # Skip this entry
        'Czech Republic': 'Czechia',
        'Slovak Republic': 'Slovakia',
        'United States': 'United States of America',
        "Cote d'Ivoire": "Côte d'Ivoire",
        'Democratic Republic of Congo': 'Dem. Rep. Congo',
        'South Sudan': 'S. Sudan',
        'Bosnia and Herzegovina': 'Bosnia and Herz.',
        'Equatorial Guinea': 'Eq. Guinea',
        'Central African Republic': 'Central African Rep.',
        'Dominican Republic': 'Dominican Rep.',
        'Korea, Rep.': 'South Korea'
    }

    for _, row in carbon_intensity_2022.iterrows():
        if row.Entity in country_mapping and country_mapping[row.Entity] is None:
            continue

        country = country_mapping.get(row.Entity, row.Entity)
        country_mask = world[world['name'] == country].unary_union

        if country_mask is None or (hasattr(country_mask, 'is_empty') and country_mask.is_empty):
            print(f"Warning: Empty or None geometry for country {country}. Skipping.")
            continue

        raster_mask = np.zeros((rows, cols), dtype=np.uint8)
        rasterize([(country_mask, 1)], out=raster_mask,
                 out_shape=(rows, cols), transform=gti.transform)

        carbon_intensity_value = row['Carbon intensity of electricity - gCO2/kWh'] / 1000

        if config['mode'] == 'SiPV':
            result[raster_mask == 1] += (
                tech_params['emissions_SiPV'] /
                (carbon_intensity_value *
                 np.where(kwh_lifetime_SiPV[raster_mask == 1] != 0,
                         kwh_lifetime_SiPV[raster_mask == 1], 1) /
                 (tech_params['lifetime_SiPV'] * 12))
            )
        elif config['mode'] == 'OPV':
            result[raster_mask == 1] += (
                tech_params['emissions_OPV'] /
                (carbon_intensity_value *
                 np.where(kwh_lifetime_OPV[raster_mask == 1] != 0,
                         kwh_lifetime_OPV[raster_mask == 1], 1) /
                 (tech_params['lifetime_OPV'] * 12))
            )

    return result


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
                                               [(0, 'green'), (0.5, 'yellow'), (1, 'purple')])

    levels = np.linspace(4, 40, 200)
    cf = ax.contourf(np.flipud(result), extend='max', levels=levels,
                     cmap=colormap, extent=extent, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',
                       aspect=30, fraction=0.05, pad=0.15)
    cbar.set_label(f'CPBT - {config["mode"]} in {config["years"][config["index"]]}',
                   rotation=0, labelpad=-100, va='top', fontsize=28, fontfamily='DejaVu Sans')
    cbar.ax.tick_params(labelsize=26)
    cbar.locator = mtick.LinearLocator(numticks=7)
    cbar.formatter = mtick.FuncFormatter(lambda x, pos: '%.0f' % x)
    cbar.update_ticks()

    ax.axis('off')
    plt.text(0.85, 0.12, 'Months', fontsize=24, transform=plt.gcf().transFigure)

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
    gti, extent, reprojected_temp, src = load_raster_data(config)
    temp_eff = (1 - config['temp_coeff'] * reprojected_temp / 100)

    # Calculate energy production
    kwh_lifetime_SiPV, kwh_lifetime_OPV = calculate_energy_production(
        gti, temp_eff, tech_params)

    print('kwh_lifetime_SiPV', np.mean(kwh_lifetime_SiPV),
          '\n kwh_lifetime_OPV', np.mean(kwh_lifetime_OPV))

    # Calculate initial result
    if config['mode'] == 'SiPV':
        result = (tech_params['lifetime_SiPV'] * 12) / (kwh_lifetime_SiPV / tech_params['CED_SiPV'])
    elif config['mode'] == 'OPV':
        result = (tech_params['lifetime_OPV'] * 12) / (kwh_lifetime_OPV / tech_params['CED_OPV'])

    # Apply carbon intensity calculations
    result = apply_carbon_intensity(
        result, src, tech_params, config,
        kwh_lifetime_SiPV, kwh_lifetime_OPV
    )

    print('Min/Max CPBT:', np.min(result), np.max(result))
    print('Mean CPBT:', np.mean(result))

    # Create visualization
    create_visualization(result, extent, config)


if __name__ == "__main__":
    main()