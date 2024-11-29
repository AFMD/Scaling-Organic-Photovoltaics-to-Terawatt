import os
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
        'index': 1,  # 0 = 2024, 1 = 2040
        'mode': "OPV-SiPV",  # SiPV, OPV, OPV-SiPV
        'temp_coeff': 0.34,  # %/K
        'years': [2024, 2040],
        'opv_bos_factor': 0.95,  # Cost factor for OPV BOS relative to SiPV
        'save_path': 'LCOE_{mode}_{year}_{carbon}.png',
        'carbon_price': "on",  # on/off
        'conversion_USD19_EUR22': 0.9,
        'california_carbon_price': 40 # in EUR/t
    }


def calculate_technology_parameters(index, opv_bos_factor):
    """Calculate parameters for both SiPV and OPV technologies"""
    # SiPV Parameters
    lifetime_SiPV = 25
    cost_module_SiPV = [0.149816348, 0.047021468]
    cost_bos_SiPV = [0.356692043, 0.109367308]
    costs_SiPV = cost_module_SiPV[index] + cost_bos_SiPV[index]

    CED_module_SiPV = [7.565275996, 4.721121563]
    CED_bos_SiPV = [4.235803116, 2.616081324]
    ECI_module_SiPV = [0.0729, 0.0405]
    ECI_bos_SiPV = [0.0729, 0.0405]
    emissions_SiPV = CED_module_SiPV[index] * ECI_module_SiPV[index] + CED_bos_SiPV[index] * ECI_bos_SiPV[index]

    # OPV Parameters
    lifetime_OPV = 20
    cost_module_OPV = [1.369844321, 0.032450411]
    cost_bos_OPV = np.array(cost_bos_SiPV) * opv_bos_factor
    costs_OPV = cost_module_OPV[index] + cost_bos_OPV[index]

    CED_module_OPV = [3.349163012, 0.666468894]
    CED_bos_OPV = np.array(CED_bos_SiPV) * 0.75
    ECI_module_OPV = [0.036094737, 0.020052632]
    ECI_bos_OPV = [0.0729, 0.0405]
    emissions_OPV = CED_module_OPV[index] * ECI_module_OPV[index] + CED_bos_OPV[index] * ECI_bos_OPV[index]

    return {
        'lifetime_SiPV': lifetime_SiPV,
        'lifetime_OPV': lifetime_OPV,
        'costs_SiPV': costs_SiPV,
        'costs_OPV': costs_OPV,
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


def load_carbon_tax_data():
    """Load and process carbon tax data"""
    carbon_tax = pd.read_csv('ecp_sector_CO2.csv')
    carbon_tax = carbon_tax[carbon_tax['ipcc_code'] == '1A1A1']
    carbon_tax = carbon_tax.groupby('jurisdiction')[['year', 'ecp_tax_usd_k', 'ecp_all_usd_k']].last().reset_index()
    mask = (carbon_tax['ecp_all_usd_k'] != 0)
    carbon_tax = carbon_tax[mask]
    return carbon_tax


def apply_carbon_tax(result_SiPV, result_OPV, tech_params, config, gti, rows, cols, src):
    """Apply carbon tax to results where applicable"""
    if config['carbon_price'] == "off":
        return result_SiPV, result_OPV

    carbon_tax = load_carbon_tax_data()
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    us_states = gpd.read_file(gpd.datasets.get_path('ne_110m_admin_1_states_provinces'))

    # Process each country with carbon tax
    for _, row in carbon_tax.iterrows():
        country = row.jurisdiction
        if country == 'Czech Republic':
            country = 'Czechia'
        elif country == 'Slovak Republic':
            country = 'Slovakia'
        elif country == 'Korea, Rep.':
            country = 'South Korea'

        country_mask = world[world['name'] == country].unary_union
        if country_mask is None or (hasattr(country_mask, 'is_empty') and country_mask.is_empty):
            print(f"Warning: Empty or None geometry for country {country}. Skipping rasterization.")
            continue

        result_SiPV, result_OPV = apply_carbon_costs(
            result_SiPV, result_OPV, tech_params, config, gti, rows, cols, src,
            country_mask, row.ecp_all_usd_k
        )

    # Apply California carbon price
    california = us_states[us_states['name'] == 'California'].geometry.iloc[0]
    result_SiPV, result_OPV = apply_carbon_costs(
        result_SiPV, result_OPV, tech_params, config, gti, rows, cols, src,
        california, config['california_carbon_price']
    )

    return result_SiPV, result_OPV


def apply_carbon_costs(result_SiPV, result_OPV, tech_params, config, gti, rows, cols, src, mask, carbon_price):
    """Apply carbon costs to a specific region"""
    carbon_costs_SiPV = (config['conversion_USD19_EUR22'] * 100 * carbon_price *
                        tech_params['emissions_SiPV'] / 1000)
    carbon_costs_OPV = (config['conversion_USD19_EUR22'] * 100 * carbon_price *
                       tech_params['emissions_OPV'] / 1000)

    raster_mask = np.zeros((rows, cols), dtype=np.uint8)
    rasterize([(mask, 1)], out=raster_mask, out_shape=(rows, cols), transform=src.transform)

    lifetime_SiPV = gti / 24 * 24 * 365 * tech_params['lifetime_SiPV'] / 1000
    lifetime_OPV = gti / 24 * 24 * 365 * tech_params['lifetime_OPV'] / 1000

    result_SiPV[raster_mask == 1] += carbon_costs_SiPV / np.where(
        lifetime_SiPV[raster_mask == 1] != 0, lifetime_SiPV[raster_mask == 1], 1)
    result_OPV[raster_mask == 1] += carbon_costs_OPV / np.where(
        lifetime_OPV[raster_mask == 1] != 0, lifetime_OPV[raster_mask == 1], 1)

    return result_SiPV, result_OPV


def setup_visualization():
    """Setup the base map visualization"""
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world[~world["continent"].isin(["Antarctica", "Seven seas (open ocean)"])]

    fig, ax = plt.subplots(figsize=(25, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    world.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.0)
    world.boundary.plot(ax=ax, color="black", linewidth=0.7)
    ax.set_extent([-180, 180, -60.1, 65], crs=ccrs.PlateCarree())

    return fig, ax


def create_custom_cmap():
    """Create custom colormap for visualization"""
    summer = plt.cm.PRGn(np.linspace(1.0, 0.7, 128))
    copper = plt.cm.RdYlGn(np.linspace(0.5, 0.1, 128))
    colors = np.vstack((summer, copper))
    return LinearSegmentedColormap.from_list('custom_cmap', colors)


def create_plot(ax, result, extent, config):
    """Create and customize the plot"""
    gl = ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.0, linestyle=':')
    gl.xlocator = MultipleLocator(base=60)
    gl.ylocator = MultipleLocator(base=15)

    cmap = create_custom_cmap()
    levels = np.linspace(-10, 10, 200)
    cf = ax.contourf(np.flipud(result), extend='neither', levels=levels,
                     cmap=cmap, extent=extent, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', aspect=30,
                       fraction=0.05, pad=0.15)
    cbar.set_label(f'OPV and SiPV: LCOE disparity in {config["years"][config["index"]]} '
                   f'{"including" if config["carbon_price"] == "on" else "excluding"} carbon price',
                   rotation=0, labelpad=-100, va='top', fontsize=28, fontfamily='DejaVu Sans')
    cbar.ax.tick_params(labelsize=26)
    cbar.locator = mtick.LinearLocator(numticks=9)
    cbar.formatter = mtick.FuncFormatter(lambda x, pos: '%.1f' % x)
    cbar.update_ticks()

    ax.axis('off')
    plt.text(0.85, 0.12, ' %', fontsize=24, transform=plt.gcf().transFigure)
    # Save the plot

    save_path = config['save_path'].format(
        mode=config['mode'],
        year=config['years'][config['index']],
        carbon='with_carbon' if config['carbon_price'] == "on" else 'no_carbon'
    )
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    print(f"Plot saved as: {save_path}")
    plt.close()


def main():
    """Main execution function"""
    config = setup_configuration()
    tech_params = calculate_technology_parameters(config['index'], config['opv_bos_factor'])

    gti, extent, reprojected_temp, src = load_raster_data(config)
    temp_eff = (1 - config['temp_coeff'] * reprojected_temp / 100)

    rows, cols = gti.shape
    kwh_lifetime_SiPV = temp_eff * gti / 24 * 24 * 365 * tech_params['lifetime_SiPV'] / 1000
    kwh_lifetime_OPV = gti / 24 * 24 * 365 * tech_params['lifetime_OPV'] / 1000

    result_SiPV = tech_params['costs_SiPV'] * 100 / kwh_lifetime_SiPV
    result_OPV = tech_params['costs_OPV'] * 100 / kwh_lifetime_OPV

    result_SiPV, result_OPV = apply_carbon_tax(
            result_SiPV, result_OPV, tech_params, config, gti, rows, cols, src)

    result = (result_OPV/result_SiPV - 1) * 100

    # Print statistics
    result_diff = result_OPV - result_SiPV
    print(f'Best performance difference in Cent/kWh: {round(np.min(result_diff), 2)}')
    print(f'Worst performance difference in Cent/kWh: {round(np.max(result_diff), 2)}')
    print(f'Best OPV performance in Cent/kWh: {round(np.min(result_OPV), 2)}')
    print(f'Worst OPV performance in Cent/kWh: {round(np.max(result_OPV), 2)}')
    print(f'Best SiPV performance in Cent/kWh: {round(np.min(result_SiPV), 2)}')
    print(f'Worst SiPV performance in Cent/kWh: {round(np.max(result_SiPV), 2)}')

    # Create visualization
    fig, ax = setup_visualization()
    create_plot(ax, result, extent, config)




if __name__ == "__main__":
    main()