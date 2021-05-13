from pathlib import Path
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import fiona
import rasterio
from math import pi
from scipy.ndimage import label
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio import features
from shapely.ops import unary_union
from shapely.geometry import shape
import os
from timeit import default_timer as timer
import multiprocessing as mp
import itertools

def checkGeoms(gdf):
    """
    Method: checkGeoms()
    Purpose: Verify dissolved/exploded geoms are valid, if not it corrects them
    Params: gdf - geodataframe to validate
    returns: gdf - geodataframe with valid geoms
    """
    #make sure index is unique
    cols = list(gdf)
    gdf = gdf.reset_index()
    gdf = gdf[cols]
    #df to store fixed geoms
    newGeoms = gpd.GeoDataFrame(columns=cols, crs=gdf.crs)
    #list to store indexes of geoms to remove
    idxToRemove = []
    for idx, row in gdf.iterrows():
        geom = shape(row['geometry'])
        if not geom.is_valid:
            clean = geom.buffer(0.0)
            assert clean.is_valid
            idxToRemove.append(idx)
            toAdd = [] #build list for row entry
            for c in cols:
                if c != 'geometry':
                    toAdd.append(row[c])
                else:
                    toAdd.append(clean)
            newGeoms.loc[len(newGeoms)] = toAdd #add row with updated geom
    if len(newGeoms) > 0:
        print("Corrected ", len(newGeoms), " geomtries out of ", len(gdf))
    gdf.drop(gdf.index[idxToRemove], inplace=True) #remove rows we updated
    gdf = gdf.append(newGeoms) #append fixed geoms
    gdf = gdf.reset_index() #reset index
    gdf = gdf[cols] #remove new index column
    return gdf

def extract_water(cofips, ancillary):

    landcover = list(ancillary.rglob(f'*landcover*.tif'))[0]

    if landcover.is_file():
        water_value = 1

        with rasterio.open(landcover) as lc:
            # read in arrays
            lc_arr = lc.read(1)

            # conditional statement to mask out 
            water = np.where(lc_arr == water_value, 1, 0)

        # eight neighbors 
        eight_neighbors = [[1, 1, 1], 
                           [1, 1, 1], 
                           [1, 1, 1]]

        water_grouped, num_feats = label(water, structure=eight_neighbors)

        # delete the array!
        del water

        water_grouped = water_grouped.astype('int32')

        mask = water_grouped != 0

        results = (
            {'properties': {'value': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(water_grouped, mask=mask, transform=lc.transform))
            )

        polygons = list(results)
        gdf = gpd.GeoDataFrame.from_features(polygons)
        gdf.crs = 'epsg:5070'
        gdf = gdf.dissolve(by='value', as_index=False)

        # calculate perimeter
        gdf['perimeter'] = gdf["geometry"].length

        # calculate area
        gdf['area'] = gdf["geometry"].area

        # calculate perimeter-area ratio [par] (perimeter / area)
        gdf['par'] = gdf['perimeter']/gdf['area'] # 

        # calculate polsby-popper score [pps] (4 * pi * area) / (perimeter ** 2)
        gdf['pps'] = (4 * pi * gdf['area']) / (gdf['perimeter'] ** 2)

        return gdf

def load_david_streams(cofips, ancillary, buffer=False):
    
    if buffer:
        streams_shp = list(ancillary.rglob('*buffer.shp'))[0]
    else:
        streams_shp = list(ancillary.rglob('*streams.shp'))[0]

    # streams_shp = list(streams_folder.rglob(wildcard))[0]
    if streams_shp.is_file():
        return gpd.read_file(streams_shp)

def load_nhd(cofips, nhd_folder, layer):

    if layer == 'area':
        wildcard = f'*_area.shp'
    elif layer == 'wb':
        wildcard = f'*_wb.shp'

    nhd_shp = list(nhd_folder.rglob(wildcard))[0]

    if nhd_shp.is_file():
        return gpd.read_file(nhd_shp)

def load_ches_bay(ches_bay_shp):
    if ches_bay_shp.is_file():
        return gpd.read_file(ches_bay_shp)
        # gdf = gpd.read_file(ches_bay_shp)
        # gdf['diss'] = 1
        # gdf_diss = gdf.dissolve(by='diss')


def identify_ponds_ver2(cofips, county_folder, ancillary, water, ches_bay_shp):

    ponds_gpkg = county_folder / 'ponds.gpkg'

    if not ponds_gpkg.is_file():

        # buffer facet stream
        stream = load_david_streams(cofips, ancillary, buffer=True)

        stream = checkGeoms(stream)
        water = checkGeoms(water)

        pond_ids = []
        not_pond_ids = []

        # water intersects with channel_width
        water_inside_channel = gpd.sjoin(water, stream, how='inner', op='intersects')
        # drop duplicates
        water_inside_channel = water_inside_channel.drop_duplicates(subset=['value'], keep='first')
        water_inside_channel = water_inside_channel[['value', 'perimeter', 'area', 'par', 'pps', 'geometry']]

        # NHDWaterbody
        nhd_wb = load_nhd(cofips, ancillary, 'wb')
        nhd_wb = checkGeoms(nhd_wb)
        nhd_wb['FCode'] = nhd_wb['FCode'].astype(int)
        nhd_wb_mask = nhd_wb[nhd_wb.FCode != 49300] # exclude estuary
        nhd_wb_mask = gpd.sjoin(nhd_wb_mask, stream, how='inner', op='intersects')           
        nhd_wb_mask = nhd_wb_mask[['OBJECTID', 'geometry']]
        # nhd_wb_mask = nhd_wb_mask.dropna(subset = ['GNIS_Name']) # exclude estuary
        wic_x_nhd_wb = gpd.sjoin(water_inside_channel, nhd_wb_mask, how='inner', op='intersects')
        wic_x_nhd_wb = wic_x_nhd_wb.drop_duplicates(subset=['value'], keep='first')

        # exclude any chespeake bay and esturine wb's classed as ponds
        # esturine = nhd_wb[nhd_wb.FCode == 49300]
        ches_bay = load_ches_bay(ches_bay_shp)
        wic_x_nhd_wb_tmp = wic_x_nhd_wb.copy()
        wic_x_nhd_wb_tmp = wic_x_nhd_wb_tmp[['value', 'perimeter', 'area', 'par', 'pps', 'geometry']]
        tmp_segs_in_bay = gpd.sjoin(wic_x_nhd_wb_tmp, ches_bay, how='inner', op='intersects')

        wic_x_nhd_wb_ids = list(wic_x_nhd_wb.value.unique())
        pond_ids = [id for id in wic_x_nhd_wb_ids if id not in tmp_segs_in_bay.value.to_list()]
        
        # NHDArea
        nhd_area = load_nhd(cofips, ancillary, 'area')
        nhd_area = checkGeoms(nhd_area)
        nhd_area['FCode'] = nhd_area['FCode'].astype(int)
        nhd_area_mask = nhd_area[nhd_area.FCode.isin([46000, 46003, 46006, 46007])] # streams and rivers only

        # identify pond/lake/river like polygons within the "water-in-channel universe"
        tmp_gdf = gpd.sjoin(water_inside_channel, nhd_area_mask, how='inner', op='intersects')
        non_pond_like_polys = list(set(tmp_gdf.value))
        wic_x_nhd_area = water_inside_channel[~water_inside_channel.value.isin(non_pond_like_polys)]
        wic_x_nhd_area = wic_x_nhd_area[~((wic_x_nhd_area.pps > 0.01) & (wic_x_nhd_area.par > 0.16))] # this selection is everything that's IS A POND
        ids_in_nhd_area = [id for id in wic_x_nhd_area.value.unique() if id not in tmp_segs_in_bay.value.to_list()]
        pond_ids += ids_in_nhd_area

        # water outside the channel
        water_outside_channel = water[~water.value.isin(water_inside_channel.value.to_list())]
        woc_x_nhd_area = gpd.sjoin(water_outside_channel, nhd_area_mask, how='inner', op='intersects')
        water_outside_channel = water_outside_channel[~water_outside_channel.value.isin(woc_x_nhd_area.value.to_list())]
        # exclude 1/10th (10% or 405 sq.m) or 1/20th (5% or 202 sq.m)
        water_outside_channel = water_outside_channel[water_outside_channel.area >= 202]
        # remove any segment that intersects with NHD Area
        tmp_segs_in_nhdarea = gpd.sjoin(water_outside_channel, nhd_area, how='inner', op='intersects')
        woc_ids_in_nhdarea = [id for id in water_outside_channel.value.unique() if id not in tmp_segs_in_nhdarea.value.to_list()]

        pond_ids += woc_ids_in_nhdarea

        # update pond values
        ponds = water.copy()
        ponds = ponds[['value', 'geometry', 'perimeter', 'area', 'par', 'pps']]
        ponds.loc[ponds.value.isin(pond_ids), 'pond'] = 1
        ponds.loc[ponds.pond.isnull(), 'pond'] = 0

        ponds.to_file(ponds_gpkg, layer='ponds', driver='GPKG')

        return ponds
    else:
        return gpd.read_file(ponds_gpkg, layer='ponds', driver='GPKG')

def load_nwi_wetlands(cofips, ancillary):
    nwi_wetlands_shp = list(ancillary.rglob(f'nwi.shp'))[0]

    if nwi_wetlands_shp.is_file():
        nwi = gpd.read_file(nwi_wetlands_shp)

        # remove freshwater ponds and lakes from NWI layer
        exclude_types = ['Freshwater Pond', 'Lake']

        nwi = nwi[~nwi.WETLAND_TY.isin(exclude_types)]
        nwi = nwi.explode()
        nwi = checkGeoms(nwi)
        nwi.loc[:, 'wid'] = range(1, nwi.shape[0] + 1)

        # calculate perimeter
        nwi['perimeter'] = nwi["geometry"].length

        # calculate area
        nwi['area'] = nwi["geometry"].area

        # calculate perimeter-area ratio [par] (perimeter / area)
        nwi['par'] = nwi['perimeter']/nwi['area'] # 

        # calculate polsby-popper score [pps] (4 * pi * area) / (perimeter ** 2)
        nwi['pps'] = (4 * pi * nwi['area']) / (nwi['perimeter'] ** 2)

        return nwi


def apply_nwi_conditions(nwi, ponds, tmp):
    nwi_filtered_gpkg = tmp / "nwi_filtered.gpkg"

    if not nwi_filtered_gpkg.is_file():

        # filter wetland types based on 
        wtypes = [
            'Freshwater Emergent Wetland',
            'Freshwater Forested/Shrub Wetland', 
            'Riverine',
        ]

        # remove elongated stream/string like features
        ids_to_keep = nwi[~nwi.WETLAND_TY.isin(wtypes)]['wid'].to_list()
        ids_to_keep += nwi.loc[(nwi.WETLAND_TY.isin(wtypes)) & (nwi.pps > 0.1)]['wid'].to_list()

        # clean NWI
        nwi = nwi[nwi.wid.isin(ids_to_keep)]

        # remove ponds from NWI that intersect wetlands
        ponds = ponds[['value', 'pond', 'geometry']]
        ponds = ponds[ponds.pond == 1]
        wetlands_x_ponds = gpd.sjoin(nwi, ponds, how='inner', op='intersects') # run intersect
        ids_to_remove = list(wetlands_x_ponds['wid'].unique())
        # remove all the ids from gdf & append gdf
        nwi = nwi[~nwi.wid.isin(ids_to_remove)]

        nwi.to_file(nwi_filtered_gpkg, layer='nwi_filtered', driver='GPKG')
        return nwi
    else:
        return gpd.read_file(nwi_filtered_gpkg, layer='nwi_filtered', driver='GPKG')

    return nwi

def extract_emergent_wetlands(cofips, ancillary, tmp):

    emergent_gpkg = tmp / 'emergent_wetlands.gpkg'

    if not emergent_gpkg.is_file():

        landcover = list(ancillary.rglob(f'*landcover*.tif'))[0]

        if landcover.is_file():
            emergent_value = 2

            if not emergent_value:
                return gpd.GeoDataFrame() # return empty geodataframe
            else:
                with rasterio.open(landcover) as lc:
                    # read in arrays
                    lc_arr = lc.read(1)

                    # conditional statement to mask out 
                    emergent_wetlands = np.where(lc_arr == emergent_value, 1, 0)

                    # eight neighbors 
                    eight_neighbors = [[1, 1, 1], 
                                    [1, 1, 1], 
                                    [1, 1, 1]]

                    emergent_wetlands_grouped, num_feats = label(emergent_wetlands, structure=eight_neighbors)

                    # delete the array!
                    del emergent_wetlands

                    emergent_wetlands_grouped = emergent_wetlands_grouped.astype('int32')

                    mask = emergent_wetlands_grouped != 0

                    results = (
                        {'properties': {'value': v}, 'geometry': s}
                        for i, (s, v) in enumerate(shapes(emergent_wetlands_grouped, mask=mask, transform=lc.transform))
                        )
                    
                    polygons = list(results)
                    gdf = gpd.GeoDataFrame.from_features(polygons)
                    gdf.crs = 'epsg:5070'
                    gdf = gdf.dissolve(by='value', as_index=False)

                    # calculate perimeter
                    gdf['perimeter'] = gdf["geometry"].length

                    # calculate area
                    gdf['area'] = gdf["geometry"].area

                    # calculate perimeter-area ratio [par] (perimeter / area)
                    gdf['par'] = gdf['perimeter']/gdf['area'] # 

                    # calculate polsby-popper score [pps] (4 * pi * area) / (perimeter ** 2)
                    gdf['pps'] = (4 * pi * gdf['area']) / (gdf['perimeter'] ** 2)
                    
                    gdf = checkGeoms(gdf)

                    gdf.to_file(emergent_gpkg, layer='emergent_wetlands', driver='GPKG')

                    return gdf
    else:
        return gpd.read_file(emergent_gpkg, layer='emergent_wetlands', driver='GPKG')


def nwi_tidal_overlay(nwi, county_folder):

    tidal_wetland_types = [
        'Estuarine and Marine Wetland', 
        'Estuarine and Marine Deepwater'
    ]
    wetland_types = list(nwi.WETLAND_TY.unique())
    is_tidal = any(wtype in tidal_wetland_types for wtype in wetland_types)

    if is_tidal:
        nwi_tidal_gpkg = county_folder / "nwi_tidal_overlay.gpkg"

        if not nwi_tidal_gpkg.is_file():
            nwi_esturine = nwi[nwi.WETLAND_TY.isin(tidal_wetland_types)]
            nwi_esturine_geom = unary_union(nwi_esturine['geometry'])
            nwi_emergent = nwi[nwi.WETLAND_TY == "Freshwater Emergent Wetland"] # Freshwater Emergent Wetland

            # dissolve nwi_emergent
            nwi_emergent_geom = unary_union(nwi_emergent['geometry'])
            nwi_emergent = gpd.GeoDataFrame({'geometry': [nwi_emergent_geom]}, crs=nwi_emergent.crs)
            nwi_emergent = nwi_emergent.explode()
            nwi_emergent.loc[:, 'wid'] = range(1, nwi_emergent.shape[0] + 1)

            tidal_emergent_ids = []
            for idx, row in nwi_emergent.iterrows():
                geom = row['geometry']
                touches_emergent = geom.touches(nwi_esturine_geom)
                if touches_emergent:
                    tidal_emergent_ids.append(row['wid'])
            
            nwi_tidal = nwi_esturine.copy()

            if tidal_emergent_ids:
                nwi_emergent = nwi_emergent[nwi_emergent.wid.isin(tidal_emergent_ids)]
                nwi_tidal  = nwi_tidal.append(nwi_emergent)

            if not nwi_tidal.empty:
                nwi_tidal = nwi_tidal.reset_index().drop(['index'], axis=1)
                nwi_tidal.to_file(nwi_tidal_gpkg, layer='nwi_tidal_overlay', driver='GPKG')
            return
    else:
        pass


def load_local_wetlands(cofips, ancillary, county_folder, pa_wetlands):

    local_wetlands_gpkg = ancillary / f'{cofips}_local_wetlands.gpkg'

    if local_wetlands_gpkg.is_file():
        return gpd.read_file(local_wetlands_gpkg, layer= f'{cofips}_local_wetlands', driver='GPKG')
    else:
        county_shp = ancillary / 'county_mask.shp'

        if county_shp.is_file():
            with fiona.open(county_shp, "r") as shapefile:
                county_geoms = [feature["geometry"] for feature in shapefile]

            # use LC's Affine
            landcover = list(ancillary.rglob(f'*landcover*.tif'))[0]

            lc = rasterio.open(landcover)

            with rasterio.open(pa_wetlands, 'r') as src:
                clip, clip_transform = mask(src, county_geoms, crop=True)
                clip = clip[0]

                county_mask = clip != 0

                results = (
                    {'properties': {'value': v}, 'geometry': s}
                    for i, (s, v) in enumerate(shapes(clip, mask=county_mask, transform=lc.transform))
                )

            polygons = list(results)
            gdf = gpd.GeoDataFrame.from_features(polygons)
            gdf.crs = 'epsg:5070'
            gdf = gdf[gdf.value < 4.0]
            gdf = gdf.explode()
            gdf.loc[:, 'wid'] = range(1, gdf.shape[0] + 1)

            gdf.to_file(local_wetlands_gpkg, layer= f'{cofips}_local_wetlands', driver='GPKG')

            return gdf

        # code to test output
        # out_meta = src.meta.copy()
        # out_meta['height'] = clip.shape[1]
        # out_meta['width'] = clip.shape[2]
        # out_meta['driver'] = 'GTiff'
        # out_meta['transform'] = clip_transform
        # with rasterio.open(file, 'w', **out_meta) as sink
        #     sink.write(1, array)

def dissolve_wetlands(cofips, no_of_wetlands, tmp):

    wetlands_diss_wo_ponds = tmp / 'wetlands_diss_wo_ponds.gpkg'

    if not wetlands_diss_wo_ponds.is_file():
        # if more than one gdf then perform overlay and then return overlay
        if len(no_of_wetlands) == 1:
            tmp = no_of_wetlands[0].copy()
            dissolve_geoms = unary_union(tmp.geometry)
        elif len(no_of_wetlands) == 2:
            nwi = no_of_wetlands[0].copy()
            ew = no_of_wetlands[1].copy()
            # tmp = gpd.overlay(no_of_wetlands[0], no_of_wetlands[1], how='union')
            dissolve_geoms = unary_union(list(nwi.geometry) + list(ew.geometry))
        elif len(no_of_wetlands) == 3:
            # dissolve all the three wetland layers by uid
            nwi = no_of_wetlands[0].copy()
            ew = no_of_wetlands[1].copy()
            local = no_of_wetlands[2].copy()
            dissolve_geoms = unary_union(list(nwi.geometry) + list(ew.geometry) + list(local.geometry))

        # print(type(dissolve_geoms), no_of_wetlands[0].crs)
        tmp = gpd.GeoDataFrame({'geometry': [dissolve_geoms]}, crs=no_of_wetlands[0].crs)
        tmp = tmp.explode()
        tmp.loc[:, 'wid'] = range(1, tmp.shape[0] + 1)
        tmp = tmp[['wid', 'geometry']]
        tmp.to_file(wetlands_diss_wo_ponds, layer='wetlands_diss_wo_ponds', driver='GPKG')
        return tmp
    else:
        return gpd.read_file(wetlands_diss_wo_ponds, layer='wetlands_diss_wo_ponds', driver='GPKG')

def load_ssurgo_layers(cofips, ancillary):
    ssurgo_shp = list(ancillary.rglob('ssurgo.shp'))[0]

    if ssurgo_shp.is_file():
        ssurgo = gpd.read_file(ssurgo_shp)
        ssurgo = ssurgo[['hydclprs', 'flodfreqdc', 'geometry']]
        ssurgo['hydclprs'] = ssurgo['hydclprs'].astype('int')

        hydric = ssurgo[ssurgo.hydclprs >= 1].reset_index()
        ff_soils = ssurgo[ssurgo.flodfreqdc == "Frequent"].reset_index()

        return hydric, ff_soils

def load_fema_layers(cofips, ancillary):
    fema_shp = list(ancillary.rglob('fema.shp'))[0]

    if fema_shp.is_file():
        fema = gpd.read_file(fema_shp)
        fema = fema[['SFHA_TF', 'geometry']].reset_index()

        # fema_geom = unary_union(fema.geometry)

        # fema = gpd.GeoDataFrame({'layer': ['fema'], 'geometry': [fema_geom]}, crs=fema.crs)

        return fema

def create_riverine_layers(cofips, ancillary, tmp):

    ssurgo_gpkg = tmp / 'ssurgo_layer.gpkg'
    fema_gpkg = tmp / 'fema_x_streams.gpkg'
    streams_gpkg = tmp / 'streams.gpkg'

    do_riverine_layers_exist = all([ssurgo_gpkg.is_file(), fema_gpkg.is_file(), streams_gpkg.is_file()])

    if not do_riverine_layers_exist:

        hydric, ff_soils = load_ssurgo_layers(cofips, ancillary)

        fema = load_fema_layers(cofips, ancillary)

        streams = load_david_streams(cofips, ancillary, buffer=False)

        # check geometries
        hydric = checkGeoms(hydric)
        ff_soils = checkGeoms(ff_soils)
        fema = checkGeoms(fema)
        streams = checkGeoms(streams)

        # hydrics polygons that intersect with streams
        hydric_x_streams = gpd.sjoin(hydric, streams, how='inner', op='intersects')
        hydric_x_streams = hydric_x_streams.drop_duplicates(subset=['index'], keep='first')

        # fema polygons that intersect with streams
        fema_x_streams = gpd.sjoin(fema, streams, how='inner', op='intersects')
        fema_x_streams = fema_x_streams.drop_duplicates(subset=['index'], keep='first')

        # add id and fix columns
        fema_x_streams = fema_x_streams[['index', 'geometry']]
        ff_soils = ff_soils[['index', 'geometry']]
        hydric_x_streams = hydric_x_streams[['index', 'geometry']]

        ssurgo_layer = gpd.overlay(ff_soils, hydric_x_streams, how='union').reset_index()
        ssurgo_layer = ssurgo_layer[['index', 'geometry']] # clean columns

        ssurgo_layer.to_file(ssurgo_gpkg, layer='ssurgo_layer', driver='GPKG')
        fema_x_streams.to_file(fema_gpkg, layer='fema_x_streams', driver='GPKG')
        streams.to_file(streams_gpkg, layer='streams', driver='GPKG')

        riverine_layers = {
            'ssurgo': ssurgo_layer, 
            'fema': fema_x_streams,
            'streams': streams,
        }

        return riverine_layers
    else:

        riverine_layers = {
            'ssurgo': gpd.read_file(ssurgo_gpkg, layer='ssurgo_layer', driver='GPKG'),
            'fema': gpd.read_file(fema_gpkg, layer='fema_x_streams', driver='GPKG'),
            'streams': gpd.read_file(streams_gpkg, layer='streams', driver='GPKG'),
        }
        
        return riverine_layers

def spatial_join_fema(arg):
    layer, overlay = arg
    tmp = gpd.sjoin(layer, overlay, how='inner', op='intersects') # run intersect
    return list(tmp.wid.unique())

def is_polygon_riverine_or_terrene(layer, riverine_layers, ID):

    # polygons that intersect ssurgo
    riverine_x_ssurgo = gpd.sjoin(layer, riverine_layers['ssurgo'], how='inner', op='intersects') # run intersect
    ids_x_ssurgo = list(riverine_x_ssurgo[ID].unique())

    # polygons that streams
    riverine_x_streams = gpd.sjoin(layer, riverine_layers['streams'], how='inner', op='intersects') # run intersect
    ids_x_streams = list(riverine_x_streams[ID].unique())

    # create mp args
    fema = riverine_layers['fema']
    gdfs = []
    for idx, row in fema.iterrows():
        data = {
            'index': [row['index']], 
            'geometry': [row['geometry']]
        }
        gdf = gpd.GeoDataFrame(
            data,crs=fema.crs
            )
        gdfs.append(gdf)

    st= timer()
    # mp polygons that intersect with fema
    args = [(layer, gdf) for gdf in gdfs]
    pool = mp.Pool(processes=8)
    results = pool.map(spatial_join_fema, args)
    pool.close()
    print(timer()-st)

    # flatten list
    ids_x_fema = list(itertools.chain(*results))

    # ids in riverine
    ids_in_riverine = list(set(ids_x_ssurgo + ids_x_fema + ids_x_streams)) 

    layer.loc[layer[ID].isin(ids_in_riverine), 'w_type'] = 'riverine'
    layer.loc[~layer[ID].isin(ids_in_riverine), 'w_type'] = 'terrene'

    return layer

def comparison(name, old, new):
    old, new = old.shape[0], new.shape[0]

    if old != new:
        result = f'Shapes mismatch! {old}(old) // {new}(new)'
    else:
        result = 'Shapes are same'
    print(
        f'''\t {name} >> {result}'''
    )

def compare_files(temp, t2, cofips):
    old = t2 / str(cofips)
    new = temp / str(cofips)

    # comparison
    old_ponds = gpd.read_file(old / "ponds.gpkg", layer="ponds", driver="GPKG")
    new_ponds = gpd.read_file(new / "ponds.gpkg", layer="ponds", driver="GPKG")

    old_pondss = old_ponds[old_ponds.pond == 1]
    new_pondss = new_ponds[new_ponds.pond == 1]
    comparison('ponds', old_pondss, new_pondss)

    old_nwi_filtered = gpd.read_file(old / "nwi_filtered.gpkg", layer="nwi_filtered", driver="GPKG")
    new_nwi_filtered = gpd.read_file(new / "nwi_filtered.gpkg", layer="nwi_filtered", driver="GPKG")
    comparison('nwi_filtered', old_nwi_filtered, new_nwi_filtered)

    try:
        old_nwi_tidal_overlay = gpd.read_file(old / "nwi_tidal_overlay.gpkg", layer="nwi_tidal_overlay", driver="GPKG")
        new_nwi_tidal_overlay = gpd.read_file(new / "nwi_tidal_overlay.gpkg", layer="nwi_tidal_overlay", driver="GPKG")
        comparison('nwi_tidal_overlay', old_nwi_tidal_overlay, new_nwi_tidal_overlay)
    except:
        print('\tnwi_tidal does not exist')

    old_wetlands_diss_wo_ponds = gpd.read_file(old / "wetlands_diss_wo_ponds.gpkg", layer="wetlands_diss_wo_ponds", driver="GPKG")
    new_wetlands_diss_wo_ponds = gpd.read_file(new / "wetlands_diss_wo_ponds.gpkg", layer="wetlands_diss_wo_ponds", driver="GPKG")
    comparison('wetlands_diss_wo_ponds', old_wetlands_diss_wo_ponds, new_wetlands_diss_wo_ponds)

    old_nontidal_wetlands = gpd.read_file(old / "nontidal_wetlands.gpkg", layer="nontidal_wetlands", driver="GPKG")
    new_nontidal_wetlands = gpd.read_file(new / "nontidal_wetlands.gpkg", layer="nontidal_wetlands", driver="GPKG")
    comparison('nontidal_wetlands', old_nontidal_wetlands, new_nontidal_wetlands)


if __name__ == '__main__':

    # conda activate cblcm_env

    # fipss = [
    #     10005, 24005, 24035, 24045, 36017, 42015, 
    #     42033, 42041, 42071, 51015, 51073, 51107, 
    #     54003, 54031, 51790, 51820
    #     ]

    # cofipses = [
    #     "augu_51015", "balt_24005", "berk_54003", "brad_42015", 
    #     "chen_36017", "clea_42033", "cumb_42041", "glou_51073", 
    #     "hard_54031", "lanc_42071", "loud_51107", "quee_24035", 
    #     "suss_10005", "wico_24045",
    # ]

    cofipses = [
        "brad_42015", "wico_24045",
    ]

    for cofips in cofipses:
        print(f"\n {cofips}: ")

        # folder paths
        data = Path(r'X:\ancillary\wetlands\data')
        county_folder = data / cofips
        tmp = county_folder / "tmp"
        ancillary = county_folder / "ancillary"

        ches_bay_shp =  Path(r"X:\ancillary\wetlands\misc_data\Chesapeake_Bay_104_Segments_albers_dissolved.shp")
        # pa_wetlands = Path(r"G:\ImageryServer\Wetlands_PA\Wetlands_2nd_Update_110216\wetlands_2013_pennsylvania_chesapeakebay.img")

        # compare_files(temp, t2, cofips)
        
        # key files:
        # ponds_gpkg = co_temp / 'ponds.gpkg'
        # emergent_gpkg = co_temp / 'emergent_wetlands.gpkg'
        # nwi_filtered_gpkg = co_temp / "nwi_filtered.gpkg"
        # nwi_tidal_gpkg = co_temp / "nwi_tidal_overlay.gpkg"
        # wetlands_diss_wo_ponds = co_temp / 'wetlands_diss_wo_ponds.gpkg'
        non_tidal_wetlands = county_folder / 'nontidal_wetlands.gpkg'

        # # deletes existing files
        # overwrite = True

        # if overwrite:
        #     try:
        #         # ponds_gpkg.unlink()
        #         nwi_filtered_gpkg.unlink()
        #         nwi_tidal_gpkg.unlink()
        #         wetlands_diss_wo_ponds.unlink()
        #         non_tidal_wetlands.unlink()
        #     except FileNotFoundError as e:
        #         print(e)
        #         pass

        # extract water
        water_gpkg = tmp / f'water.gpkg'
        if not water_gpkg.is_file():
            water = extract_water(cofips, ancillary)
            water.to_file(water_gpkg, layer='water', driver='GPKG')
        else:
            water = gpd.read_file(water_gpkg, layer='water', driver='GPKG')

        print('identify_ponds...')
        ponds = identify_ponds_ver2(cofips, county_folder, ancillary, water, ches_bay_shp)

        ### wetlands
        print('wetlands...')
        no_of_wetlands = []
        nwi = load_nwi_wetlands(cofips, ancillary)
        nwi_tidal_overlay(nwi, county_folder)
        nwi = apply_nwi_conditions(nwi, ponds, tmp)
        no_of_wetlands.append(nwi)
        
        # add emergent wetlands
        emergent_wetlands = extract_emergent_wetlands(cofips, ancillary, tmp)
        if not emergent_wetlands.empty:
            no_of_wetlands.append(emergent_wetlands)

        if cofips.split('_')[1][:2] in ['42']:
            print('local wetlands...')
            local_wetlands = load_local_wetlands(cofips, ancillary, county_folder, pa_wetlands)
            no_of_wetlands.append(local_wetlands)

        # remove ponds from wetlands
        print('wetlands_wo_ponds...')
        print(len(no_of_wetlands))
        wetlands_wo_ponds = dissolve_wetlands(cofips, no_of_wetlands, tmp)

        # print('riverine_layers...')
        riverine_layers = create_riverine_layers(cofips, ancillary, tmp)

        print("is_polygon_riverine_or_terrene")
        wetlands = is_polygon_riverine_or_terrene(wetlands_wo_ponds, riverine_layers, "wid")
        # wetlands = gpd.read_file(co_temp / f'{cofips}_wetlands.shp')
        wetlands.to_file(non_tidal_wetlands, layer='nontidal_wetlands', driver='GPKG')
