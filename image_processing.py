import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from skimage.filters import threshold_multiotsu
import numpy as np
from osgeo import gdal
import config
import re
from datetime import datetime
import glob

nodata_value = -9999
nodata = None

def reproject_shapefile_to_match_raster(gdf, raster_crs):
    return gdf.to_crs(raster_crs)


def rename_for_sorting(filepath):
    filename = os.path.basename(filepath)
    # Match format like: S2_2020-1-01_2020-2-01_NDWI_10m_cut
    match = re.search(r'^S2_(\d{4}-\d{1,2}-\d{1,2})_(\d{4}-\d{1,2}-\d{1,2})_(.*)$', filename)
    if match:
        start_date = datetime.strptime(match.group(1), '%Y-%m-%d')
        end_date = datetime.strptime(match.group(2), '%Y-%m-%d')
        rest_of_filename = match.group(3)
        new_filename = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_{rest_of_filename}"
        new_filepath = os.path.join(os.path.dirname(filepath), new_filename)
        
        # Check if file already exists and delete if it does
        if os.path.exists(new_filepath):
            os.remove(new_filepath)
        
        os.rename(filepath, new_filepath)
        return new_filepath

    # Return the filepath itself if it doesn't match the pattern
    return filepath

def open_tif_file(filepath):
    if filepath.endswith(('.tif', '.tiff')):
        return rasterio.open(filepath)
    return None

def handle_error(error_file, filepath, error_message, custom_message):
    print(f"Error processing {filepath}: {error_message}")
    error_file.write(f"{filepath}: {custom_message}: {error_message}\n")


def clip_rasters(input_folder, output_folder, shp_file, error_file_path):
    gdf = gpd.read_file(shp_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    clipped_rasters = []
    for file in os.listdir(input_folder):
        if file.endswith(('.tif', '.tiff')):
            try:
                input_raster_path = os.path.join(input_folder, file)
                
                with rasterio.open(input_raster_path) as src:
                    if gdf.crs != src.crs:
                        gdf = reproject_shapefile_to_match_raster(gdf, src.crs)
                    out_image, out_transform = mask(src, [mapping(gdf.geometry[0])], crop=True, nodata=nodata_value)
                    out_meta = src.meta.copy()

                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": np.uint16(65535)
                })

                out_image = out_image.astype(np.float32)
                out_image[out_image == nodata_value] = np.nan

                output_raster_path = os.path.join(output_folder, os.path.splitext(file)[0] + '_cut.tif')
                with rasterio.open(output_raster_path, 'w', **out_meta) as dest:
                    dest.write(out_image.astype(rasterio.float32))

                clipped_rasters.append(output_raster_path)
                print(f"Clipped raster saved as: {output_raster_path}")
                
            except Exception as e:
                with open(error_file_path, "a") as error_file:
                    handle_error(error_file, file, str(e), "Error during raster clipping")

    return clipped_rasters

def calculate_ndwi(input_folder: str, output_folder: str, error_file_path: str,typeSen:str) -> list:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ndwi_files = []
    for file in os.listdir(input_folder):
        try:
            if file.endswith(('.tif', '.tiff')):
                gf2_image_filepath = os.path.join(input_folder, file)
                with rasterio.open(gf2_image_filepath) as src:
                    if typeSen == "GF":
                        green_band = src.read(2)
                        nir_band = src.read(4)
                        profile = src.profile
                    else:
                        green_band = src.read(3)
                        nir_band = src.read(8)
                        profile = src.profile

                epsilon = 1e-8
                ndwi = (green_band.astype(float) - nir_band.astype(float)) / (green_band.astype(float) + nir_band.astype(float) + epsilon)
                output_filename_ndwi = os.path.join(output_folder, os.path.splitext(file)[0] + "_NDWI.tif")

                index_profile = profile.copy()
                index_profile.update(dtype=rasterio.float32, count=1)

                with rasterio.open(output_filename_ndwi, 'w', **index_profile) as dst:
                    dst.write(ndwi.astype(rasterio.float32), 1)

                ndwi_files.append(output_filename_ndwi)
                print(f"Generated file: {output_filename_ndwi}")
        except ValueError as e:
            print(f"Error processing {file}: {str(e)}")
            if 'After discretization into bins' in str(e):
                with open(error_file_path, "a") as error_file:
                    error_file.write(f"{file}: 无法进行阈值判断: {str(e)}\n")
    return ndwi_files

def classify_images(input_folder: str, output_directory: str, error_file_path: str) -> list:
    os.makedirs(output_directory, exist_ok=True)

    classified_files = []
    for file in os.listdir(input_folder):
        try:
            if file.endswith(('.tif', '.tiff')):
                input_tif = os.path.join(input_folder, file)
                with rasterio.open(input_tif) as src:
                    data = src.read().astype(np.float32)
                    profile = src.profile

                data[data == -9999] = np.nan
                data[np.isnan(data)] = np.nan 
                scaled_data = np.empty_like(data, dtype=np.uint8)
                for i in range(data.shape[0]):
                    scaled_data[i] = ((data[i] - np.nanmin(data[i])) / (np.nanmax(data[i]) - np.nanmin(data[i])) * 255).astype(np.uint8)
                thresholds = [threshold_multiotsu(band, nbins=1024)[-1] for band in scaled_data]
                output_txt = os.path.join(output_directory, 'thresholds2.txt')
                with open(output_txt, 'w') as txt_file:
                    txt_file.write(f'NDWI Threshold: {thresholds[0]}\n')

                classification = np.full_like(scaled_data[0], 0, dtype=np.uint8)
                classification[(scaled_data[0] > thresholds[0])] = 1
                classification[np.isnan(data[0])] = 255

                output_tif = os.path.join(output_directory, os.path.splitext(file)[0] + '_classified.tif')
                profile.update(dtype=rasterio.uint8, count=1, nodata=255)

                with rasterio.open(output_tif, 'w', **profile) as dst:
                    dst.write(classification, 1)

                classified_files.append(output_tif)
                print(f"Generated file: {output_tif}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            with open(error_file_path, "a") as error_file:
                error_file.write(f"{file}: 分类错误: {str(e)}\n")
    return classified_files


def clip_and_calculate_width(tif: str, shp_file: str,contype) -> dict:
    if "_clipped" not in tif:
        gdf = gpd.read_file(shp_file)
        
        with rasterio.open(tif) as src:
            if gdf.crs != src.crs:
                gdf = reproject_shapefile_to_match_raster(gdf, src.crs)
                
            out_image, out_transform = mask(src, [mapping(gdf.geometry[0])], crop=True, nodata=-9999)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": 255
        })

        out_image = out_image.astype(np.float32)
        out_image[out_image == -9999] = np.nan

        clipped_raster_path = os.path.join(os.path.dirname(tif), os.path.splitext(os.path.basename(tif))[0] + '_clipped.tif')
        with rasterio.open(clipped_raster_path, 'w', **out_meta) as dest:
            dest.write(out_image.astype(rasterio.float32))

        raster = gdal.Open(clipped_raster_path)
        band = raster.GetRasterBand(1)

        nodata_value = band.GetNoDataValue()
        transform = raster.GetGeoTransform()
        pixel_width = transform[1]
        pixel_height = -transform[5]

        array = band.ReadAsArray()
        array[array == nodata_value] = 0
        index_1 = np.where(array == 1)

        height = len(np.unique(index_1[0]))
        width = len(np.unique(index_1[1]))
        if contype == "GF":
            distance = np.sqrt(np.power(height * pixel_height * 111.31955 * 1000, 2) + np.power(width * pixel_width * 111.31955 * 1000, 2))
        elif contype == "S2GEE":
            distance = max(np.sqrt(np.power(height * 10, 2) + np.power(width * 10, 2)) - 10, 0)
        else:
            distance = max(np.sqrt(np.power(height * pixel_height, 2) + np.power(width * pixel_width, 2)) - 10, 0)
         
        filename_parts = os.path.basename(clipped_raster_path).split("_")

        return {
            "filename_part_0": filename_parts[0],
            "filename_part_4": filename_parts[4],
            "distance": distance,
            "clipped_raster_path": clipped_raster_path
        }
    
    print(f"No data found for {tif}")
    return {}
    


def width_calculate(input_folder, shp_file, error_file_path, contype):
    tifs = glob.glob(os.path.join(input_folder, "*.tif"))
    widths = {}
    
    with open(error_file_path, "a") as error_file:
        for tif in tifs:
            try:
                width_info = clip_and_calculate_width(tif, shp_file, contype)
                distance = width_info.get('distance', 0)
                widths[tif] = width_info
                
                if distance < 20:
                    handle_error(error_file, tif, "长度小于20，请核实", "宽度计算错误")
                print(f"{tif} ----- {distance}")
            except Exception as e:
                handle_error(error_file, tif, str(e), "宽度计算错误")
    return widths