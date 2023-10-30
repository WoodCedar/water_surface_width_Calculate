import os
import config
import pandas as pd
from image_processing import clip_rasters, calculate_ndwi, classify_images, width_calculate, rename_for_sorting

def main():
    error_file_path = config.error_txt
    

    if os.path.exists(error_file_path):
        os.remove(error_file_path)
    

    
    clipped_folder = os.path.join(config.output_folder, "clipped")
    clipped_files = clip_rasters(config.input_folder, clipped_folder, config.shp_file, error_file_path)
    
    new_filepaths = [rename_for_sorting(f) for f in clipped_files]
 
    # Sort the date
    

    ndwi_folder = os.path.join(config.output_folder, "ndwi")
    ndwi_files = calculate_ndwi(clipped_folder, ndwi_folder, error_file_path, config.type)


    classified_folder = os.path.join(config.output_folder, "classified")
    classified_files = classify_images(clipped_folder, classified_folder, error_file_path)

    widths = width_calculate(classified_folder, config.Width_shp, error_file_path, contype=config.type)
    
    widths_df = pd.DataFrame(list(widths.values()))
    widths_df.to_excel(os.path.join(config.output_folder, "widths.xlsx"), index=False)

    print("All operations completed successfully.")

if __name__ == "__main__":
    main()
