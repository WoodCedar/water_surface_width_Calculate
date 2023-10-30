# water_surface_width_Calculate

The program focuses on the calculation of the water surface width for high resolution (2m) and sentinel images (10m), which is based on the line lengths of the line files and eliminates the possible intervening land in the water.
程序主要针对高分辨率（2m）及哨兵影像（10m）的水面宽度计算，水面宽度是基于线文件的线长度并剔除中间可能存在的水中陆地后的宽度

The program supports fully automated processing of raw multi-band images. If the data is ndwi data, you need to comment out the ndwi calculation process in the program and add new ndwi paths to the classify session.
程序支持原始多波段影像的全自动处理，如果为ndwi数据则需要自行注释掉程序中的ndwi计算流程并添加新的ndwi路径给classify环节
