import os
import glob
from osgeo import gdal, gdalconst
import subprocess
import gdalTools
import cv2 as cv
import numpy as np
from tqdm import tqdm
from config_eval import ConfigEval
import sys
from Redundancy_predict_edge_only import predict
from Redundancy_predict_segmentation_only import predict_seg
from osgeo import ogr, osr
import os
import gdalTools
from shutil import rmtree, copyfile


def polygonize(imagePath, raster_path, forest_shp_path):
    pwd = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.realpath(os.path.join(pwd, 'polygonize/')))
    polygonize_exe = os.path.realpath(os.path.join(pwd, 'polygonize/polygonize0529.exe'))
    # polygonize_exe = os.path.realpath(os.path.join(pwd, 'polygonize/polygonize1126.exe'))
    polygonize_path = os.path.realpath(os.path.join(pwd, 'polygonize/polygonize.config'))

    rmHole = "400"
    simpoly = "2"

    scale = "3"
    with open(polygonize_path,'w') as f_config:
        f_config.write("--image=" + imagePath+'\n')
        f_config.write("--edgebuf="+raster_path+'\n')
        f_config.write("--line="+forest_shp_path+'\n')
        f_config.write("--rmHole=" + rmHole + '\n')
        f_config.write("--simpoly=" + simpoly + '\n')
        f_config.write("--scale=" + scale)
    f_config.close()
    subprocess.call(polygonize_exe)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    import time
    t1 = time.time()
    print('starting parsing arguments')
    cfg = ConfigEval()
    f = open('config_test.txt')
    data = f.readlines()
    cfg.data_path = data[0].replace('\n', '')    ####  the path of your image, such as "D:/test/test.png"
    cfg.model_path = data[1].replace('\n', '')   ####  the path of edge model, such as "D:/test/edge.pth"
    outPath = data[2].replace('\n', '')           ####  the path of out shapefile, such as "D:/test/result.shp"
    cfg.seg_model_path = data[3].replace('\n', '')    ####  the path of segmentation model, such as "D:/test/seg.pth"
    outRoot = os.path.split(outPath)[0]
    outName = os.path.split(outPath)[-1].split('.')[0]

    print('starting inferring edge')
    predict(cfg)
    print('starting inferring binary map')
    predict_seg(cfg)
    out_shp_path = cfg.save_path + '_shp'
    image_root = cfg.data_path
    if os.path.exists(out_shp_path):
        rmtree(out_shp_path)

    mkdir(out_shp_path)

    imgList = [cfg.data_path]
    print('starting polygon')
    for imgPath in imgList:
        # imageName = os.path.split(imgPath)[-1].split('.')[0]
        # rasterPath = glob.glob(cfg.save_path + f'_ms/{imageName}*tif')[0]
        # assert os.path.exists(imgPath), print(f'please cheack {imgPath}')
        # assert os.path.exists(rasterPath), print(f'please cheack {rasterPath}')
        # print(rasterPath)
        # shpName = imageName + '.shp'
        # shpPath = os.path.join(out_shp_path, shpName)
        # polygonize(imgPath, rasterPath, shpPath, cfg.pwd)

        imageName = os.path.split(imgPath)[-1].split('.')[0]
        rasterPath = glob.glob(cfg.save_path + f'_ms/{imageName}*tif')[1]
        assert os.path.exists(imgPath), print(f'please cheack {imgPath}')
        assert os.path.exists(rasterPath), print(f'please cheack {rasterPath}')
        print(rasterPath)
        shpName = imageName + '_final.shp'
        shpPath = os.path.join(out_shp_path, shpName)
        polygonize(imgPath, rasterPath, shpPath)

        rasterPath_seg = glob.glob(cfg.save_path + f'_seg/{imageName}*tif')[0]
        gdalTools.ZonalStatisticsAsTable(rasterPath_seg, shpPath)

        shp_baseName = os.path.basename(shpPath).split('.')[0]
        shp_root = os.path.split(shpPath)[0]
        mkdir(outRoot)
        shpList = glob.glob(f'{shp_root}/{shp_baseName}*')
        files = os.listdir(shp_root)
        for f in files:
            if f[-4:] == '.shp':
                shp_path = os.path.join(shp_root, f)
                ds = ogr.Open(shp_path, 0)
                layer = ds.GetLayer()
                layer.SetAttributeFilter("majority = 1 or majority = 2")

                driver = ogr.GetDriverByName('ESRI Shapefile')
                out_ds = driver.CreateDataSource(outPath)
                out_layer = out_ds.CopyLayer(layer, 'temp')
                del layer, ds, out_layer, out_ds

        polygonPath = outPath
        imgPath = rasterPath_seg
        linePath = os.path.join(outRoot, 'line.shp')
        out_polygon_path = os.path.join(outRoot, 'polygon.tif')
        out_line_path = os.path.join(outRoot, 'line.tif')
        out_line_dn_path = os.path.join(outRoot, 'line_dn.tif')
        gdalTools.pol2line(polygonPath, linePath)
        gdalTools.shp2Raster(linePath, imgPath, out_line_path, nodata=0)
        gdalTools.shp2Raster(polygonPath, imgPath, out_polygon_path, nodata=0)
        copyfile(rasterPath, out_line_dn_path)

        print(f'spend time:{time.time()-t1}s')
