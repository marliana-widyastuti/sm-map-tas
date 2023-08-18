""" utils.py
This module displays the summary of the tabular data contained in a CSV file 
"""
import ee
import pandas as pd
import numpy as np
from datetime import datetime
from keras.models import load_model
import os
import joblib
import rasterio
from google.cloud import storage
from rasterio.merge import merge
from keras import backend as K
import tensorflow_probability as tfp
import time
import shutil
import urllib.request

print("package utils is loaded")

ee.Initialize()
class runArea:

    # init method or constructor
    def __init__(self, date, res, xmin, xmax, ymin, ymax, model_path, output): #output : os.getcwd()
        if output.__contains__("\\"):
            self.sep = "\\"
        else:
            self.sep = "/"
        try:
            os.mkdir(output + self.sep + 'output')
        except FileExistsError:
            print("direcory already exist")
        try:
            os.mkdir(output + self.sep + 'output'+ self.sep + date)
        except FileExistsError:
            print("direcory already exist")
        self.out = output + self.sep + 'output'+ self.sep + date ## directory for placing the outputs

        self.date = str(date)  ## format YYYYmmdd as input
        self.res = int(res)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.datetime = datetime.strptime(date, '%Y%m%d')  
        self.smap_bands = [#'smap', 'ssmap','smap1', 'ssmap1', 'smap2', 'ssmap2', 'smap3', 'ssmap3',
                           'smap4', 'ssmap4', 'smap5', 'ssmap5', 'smap6', 'ssmap6',
                           'smap7', 'ssmap7']
        self.pattern = ('{}_{}_{}-{}_{}-{}').format(self.date, self.res, int(self.xmin*100), int(self.xmax*100), int(self.ymin*100), int(self.ymax*100))
        smap_cols = ['smap{}'.format(str(n)) for n in range(4,8)]
        ssmap_cols = ['ssmap{}'.format(str(n)) for n in range(4,8)]
        rains_cols = ['rain']+['rain{}'.format(str(n)) for n in range(1,4)]
        lstm_cols = smap_cols+ssmap_cols
        mlp_cols = rains_cols+['tmin','tmax',
                    'irri', 'past', 'agri', 'fore', 'sava',
                    'elevation',
                    'AWC1', 'AWC2', 'AWC3', 'AWC4', 
                    'SOC1', 'SOC2', 'SOC3', 'SOC4', 
                    'CLY1', 'CLY2', 'CLY3', 'CLY4']

        self.var_in = lstm_cols + mlp_cols
        self.lmod = os.listdir(model_path)
        self.key = 'sm-tassie-e4591e32eeab.json'
        

    def reclass_LU():
        DEM = ee.Image("CGIAR/SRTM90_V4")
        LU = ee.Image("users/ignaciofuentessanroman/AU/clum_50m1218m")
        MCD = ee.ImageCollection("MODIS/006/MCD12Q1")

        MCD = MCD.select('LC_Type1').sort(prop='system:time_start', opt_ascending=False).first()
        #Land Use
        LUx = LU.where(LU.lte(210), 1) #pasture
        LUx = LUx.where(LU.gt(210).And(LU.lte(314)), 2) # forest
        LUx = LUx.where(LU.gt(319).And(LU.lte(325)), 1)
        LUx = LUx.where(LU.gte(330).And(LU.lte(360)), 3) # agri
        LUx = LUx.where(LU.gt(359).And(LU.lte(400)), 1) # 
        LUx = LUx.where(LU.gte(410).And(LU.lt(420)), 4) # irri
        LUx = LUx.where(LU.gte(420).And(LU.lt(430)), 4) # 
        LUx = LUx.where(LU.gte(430).And(LU.lte(465)), 4)
        LUx = LUx.where(LU.gte(510).And(LU.lt(600)), 1) 
        LUx = LUx.where(LU.gt(600).And(LU.lt(650)), 1)
        LUx = LUx.where(LU.gte(650), 1)
        LUx = LUx.where(LUx.eq(1).And(MCD.lte(5)), 2)
        LUx = LUx.where(LUx.eq(1).And(MCD.gte(5)).And(MCD.lte(6)), 5)
        LUx = LUx.where(LUx.eq(1).And(MCD.gte(7)).And(MCD.lte(9)), 6)
        irri = LUx.eq(4)
        past = LUx.eq(1)
        agri = LUx.eq(3)
        fore = LUx.eq(2)
        sava = LUx.gte(5)
        LUx = ee.Image.cat([irri.rename('irri'), past.rename('past'), agri.rename('agri'),
                            fore.rename('fore'), sava.rename('sava'), DEM.rename('elevation')])
        return LUx
    
    def get_daily_smap(self):
        SMAP = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007")
        band_smap = SMAP.select(['sm_surface', 'sm_rootzone'])
        startDate = self.datetime - pd.DateOffset(days=7)
        startDate = ee.Date(startDate.strftime('%Y-%m-%d'))
        endDate = self.datetime - pd.DateOffset(days=4)
        endDate = ee.Date(endDate.strftime('%Y-%m-%d'))
        numberOfDays = endDate.difference(startDate, 'days')


        def week(i):
            start = startDate.advance(i, 'days')
            end = start.advance(1, 'days')
            day_smap = band_smap.filterDate(start, end).mean()
            return day_smap.set('system:time_start', start)

        daily_smap = ee.ImageCollection(ee.List.sequence(0, numberOfDays).map(week))
        sort_smap = daily_smap.sort(prop='system:time_start', opt_ascending=False)
        sort_smap = sort_smap.toBands().slice(0,8).rename(self.smap_bands)
        return sort_smap
    
    def get_soil():
        AWC1 = ee.Image('users/marlianatw/AWC_Tas_x0to5cm_predicted_mean').rename('AWC1')
        AWC2 = ee.Image('users/marlianatw/AWC_Tas_x5to15cm_predicted_mean').rename('AWC2')
        AWC3 = ee.Image('users/marlianatw/AWC_Tas_x15to30cm_predicted_mean').rename('AWC3')
        AWC4 = ee.Image('users/marlianatw/AWC_Tas_x30to60cm_predicted_mean').rename('AWC4')

        SOC1 = ee.Image('users/marlianatw/SOC_0to5cm').rename('SOC1')
        SOC2 = ee.Image('users/marlianatw/SOC_5to15cm').rename('SOC2')
        SOC3 = ee.Image('users/marlianatw/SOC_15to30cm').rename('SOC3')
        SOC4 = ee.Image('users/marlianatw/SOC_30to60cm').rename('SOC4')

        CLY1 = ee.Image('users/marlianatw/Clay_Tas_x0to5cm_mean').rename('CLY1')
        CLY2 = ee.Image('users/marlianatw/Clay_Tas_x5to15cm_mean').rename('CLY2')
        CLY3 = ee.Image('users/marlianatw/Clay_Tas_x15to30cm_mean').rename('CLY3')
        CLY4 = ee.Image('users/marlianatw/Clay_Tas_x30to60cm_mean').rename('CLY4')
        return AWC1.addBands([AWC2, AWC3, AWC4, SOC1, SOC2, SOC3, SOC4, CLY1, CLY2, CLY3, CLY4])
    
    def get_weather(self):
        str_date = self.datetime.strftime('%Y%m%d')
        str_date1 = (self.datetime - pd.DateOffset(days=1)).strftime('%Y%m%d')
        str_date2 = (self.datetime - pd.DateOffset(days=2)).strftime('%Y%m%d')
        str_date3 = (self.datetime - pd.DateOffset(days=3)).strftime('%Y%m%d')

        RAIN = ee.Image(f'users/marlianatw/TAS/RainPrediction24hr_{str_date}0900AEST').rename('rain')
        RAIN1 = ee.Image(f'users/marlianatw/TAS/RainPrediction24hr_{str_date1}0900AEST').rename('rain1')
        RAIN2 = ee.Image(f'users/marlianatw/TAS/RainPrediction24hr_{str_date2}0900AEST').rename('rain2')
        RAIN3 = ee.Image(f'users/marlianatw/TAS/RainPrediction24hr_{str_date3}0900AEST').rename('rain3')

        TMAX = ee.Image(f'users/marlianatw/TAS/TmaxPrediction_{str_date}090000AEST').rename('tmax')
        TMIN = ee.Image(f'users/marlianatw/TAS/TminPrediction_{str_date}090000AEST').rename('tmin')

        return RAIN.addBands(RAIN1).addBands(RAIN2).addBands(RAIN3).addBands(TMAX).addBands(TMIN)
    
    def combineImages(self):
        LUx = runArea.reclass_LU()
        smap = self.get_daily_smap()
        soil = runArea.get_soil()
        clim = self.get_weather()
        img = clim.addBands(smap).addBands(LUx).addBands(soil)
        return img.select(self.var_in)

    def exportImage(self):
        geo_exp = ee.Geometry.BBox(self.xmin, self.ymin, self.xmax, self.ymax)
        img = self.combineImages().clip(geo_exp)
        # crs = img.projection().getInfo()['crs']
        reImg = img.resample('bilinear')
        task = ee.batch.Export.image.toCloudStorage(reImg.toFloat(), 
                                             bucket='sm-tassie',
                                             fileNamePrefix=('raw_input_data/img_{}').format(self.pattern),
                                             fileFormat='GeoTIFF',
                                             region=geo_exp,
                                             scale=self.res,
                                             maxPixels=1e13)
        task.start()
        print('Exporting images of date: ', self.date)
        # Monitor the task.
        while task.status()['state'] in ['READY', 'RUNNING']:
            print(task.status()['state'])
            time.sleep(10)
        else:
            print(task.status()['state'])

## Apply model on images saved in Cloud Storage ------------------------------------------#######
    def run_mod(raster_path, tmod, out_path):
        scaleCov = joblib.load('scalerTAS.save')
        maxs = scaleCov.data_max_
        mins = scaleCov.data_min_
        with rasterio.open(raster_path) as src:
            out_meta = src.meta.copy()
        out_meta.update({
            "count": 2,
        })
        
        with (rasterio.open(raster_path, "r") as src, rasterio.open(out_path, "w", **out_meta) as dst):
            for _, win in src.block_windows():
                arr = src.read(window=win)
                normalised = ((arr - mins.reshape((mins.shape[0], 1, 1)))/(maxs.reshape((maxs.shape[0], 1, 1)) - mins.reshape((mins.shape[0], 1, 1))))
                container = []
                container2 = []
                for n in range(arr.shape[2]):
                    pred = tmod.predict(np.swapaxes(normalised[:,:,n], 1, 0),verbose=0)
                    container.append(pred[:, 0].ravel()) # pred1
                    container2.append(pred[:, 1].ravel()) # pred2

                dst.write(np.swapaxes(np.vstack(container), 1, 0), indexes=1, window=win)
                dst.write(np.swapaxes(np.vstack(container2), 1, 0), indexes=2, window=win)
                print(win)

    def get_list_img(self, dir):
        client = storage.Client.from_service_account_json(json_credentials_path=self.key)
        blobs = client.list_blobs('sm-tassie', prefix=dir)
        blob = []
        for x in blobs:
            blob.append(x.name)
        return blob
    
    def tile(self):
        blob = self.get_list_img(dir='raw_input_data')
        blob = [x for x in blob if self.pattern in x]
        n = len(blob)
        return n

    def exe_mod(self, model_path):
        try:
            os.mkdir(self.out + self.sep + 'raw' )
        except FileExistsError:
            print("direcory already exist")

        def ccc(y_true, x_true):
            uy, ux = K.mean(y_true), K.mean(x_true)
            sxy = tfp.stats.covariance(y_true, x_true)
            sy, sx = tfp.stats.variance(y_true), tfp.stats.variance(x_true)
            E = 2*sxy/(sy+sx+K.pow(uy-ux, 2))
            return 1-E
        
        for i, mod in enumerate(self.lmod):
            path_mod = model_path+mod
            tmod = load_model(path_mod, custom_objects={'ccc':ccc})
            
            blob = self.get_list_img(dir='raw_input_data')
            blob = [x for x in blob if self.pattern in x]
            n = self.tile()
            if n > 1:
                for count, ob in enumerate(blob):
                    fname = ob[19:]  ## raw_input_data/img_{fname}
                    out_path = (f'{self.out}/raw/SM_{mod}_{fname}')
                    raster_path = f'https://storage.googleapis.com/sm-tassie/{ob}'
                    
                    print((f'Processing.. model {mod} tile {count+1}'))
                    runArea.run_mod(raster_path, tmod, out_path)
            else:
                ob = blob[0]
                fname = ob[19:] ## raw_input_data/img_{fname}
                out_path = (f'{self.out}/raw/SM_{mod}_{fname}')
                raster_path = f'https://storage.googleapis.com/sm-tassie/{ob}'
                
                print(('Processing.. model {}').format(mod))
                runArea.run_mod(raster_path, tmod, out_path)

### Calculate SM average and standard deviation
    def combined_SM(self):
        try:
            os.mkdir(self.out + self.sep + 'merged')
        except FileExistsError:
            print("direcory already exist")

        list_raw = os.listdir(self.out + self.sep + 'raw')
        n = self.tile()
        if n > 1:
            for _, mod in enumerate(self.lmod):
                pattern = (f'{mod}_{self.pattern}')
                img_1mod = [x for x in list_raw if pattern in x]
                ras=[]
                for _, tile in enumerate(img_1mod):
                    path = f'{self.out}/raw/{tile}'
                    img_tile = rasterio.open(path)
                    ras.append(img_tile)
                    with rasterio.open(path) as src:
                        out_meta = src.meta.copy()

                mosaic, output = merge(ras)
                merge_meta = out_meta
                merge_meta.update({
                    'height': mosaic.shape[1],
                    'width':mosaic.shape[2],
                    "transform": output})

                output_path = (f'{self.out}/merged/SM_{pattern}.tif')
                with rasterio.open(output_path, 'w', **merge_meta) as m:
                    m.write(mosaic)
        else:
            src = f"{self.out}/raw/"
            dest = f"{self.out}/merged/"
            try:    
                [shutil.copy(src+fn, dest) for fn in list_raw]
            except:
                print('files already exist in', dest)
    
    def calc_sm(self):
        try:
            os.mkdir(self.out + self.sep + 'map')
        except FileExistsError:
            print("direcory already exist")
        list_sm = [f'{self.out}/merged/'+x for x in os.listdir(f'{self.out}/merged/') if self.pattern in x] ## filter based on pattern
        print(list_sm)
        dsur= list()
        dsub= list()
        for sm in list_sm:
            with rasterio.open(sm) as src:
                sur = src.read(1)
                sub = src.read(2)
            dsur.append(sur)
            dsub.append(sub)

        dsur_mean = np.mean(dsur, axis =0)
        dsur_sd = np.std(dsur, axis =0)
        dsub_mean = np.mean(dsub, axis =0)
        dsub_sd = np.std(dsub, axis =0)
        band_sm = [dsur_mean, dsur_sd, dsub_mean, dsub_sd]
        print(len(band_sm))

        self.saveMap(dsur_mean, f'L1_{self.res}_mean_SM_{self.date}')
        self.saveMap(dsur_sd, f'L1_{self.res}_sd_SM_{self.date}')
        self.saveMap(dsub_mean, f'L2_{self.res}_mean_SM_{self.date}')
        self.saveMap(dsub_sd, f'L2_{self.res}_sd_SM_{self.date}')

    def saveMap(self, src, fname):
        list_sm = [f'{self.out}/merged/'+x for x in os.listdir(f'{self.out}/merged/') if self.pattern in x] ## filter based on pattern
        upmeta = rasterio.open(list_sm[0])
        upmeta = upmeta.meta.copy()
        upmeta.update(dtype=rasterio.float32)
        upmeta.update({
                        "count": 1,
                    })
        out_img = (f'{self.out}/map/{fname}.tif')
        with rasterio.open(out_img, 'w', **upmeta) as dest:
            dest.write(src, indexes=1)

    def SMtoGCS(self):
        lmap = os.listdir(self.out + self.sep + 'map')
        for fname in lmap:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=self.key
            storage_client = storage.Client()
            bucket = storage_client.get_bucket("sm-tassie") # your bucket name
            blob = bucket.blob(f'output/{fname}')
            blob.upload_from_filename(f'{self.out}{self.sep}map{self.sep}{fname}')
        
        a = os.stat(f'{self.out}{self.sep}map{self.sep}{fname}')
        sleep = int(a.st_size/1000000)
        time.sleep(sleep)


#### download weather data
class tasData:
    def __init__(self, date, output):
        if output.__contains__("\\"):
            self.sep = "\\"
        else:
            self.sep = "/"
        try:
            os.mkdir(output + self.sep + 'weather_data')
        except FileExistsError:
            print("direcory already exist")
    
        self.date = date
        self.pathWeather = output + self.sep + 'weather_data'
        self.key = 'sm-tassie-e4591e32eeab.json'
    
    def localToGCS(self, fname):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=self.key
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("sm-tassie") # your bucket name
        blob = bucket.blob(f'weather_data/{fname}')
        blob.upload_from_filename(f'{self.pathWeather}/{fname}')
        
        a = os.stat(f'{self.pathWeather}/{fname}')
        sleep = int(a.st_size/1000000)
        time.sleep(sleep)

    def GCStoGEE(self, fname):
        gcloud_uri = f"gs://sm-tassie/weather_data/{fname}"

        request_id = ee.data.newTaskId()[0]
        params = {
            "name": f"projects/earthengine-legacy/assets/users/marlianatw/TAS/{fname[:-4]}",
            "tilesets": [{"sources": [{"uris": [gcloud_uri]}]}]
        }
        ee.data.startIngestion(request_id=request_id, params=params)

    def uploadToGEE(self):
        list_fname = [file for file in os.listdir(self.pathWeather) if self.date in file]
        print(list_fname)
        for _, fname in enumerate(list_fname):
            self.localToGCS(fname)
            self.GCStoGEE(fname)
            time.sleep(60)
            print(f'File {fname} has been uploaded to GEE')
    
    def downloadWeather(self):
        #rain
        url = f'https://www.dropbox.com/scl/fo/izet3gk0uh0hko1t2urk7/h/RainPrediction_{self.date}0900AEST.tif?rlkey=nbvlekulead5tly8tho1hy7o9&dl=1'
        file = f'{self.pathWeather}/RainPrediction24hr_{self.date}0900AEST.tif'
        urllib.request.urlretrieve(url, file)
        time.sleep(20)
        
        #tmin
        url = f'https://www.dropbox.com/scl/fo/izet3gk0uh0hko1t2urk7/h/TmaxPrediction_{self.date}090000AEST.tif?rlkey=nbvlekulead5tly8tho1hy7o9&dl=1'
        file = f'{self.pathWeather}/TmaxPrediction_{self.date}090000AEST.tif'
        urllib.request.urlretrieve(url, file)
        time.sleep(20)
        
        #tmax
        url = f'https://www.dropbox.com/scl/fo/izet3gk0uh0hko1t2urk7/h/TminPrediction_{self.date}090000AEST.tif?rlkey=nbvlekulead5tly8tho1hy7o9&dl=1'
        file = f'{self.pathWeather}/TminPrediction_{self.date}090000AEST.tif'
        urllib.request.urlretrieve(url, file)
        time.sleep(20)
        print(f'Weather data for {self.date} has been downloaded.')

def calculate_execution_time(start: float, stop: float):
    if stop - start < 60:
        execution_duration = ("%1d" % (stop - start))
        print(f"Process completed in {execution_duration} seconds")
        exit(0)
    elif stop - start < 3600:
        execution_duration = ("%1d" % ((stop - start) / 60))
        print(f"Process completed in {execution_duration} minutes")
        exit(0)
    else:
        execution_duration = ("%1d" % ((stop - start) / 3600))
        print(f"Process complete in {execution_duration} hours")
        exit(0)