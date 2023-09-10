library(raster)

correctCRS <- function(fname){
  raw = raster(paste0('SMAP/raw/',fname))
  ref = raster('weather_data/RainPrediction24hr_202309090900AEST.tif')
  out_file = paste0('SMAP/adjusted/',fname)

  cor = raster::crop(raw, ref)
  raster::writeRaster(cor, filename = out_file, format='GTiff', overwrite=T)
}

flist <- list.files('SMAP/raw/',pattern = '.tif')
lapply(flist, FUN = correctCRS)
