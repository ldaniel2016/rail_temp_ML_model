#Program for retrieving forecasts from MOS archive for vaylavirasto track temperature NN model.

rm(list = ls())   # remove all previous lists 

### Loading needed libraries ###
# Libraries for data retrieval and handling
library(doParallel)
library(MOSpointutils)
library(lubridate)
library(dplyr)
library(stringr)

#Load MOS-archive data retrieval function
setwd("/home/daniel/projects/rails/scr/")
source("retrieve_data_MOS_archive.R")
source("mos_utils_functions.R")

begin_date=as.POSIXct("2019-09-07", tz="GMT") # observation starts from "2019-09-17"begin_date=as.POSIXct("2019-09-07", tz="GMT") # observation starts from "2019-09-17"
end_date=as.POSIXct("2023-06-04", tz="GMT") # observation ends on "2023-06-01"

#Set values for needed variables and lists
timestamps_series <- define_time_series(begin_date, end_date,interval_in_hours=12,even_hours=TRUE)
station_list_rail <- read.csv("mos_stations_rail_stations_väylävirasto.csv",header=TRUE)
#Variables list with columns for fmikey names that are used in mos archive
variable_list <- read.csv("MOS_archive_FMIKeys.csv",header=TRUE)

#Subset of maximum 50 stations
station_list_retrieved <- station_list_rail #station_list[1:8,]

cat("SID:",station_list_retrieved$SID,"\n") ##KY##
### RETRIEVING PREDICTOR DATA ###
start_time <- Sys.time()
predictor_data1 <- retrieve_data_MOS_archive(variable_list,station_list_retrieved,timestamps_series,download_quiet=T)
print(Sys.time() - start_time)

#Remove duplicates (niitä ei pitäisi tulla, mutta näköjään voi serverin epävakauden takia tulla)
predictor_data1 <- predictor_data1[!duplicated(predictor_data1), ]

# Create a data frame of all variables. Each column contains data for one station. 
latitudes <- as.numeric(strsplit(substring(predictor_data1$lat[1],2,nchar(as.character(predictor_data1$lat[1]))-1), " ", fixed=T)[[1]])
longitudes <- as.numeric(strsplit(substring(predictor_data1$lon[1],2,nchar(as.character(predictor_data1$lon[1]))-1), " ", fixed=T)[[1]])
s <- str_sub(predictor_data1$value,2,-2)
values <- t(data.frame(sapply(1:length(predictor_data1$value), function(x) as.numeric(strsplit(s[x], " ", fixed=T)[[1]]))))
analysis_time <- predictor_data1$analysis_time
forecast_time <- as.POSIXct(predictor_data1$time,tz="UTC",format="%Y%m%dT%H%M%S")
forecast_period <- predictor_data1$forecast_period
param_name <- predictor_data1$param_name
rm(s,predictor_data1)

#mosdata dataframe consists data for one station at a time
df_mos_all <- numeric()
for (j in 1:dim(station_list_retrieved)[1]){
#  j <- 1 #test for one station
  station_fmisid <- station_list_retrieved$SID[j]
 
  #columns:station_id, analysis_time, forecast_time, forecast_period, (param_id, level_value,)=param_name, value
  mosdata <- data.frame(station_id=station_fmisid, analysis_time=analysis_time,
                        forecast_time=forecast_time,
                        forecast_period=forecast_period,
                        param_name=param_name,value=values[,j])

  #Generate wide dataframe for a station
  df_mos <- GenerateMOSArcDataFrame(station_fmisid,mosdata)
  df_mos_all <- rbind(df_mos_all,df_mos)
}  

#Remove leadtime 0
df_mos_all <- subset(df_mos_all, forecast_period != 0)

#Add columns for cos and sin of month and hour of day
df_mos_all$cosmonth <- round(cos(month(df_mos_all$forecast_time)*2*pi/12),2) #kerrotaan kahdella koska ympyrän kehä on 2*pi eli 360 astetta
df_mos_all$sinmonth <- round(sin(month(df_mos_all$forecast_time)*2*pi/12),2) #kerrotaan kahdella koska ympyrän kehä on 2*pi eli 360 astetta
df_mos_all$coshour <- round(cos(hour(df_mos_all$forecast_time)*2*pi/24),2)
df_mos_all$sinhour <- round(sin(hour(df_mos_all$forecast_time)*2*pi/24),2)

#Add columns for lat and lon values using merge
station_latlons <- station_list_retrieved[,c("SID","lat","lon")]
colnames(station_latlons)[colnames(station_latlons)=="SID"] <- "station_id"
df_mos_all <- merge(df_mos_all, station_latlons, by="station_id")

# calculating the analysis_date
df_mos_all$analysis_date <- as.POSIXct(df_mos_all$forecast_time, tz="GMT") - (df_mos_all$forecast_period*3600)
df_mos_all <- df_mos_all[,c(1:4,27,5:26)]

colnames(df_mos_all)
write.csv(df_mos_all, paste0("/home/daniel/projects/rails/data/mos_archive_data_for_all_rail_stations.csv"), row.names=FALSE)
