##Select one season of timestamps_series
SelectSeason <- function(timestamp_series_all, ss) {
  if (season==1){
    timestamps_series_ss <- timestamps_series_all[month(timestamps_series_all) %in% c(12,1,2)]
  } else if (season==2){
    timestamps_series_ss <- timestamps_series_all[month(timestamps_series_all) %in% c(3,4,5)]
  } else if (season==3){
    timestamps_series_ss <- timestamps_series_all[month(timestamps_series_all) %in% c(6,7,8)]
  } else if (season==4){
    timestamps_series_ss <- timestamps_series_all[month(timestamps_series_all) %in% c(9,10,11)]
  } else {
    stop("Season not defined correctly (must be number: 1,2,3, or 4)")
  }
  invisible(timestamps_series_ss)
}

Create_M1_values <- function(mos_data,value){
  if (value %in% unique(mos_data$param_name)){
    M1_data <- mos_data[mos_data$param_name==value,]
    M1_data$value <- M1_data$value[c(NA,1:(length(M1_data$value)-1))]
    M1_data$value[M1_data$forecast_period==0] <- NA
    M1_data$param_name <- paste0(value,"_M1")
  } else {
    M1_data <- NULL
  }

  invisible(M1_data)
}

InterpolateMinMaxValues_smartmet <- function(station_fmisid, obsdata) {
  param_list <- unique(obsdata[["parameter"]]) 
  for (param_id_extr in c("TAMAX12H","TAMIN12H")) {
    if (param_id_extr %in% param_list) {
      obsdata <- ReturnInterpolatedMinMaxValues_smartmet(station_fmisid, obsdata,"parameter",param_id_extr)
    }
  }
  rm(param_id_extr)
  invisible(obsdata)
}

ReturnInterpolatedMinMaxValues_smartmet <- function(station_id, obsdata, column_name, param_id_extr) {
  com_string <- paste0("df_obs <- subset(obsdata,",column_name,"==param_id_extr)")
  eval(parse(text=com_string))
  rm(com_string)
#  start <- df_obs$obstime[1]
#  end <- df_obs$obstime[dim(df_obs)[1]]
  obs_interp <-  as.data.frame(cbind(df_obs$obstime,NA))
  obs_interp[,1] <- df_obs$obstime
  colnames(obs_interp) <- c("obstime","value")
  
  df_obs <- na.omit(df_obs)
  
  for (rownumber in 1:dim(obs_interp)[1]) {
    differences_in_hours <- difftime(obs_interp$obstime[rownumber],df_obs$obstime,units="hours") #(obs_interp$obstime[rownumber] - df_obs$obstime)
    assigned_value <- df_obs$value[head(which((differences_in_hours <= 0) & (differences_in_hours > -12)),1)]
    if (!length(assigned_value)==FALSE) {
      obs_interp$value[rownumber] <- assigned_value
    }
    rm(assigned_value)
    rm(differences_in_hours)
  }
  rm(rownumber)
  df_obs <- data.frame(station_id,obs_interp[["obstime"]],param_id_extr,obs_interp[["value"]])
  colnames(df_obs) <- c("station_id","obstime",column_name,"value")
  com_string <- paste0("obsdata <- subset(obsdata,",column_name,"!=param_id_extr)")
  eval(parse(text=com_string))
  rm(com_string)
  obsdata <- rbind(obsdata,df_obs)
  return(obsdata)
}

GenerateMOSArcDataFrame <- function(station_id,mosdata) { 
  # Generate MOS dataframe from the mosdata retrived
  # This dataframe contains the MOS data for 00-64 forecast periods
  # station_id: station_id
  # atime: analysis_time can have values "00" and "12" UTC
  # mos_season_data: MOS data for a particular season
  # 
  # Returns:
  #   data: MOS data with independent variables of the linear regression
  #   timevector: a POSIXct vector with UTC-based datetimes of the observations 
  
  # param_id, level_id describes a particular parameter which is an independent variable of the MOS data
  parameter_list <- unique(mosdata[["param_name"]])
#  level_list <- unique(mos_season_data[["level_value"]])
  analysis_times <- unique(mosdata[["analysis_time"]])
  forecast_periods <- unique(mosdata[["forecast_period"]])
  
#  mos_vars <- all_variable_lists[["MOS"]]                     # Getting the mos variable names from all_variable_lists
#  mos_D_vars <- all_variable_lists[["derived_variables_all"]] # Getting the derived variable names from all_variable_lists
  
  # Finding the parameter name and its values are added to the MOS dataframe
  # loop variable is zero for the first time when the first parameter value is added to the dataframe 
  loop1 <- 0  
  for (i in 1:length(parameter_list)) {
    df <- (mosdata %>%
             dplyr::filter(param_name == parameter_list[i]) %>%
             dplyr::select(station_id, analysis_time,  forecast_time, forecast_period, value))
      
    if (nrow(df) != 0) {
      names(df)[ncol(df)]<- as.character(parameter_list[i])
      if (loop1 == 0) { 
        df_mos <- df
        loop1 <- 1
      } else {
        df_mos<- merge(df_mos, df,by.x=c("station_id", "analysis_time", "forecast_time", "forecast_period"), by.y=c("station_id","analysis_time","forecast_time", "forecast_period"), all.x = TRUE)
      }
    }
  }

  return(df_mos)
}

FetchData_mos_obs <- function(station_fmisid, df_mos, obsdata, param_name) {
  # # This is faster, but does not guarantee that the specific season would have all the parameters from the param_list available
  df_obs <- subset(obsdata, parameter == param_name)[which(!names(obsdata) %in% "parameter")] # dplyr::filter(obs_season_data, parameter == param_id)

  if (dim(df_obs)[1]==0) {
    return(cbind(df_mos,NA))
  }
  colnames(df_obs)[ncol(df_obs)] <- param_name
  
  df_mos_obs<- merge(df_mos, df_obs,by.x=c("station_id","forecast_time"), by.y=c("station_id","obstime"), all.x = TRUE)
  return(df_mos_obs)
}

CleanData <- function(data_cleaned) {
  # Cleans a data  from bad observations and variables and  NAs 
  # Args:
  #   data: data retrived from the database
  #
  #
  # Returns:
  #   A cleaned data object of the same form as the input argument
  
  # 1) Handles exceptions
  # 2) remove predictors that have too much missing values
  # 3) Removes clear outliers from predictors that might affect fits in the linear regression
  # 4) Remove rows that do not have a complete set of predictand+predictor variables
  
  ##KY##
  #Remove rows where observation is NaN
  data_cleaned <- data_cleaned[!is.na(data_cleaned[ncol(data_cleaned)]),]
  
  if (dim(data_cleaned)[1] < 10){
    return(data_cleaned)
    break
  }
  
  # Change the column of the predictor variable from last as the first variable
  station_info <- data_cleaned[,c(1:4)]
  station_data <- data_cleaned[,5:(ncol(data_cleaned))]
  station_data <- station_data[,c(ncol(station_data),1:(ncol(station_data)-1))]
  
  # 1) Exceptions
  # 1.1) ALL ENSMEAN VARIABLES ARE SET MISSING FOR THE FORECAST_PERIODS < 144h
  if ((length(grep("_ENSMEAN",colnames(station_data)))>0) & (length(which((as.integer(station_info$forecast_period))<144))>0)) {
    station_data[which((as.integer(station_info$forecast_period))<144),grep("_ENSMEAN",colnames(station_data))] <- NA
  }
  
  # Removing those predictors that are either constant or completely missing. Always leave predictand column as it is (predictand)
  station_data <- station_data[,sort(unique(c(1,which(apply(X = station_data, MARGIN = 2, FUN = function(v) length(unique(v)))!=1))))]
  
  # Total sample size (no.of.obs) and the number of missing data for each variable (no.of.na)
  no.of.obs <-  nrow(station_data)
  no.of.na <- apply(X = station_data[,], MARGIN = 2, FUN = function(x) sum(is.na(x)))
  
  # na.tolerance specifies the maximum number of data points that are allowed missing for an individual predictor
  # na.tolerance2 specifies the maximum number of data points that are allowed missing for ensmean predictors
  na.tolerance <- no.of.obs * 0.20
  na.tolerance2 <- no.of.obs * 0.70
  
  # taking only the predictor variables that have no.of.na < na.tolerance. Always leave predictand column as it is (predictand)
  good.variables <- sort(unique(c(1,which(no.of.na < na.tolerance))))
  
  # EXCEPTION 2: ALLOW ALL ENSMEAN VARIABLES TO TRAINING DATA IF THEY HAVE MORE DATA POINTS THAN na.tolerance2
  if (length(grep("_ENSMEAN",colnames(station_data)))>0) {
    column_indices <- grep("_ENSMEAN",colnames(station_data))
    no.of.na <- as.integer(apply(X = as.data.frame(station_data[,column_indices]), MARGIN = 2, FUN = function(x) sum(is.na(x))))
    good.variables2 <- unique(which(no.of.na < na.tolerance2))
    good.variables <- sort(unique(c(good.variables,column_indices[good.variables2])))
  }
  
  # 2) remove predictors that have too much missing values
  station_data <- station_data[, good.variables]
  
  if (dim(station_data)[2] < 3){
    return(data_cleaned[0,])
    break
  }
  
  # 3) Removes clear outliers from predictors that might affect fits in the linear regression
  for (column in 2:dim(station_data)[2]) {
    analyzed_predictor <- station_data[,column]
    # For cloudiness this kind of test is needed: If almost all (more than 95% defined here) values in history are zero, set also the few remaining ones to zero
    if ((sum(analyzed_predictor==0,na.rm=TRUE) / length(analyzed_predictor))>0.95) {
      analyzed_predictor[] <- 0
    }
    # Outlier tests, replace outliers with NA values (and not sample mean values)
    # Separately for cloudiness and wind (which can have a heavily skewed distribution)
    non_NA_indices <- which(!is.na(analyzed_predictor))
    non_NA_vector <- as.vector(na.omit(analyzed_predictor))
    if (colnames(station_data)[column] %in% c("LCC", "MCC", "HCC", "TCC", "U10", "V10")) {
      analyzed_predictor[non_NA_indices[(outliers::scores(non_NA_vector, type="t", prob=0.99999))]] <- NA # mean(analyzed_predictor,na.rm=TRUE)
    } else {
      analyzed_predictor[non_NA_indices[(outliers::scores(non_NA_vector, type="t", prob=0.999))]] <- NA # mean(analyzed_predictor,na.rm=TRUE)
    }
    rm(non_NA_indices)
    rm(non_NA_vector)
    
    # Replace original data with the outlier removed data
    station_data[,column] <- analyzed_predictor
    rm(analyzed_predictor)
  }
  
  # 4) Remove rows that do not have a complete set of predictand+predictor variables
  complete.rows <- complete.cases(station_data)
  station_data <- station_data[complete.rows,]
  station_info <- station_info[complete.rows,]
  
  # returning the cleaned data by combining the new station_info dataframe and the new station_data dataframe
  data_cleaned <- cbind.data.frame(station_info, station_data)
  
  return(data_cleaned)
}

T_QC_road_stations <- function(obsdata_road){
  #Quality control for road weather temperature (and dewpoint temperature) observations
  #
  #Args: obsdata_road for one station with columns "station_id" "obstime" "value" "parameter"
  #Returns: obsdata_road_qc (same columns)
  #
  #1) Checks daily maximum and minimum values
  #2) If they are outside realisitic range in Finland, all observations from those days are removed
  #3) If some day is between days that are both removed the day that is between is also removed
  
  
  # 2m temperature thresholds for Finnish stations (source: FMI's webpage, records until 2020)
  T_low_thr <- c(-51.5,-49,-44.3,-36,-24.6,-7,-5,-10.8,-18.7,-31.8,-42,-47)
  T_high_thr <- c(10.9,11.8,17.5,25.5,31,33.8,37.2,33.8,28.8,21.1,16.6,11.3)
  T_sd_thr <- 8 #quite random choice for standard deviation threshold 
  
  ## Add date column to obsdata_road table and create all_dates variable
  obsdata_road$date <- as.Date(obsdata_road$obstime)
  obsdata_road_TA <- obsdata_road[obsdata_road$parameter=="TA",]
  all_dates <- unique(obsdata_road_TA$date)
  
  #Add column celsius (just to make values easier to intepret)
  obsdata_road_TA <- na.omit(obsdata_road_TA)
  obsdata_road_TA$celsius <- obsdata_road_TA$value-273.15
  
  ## Calculate daily min, max and sd of hourly temperatures
  agg_max_per_date <- aggregate(celsius ~ date, data = obsdata_road_TA, max)
  agg_min_per_date <- aggregate(celsius ~ date, data = obsdata_road_TA, min)
  agg_sd_per_date <- aggregate(celsius ~ date, data = obsdata_road_TA, sd)
  
  ## Find dates that have min/max values outside the thresholds
  rejected_max_dates <- vector()
  rejected_min_dates <- vector()
  rejected_sd_dates <- vector()
  for (mm in unique(month(agg_max_per_date$date))){
    agg_max_month <- agg_max_per_date[month(agg_max_per_date$date)==mm,]
    rejected_max_dates <- c.Date(rejected_max_dates,agg_max_month$date[agg_max_month$celsius > T_high_thr[mm]])
    agg_min_month <- agg_min_per_date[month(agg_min_per_date$date)==mm,]
    rejected_min_dates <- c.Date(rejected_min_dates,agg_min_month$date[agg_min_month$celsius < T_low_thr[mm]])
    agg_sd_month <- agg_sd_per_date[month(agg_sd_per_date$date)==mm,]
    rejected_sd_dates <- c.Date(rejected_sd_dates,agg_sd_month$date[agg_sd_month$celsius > T_sd_thr])
  }
  rejected_dates <- sort(na.omit(unique(c.Date(rejected_max_dates,rejected_min_dates,rejected_sd_dates))))
  
  ##If following and previous day is rejected then also the day that is between them will be rejected
  day_diffs <- rejected_dates[-1] - rejected_dates[-(length(rejected_dates))]
  if(any(day_diffs==2)){
    for (i in which(day_diffs==2)){
      rejected_dates <- c.Date(rejected_dates,rejected_dates[i]+1)
    }
    sort(rejected_dates)
  }
  
  ##All temperature variables (TA,TD) are rejected if 2m temperature has false values
  obsdata_road_qc <- obsdata_road
  obsdata_road_qc$value[obsdata_road_qc$date %in% rejected_dates] <- NA
  obsdata_road_qc <- na.omit(obsdata_road_qc)
  obsdata_road_qc <- obsdata_road_qc[,c("station_id","obstime","value","parameter")]
  
  return(obsdata_road_qc)
}


FitWithGlmnR1purrr_new <- function(training.set, max_variables=10, response_name_EC) {
  # The Lasso regression with 11 predictors and lambda lse
  #
  # Args:
  #   training.set: glmnet takes the model data and observations as arguements
  #   response_name_EC: name variable_EC for calculating the measure of raw forecast
  #
  # Returns:
  #   Two lists: coefficients and measures (validation score MAE)
  
  # QC FOR CONSTANT VALUES: SETTING VALUES TO NA IF >10 SONSECUTIVE DATA POINTS HAVE SIMILAR (NON-MISSING) VALUE (NO CLOUDINESS)
  for (data_column in (5:dim(training.set)[2])) {
    if (colnames(training.set)[data_column] %in% c("LCC","MCC")){
      next
    }
    chunks <- rle(training.set[,data_column])
    replacable_chunks <- which(chunks$lengths>10 & is.na(chunks$values)==FALSE)
    if (length(replacable_chunks)>0) {
      for (replacable_chunk in replacable_chunks) {
        row1 <- sum(chunks$lengths[0:(replacable_chunk-1)])+1
        row2 <- sum(chunks$lengths[1:(replacable_chunk)])
        training.set[row1:row2,data_column] <- NA
      }
      rm(replacable_chunk)
    }
    rm(chunks)
    rm(replacable_chunks)
  }
  rm(data_column)
  complete.rows <- complete.cases(training.set)
  training.set <- training.set[complete.rows,]
  
  # Checking whether training set contains enough data points
  if (dim(training.set)[1] > modelobspairs_minimum_sample_size) {
    training.matrix <- as.matrix(training.set[5:ncol(training.set)])
    glmnet.model <- suppressWarnings(glmnet(training.matrix[,-1], training.matrix[,1], family = "gaussian", alpha = 1, standardize = TRUE, pmax = max_variables+1))
    filter.for.folds <- IndexVectorToFilter(SplitDataEvenly(training.matrix[,1]))
    cv.glmnet.model <- suppressWarnings(cv.glmnet(training.matrix[,-1],training.matrix[,1], alpha = 1, foldid = filter.for.folds, pmax =max_variables+1))
    best.lambda <- cv.glmnet.model$lambda.min
    # Model measures
    mae.mos <- assess.glmnet(cv.glmnet.model,training.matrix[,-1],training.matrix[,1])$mae[[1]]
    ###Hae havaintonimen colnames(training.matrix)[1] perusteella taulukosta oikea ennustesuure
    if (response_name_EC %in% colnames(training.matrix)){
      mae.raw <- mean(abs(training.matrix[,1]-training.matrix[,c(response_name_EC)]),na.rm=T)
    } else {
      mae.raw <- NA
    }
    # choosing the best coefficients
    all.coefficients <- coef(cv.glmnet.model, s = best.lambda)
    all.coef.names <- rownames(all.coefficients)
    nonzero.indices <- which(all.coefficients != 0)
    coefficients <- all.coefficients[nonzero.indices]
    names(coefficients) <- all.coef.names[nonzero.indices] 
    
    names(coefficients)[1] <- "Intercept"
    results <- list("coefficients" = coefficients, "measures" = data.frame(mae.mos=mae.mos, mae.raw=mae.raw))
    return(results)  
  } else {
    results <- list("coefficients" = NA, "measures" = NA)
    return(results)
  }
  
}

