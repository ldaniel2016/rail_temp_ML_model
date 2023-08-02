retrieve_data_MOS_archive <- function(variable_list,station_list_retrieved,timestamps_series,download_quiet=TRUE) {
  # This function retrieves data from MOS archive (https://mos-archive.apps.ocp.fmi.fi) using timeseries tool
  
  # Check that doParallel library is installed
  library(doParallel)
  
  # Checking if variable list exists
  if (dim(variable_list)[1] == 0) {
    stop("Empty variable list!")
  }

  # Station information
  retrieved_latlons <- paste(station_list_retrieved$lat,station_list_retrieved$lon,sep=",",collapse=",")
  retrieved_sids <- station_list_retrieved$SID
  retrieved_parameters <- variable_list$FMIKey_short
  
  # detectCores()
  #Devmos 12 cpus 
  #OBS! It’s usually not a good idea to use ALL of the cores available since that will use all of your computer’s resources.
  cl <- makeCluster(8) 
  registerDoParallel(cl) 
  getDoParWorkers()
#  showConnections()
  
  # Parallel data retrieval from mos-archive to variable r
  r <- foreach(i=1:length(timestamps_series), .combine=bind_rows) %:%
    foreach(j=1:length(retrieved_parameters), .combine=bind_rows) %dopar% {
      destfile <- tempfile(fileext = ".csv", tmpdir='/dev/shm') #destfile <-  tempfile(fileext = ".txt")
      timestamp <- format(timestamps_series[i],'%Y%m%d%H%M')
      fmikey <- retrieved_parameters[j]
#      if (timestamps_series[i] < "2016-03-08 GMT"){
#        #muokkaa starttime ja endtime, jos se nopeuttaisi hakuja.
#      }
      url <- paste("https://mos-archive.apps.ocp.fmi.fi/timeseries?format=ascii&tz=gmt&precision=double&separator=;&latlons=",retrieved_latlons,"&param=origintime,time,lat,lon,",fmikey,",@I-4&starttime=data&endtime=data&origintime=",timestamp,"&timestep=all&grouplocations=1",sep="")
      if (length(url) != 1L || typeof(url) != "character") stop("'url' must be a length-one character vector")
      if (length(destfile) != 1L || typeof(destfile) != "character") stop("'destfile' must be a length-one character vector")
      extra <- '--no-proxy --no-check-certificate'
      if (download_quiet) extra <- c(extra, "--quiet")
      
      status <- system(paste("wget", paste(extra, collapse = " "), 
                             shQuote(url), "-O", shQuote(path.expand(destfile))))
      
      #If status is not OK, try again max. 10 times with test url (small data set)
      if (status) {
        cat("'wget' call had nonzero exit status for url:",url,"\n")
        Sys.sleep(180)
        
        destfile2 <- tempfile(fileext = ".csv", tmpdir='/dev/shm')
        url_test <- "https://mos-archive.apps.ocp.fmi.fi/timeseries?format=ascii&place=Oulu&param=name,time&timesteps=0"
        tries <- 0
        while(tries<10){
          status_test <- system(paste("wget", paste(extra, collapse = " "),
                                      shQuote(url_test), "-O", shQuote(path.expand(destfile2))))
          if (status_test) {
            tries <- tries+1
            Sys.sleep(30)
          } else {
            status <- system(paste("wget", paste(extra, collapse = " "), 
                                   shQuote(url), "-O", shQuote(path.expand(destfile))))
            break
          }
        }
        file.remove(destfile2)
      }
      
      #If server returns "High Load", wait 3 minutes and try again
      if (!is.na(file.info(destfile)$size)){#don't do anything with empty files 
        if (file.size(destfile)>10){
          out <- read.csv(destfile,sep=";",header=F)
          if(any(out$V1 == "<html><head><title>High Load</title></head><body><h1>1234 High Load</h1></body></html>")){
            Sys.sleep(180)
            status <- system(paste("wget", paste(extra, collapse = " "), 
                                   shQuote(url), "-O", shQuote(path.expand(destfile))))
          }
        }
      }
          

      if (!is.na(file.info(destfile)$size)){#don't do anything with empty files 
        if (file.size(destfile)>10){
          out <- read.csv(destfile,sep=";",header=F)
          file.remove(destfile)
          out
        } else {
          file.remove(destfile)
          NULL
        } 
      } else {
        file.remove(destfile)
        NULL
      }
            
      # txt2 <- paste("https://mos-archive.apps.ocp.fmi.fi/timeseries?format=ascii&tz=gmt&precision=double&separator=;&latlons=",retrieved_latlons,"&param=origintime,time,lat,lon,",fmikey,",@I-4&starttime=data&endtime=data&origintime=",timestamp,"&timestep=all&grouplocations=1",sep="")
      # tryCatch(
      #   {     
      #     download.file(url=txt2,destfile=destfile, method="wget",extra = '--no-proxy --no-check-certificate',quiet=download_quiet,options(timeout=60))
      #     if (file.info(destfile)$size>100){ #don't do anything with empty files 
      #       out <- read.csv(destfile,sep=";",header=F)
      #       file.remove(destfile)
      #       out
      #     } else {
      #       file.remove(destfile)
      #       return(NULL)
      #     }
      #   },
      #   # how to handle warnings 
      #   warning = function(cond) {
      #     message(cond)
      #   },
      #   # how to handle errors
      #   error = function(cond) {
      #     message(cond)
      #   }
      # )
    }
#  print(dim(r))
  stopCluster(cl) # DONT FORGET TO STOP YOUR CLUSTERS!

  # Add columns for forecast_period and analysis time
  r$forecast_period <- as.integer(difftime(as.POSIXct(r$V2, format =  "%Y%m%dT%H%M%S", tz="gmt"), as.POSIXct(r$V1, format =  "%Y%m%dT%H%M%S", tz="gmt"), units="hours")) #forecast_period
  r$analysis_time <- format(strptime(r$V1,"%Y%m%dT%H%M%S"),'%H')

  #Modify FMIKeys to MOS variable names using variable_list given as function input
  param_conv <- rep(variable_list$variable_EC,2)
  names(param_conv) <- c(variable_list$FMIKey1,variable_list$FMIKey2)
  
  r$param_name <- as.character(param_conv[as.character(r$V6)])
  
  if (sum(is.na(r$param_name)) > 0){
    cat(sum(is.na(r$param_name))," of the variables are not found in the FMIKey list!\n")
    cat(r$param_name[(is.na(r$param_name))],"\n")
    r <- na.omit(r)
  } else {
    cat("All the variables are found in the FMIKey list!\n")
  }

  colnames(r) <- c("origintime","time","lat","lon","value","fmikey","forecast_period","analysis_time","param_name")

  # Returning
  invisible(r)
}
