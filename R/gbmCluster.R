gbmCluster <- function(n){
    # If number of cores (n) not given, try to work it out from the number
    # that appear to be available and the number of CV folds.
    if (is.null(n)){
      n <- max(1, parallel::detectCores() - 1)
    }
    if (n == 1){ NULL }
    else { parallel::makeCluster(n) }
}
