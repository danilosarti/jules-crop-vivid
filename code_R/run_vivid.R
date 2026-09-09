# ------------------------------------------------------------------
# VIVID analysis (R vivid package) for the JULES-crop emulator.
# The network forward pass is reconstructed in pure R from the exported weights,
# so this script depends only on the vivid and ggplot2 packages. Runs vivi() + figures.
# ------------------------------------------------------------------
dir <- "/Users/danilosarti/Documents/amauri/R_vivid"
setwd(dir)
logf <- file.path(dir, "run.log")
cat("START", format(Sys.time()), "\n", file = logf)
lg <- function(...) cat(..., "\n", file = logf, append = TRUE)
ok <- TRUE
tryCatch({
  suppressMessages({ library(vivid); library(ggplot2) })
  lg("vivid version:", as.character(packageVersion("vivid")))

  d <- read.csv("model_data_v2.csv", stringsAsFactors = FALSE)
  CONT <- c("gdd_cum","tmean","tmax","srad","diff_rad","q")
  CAT  <- c("cultivar","county","treatment")
  RESP <- c("sdm","ldm","gdm")
  feat <- c("gdd_cum","tmean","tmax","srad","diff_rad","q",
            "cultivar_AG1051","cultivar_DKB363","cultivar_DKB393","cultivar_LG36790","cultivar_P4285YH",
            "county_Chapadinha","county_Piracicaba","county_Selviria",
            "treatment_irrigated","treatment_rainfed")
  cmin <- c(gdd_cum=11.222976600000038, tmean=11.540000012500016, tmax=17.486483899999996,
            srad=1.85185185175, diff_rad=1.027851069875, q=0.004724311)
  cmax <- c(gdd_cum=2089.1103950000024, tmean=31.966200000000025, tmax=40.67623000000003,
            srad=362.73148153375, diff_rad=316.891784427125, q=0.0240575835)
  ymin <- c(sdm=0.0, ldm=0.0009002377, gdm=0.0)
  ymax <- c(sdm=6.839514, ldm=4.7077255, gdm=9.577873)

  W1<-as.matrix(read.csv("W1.csv",header=FALSE)); b1<-scan("b1.csv",quiet=TRUE)
  W2<-as.matrix(read.csv("W2.csv",header=FALSE)); b2<-scan("b2.csv",quiet=TRUE)
  W3<-as.matrix(read.csv("W3.csv",header=FALSE)); b3<-scan("b3.csv",quiet=TRUE)
  W4<-as.matrix(read.csv("W4.csv",header=FALSE)); b4<-scan("b4.csv",quiet=TRUE)
  relu <- function(x) { x[x<0]<-0; x }
  sig  <- function(x) 1/(1+exp(-x))
  addb <- function(M,b) sweep(M,2,b,"+")
  fwd <- function(X){                     # X: n x 16 -> n x 3 (normalized [0,1])
    a <- relu(addb(X %*% W1, b1))
    a <- relu(addb(a %*% W2, b2))
    a <- relu(addb(a %*% W3, b3))
    sig(addb(a %*% W4, b4))
  }
  encode <- function(df){
    X <- matrix(0, nrow=nrow(df), ncol=length(feat), dimnames=list(NULL,feat))
    for(v in CONT) X[,v] <- (df[[v]]-cmin[[v]])/(cmax[[v]]-cmin[[v]])
    for(f in feat) for(v in CAT) if(startsWith(f, paste0(v,"_"))){
      lev <- sub(paste0(v,"_"), "", f); X[,f] <- as.integer(as.character(df[[v]])==lev)
    }
    X
  }
  # data for vivi: original predictors + normalized response
  dat <- d[, c(CONT,CAT)]
  for(v in CAT) dat[[v]] <- factor(dat[[v]])

  make_pf <- function(ri) function(fit, data, ...) as.numeric(fwd(encode(data))[,ri])

  vivis <- list()
  for(i in seq_along(RESP)){
    rp <- RESP[i]
    dsub <- dat
    dsub[[rp]] <- (d[[rp]]-ymin[[rp]])/(ymax[[rp]]-ymin[[rp]])
    vivis[[rp]] <- vivi(data=dsub, fit=list(), response=rp,
                        predictFun=make_pf(i), vars=c(CONT,CAT),
                        gridSize=20, nmax=400, numPerm=5)
    write.csv(as.matrix(vivis[[rp]]), paste0("vivi_R_",rp,".csv"))
    lg("vivi ok:", rp, "| Vimp max", round(max(diag(vivis[[rp]])),4))
  }
  # common color scales
  alldiag <- unlist(lapply(vivis, diag))
  offs <- unlist(lapply(vivis, function(m){ mm<-m; diag(mm)<-NA; as.numeric(mm[!is.na(mm)]) }))
  impLims <- range(alldiag); intLims <- c(0, max(offs)); thr <- 0.30*max(offs)
  lg("impLims", round(impLims,4), "| intLims", round(intLims,4), "| thr", round(thr,4))

  for(rp in RESP){
    m <- vivis[[rp]]
    h <- tryCatch(viviHeatmap(m, impLims=impLims, intLims=intLims),
                  error=function(e) viviHeatmap(m))
    h <- h + ggtitle(paste("VIVID heatmap -", toupper(rp)))
    ggsave(paste0("heatR_",rp,".png"), h, width=7.6, height=6.6, dpi=300)
    ggsave(paste0("heatR_",rp,".pdf"), h, width=7.6, height=6.6)
    n <- tryCatch(viviNetwork(m, intThreshold=thr, impLims=impLims, intLims=intLims),
                  error=function(e) tryCatch(viviNetwork(m, intThreshold=thr),
                  error=function(e2) viviNetwork(m)))
    n <- n + ggtitle(paste("VIVID network -", toupper(rp)))
    ggsave(paste0("netR_",rp,".png"), n, width=8.6, height=6.4, dpi=300)
    ggsave(paste0("netR_",rp,".pdf"), n, width=8.6, height=6.4)
    lg("figs ok:", rp)
  }
}, error=function(e){ ok<<-FALSE; lg("ERROR:", conditionMessage(e)) })
cat(if(ok) "DONE_OK" else "DONE_ERR", format(Sys.time()), "\n", file=logf, append=TRUE)
file.create(file.path(dir, if(ok) "DONE_OK" else "DONE_ERR"))
