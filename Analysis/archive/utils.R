### Fred Callaway's R utilities ###

# makes a barplot with error bars and all
my_bar_plot <- function(data_frame, title, x, y, fill) {
  if (missing(fill)) {
    groupvars = x
    fill = x
  }
  else {
    groupvars = c(x, fill)
  }
  dfc = summarySE(data_frame, measurevar=y, groupvars=groupvars)
  ggplot(dfc, aes(x=get(x), y=get(y), fill=get(fill)) + 
    geom_bar(position=position_dodge(), stat="identity") +
    geom_errorbar(aes(ymin=get(y)-se, ymax=get(y)+se),
                  width=0.2, position=position_dodge(0.9)) +
    ggtitle(title))
}


# Summarizes data.  Gives count, mean, standard deviation, standard error
# of the mean, and confidence interval (default 95%).  data: a data
# frame.  measurevar: the name of a column that contains the variable to
# be summariezed groupvars: a vector containing names of columns that
# contain grouping variables na.rm: a boolean that indicates whether to
# ignore NA's conf.interval: the percent range of the confidence interval
# (default is 95%)
# from http://www.cookbook-r.com/Graphs/Plotting_means_and_error_bars_(ggplot2)/
summarySE <- function(data = NULL, measurevar, groupvars = NULL, na.rm = FALSE, 
                      conf.interval = 0.95, .drop = TRUE) {
  require(plyr)
  
  # New version of length which can handle NA's: if na.rm==T, don't count
  # them
  length2 <- function(x, na.rm = FALSE) {
    if (na.rm) 
      sum(!is.na(x)) else length(x)
  }
  
  # This does the summary. For each group's data frame, return a vector with
  # N, mean, and sd
  datac <- ddply(data, groupvars, .drop = .drop, .fun = function(xx, col) {
    c(N = length2(xx[[col]], na.rm = na.rm), mean = mean(xx[[col]], na.rm = na.rm), 
      sd = sd(xx[[col]], na.rm = na.rm))
  }, measurevar)
  
  # Rename the 'mean' column
  datac <- rename(datac, c(mean = measurevar))
  
  datac$se <- datac$sd/sqrt(datac$N)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error Calculate t-statistic
  # for confidence interval: e.g., if conf.interval is .95, use .975
  # (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + 0.5, datac$N - 1)
  datac$ci <- datac$se * ciMult
  
  return(datac)
}


# Use to get bootstrap confidence intervals, yay!
# data is the subset of your data w/ dv measurements (e.g. data$explanation 
# or some such thing)--note that you'll want to use this function multiple 
# times for different levels of IV(s) (like doing data$explanation[data$cond==cond1], 
# then cond2 or whatever you have things named).
# Levels are the different possible measurements of the dv (e.g., if you're 
# getting free responses and have sorted them into different categories, 
# levels will be something like c("explanation type 1", "explanation type 2",etc.))
# R is number of times to resample (usually 1000)
# returns a matrix of confidence intervals for each level
# credit to Erin Bennet from Stanford CoCo Lab
bootstrap_ci <- function(data,levels,R) {
  #initialise stuffs
  smp <- matrix(nrow=length(levels),ncol=R)
  cis <- c()
  #so much repeating wow
  for (i in 1:R) {
    #resample
    cur_boot <- sample(data,replace=TRUE)
    #add freq count for each level
    for (j in 1:length(levels)) {
      smp[j,i] <- length(cur_boot[cur_boot==levels[j]])
    }
  }
  #calculate confidence interval
  for (j in 1:length(levels)) {
    cis <- rbind(cis,quantile(smp[j,],probs=c(.025,.975)))
  }
  #helpfully name the rows things that make sense
  rownames(cis) <- levels
  return(cis)
}