## SETUP
library(ggplot2)
library(reshape2)
source('/Users/fred/Dropbox/coding/R/utils.R')
theme_set(theme_bw())
setwd("~/Drive/Danish-SRN/Analysis")
netlog = read.csv('net-log.csv')
netlog[, c('hidden', 'rate', 'momentum', 'rand_range', 'num_train')] =
         lapply(netlog[, c('hidden', 'rate', 'momentum', 'rand_range', 'num_train')], 
                as.factor)


## HOW DOES LANGUAGE IMPACT WORD SEGMENTATION?
df = netlog
with(df, tapply(word_F, language, mean))
out = aov(word_F ~ language, data=df)
summary(out)
lm(word_F ~ language, data=df)

dfc = summarySE(netlog, measurevar="word_F", groupvars="language")
ggplot(dfc, aes(x=language, y=word_F, fill=language)) + 
    geom_bar(position=position_dodge(), stat="identity") +
    geom_errorbar(aes(ymin=word_F-se, ymax=word_F+se),
                  width=.2, position=position_dodge(.9)) +
    ggtitle("Word segmentation")
# english does better


## HOW DO CONDITION AND LANGUAGE IMPACT WORD CHOICE ACCURACY?
df = melt(netlog,id.vars=c('language','subject'),measure.vars=c('expA_accuracy','expB_accuracy'),
          variable.name='condition', value.name='accuracy')
with(df, tapply(accuracy, list(language,condition), mean))
out = aov(accuracy ~ condition*language, data=df)
summary(out)
lm(accuracy ~ language, data=df)

english = split(netlog,netlog$language)[2]
df = melt(english,id.vars=c('subject'),measure.vars=c('expA_accuracy','expB_accuracy'),
          variable.name='condition', value.name='accuracy')
out = aov(accuracy ~ condition, data=df)
summary(out)

dfc = summarySE(df, measurevar="accuracy", groupvars=c("language","condition"))
ggplot(dfc, aes(x=language, y=accuracy, fill=condition)) + 
    geom_bar(position=position_dodge(), stat="identity") +
    geom_errorbar(aes(ymin=accuracy-se, ymax=accuracy+se),
                  width=.2, position=position_dodge(.9)) +
    coord_cartesian(ylim=c(0.5,0.8)) +
    ggtitle("ALL accuracy")
# it looks like danish may do slightly better, p < 0.1



## WORD ERRORS
pd = position_dodge(.1)
word.errors = read.csv('word-error-log.csv')
dfc = summarySE(word.errors, measurevar="error", groupvars=c("language","word","condition","type"))
ggplot(dfc, aes(x=language, y=error, fill=language)) + 
    geom_bar(position=position_dodge(), stat="identity") +
    geom_errorbar(aes(ymin=error-se, ymax=error+se),
                  width=.2, position=position_dodge(.9)) +
    ggtitle("ALL error")


## HOW DO NET PARAMETERS AFFECT WORD SEGMENTATION?
dfc = summarySE(netlog, measurevar='word_F', groupvars=c('num_train', 'language'))
ggplot(dfc, aes(x=num_train, y=word_F, fill=language)) + 
    geom_bar(position=position_dodge(), stat="identity") +
    geom_errorbar(aes(ymin=word_F-se, ymax=word_F+se),
                  width=.2, position=position_dodge(.9)) +
    ggtitle("Segmentation")
summary(aov(word_F ~ language * hidden * rate * momentum  * num_train, data=netlog))
