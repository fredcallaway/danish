"""Creates .in files for reading by lens"""


def write_lens_files(id, seed, input_, hidden, output, num_updates, learning_rate,
                     momentum, backprop_ticks, rand_range, ):

    architecture = """
# Network ID: %(id)s

## ARCHITECTURE ##
seed %(seed)s
addNet srn -i 1
setObj rand_range %(rand_range)s

addGroup input   %(input_)s  INPUT
addGroup hidden  %(hidden)s -RESET_ON_EXAMPLE
addGroup context %(hidden)s ELMAN
addGroup output  %(output)s  OUTPUT

connectGroups input hidden
connectGroups context hidden
connectGroups hidden output
elmanConnect hidden context
orderGroups bias input context hidden output

setObj learning_rate %(learning_rate)s
setObj momentum %(momentum)s
setObj batchSize 1
setObj backprop_ticks %(backprop_ticks)s
setTime -h %(backprop_ticks)s
""" % locals()

    training = architecture+"""
## TRAINING ##
setObj num_updates %(num_updates)s
loadExamples train.ex -s "train.ex"
useTrainingSet train.ex
setObj reportInterval 10000
train
saveWeights weights/%(id)s.wt -v 2
echo success
exit
""" % locals()

    testing = architecture+"""
## TESTING ##
loadWeights weights/%(id)s.wt
loadExamples test.ex -s "test.ex"
useTestingSet test.ex
touch segmentation.out
echo  > segmentation.out
proc printOutputs {group unit} {return [format "%%f " [getObj $unit.output]]}
setObj postEventProc {printUnitValues segmentation.out printOutputs output -a; echo "" >> segmentation.out}
test
setObj postEventProc {}
echo success
exit
""" % locals()

    experiment = architecture+"""

## EXP-A TRAINING ##
setObj reportInterval 10000
setObj num_updates 4548
loadWeights weights/%(id)s.wt
loadExamples exp-trainA.ex
useTrainingSet exp-trainA
train

## EXP-A TESTING ##
setObj reportInterval 1
for {set i 0} {$i < 72} {incr i} {
    echo ***$i
    loadExamples exp-testA$i.ex
    useTrainingSet exp-testA$i
    set n [eval getObject exp-testA$i.numExamples]
    setObj num_updates $n
    train
}
# mark the beginning of exp b
echo ######

## EXP-B TRAINING ##
setObj reportInterval 10000
setObj num_updates 4548
loadWeights weights/%(id)s.wt
loadExamples exp-trainB.ex
useTrainingSet exp-trainB
train

## EXP-B TESTING ##
setObj reportInterval 1
for {set i 0} {$i < 72} {incr i} {
    echo ***$i
    loadExamples exp-testB$i.ex
    useTrainingSet exp-testB$i
    set n [eval getObject exp-testB$i.numExamples]
    setObj num_updates $n
    train
}
echo success
exit
""" % locals()

    with open('lens/training.in', 'w') as net_file:
        net_file.write(training)

    with open('lens/testing.in', 'w') as net_file:
        net_file.write(testing)

    with open('lens/experiment.in', 'w') as net_file:
        net_file.write(experiment)
