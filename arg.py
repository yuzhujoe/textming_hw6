# TRAININPUT = "./HW4_data/train.csv"
TRAININPUT = "./HW4_data/train100.csv"
DEVQRY = "./HW4_data/test.queries"
# DEVPAIR = "./HW4_data/test.csv"
DEVPAIR = "./HW4_data/dev100.csv"
OUTDIR = "./eval/"
PAIRWISE_TRAIN_OUTPUTDIR = "./HW4_data/"
PAIRWISE_TRAIN_INPUT_X = "./HW4_data/_train_X_10"
PAIRWISE_TRAIN_INPUT_Y = "./HW4_data/_train_Y_10"
PAIRWISE_DEV_PREDICT = "./HW4_data/dev.predict"

K = 10
GD_MAXITER = 10
GD_NUMLATENT = 10
GD_LAMBDA_U = 0.01
GD_LAMBDA_V = 0.01
GD_STOPCRITERIA = 1e-4
LR_LEARNINGRATE = 0.001
LR_BATCHSIZE = 2000
LR_STOPCRITERIA = 1e-6
# step size should not be too large because of overshooting problem
GD_STEPSIZE = 0.00016
