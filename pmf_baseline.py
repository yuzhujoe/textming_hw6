import numpy
from scipy import sparse
import arg
import preprocessing
import time
import linearSVM
import lr_LETOR


class PMF:

    # U user latent matrix
    # V mo latent matrix
    def gradient(self,ratingMat,U,V,lambdaU,lambdaV,useU):

        row,col = ratingMat.shape

        rowidx, colidx = ratingMat.nonzero()

        gradU = None
        gradV = None
        mult = U.dot(V.transpose())
        A = ratingMat - mult
        iA = sparse.csr_matrix((numpy.array(A[rowidx,colidx])[0],(rowidx,colidx)),shape=(row,col))

        if useU:
            lu = lambdaU*U
            gradU = -iA*V + lu
        else:
            iA = iA.transpose().tocsr()
            lv = lambdaV*V
            gradV = -iA*U + lv

        return [gradU,gradV,iA]

    def optimize(self,U,V,gradU,gradV,stepsize,useU):
        oU = U
        oV = V
        if useU:
            oU = U - stepsize * gradU
        else:
            oV = V - stepsize * gradV
        return oU,oV

    def initialize(self,numLatent,ratingMat):
        row,col = ratingMat.shape

        U = numpy.random.rand(row,numLatent)
        V = numpy.random.rand(col,numLatent)

        return [U,V]

    def gradDescend(self,ratingMat,movie_pair,orderpair):
        step = 0
        maxIter = arg.GD_MAXITER
        lambdaU = arg.GD_LAMBDA_U
        lambdaV = arg.GD_LAMBDA_V
        stepsize = arg.GD_STEPSIZE

        [U,V] = self.initialize(arg.GD_NUMLATENT,ratingMat)
        useU = True

        obj_funct_old = 1
        obj_funct_new = 0

        while True:
            [gradU,gradV,iA] = self.gradient(ratingMat,U,V,lambdaU,lambdaV,useU)
            step += 1
            obj_funct_new = (iA.data **2).sum()+ lambdaU *(numpy.linalg.norm(U) ** 2) + lambdaV* (numpy.linalg.norm(V) **2)

            print obj_funct_new
            ratio = abs((obj_funct_new - obj_funct_old)/obj_funct_old)

            if step > maxIter or ratio < arg.GD_STOPCRITERIA:
                break

            obj_funct_old = obj_funct_new

            [U,V] = self.optimize(U,V,gradU,gradV,stepsize,useU)

            if useU:
                useU = False
            else:
                useU = True

        predicMovieRat = U.dot(V.T)
        return [predicMovieRat,U,V,step]

    def out(self,predicMovieRat,orderpair):
        with open("exp4_L_"+str(arg.GD_NUMLATENT),"w") as f:
            for (uid,movie_id) in orderpair:
                f.write(str(predicMovieRat[uid,movie_id])+"\n")

    def write_score(self,res_score,order):
        with open("" + str(arg.GD_NUMLATENT),"w") as f:
            for (uid,movie_id) in orderpair:
                f.write(str(res[(uid,movie_id)]) + "\n")

if __name__ == "__main__":

    train_path = arg.TRAININPUT

    dp = preprocessing.dataPrep()

    rating_mat = dp.readTrain(train_path)

    rating_mat_tran = rating_mat.transpose().tocsr()

    maxRow,maxCol = rating_mat.shape

    user_pair,movie_pair,orderpair = dp.readPair(arg.DEVPAIR)

    pmf = PMF()

    st = time.time()
    [predict_movie_mat,U,V,step] = pmf.gradDescend(rating_mat,movie_pair,orderpair)
    print "time: ", time.time() - st

    pm,pu,res = dp.readPair(arg.DEVPAIR)

    # pmf.out(predict_movie_mat,res)

    train_matrix, train_label, hmT_X, hmT_Y = dp.build_pairwise_training_set(arg.TRAININPUT,U,V)

    # dp.write_pairwise_train(train_matrix,train_label,arg.PAIRWISE_TRAIN_OUTPUTDIR)

    # svm
    lsvm = linearSVM.LSVM()
    lsvm.trainSVM(train_matrix,train_label)
    predict_user_Pair,predict_movie_Pair,reslist =  dp.readPair(arg.DEVPAIR)
    res_score = lsvm.svm_make_predict_use_w(U,V,predict_user_Pair)
    lsvm.write_predict_to_file(res_score,arg.PAIRWISE_DEV_PREDICT)

    # LR
    lr = lr_LETOR.lr_LETOR()
    w_final = lr.trainLRLETOR(train_matrix,train_label)
    res_score = lr.lr_make_predict_use_w(w_final,U,V,predict_user_Pair)
    lr.write_predict_to_file(res_score,arg.PAIRWISE_DEV_PREDICT)
