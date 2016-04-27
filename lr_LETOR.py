import arg
import numpy
import numpy.matlib
from scipy import sparse
num_cat = 2
max_iter = 10
eval_percent = 0.05
lmd = 0.01


class lr_LETOR:

    def val2idx(self,val):
        if val == 1:
            return 0
        elif val == -1:
            return 1

    def star_to_vector(self,star):
        numrow = len(star)
        Y = numpy.zeros((numrow,num_cat))
        for i in xrange(numrow):
            idx = self.val2idx(star[i])
            Y[i,idx] = 1

        return Y

    def partition_data_2_batch(self,batch_size,numrow):
        res = []
        tmp = []
        size = 0
        startidx = int(eval_percent * numrow)
        for i in xrange(startidx,numrow):
            tmp.append(i)
            size += 1
            if size == batch_size or i == numrow - 1:
                res.append(tmp)
                tmp = []
                size = 0

        return res

    def partition_data_2_validateset(self,numrow):
        res = []
        # todo change to other selection method
        len = int(eval_percent * numrow)
        for i in xrange(len):
            res.append(i)

        return res

    def gradient_mini_batch_all_cat(self,W,X,Y):
        # st = time.time()
        print "W: ",numpy.linalg.norm(W)
        # print "X: ",numpy.linalg.norm(X)

        xw = numpy.dot(X,W.transpose())
        xw = numpy.exp(xw)
        xwsum = numpy.sum(xw,1,keepdims=True)

        xws = xw/xwsum

        # print "xws: ",xws

        g = numpy.dot((Y - xws).transpose(), X) - lmd * W


        return g

    def log_likelihood(self,W,X,Y):

        xw = numpy.dot(X, W.transpose())
        xw = numpy.exp(xw)
        xwsum = numpy.sum(xw,1,keepdims=True)

        xws = xw/xwsum

        xwsp = numpy.multiply(xws,Y)

        xwspsum = numpy.sum(xwsp,1,keepdims=True)

        xwspslog = numpy.log(xwspsum)

        xwspslogs = numpy.sum(xwspslog)
        nm = numpy.linalg.norm(W)
        return xwspslogs - lmd/2 * nm * nm

    def check_stop_criteria_loglikelihood(self,llold,llnew):
        if llold == None:
            return False
        elif abs((llnew - llold)/llold) < arg.LR_STOPCRITERIA:
            print "ratio: ", abs((llnew - llold)/llold)
            return True
        else:
            return False



    def write_predict_to_file(self,res,order_pair,outputpath):
        with open(outputpath +"_" +str(arg.GD_NUMLATENT),"w") as f:
            for (uid,movie_id) in order_pair:
                f.write(str(res[(uid,movie_id)]) + "\n")

    # def write_predict_2_file(self,W,X,filename):
    #
    #     with open(filename,"w") as f:
    #         numrow, numcol = X.shape
    #         soft = self.soft_predict(W,X)
    #         for i in xrange(numrow):
    #             f.write(str(soft[i]) + "\n")

    def lr_make_predict_use_w(self,W,U,V,predict_user_Pair):
        res_score = {}
        wtemp = W[0,:] - W[1,:]
        for uid in predict_user_Pair:
            movie_list = predict_user_Pair[uid]
            for movie_id in movie_list:
                v  = numpy.multiply(U[uid,:],V[movie_id,:])
                res_score[(uid,movie_id)] = numpy.dot(wtemp,v)

        return res_score


    def sgd_mini_batch(self,alpha,batch_size,W,s,Y):
        numrow,numcol = s.shape
        wrow,wcol = W.shape
        break_outer_loop = False

        validate_idx = self.partition_data_2_validateset(numrow)

        Yval = [Y[i] for i in validate_idx]

        Yval_vec = self.star_to_vector(Yval)

        batch_idx = self.partition_data_2_batch(batch_size,numrow)

        Wold = W

        Y_vec = self.star_to_vector(Y)
        count = 0
        llnew = None
        llold = None

        iter = 0
        while iter < max_iter:
            for j in xrange(len(batch_idx)):
                data_idx = batch_idx[j]

                wg = self.gradient_mini_batch_all_cat(Wold,s[data_idx,:],Y_vec[data_idx,:])
                Wnew = Wold + alpha * wg

                Wold = Wnew.copy()
                llnew = self.log_likelihood(Wold,s[validate_idx,:],Yval_vec)

                print "lnew: ",llnew
                print "lold: ",llold
                if self.check_stop_criteria_loglikelihood(llold,llnew):
                    break_outer_loop = True
                    break
                llold = llnew
                print "round : %s" %count, " log likelihood: %s" %llold
                count += 1
            if break_outer_loop:
                break
            iter += 1
        return Wold

    def trainLRLETOR(self,X,Y):

        num_feature = arg.GD_NUMLATENT
        W_init = numpy.ones((num_cat,num_feature))*(1.0/num_cat)
        W_final = self.sgd_mini_batch(arg.LR_LEARNINGRATE,arg.LR_BATCHSIZE,W_init,X,Y)
        return W_final
