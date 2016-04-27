import preprocessing
import arg
from sklearn.svm import LinearSVC
import numpy

class LSVM:
    svc = LinearSVC()

    def trainSVM(self,X,Y):
        print "begin training svm"
        self.svc.fit(X,Y)
        print "finish training svm"

    def predictSVM(self,svc,X_dev):
        res = svc.predict(X_dev)
        return res

    def write_predict_to_file(self,res,outputpath):
        with open(outputpath,"w") as f:
            for i in xrange(len(res)):
                f.write(str(res[i]) + "\n")

    def svm_comparator(self,v1,v2):
        x = v1 - v2
        reslist =  self.svc.predict(x)
        return reslist[0]

    # def svm_make_predict(self,U,V,model):
    #     predict_user_Pair,predict_movie_Pair,reslist =  dp.readPair(arg.DEVPAIR)
    #
    #     res = {}
    #     res_score = {}
    #
    #     for uid in predict_user_Pair:
    #         movie_list = predict_user_Pair[uid]
    #         vlist = []
    #         for movie_id in movie_list:
    #             v = numpy.multiply(U[uid,:],V[movie_id,:])
    #             vlist.append(v)
    #
    #         sorted_v_idx_list= sorted(range(len(movie_list)),key= lambda x: vlist[x],cmp=model.svm_comparator)
    #         movied_id_sorted_list = [movie_list[i] for i in sorted_v_idx_list]
    #         res[uid] = movied_id_sorted_list
    #         l = len(movied_id_sorted_list)
    #         for idx in xrange(l):
    #             res_score[(uid,movied_id_sorted_list[idx])] = 1 + 4* idx*1.0/(l-1)
    #
    #     return res,res_score

    def svm_make_predict_use_w(self,U,V,predict_user_Pair):

        res_score = {}
        for uid in predict_user_Pair:
            movie_list = predict_user_Pair[uid]
            for movie_id in movie_list:
                v  = numpy.multiply(U[uid,:],V[movie_id,:])
                res_score[(uid,movie_id)] = numpy.dot(self.svc.coef_[0],v)

        return res_score


if __name__ == "__main__":
    dp = preprocessing.dataPrep()




