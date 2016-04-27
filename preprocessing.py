from scipy import sparse
import numpy
import arg
import numpy.matlib
import collections

class dataPrep:
    'class contains function to process data'

    # TODO: without imputation, subtract 3 from non empty cell
    def readTrain(self,path):
        row = []
        col = []
        val = []

        maxRow = 0
        maxCol = 0

        with open(path) as f:
            for line in f:
                movid_id,uid,rate_score,date = line.strip().split(",")
                movid_id = int(movid_id)
                uid = int(uid)
                rate_score = int(rate_score)
                row.append(uid)
                col.append(movid_id)
                val.append(float(rate_score))

                if movid_id > maxCol:
                    maxCol = movid_id
                if uid > maxRow:
                    maxRow = uid

        row = numpy.array(row)
        col = numpy.array(col)
        val = numpy.array(val)
        maxRow += 1
        maxCol += 1


        A = sparse.coo_matrix((val,(row,col)),shape=(maxRow,maxCol),dtype='d')

        return A.tocsr()

    def readPair(self,path):
        reslist = []

        predict_user_Pair = {}
        predict_movie_Pair = {}

        with open(path) as f:
            for line in f:
                movie_id, uid = line.strip().split(",")
                movie_id = int(movie_id)
                uid = int(uid)
                reslist.append((uid,movie_id))
                if uid not in predict_user_Pair:
                    predict_user_Pair[uid] = [movie_id]
                else:
                    predict_user_Pair[uid].append(movie_id)

                if movie_id not in predict_movie_Pair:
                    predict_movie_Pair[movie_id] = [uid]
                else:
                    predict_movie_Pair[movie_id].append(uid)

        return  predict_user_Pair,predict_movie_Pair,reslist

    def statTrain(self,path):

        maxRow = 0
        maxCol = 0

        hm = {}
        total_T = 0
        with open(path) as f:
            for line in f:
                movid_id,uid,rate_score,date = line.strip().split(",")
                movid_id = int(movid_id)
                uid = int(uid)
                rate_score = int(rate_score)

                if movid_id > maxCol:
                    maxCol = movid_id
                if uid > maxRow:
                    maxRow = uid

                if rate_score == 5:
                    if (uid,5) not in hm:
                        hm[(uid,5)] = 1
                    else:
                        hm[(uid,5)] += 1
                elif rate_score == 1:
                    if (uid,1) not in hm:
                        hm[(uid,1)] = 1
                    else:
                        hm[(uid,1)] += 1

        maxRow += 1
        maxCol += 1

        for i in xrange(maxRow):
            num_1 = 0
            num_5 = 0
            if (i,1) in hm:
                num_1 = hm[(i,1)]
            if (i,5) in hm:
                num_5 = hm[(i,5)]
            total_T += (num_1 * num_5) * 2


        print total_T

        num_1 = hm[(1234,1)]
        num_5 = hm[(1234,5)]

        print "id:1234 ", (num_1 * num_5) * 2

        num_1 = hm[(4321,1)]
        num_5 = hm[(4321,5)]

        print "id:4321 ", (num_1 * num_5) * 2

        return hm

    def build_pairwise_testing_set(self,path,U,V):

        hm = {}

        with open(path) as f:
            for line in f:
                movie_id,uid = line.strip().split(",")
                movie_id = int(movie_id)
                uid = int(uid)
                if uid not in hm:
                    hm[uid] = [movie_id]
                else:
                    hm[uid].append(movie_id)

        test_X = []
        for uid in hm:
            for movie_id1 in hm[uid]:
                for movie_id2 in hm[uid]:
                    x1 = numpy.multiply(U[uid,:],V[movie_id1,:])
                    x2 = numpy.multiply(U[uid,:],V[movie_id2,:])
                    x  = x1 - x2



    def build_pairwise_training_set(self,path,U,V):
        maxRow = 0
        maxCol = 0

        hm = {}
        total_T = 0
        with open(path) as f:
            for line in f:
                movid_id,uid,rate_score,date = line.strip().split(",")
                movid_id = int(movid_id)
                uid = int(uid)
                rate_score = int(rate_score)
                if movid_id > maxCol:
                    maxCol = movid_id
                if uid > maxRow:
                    maxRow = uid
                # todo hm[uid,rate_score].append(movie_id)
                if rate_score == 5:
                    if (uid,5) not in hm:
                        hm[(uid,5)] = []
                    hm[(uid,5)].append(movid_id)
                elif rate_score == 1:
                    if (uid,1) not in hm:
                        hm[(uid,1)] = []
                    hm[(uid,1)].append(movid_id)
        maxRow += 1
        maxCol += 1

        hmT_X = {}
        hmT_Y = {}

        train_matrix_list = []
        train_label_list = []
        # with open(outputpath+"_train_X","w") as f1:
        #     with open(outputpath+"_train_Y","w") as f2:
        for u in xrange(maxRow):
        # for u in xrange(1):
            ui = U[u,:]
            if (u,1) in hm:
                movie_with_rate_1 = hm[(u,1)]
            else:
                movie_with_rate_1 = []
            if (u,5) in hm:
                movie_with_rate_5 = hm[(u,5)]
            else:
                movie_with_rate_5 = []

            for i1 in movie_with_rate_1:
                for i5 in movie_with_rate_5:
                    # use U V find x vector
                    t1 = numpy.multiply(ui,V[i1,:])
                    t5 = numpy.multiply(ui,V[i5,:])
                    hmT_X[(u,i1,i5)] = t1 - t5
                    hmT_Y[(u,i1,i5)] = -1
                    hmT_X[(u,i5,i1)] = t5 - t1
                    hmT_Y[(u,i5,i1)] = 1

                    train_matrix_list.append(hmT_X[(u,i1,i5)].tolist())
                    train_matrix_list.append(hmT_X[(u,i5,i1)].tolist())
                    train_label_list.append(-1)
                    train_label_list.append(1)

        train_matrix = numpy.array(train_matrix_list)
        train_label = numpy.array(train_label_list)

        return train_matrix,train_label,hmT_X,hmT_Y

    # TODO: write follow libSVM import format: X sparse matrix: [num_sample, num_feature]
    def write_pairwise_train(self,train_matrix,train_lebel,outputpath):
        numpy.savetxt(outputpath + "_train_X_" + str(arg.GD_NUMLATENT),train_matrix,fmt = "%1.4e")
        numpy.savetxt(outputpath + "_train_Y_" +str(arg.GD_NUMLATENT),train_lebel,fmt= "1.1e")

    def write_pairwise_train_libsvm(self,train_matrix,train_label,outputpath):
        with open(outputpath + "_train_libsvm_" + str(arg.GD_NUMLATENT),"w") as f:
            numrow, numcol = train_matrix.shape
            for i in xrange(numrow):
                f.write(train_label[i]+ " ")
                for j in xrange(numcol):
                    f.write(str(i+1) + ":" + str(train_matrix[i,j]) + " ")
                f.write("\n")


    def load_pariwise_train(self,train_X_fname,train_Y_fname):
        train_matrix = numpy.loadtxt(train_X_fname)
        train_label = numpy.loadtxt(train_Y_fname)
        return  train_matrix, train_label



if __name__ == "__main__":
    dp = dataPrep()
    # dp.statTrain(arg.TRAININPUT)
    train_X, train_Y = dp.load_pariwise_train(arg.PAIRWISE_TRAIN_INPUT_X,arg.PAIRWISE_TRAIN_INPUT_Y)
    print train_X.shape
    print train_Y.shape

