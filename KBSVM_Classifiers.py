import numpy as np
import pandas as pd
import math
import random
import cvxpy as cp

def confusion_matrix1(y_true, y_pred):

    x = list(filter(None.__ne__, set(y_true)))

    a = [[0 for x in range(2)] for y in range(2)]
    for j in range(len(x)):
        s1, s2, k = 0, 0, 0
        for i in range(len(y_true)):
            if ((x[j] == y_true[i]) & (y_true[i] == y_pred[i])):
                s1 += 1
            elif ((x[j] == y_true[i]) & (y_true[i] != y_pred[i])):
                s2 += 1

        a[j][j] = s1
        a[j][~j] = s2
    a = np.array(a).reshape((2, 2))
    print("CONFUSION MATRIX :: ")
    print(a)
    sum = 0
    for i in range(len(a)):
        sum += a[i, i]
    error = 1 - (sum / len(y_true))
    return error

def cross_valid(training_data_example, training_data_label):

    #---------------------------------------------K-FOLD CROSS-VALIDATION----------------------------------------------
    # Length of the training dataset : 500
    n1 = len(training_data_label)
    attr = len(training_data_example[0, :])
    # Number of folds can be changed by changing the kfold value....................................................
    kfold = 10;
    div = math.floor(n1 / kfold);
    # Assigning different hyper-parameter values
    Cdata_val=[]
    Cadv_val=[]
    # Dictionary to store (Cdata, Cadv) pair........................................................................
    dic={}
    count=0
    # Number of different Cdata & Cadv value........................................................................
    n_c=8
    for i in range(n_c):
        Cdata_val.append(math.pow(10,(-4+i)))
        Cadv_val.append(math.pow(10,(-4+i)))

    #--------------------------------------------VARIABLES -------------------------------------------------------------
    w = cp.Variable(shape=(attr, 1))
    eta = cp.Variable((attr, 1))
    zeta = cp.Variable((1, 1))
    b = cp.Variable((1, 1))
    u1 = cp.Variable((2, 1))
    u2 = cp.Variable((2, 1))
    u3 = cp.Variable((1,1))
    # -----------------------------------------ADVICE SET----------------------------------------------------------------
    # Advice set 1...........
    l = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    D1 = np.array(l)
    D1 = np.reshape(D1, (2, attr))
    l = [100, 25]
    d1 = np.array(l)
    d1 = np.reshape(d1, (2, 1))
    # Advice set 2...........
    l = [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0]
    D2 = np.array(l)
    D2 = np.reshape(D2, (2, attr))
    l = [-126, -30]
    d2 = np.array(l)
    d2 = np.reshape(d2, (2, 1))
    # Advice set 3............
    l = [0, 0, 0, 0, 0, 0, 0, -1]
    D3 = np.array(l)
    D3 = np.reshape(D3, (1, attr))
    l = [-45]
    d3 = np.array(l)
    d3 = np.reshape(d3, (1, 1))
    # -----------------------------INDEXES FOR CROSS-VALIDATION------------------------------------------------------------
    cv_test_idx=[] #---CV TEST DATA SET
    cv_train_idx=[] #---CV TRAIN DATA SET
    for i in range(kfold):
        hold = random.sample(range(n1), div)
        cv_test_idx.append(list(hold))
        cv_train_idx.append(list(np.setdiff1d(range(n1),hold)))
    #---- PARAMETERS FOR CROSS-VALIDATION--------------------------------------------------------------------------------
    n2 = len(cv_train_idx[0]);
    xi = cp.Variable((n2, 1));
    Error = np.zeros((kfold, n_c, n_c))
    # ---------------------------------- OBJECTIVE & CONSTRAINTS TERMS --------------------------------------------------
    term12 = cp.abs(w)
    term1 = cp.sum(term12)
    term31 = cp.sum(cp.abs(eta))
    term32 = cp.sum(zeta)
    c2 = -w + (D1.T * u1) + eta
    c3 = b - 1 - (d1.T * u1) + zeta
    c4 = w + (D2.T * u2) + eta
    c5 = -b - 1 - (d2.T * u2) + zeta
    c6 = w + (D3.T * u3) + eta
    c7 = -b - 1 - (d3.T * u3) + zeta

    #--------------------------------- CROSS-VALIDATION -----------------------------------------------------------------
    for i in range(len(Cdata_val)):
        Cdata = Cdata_val[i]
        for j in range(len(Cadv_val)):
            Cadv = Cadv_val[j]
            #
            # Dictionary....
            key= count
            dic.setdefault(key, ())
            dic[key]=(Cdata,Cadv)
            #
            #
            for k in range(kfold):
                #
                #
                x = training_data_example[cv_train_idx[k],:]
                y = training_data_label[cv_train_idx[k]]
                c1 = np.diag(y) * (x * w + b) - 1 + xi
                term2 = Cdata * cp.sum(xi)
                term3 = Cadv * (term31 + term32)
                Objective = cp.Minimize(term1 + term2 + term3)
                Constraints = [xi >= 0, u1 >= 0, u2 >= 0, zeta >= 0, c1 >= 0, c2 == 0, c3 >= 0, c4 == 0, c5 >= 0]
                Prob = cp.Problem(Objective, Constraints)
                x_cv_test = training_data_example[cv_test_idx[k],:]
                y_cv_test = training_data_label[cv_test_idx[k]]
                Prob.solve(solver=cp.GLPK, verbose=True)
                # OPTIMIZED VALUES OF w & b ----------------
                #
                w_Opt = np.array(w.value, dtype=float)
                b_Opt = np.array(b.value, dtype=float)
                #
                #
                y_cv_pred = np.sign(np.matmul(x_cv_test, w_Opt) + b_Opt);
                y_cv_pred = np.reshape(y_cv_pred,(1,len(y_cv_pred)))
                y_cv_pred = y_cv_pred[0]
                #
                #
                Err = confusion_matrix1(y_cv_test, y_cv_pred)
                Error[k,i,j] = Err
            #
            count+=1
            #

    # ERROR OF ALL (Cdata, Cadv) PAIR AFTER KFOLD CV IN Error.txt FILE ---------------------------------
    #
    with open('Error.txt', 'w') as outfile:
        for j in Error:
            j=np.reshape(j,(1,n_c * n_c))
            np.savetxt(outfile, j)
    #
    # MEAN ERROR OF ALL (Cdata, Cadv) PAIR AFTER KFOLD CV IN Mean_Error.txt FILE ------------------------
    err = np.mean(Error, axis=0)
    err=np.reshape(err,(1,n_c * n_c))
    with open('Mean_Error.txt', 'w') as outfile:
        np.savetxt(outfile, err)
    #
    # RETURNING THE BEST (Cdata, Cadv) PAIR --------------------------------------------------------------
    err1=err[0]
    min_error= np.min(err1)
    idx1= np.where(err1 == min_error)
    idx=idx1[0]
    hold = dic[idx[0]]
    min_Cdata = hold[0]
    min_Cadv = hold[1]
    #
    #
    return (min_Cdata, min_Cadv)

def train_svm_with_data_and_advice(training_data_example, training_data_label, test_data_example, test_data_label):

    #
    # --------------------------------------CROSS-VALIDATION------------------------------------------------
    # BEST (Cdata, Cadv) PAIR ...................
    Cdata, Cadv = cross_valid(training_data_example, training_data_label)

    #Cdata, Cadv = 0.1, 0.1
    #
    #
    # --------------------------------------------VARIABLES -------------------------------------------------------------
    n1 = len(training_data_label)
    n2 = len(test_data_label)
    attr = len(training_data_example[0,:])
    idx_glucose = 1
    idx_BMI = 5
    w = cp.Variable(shape=(attr,1))
    xi = cp.Variable((n1,1))
    eta = cp.Variable((attr,1))
    zeta = cp.Variable((1,1))
    x = training_data_example
    y = training_data_label
    b = cp.Variable((1,1))
    u1 = cp.Variable((2,1))
    u2 = cp.Variable((2,1))
    u3 = cp.Variable((1,1))

    #-----------------------------------------ADVICE SET----------------------------------------------------------------
    # Advice set 1 (Negative Advice)...........
    l=[0,1,0,0,0,0,0,0, 0,0,0,0,0,1,0,0]
    D1 = np.array(l)
    D1=np.reshape(D1,(2,attr))
    l = [100, 25]
    d1 = np.array(l)
    d1=np.reshape(d1,(2,1))
    # Advice set 2 (Positive Advice)...........
    l = [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0]
    D2 = np.array(l)
    D2 = np.reshape(D2, (2, attr))
    l = [-126, -30]
    d2 = np.array(l)
    d2 = np.reshape(d2, (2, 1))
    # Advice set 3 (Positive Advice)............
    l = [0, 0, 0, 0, 0, 0, 0, -1]
    D3 = np.array(l)
    D3 = np.reshape(D3, (1, attr))
    l = [-45]
    d3 = np.array(l)
    d3 = np.reshape(d3, (1, 1))
    #----------------------------------- OBJECTIVE & CONSTRAINT TERMS -------------------------------------------------
    term12=cp.abs(w)
    term1 = cp.sum(term12)
    #term1 = 0.5 * (w.T * w)
    term2 = Cdata * cp.sum(xi)
    term31 = cp.sum(cp.abs(eta))
    term32 = cp.sum(zeta)
    term3 = Cadv * (term31 + term32)
    Objective = cp.Minimize(term1 + term2 + term3)
    c1 = np.diag(y) * (x*w+b) - 1 + xi
    c2 = -w + (D1.T * u1) + eta
    c3 = b -1 - (d1.T * u1) + zeta
    c4 = w + (D2.T * u2) + eta
    c5 = -b - 1 - (d2.T * u2) + zeta
    c6 = w + (D3.T * u3) + eta
    c7 = -b - 1 - (d3.T * u3) + zeta
    Constraints = [xi >= 0, u1 >= 0, u2>=0, zeta >= 0 , c1 >= 0, c2 == 0, c3 >= 0, c4 == 0, c5 >= 0]
    Prob = cp.Problem(Objective, Constraints)
    Prob.solve(solver=cp.GLPK, verbose=True)
    #
    #------ OPTIMIZED VALUES OF w & b ----------------
    w_Opt = np.array(w.value, dtype = float)
    b_Opt = np.array(b.value, dtype = float)
    print("\n-----------------------------------------------\n ------::: PROBLEM STATUS :::------\n",Prob.status)
    print("\n OPTIMIZED VALUE = ",Prob.value,'\n')
    print("The best Cdata, Cadv pair is = ", "( ",Cdata, " , ",Cadv," )" "\n")
    #---- PREDICTED LABELS OF THE TEST DATA ---------------------------------------
    x_test = test_data_example
    y_test = test_data_label
    y_pred = np.sign(np.matmul(x_test, w_Opt) + b_Opt)
    y_pred = np.reshape(y_pred, (1, len(y_pred)))
    y_pred = y_pred[0]
    #
    #
    Error = confusion_matrix1(y_test, y_pred)
    print("\n Error on the test data set = ",Error)
    

if __name__ == '__main__':

    #--------------------------------------------------------------------------------------------------------------------------------------

    training_data = pd.read_csv('diabetes_dataset/training_data.csv',header=None)
    training_data=np.array(training_data)
    training_data_example = training_data[:, 0:8]
    training_data_label = training_data[:, 8]
    test_data = pd.read_csv('diabetes_dataset/test_data.csv',header=None)
    test_data = np.array(test_data)
    test_data_example = test_data[:, 0:8]
    test_data_label = test_data[:, 8]
    train_svm_with_data_and_advice(training_data_example, training_data_label, test_data_example, test_data_label)

