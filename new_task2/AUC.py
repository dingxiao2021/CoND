import numpy as np

def Calculation_AUC(MatrixAdjacency_Train,MatrixAdjacency_Test,Matrix_similarity,MaxNodeNum):

    AUCnum = 10000
    
    Matrix_similarity = np.triu(Matrix_similarity - Matrix_similarity * MatrixAdjacency_Train)
    Matrix_NoExist = np.ones(MaxNodeNum) - MatrixAdjacency_Train - MatrixAdjacency_Test - np.eye(MaxNodeNum)
     
    Test = np.triu(MatrixAdjacency_Test)
    NoExist = np.triu(Matrix_NoExist)
    


    Test_num = len(np.argwhere(Test == 1))
    NoExist_num = len(np.argwhere(NoExist == 1))

      
    Test_rd = [int(x) for index,x in enumerate((Test_num * np.random.rand(1,AUCnum))[0])]
    NoExist_rd = [int(x) for index,x in enumerate((NoExist_num * np.random.rand(1,AUCnum))[0])]

    TestPre= Matrix_similarity * Test
    NoExistPre = Matrix_similarity * NoExist
    
    TestIndex = np.argwhere(Test == 1)
    Test_Data = np.array([TestPre[x[0],x[1]] for index,x in enumerate(TestIndex)]).T
    NoExistIndex = np.argwhere(NoExist == 1)
    NoExist_Data = np.array([NoExistPre[x[0],x[1]] for index,x in enumerate(NoExistIndex)]).T

    
    Test_rd = np.array([Test_Data[x] for index,x in enumerate(Test_rd)])
    NoExist_rd = np.array([NoExist_Data[x] for index,x in enumerate(NoExist_rd)])

    n1,n2 = 0,0
    for num in range(AUCnum):
        if Test_rd[num] > NoExist_rd[num]:
            n1 += 1
        elif Test_rd[num] == NoExist_rd[num]:
            n2 += 0.5
        else:
            n1 += 0
    auc = float(n1+n2)/AUCnum

    return auc
