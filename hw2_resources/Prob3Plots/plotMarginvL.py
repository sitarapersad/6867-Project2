import pylab as pl

def problem3_plot(L, margin1,margin2,margin3):
    pl.figure()
    pl.subplot('111', axisbg='whitesmoke')
    line1 = pl.plot(L,margin1, linestyle = '-',c='r',marker = 'o', label='dataset1')
    line2 = pl.plot(L,margin2, linestyle = '-',c='b',marker = 'o',label='dataset2')    
    line3 = pl.plot(L,margin3, linestyle = '-',c='g',marker = 'o',label='dataset3')
    pl.legend([line1, line2,line3], ['Dataset 1', 'Dataset2', 'Dataset 3'])
    title = 'Effect of $\gamma$ on number of support vectors using Quadratic Program '
    pl.title(title)
    pl.legend()
    pl.ylabel('Number of Support Vectors')
    pl.xlabel('$ \gamma$')
    pl.axis('tight')
    
    pl.savefig('QP_NumSVM.png')


L = [0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
L = range(-10,3)
margin1 = [0.02229214019686376, 0.044483274578318868, 0.088454804106533044, 0.17476529108767841, 0.34425335505605381, 0.6655983335118586, 1.0000796101186362, 1.1737369747102224, 1.3567466596512512, 1.610065403023474, 1.8188371281265749, 2.1128634478447959, 2.6421649934902236]

margin2 = [0.7008691198039696, 0.73646152628399131, 0.75800082590500506, 0.78417697095489758, 0.80963115478927483, 0.87544786569418109, 0.94117608147971854, 1.0807438455487299, 1.2452649476808995, 1.5135213523239373, 1.9195464149181902, 2.6569540219033021, 4.3292416344259355]

margin3 = [0.21926152339439561, 0.26680398001980099, 0.33274760014837718, 0.41916230130903542, 0.46842503966075777, 0.56302304802353742, 0.67107515767315573, 0.78425240543106189, 0.93330615202323763, 1.1384950351761223, 1.4319973317183956, 2.1650148925217003, 4.0183537460617362]
#print len(L), len(margin3)
#problem3_plot(L,margin1,margin2,margin3)


## QP NUM SVMS
gamma = [2**i for i in range(-2,3)]
QPsvm_1 = [76, 91, 124, 180, 225]
QPsvm_2 = [213, 210, 217, 241, 290]
QPsvm_3 = [113, 109, 122, 178, 250]

problem3_plot(gamma, QPsvm_1, QPsvm_2,QPsvm_3)

## Pegasos NUM_SVMS

#gamma = 
#svm_1 = 
#svm_2 = 
#svm_3 = 

## PROB2 EFFECT OF C on NUM SVM
C = [0.01,0.1,1,10,100]
C = [-2,-1,0,1,2]
svm_1 = [399, 152, 49, 53, 52]
svm_2 = [390, 237, 125, 85, 109]
svm_3 = [385, 136, 58, 45, 54]

def problem3_plotsvm(L, margin1,margin2,margin3):
    pl.figure()
    pl.subplot('111', axisbg='whitesmoke')
    line1 = pl.plot(L,margin1, linestyle = '-',c='r',marker = 'o', label='dataset1')
    line2 = pl.plot(L,margin2, linestyle = '-',c='b',marker = 'o',label='dataset2')    
    line3 = pl.plot(L,margin3, linestyle = '-',c='g',marker = 'o',label='dataset3')
    pl.legend([line1, line2,line3], ['Dataset 1', 'Dataset2', 'Dataset 3'])
    title = 'Effect of $C$ on number of support vectors using Quadratic Program '
    pl.title(title)
    pl.legend()
    pl.ylabel('Number of Support Vectors')
    pl.xlabel('$\log_{10} C$')
    pl.axis('tight')
    
    pl.savefig('C_NumSVM.png')
    
#problem3_plotsvm(C, svm_1,svm_2,svm_3)

#def heatmap