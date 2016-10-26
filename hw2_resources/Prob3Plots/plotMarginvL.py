import pylab as pl

def problem3_plot(L, margin1,margin2,margin3,margin4):
    pl.figure()
    pl.subplot('111', axisbg='whitesmoke')
    line1 = pl.plot(L,margin1, linestyle = '-',c='r',marker = 'o', label='dataset1')
    line2 = pl.plot(L,margin2, linestyle = '-',c='b',marker = 'o',label='dataset2')    
    line3 = pl.plot(L,margin3, linestyle = '-',c='g',marker = 'o',label='dataset3')
    line4 = pl.plot(L,margin4, linestyle = '-',c='y',marker = 'o',label='dataset4') 
    pl.legend([line1, line2,line3,line4], ['Dataset 1', 'Dataset2', 'Dataset 3','Dataset 4'])
    title = 'Effect of $\lambda$ on number of support vectors using Linear Pegasos '
    pl.title(title)
    pl.legend()
    pl.ylabel('Number of Support Vectors')
    pl.xlabel('$ \log_2 \lambda$')
    pl.axis('tight')
    
    pl.savefig('Efect_of_L_on_Margin.png')


L = [0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
L = range(-10,3)
margin1 = [0.02229214019686376, 0.044483274578318868, 0.088454804106533044, 0.17476529108767841, 0.34425335505605381, 0.6655983335118586, 1.0000796101186362, 1.1737369747102224, 1.3567466596512512, 1.610065403023474, 1.8188371281265749, 2.1128634478447959, 2.6421649934902236]

margin2 = [0.7008691198039696, 0.73646152628399131, 0.75800082590500506, 0.78417697095489758, 0.80963115478927483, 0.87544786569418109, 0.94117608147971854, 1.0807438455487299, 1.2452649476808995, 1.5135213523239373, 1.9195464149181902, 2.6569540219033021, 4.3292416344259355]

margin3 = [0.21926152339439561, 0.26680398001980099, 0.33274760014837718, 0.41916230130903542, 0.46842503966075777, 0.56302304802353742, 0.67107515767315573, 0.78425240543106189, 0.93330615202323763, 1.1384950351761223, 1.4319973317183956, 2.1650148925217003, 4.0183537460617362]

margin4 = [2.0004418064211453, 2.019385769342763, 2.0896661656485152, 2.1971670148476723, 2.8451327299347278, 3.0900347254200549, 3.3513133698392772, 3.8731436812631714, 3.8997329990513476, 6.1370214534916663, 11.984866018765514, 24.149207583203253, 48.178966222652733]
#print len(L), len(margin3)
problem3_plot(L,margin1,margin2,margin3,margin4)


### QP NUM SVMS
gamma = [2**i for i in range(-2,3)]
QPsvm_1 = [76, 91, 124, 180, 225]
QPsvm_2 = [213, 210, 217, 241, 290]
QPsvm_3 = [113, 109, 122, 178, 250]
QPsvm_4 = [168, 194, 247, 343, 385]
#
#problem3_plot(gamma, QPsvm_1, QPsvm_2,QPsvm_3)
##

# Pegasos NUM_SVMS

gamma = [2**i for i in range(-2,3)]
svm_1 = [8,40,38,55,73]
svm_2 = [142,138,399,128,304]
svm_3 = [200,30,29,38,57]
svm_4 = [400,137,161,201,67]

## PROB2 EFFECT OF C on NUM SVM
#C = [0.01,0.1,1,10,100]
#C = [-2,-1,0,1,2]
#svm_1 = [399, 152, 49, 53, 52]
#svm_2 = [390, 237, 125, 85, 109]
#svm_3 = [385, 136, 58, 45, 54]
#svm_4 = [400, 273, 145, 106, 93]

def problem3_plotsvm(L, margin1,margin2,margin3,margin4):
    pl.figure()
    pl.subplot('111', axisbg='whitesmoke')
    line1 = pl.plot(L,margin1, linestyle = '-',c='r',marker = 'o', label='dataset1')
    line2 = pl.plot(L,margin2, linestyle = '-',c='b',marker = 'o',label='dataset2')    
    line3 = pl.plot(L,margin3, linestyle = '-',c='g',marker = 'o',label='dataset3')    
    line4 = pl.plot(L,margin4, linestyle = '-',c='y',marker = 'o',label='dataset4')
    pl.legend([line1, line2,line3,line4], ['Dataset 1', 'Dataset2', 'Dataset 3','Dataset 4'])
    title = 'Effect of $\gamma$ on number of support vectors using Pegasos'
    pl.title(title)
    pl.legend()
    pl.ylabel('Number of Support Vectors')
    pl.xlabel('$\gamma$')
    pl.axis('tight')
    
    pl.savefig('Peg_Gam_NumSVM.png')
    
problem3_plotsvm(gamma, svm_1, svm_2, svm_3, svm_4)
#
##def heatmap