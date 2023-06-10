import matplotlib.pyplot as plt
import copy
import numpy as np

def create_boxplot(args, train_local_loss):
    m = []
    for i in train_local_loss.keys():
        m.append(train_local_loss[i])
    
    arr_lens = []
    for i in range(50):
        arr_lens.append(len(m[i]))
    
    max_ep = max(arr_lens)
    
    
    m_c = copy.deepcopy(m)
    for i in range(args.num_users):
        if len(m_c[i]) < max_ep:
            for j in range(max_ep - len(m_c[i])):
                m_c[i].append(0)
    
    
    bp = []
    for i in range(max_ep):
        tmp = []
        for j in range(args.num_users):
            tmp.append(m_c[j][i])
        bp.append(tmp)
    
    
    for i in range(len(bp)):
        bp[i] = [j for j in bp[i] if j != 0]
    
    
    data = []
    temp = np.arange(0, 44, 3)
    for i in range(0, 44):
        if i in temp:
            data.append(bp[i])
        else:
            data.append([])
            
    
    
    fig = plt.figure(figsize =(15, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(bp)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'boxplot_epoch_{args.epochs}_numUser_{args.num_users}_dataDist_{args.data_dist}_dataset_{args.dataset}')
    # plt.show()
    plt.close()
