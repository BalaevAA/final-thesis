import matplotlib.pyplot as plt
import copy
import numpy as np


def create_hist_curves(args, train_local_loss):
    m = []
    for i in train_local_loss.keys():
        m.append(train_local_loss[i])
    
    arr_lens = []
    for i in range(args.num_users):
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
    
    
    x_ep = np.arange(0, max_ep)
    y_cur = [len(i) for i in bp]
    plt.title("hist number of curves in each epoch")
    plt.xlabel('epoch')
    plt.ylabel('clients')
    plt.bar(x_ep, y_cur, color ='maroon')
    plt.savefig(f"hist_num_users{args.num_users}_iid_{args.data_dist}_epochs_{args.epochs}.png") 
    # plt.show()
    plt.close()