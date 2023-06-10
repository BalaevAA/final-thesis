class Arguments:
    def __init__(self):
        self.epochs = 60
        self.num_users = 20
        self.frac = 0.2
        self.local_ep = 5
        self.local_bs = 32
        self.test_bs = 32
        self.lr = 0.01
        self.momentum = 0.9
        self.optimizer = 'user'
        
        self.model = 'vgg19'
        
        self.dataset = 'imagenet'
        self.data_dist = 'noniid1'
        self.alpha = 0.8
        self.num_classes = 10
        self.num_channels = 3
        self.gpu = 0
        self.verbose = 1
        self.seed = 1
        