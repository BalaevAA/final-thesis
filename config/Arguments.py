class Arguments:
    def __init__(self):
        self.epochs = 60
        self.num_users = 20
        self.frac = 0.2
        self.local_ep = 5
        self.local_bs = 32
        self.test_bs = 32
        self.lr = 0.001
        self.momentum = 0.9
        self.split = 'user'
        
        self.model = 'mobilenet'
        
        self.rebuild = 1
        self.struct = 1
        self.dataset = 'imagenet'
        self.iid = True
        self.alpha = 0.8
        self.num_classes = 200
        self.num_channels = 3
        self.gpu = 0
        self.stopping_rounds = 10
        self.verbose = 1
        self.debug = 1
        self.seed = 1
        