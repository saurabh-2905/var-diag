from libraries.exeint import exeInt
from libraries.state_transition import StateTransition as st


class hybrid:
    def __init__(self):
        ### initialize exeinz
        self.ei = exeInt()
        self.model = st()

        ### ei outputs
        self.thresholds = None
        self.exe_list = None

    def train(self, train_data_path):
        ####### train ei model #######
        exe_list, filewise_exe_list = self.ei.get_exeint(train_data_path)
        self.exe_list = exe_list
        ###calculate dynamic thresholds 
        thresholds = self.ei.get_dynamicthresh(exe_list)
        self.thresholds = thresholds    ### update to class var so that it can be accessed by other functions in the class

    def viz_thresholds(self):
        self.ei.viz_thresholds(self.exe_list, thresholds=self.thresholds)
    