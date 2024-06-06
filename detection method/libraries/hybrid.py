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
        self.ei_detections = []

        ### st outputs
        self.transitions = None

    def train(self, train_data_path):
        '''
        file_paths -> list: 
            complete path to the sample data files (.npy)
        '''
        ####### train ei model #######
        exe_list, filewise_exe_list = self.ei.get_exeint(train_data_path)
        self.exe_list = exe_list
        ###calculate dynamic thresholds 
        thresholds = self.ei.get_dynamicthresh(exe_list)
        self.thresholds = thresholds    ### update to class var so that it can be accessed by other functions in the class

        ####### train st model #######
        self.model.train(train_data_path)
        self.transitions = self.model.transitions

    def test_single(self, sample_path, thresholds): 
        '''
        sample_path: path to the test trace file -> str
        thresholds: dictionary containing the threshold values for each variable -> dict

        return:
        detected_anomalies: list of detected anomalies -> list
        '''
        #### ei testing
        self.ei_detections = self.ei.test_single(sample_path, thresholds)
        self.st_detection = self.model.test_single(sample_path)

        return self.st_detection, self.ei_detections

    def viz_thresholds(self):
        self.ei.viz_thresholds(self.exe_list, thresholds=self.thresholds)
    