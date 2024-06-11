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
        self.st_detections = []

        ### hybrid outputs  
        self.hybrid_detections = []

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
        merged_detection, grouped_det = self.ei.merge_detections(self.ei_detections, diff_val=5.0)  ### merge detections for multiple variables
        self.ei_detections = merged_detection
        self.st_detections = self.model.test_single(sample_path)

        self.hybrid_detections = self.combine_detections(self.ei_detections, self.st_detections)

        return self.hybrid_detections
    
    def combine_detections(self, ei_detection, st_detection):
        '''
        ei_detections: list of ei detections -> list
        st_detections: list of st detections -> list

        return:
        combined_detections: list of combined detections -> list
        '''
        ####### Approach 1: check if any st and ei detections intersect, give high priority to st detections since they have high precision.
        all_detections = []
        for ei_det in ei_detection:
            #### structure of detection, get elements
            ei_var = ei_det[0]
            eits1, eits2 = ei_det[1]
            # print('EI', eits1, eits2)

            #### get all st detections within the ei detection
            st_detections_within_ei = []
            for i, st_det in enumerate(st_detection):
                st_var = st_det[0]
                stts1, stts2 = st_det[1]
                # print('ST', i, stts1, stts2)

                if eits1 <= stts1 and eits2 >= stts2:
                    st_detections_within_ei.append(st_det)
                    
            if len(st_detections_within_ei) > 0:
                for det in st_detections_within_ei:
                    # print('Removed', det)
                    st_detection.remove(det)
                all_detections.extend(st_detections_within_ei)
                # print(ei_det, 'replaced with', st_detections_within_ei)
            else:
                all_detections.append(ei_det)
                # print('added', ei_det)

        print('Any Detections in ST that are not in EI:')
        print(st_detection)
        all_detections.extend(st_detection)
        ############# Approach 1 end #############        

        return all_detections
    

    def get_correct_detections(self, ei_detections, st_detections):
        correct_pred, rest_pred, y_pred, y_true = self.ei.get_correct_detections(ei_detections, st_detections)
        return correct_pred, rest_pred, y_pred, y_true

    def viz_thresholds(self):
        self.ei.viz_thresholds(self.exe_list, thresholds=self.thresholds)
    