import abc
from torch.utils.data import Dataset


class TESTGROUP_Dataset(Dataset, abc.ABC):

    def __init__(self, data_clean, data_red):
        # data = num x dim
        self.m_datanum = data_clean.shape[0]
        self.m_clean_data = data_clean
        self.m_red_data = data_red

    def __repr__(self):
        msg = f'TESTGROUP_Dataset\n' \
              f'(number of data, data dim): {self.m_data.shape}\n'
        return msg

    def __len__(self):
        return self.m_datanum

    def __getitem__(self, index):
        OutputData = {
            'clean':self.m_clean_data[index],
            'red':self.m_red_data[index]
        }
        return OutputData, index