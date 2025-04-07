import torch
import torch.nn as nn

class Deep_LSTM(nn.Module):
    def __init__(self):
        super(Deep_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=39, hidden_size=128, num_layers=1, batch_first=True)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        
        self.lstm4 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        
        self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.batchnorm5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.5)
        
        self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.batchnorm6 = nn.BatchNorm1d(64)
        self.dropout6 = nn.Dropout(0.5)
        
        self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.batchnorm7 = nn.BatchNorm1d(32)
        self.dropout7 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(32,16)
        self.dropout8 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16,4)

    def forward(self, x) :
        batch_size, seq_len, _ = x.shape
        
        x, _ = self.lstm1(x)
        x = x.contiguous().view(-1, x.shape[-1])
        x = self.batchnorm1(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = x.contiguous().view(-1, x.shape[-1])
        x = self.batchnorm2(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)
        x = x.contiguous().view(-1, x.shape[-1])
        x = self.batchnorm3(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.dropout3(x)
        
        x, _ = self.lstm4(x)
        x = x.contiguous().view(-1, x.shape[-1])
        x = self.batchnorm4(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.dropout4(x)
        
        x, _ = self.lstm5(x)
        x = x.contiguous().view(-1, x.shape[-1])
        x = self.batchnorm5(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.dropout5(x)

        x, _ = self.lstm6(x)
        x = x.contiguous().view(-1, x.shape[-1])
        x = self.batchnorm6(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.dropout6(x)
        
        x, _ = self.lstm7(x)
        x = x[:,-1,:]
        x = self.batchnorm7(x)
        x = self.dropout7(x)
        
        x = self.fc1(x)
        x = self.dropout8(x)
        x = self.fc2(x)
        
        return x