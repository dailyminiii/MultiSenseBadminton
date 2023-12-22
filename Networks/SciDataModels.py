import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DRefinedModel(nn.Module):
    def __init__(self, input_channels, hidden_features, output_size):
        super(Conv1DRefinedModel, self).__init__()

        self.conv1d_1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)  # Reverting the stride
        self.bn2 = nn.BatchNorm1d(64)
        # self.conv1d_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm1d(128)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        
        # 1x1 convolution for adjusting the channels of residual
        self.match_dim = nn.Conv1d(32, 64, kernel_size=1)
        # Adaptive max pooling for adjusting the time dimension of residual
        self.adaptive_pool = nn.AdaptiveMaxPool1d(72)  # Target size for residual before conv1d_2

        self.fc1 = nn.Linear(2304, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_size)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        
        # Initial convolution
        x1 = F.relu(self.conv1d_1(x))
        x1 = self.bn1(x1)
        x1 = self.maxpool1d(x1)
        x1 = self.dropout(x1)
        
        # Adjust residual for the next convolution
        residual = self.match_dim(x1)  # adjust channels for residual connection
        residual = self.adaptive_pool(residual)  # adjust the time dimension for residual connection
        
        # Apply residual connection
        x2 = F.relu(self.conv1d_2(x1) + residual)
        x2 = self.bn2(x2)
        x2 = self.maxpool1d(x2)
        x2 = self.dropout(x2)
        
        # Continue with the rest of the model
        # x3 = F.relu(self.conv1d_3(x2))
        # x3 = self.bn3(x3)
        # x3 = self.maxpool1d(x3)
        # x3 = self.dropout(x3)

        x3 = x2.view(x2.size(0), -1)
        x3 = self.fc1(x3)
        x3 = self.fc2(x3)

        return x3
    
class ConvLSTMRefinedModel(nn.Module):
    def __init__(self, input_channels, hidden_features, output_size):
        super(ConvLSTMRefinedModel, self).__init__()

        self.conv1d_1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)  # Reverting the stride
        self.bn2 = nn.BatchNorm1d(64)
        # self.conv1d_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm1d(128)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        
        # 1x1 convolution for adjusting the channels of residual
        self.match_dim = nn.Conv1d(32, 64, kernel_size=1)
        # Adaptive max pooling for adjusting the time dimension of residual
        self.adaptive_pool = nn.AdaptiveMaxPool1d(72)  # Target size for residual before conv1d_2

        self.fc1 = nn.Linear(4608, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_size)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        
        # Initial convolution
        x1 = F.relu(self.conv1d_1(x))
        x1 = self.bn1(x1)
        x1 = self.maxpool1d(x1)
        x1 = self.dropout(x1)
        
        # Adjust residual for the next convolution
        residual = self.match_dim(x1)  # adjust channels for residual connection
        residual = self.adaptive_pool(residual)  # adjust the time dimension for residual connection
        
        # Apply residual connection
        x2 = F.relu(self.conv1d_2(x1) + residual)
        x2 = self.bn2(x2)
        x2 = self.maxpool1d(x2)
        x2 = self.dropout(x2)
        
        # Continue with the rest of the model
        # x3 = F.relu(self.conv1d_3(x2))
        # x3 = self.bn3(x3)
        # x3 = self.maxpool1d(x3)
        # x3 = self.dropout(x3)

        # Continue with the rest of the model
        x2 = torch.transpose(x2, 1, 2)
        x3, _ = self.lstm(x2)

        a, b, c = x3.shape
        x3 = x3.reshape(a, b*c)
        x3 = self.fc1(x3)
        x3 = self.fc2(x3)

        return x3
    
class LSTMRefinedModel(nn.Module):
    def __init__(self, input_channels, hidden_features, output_size):
        super(LSTMRefinedModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=64, num_layers=2, batch_first=True)

        self.fc1 = nn.Linear(9600, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_size)

    def forward(self, x):
       
        # x2 = torch.transpose(x, 1, 2)
        x3, _ = self.lstm(x)

        a, b, c = x3.shape
        x3 = x3.reshape(a, b*c)
        x3 = self.fc1(x3)
        x3 = self.fc2(x3)

        return x3
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Linear(2 * embed_size, embed_size)
        )

    def forward(self, x, mask=None):
        attention_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm(attention_out + x)
        out = self.feed_forward(x)
        return out

class TransformerRefinedModel(nn.Module):
    def __init__(self, input_channels, embed_size, output_size):
        super(TransformerRefinedModel, self).__init__()

        # Initial embedding layer
        self.embedding = nn.Linear(input_channels, embed_size)

        # Transformer block
        self.transformer = TransformerBlock(embed_size)

        # Fully connected output layer
        self.fc = nn.Linear(embed_size, output_size)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)

        # Transformer block
        x = self.transformer(x)

        # Use only the output of the last time step
        x = x[:, -1, :]

        # Fully connected output layer
        x = self.fc(x)

        return x