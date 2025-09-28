#Only ViT-tiny
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

def vit_model(num_classes=4):
    # Step 1: Load Pretrained ViT-Tiny Model
    vit_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, img_size = 256, num_classes=num_classes)  # Adjust num_classes accordingly
    # Step 2: Modify the final classification layer to match your task
    #vit_model.head = nn.Linear(vit_model.head.in_features, 4)  # Example for 4 classes, change as per your dataset
    vit_model.head = nn.Identity()
    
    # Freeze the pre-trained weights (optional, if you want to train only the last layer)
    for param in vit_model.parameters():
        param.requires_grad = True
    return vit_model
    
# Define the CNN with ViT model using simple concatenation
class CNNWithViTConcate(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNWithViTConcate, self).__init__()

        # CNN part
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(p=0.1)

        # Vision Transformer part with custom parameters
        self.vit = vit_model(num_classes=num_classes)

        #self.vit.head = nn.Identity()

        # Feature combination layers
        self.fc_cnn = nn.Linear(128 * 16 * 16, 256)
        self.fc_vit = nn.Linear(192, 256)

        # Combined fully connected layers
        self.fc_combined = nn.Linear(512, 256)
        self.bn_combined = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        # CNN forward pass
        x_cnn = self.pool1(F.relu(self.bn1(self.conv1(x))))
        #x_cnn = self.dropout1(x_cnn)
        #print(f'After Conv1 and Pool1: {x_cnn.shape}')

        x_cnn = self.pool2(F.relu(self.bn2(self.conv2(x_cnn))))
        #print(f'After Conv2 and Pool2: {x_cnn.shape}')

        x_cnn = self.pool3(F.relu(self.bn3(self.conv3(x_cnn))))
        #print(f'After Conv3 and Pool3: {x_cnn.shape}')

        x_cnn = self.pool4(F.relu(self.bn4(self.conv4(x_cnn))))
        x_cnn = self.dropout4(x_cnn)
        #print(f'After Conv4 and Pool4: {x_cnn.shape}')

        # Flatten CNN features
        x_cnn_flat = x_cnn.view(-1, 128 * 16 * 16)
        #x_cnn = x_cnn.view(x_cnn.size(0), -1)
        #print(f'After Flatten: {x_cnn.shape}')

        x_cnn_flat = F.relu(self.fc_cnn(x_cnn_flat))
        #print(f'After fc_cnn: {x_cnn.shape}')

        # Resize the input for ViT
        #x_resized = self.resize(x)

        # ViT forward pass
        x_vit = self.vit(x)
        #print(f'After ViT: {x_vit.shape}')

        # Flatten ViT features
        x_vit = F.relu(self.fc_vit(x_vit))
        #print(f'After fc_vit: {x_vit.shape}')

        # Combine CNN and ViT features
        x_combined = torch.cat((x_cnn_flat, x_vit), dim=1)
        #print(f'After Concatenation: {x_combined.shape}')

        x_combined = F.relu(self.bn_combined(self.fc_combined(x_combined)))
        #print(f'After fc_combined: {x_combined.shape}')

        # Pass through final fully connected layers
        x_combined = F.relu(self.bn_fc1(self.fc1(x_combined)))
        #print(f'After fc1: {x_combined.shape}')

        x_combined = self.fc2(x_combined)
        #print(f'After fc2 (Output): {x_combined.shape}')

        return x_combined #x_cnn, x_vit



# Define ViT-CNN model with Late Fusion
# Define LateFusion class
class LateFusion(nn.Module):
    def __init__(self, num_classes=4):
        super(LateFusion, self).__init__()
        self.fc_cnn = nn.Linear(128 * 16 * 16, num_classes)  # Adapt input size
        self.fc_vit = nn.Linear(192, num_classes)           # Adapt input size
        self.fc_combined = nn.Linear(2 * num_classes, num_classes)

    def forward(self, x_cnn, x_vit):
        logits_cnn = self.fc_cnn(x_cnn)  # Class logits from CNN
        logits_vit = self.fc_vit(x_vit)  # Class logits from ViT
        logits_combined = torch.cat((logits_cnn, logits_vit), dim=1)
        return self.fc_combined(logits_combined)  # Final combined logits

class CNNWithViTLate(nn.Module):
    def __init__(self, num_classes=4, use_attention_fusion=False):
        super(CNNWithViTLate, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Vision Transformer part with custom parameters
        self.vit = vit_model(num_classes=num_classes)
        
        # Late fusion module
        self.late_fusion = LateFusion(num_classes)



    def forward(self, x):
        # CNN forward pass
        x_cnn = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x_cnn = self.pool2(F.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.pool3(F.relu(self.bn3(self.conv3(x_cnn))))
        x_cnn = self.pool4(F.relu(self.bn4(self.conv4(x_cnn))))

        # Flatten CNN features
        x_cnn_flat = x_cnn.view(-1, 128 * 16 * 16)
        #print(f'After Flatten: {x_cnn_flat.shape}')

        x_vit = self.vit(x)
        #print(f'After ViT: {x_vit.shape}')
        x_out = self.late_fusion(x_cnn_flat, x_vit)

        return x_out

# Define Cross-Attention Fusion class
# This class implements cross-attention fusion between CNN and ViT features

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_cnn, dim_vit, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        # Cross-attention layers
        self.cross_att_cnn_to_vit = nn.MultiheadAttention(embed_dim=dim_vit, num_heads=num_heads)
        self.cross_att_vit_to_cnn = nn.MultiheadAttention(embed_dim=dim_cnn, num_heads=num_heads)

    def forward(self, x_cnn, x_vit):
        # Add sequence dimensions for multihead attention
        x_cnn = x_cnn.unsqueeze(0)  # Shape: (1, batch_size, dim_cnn)
        x_vit = x_vit.unsqueeze(0)  # Shape: (1, batch_size, dim_vit)

        # Cross-attention
        # CNN attends to ViT features
        att_cnn_to_vit, _ = self.cross_att_cnn_to_vit(x_vit, x_cnn, x_cnn)  # Query: ViT, Key/Value: CNN
        # ViT attends to CNN features
        att_vit_to_cnn, _ = self.cross_att_vit_to_cnn(x_cnn, x_vit, x_vit)  # Query: CNN, Key/Value: ViT

        # Remove sequence dimensions after attention
        att_cnn_to_vit = att_cnn_to_vit.squeeze(0)  # Shape: (batch_size, dim_vit)
        att_vit_to_cnn = att_vit_to_cnn.squeeze(0)  # Shape: (batch_size, dim_cnn)

        # Concatenate cross-attention results
        x_combined = torch.cat((att_cnn_to_vit, att_vit_to_cnn), dim=1)  # Shape: (batch_size, dim_vit + dim_cnn)
        
        return x_combined

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


# Define the CNN with ViT model using cross-attention fusion

class CNNWithViTCross(nn.Module):
    def __init__(self, num_classes=4, dim_cnn =  128, dim_vit = 128, dropout_prob = 0.3):
        super(CNNWithViTCross, self).__init__()
        # Initialize the model parameters
        # CNN part
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.dropout4 = nn.Dropout(p=0.05)

        # Vision Transformer part
        self.vit = vit_model(num_classes=num_classes)  # Adjust to match your implementation

        # Linear layers to project features to the same dimension
        self.fc_cnn = nn.Linear(256 * 16 * 16, dim_cnn)  # Adjust input size as needed
        self.fc_vit = nn.Linear(192, dim_vit)            # Adjust input size as needed

        # Self-attention layers for CNN and ViT
        #self.self_attention_cnn = nn.MultiheadAttention(embed_dim=dim_cnn, num_heads=8)
        

        # Dropout layer for CNN features
        self.dropout_cnn = nn.Dropout(dropout_prob)
        # Spatial attention layers for CNN and ViT
        self.spatial_attention_cnn = SpatialGate()
        self.self_attention_vit = nn.MultiheadAttention(embed_dim=dim_vit, num_heads=8)

        # Dropout layer for ViT features
        self.dropout_vit = nn.Dropout(dropout_prob)


        # Cross-attention fusion
        self.cross_attention_fusion = CrossAttentionFusion(dim_cnn=dim_vit, dim_vit=dim_cnn)

        # Fully connected layers after fusion
        self.fc_combined = nn.Linear(256, 256)
        #self.bn_combined = nn.BatchNorm1d(256)
        self.bn_combined = nn.LayerNorm(256)
        
        self.fc1 = nn.Linear(256, 128)
        #self.bn_fc1 = nn.BatchNorm1d(128)
        self.bn_fc1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN forward pass
        x_cnn = self.pool1(F.relu(self.bn1(self.conv1(x))))
        #x_cnn = self.dropout1(x_cnn)
        x_cnn = self.pool2(F.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.pool3(F.relu(self.bn3(self.conv3(x_cnn))))
        x_cnn = self.pool4(F.relu(self.bn4(self.conv4(x_cnn))))
        #x_cnn = self.dropout4(x_cnn)


        x_cnn_attention = self.spatial_attention_cnn(x_cnn)
        #print(x_cnn_attention.shape)

        # Flatten CNN features and project
        x_cnn_flat = x_cnn_attention.view(-1, 256 * 16 * 16) 
        #x_cnn_flat = x_cnn.view(-1, 128 * 16 * 16)
        x_cnn_flat = F.relu(self.fc_cnn(x_cnn_flat))


        # Apply dropout after CNN features
        x_cnn_flat = self.dropout_cnn(x_cnn_flat)

        #Apply self-attention to CNN features
        #x_cnn_attention = x_cnn_flat.unsqueeze(0)  # Add sequence dimension (seq_len, batch_size, embed_dim)
        #self_attention_cnn, _ = self.self_attention_cnn(x_cnn_attention, x_cnn_attention, x_cnn_attention)
        #self_attention_cnn = self_attention_cnn.squeeze(0)  # Remove sequence dimension

        # ViT forward pass and project
        x_vit = self.vit(x)
        x_vit = F.relu(self.fc_vit(x_vit))

    
        # Apply dropout after ViT features
        x_vit = self.dropout_vit(x_vit)

        # Apply self-attention to ViT features
        x_vit_attention = x_vit.unsqueeze(0)  # Add sequence dimension (seq_len, batch_size, embed_dim)
        self_attention_vit, _ = self.self_attention_vit(x_vit_attention, x_vit_attention, x_vit_attention)
        self_attention_vit = self_attention_vit.squeeze(0)  # Remove sequence dimension
        
        # Cross-attention fusion
        x_combined = self.cross_attention_fusion(x_cnn_flat, self_attention_vit)

        # Pass through fully connected layers
        x_combined = F.relu(self.bn_combined(self.fc_combined(x_combined)))
        x_combined = F.relu(self.bn_fc1(self.fc1(x_combined)))
        x_combined = self.fc2(x_combined)

        return x_combined
    

# Define the CNN model without ViT
class CNNModel(nn.Module):
    def __init__(self,num_classes=4, dropout_prob = 0.5):
        super(CNNModel, self).__init__()
        # CNN part
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.dropout4 = nn.Dropout(p=0.2)

        self.fc = nn.Linear(128 * 16 * 16, 256)
        self.bn = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN forward pass
        x_cnn = self.pool1(F.relu(self.bn1(self.conv1(x))))
        #x_cnn = self.dropout1(x_cnn)
        x_cnn = self.pool2(F.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.pool3(F.relu(self.bn3(self.conv3(x_cnn))))
        x_cnn = self.pool4(F.relu(self.bn4(self.conv4(x_cnn))))
        #x_cnn = self.dropout4(x_cnn)

        #print(x_cnn.shape)

        x = x_cnn.view(-1, 128 * 16 * 16)

        # Pass through fully connected layers
        x = F.relu(self.bn(self.fc(x)))
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x= self.fc2(x)

        return x
