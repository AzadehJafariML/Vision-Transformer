"""From-scratch 3D Vision Transformer for four-class functional-map classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Defining patch extraction and embedding

# class is a template for creating objects and to reuse. It bundles two things together:
#1. information it stores and needs to process inputs, 2. processing steps appplied on an input


#def __init__(self): DUNDER init or double underscore init method acts as the object initializer inside a class
#Self means this particular object, everthing in front of it are attributes of the self

# ***Dunder init prepares the components; forward uses them to process and apply command on the inputs***
class PatchEmbedding3D(nn.Module): #PatchEmbedding3D has pytorch's model features
    def __init__(self, image_shape, patch_size, embedding_dim): #A function inside a class is called a method.
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.patch_size = patch_size
        if len(self.image_shape) != 3 or min(self.image_shape) <= 0:
            raise ValueError("3D image must contain three positive dimensions")
        if patch_size <= 0:
            raise ValueError("patch size must be positive and larger than zero")
        depth, height, width = self.image_shape
        pad_d = (-depth) % patch_size# minus makes the remaining positive -58 % 8= +6
        pad_h = (-height) % patch_size#-80 % 8= 0
        pad_w = (-width) % patch_size# -54%8 = +2
        #F.pad expects padding instructions starting with the last dimension. this means width first, then height, then depth
        self.padding = (0, pad_w, 0, pad_h, 0, pad_d)
        grid_d = (depth + pad_d) // patch_size #rounding down the result to the nearest whole number
        grid_h = (height + pad_h) // patch_size
        grid_w = (width + pad_w) // patch_size
        self.num_patches = grid_d * grid_h * grid_w #after extracting patches, the image dimensions are converted to grids (number of patches in each coordination)
        self.projection = nn.Linear(patch_size ** 3, embedding_dim) ##nn.Linear(input_size, output_size), in this step each patch
        #contained 512 (8** 3) values

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError("Expected shape (batch, 1, depth, height, width).")
        if tuple(x.shape[2:]) != self.image_shape:
            raise ValueError("Map dimensions differ from the model setup.")
        batch = x.shape[0]
        p = self.patch_size
        
        # apply padding AFTER the original image at the high-index edges.
        # Constant mode fills the added positions with the value(0).
        padded = F.pad(x, self.padding, mode="constant", value=0)
        
        #"constant"==> 0 0 1 2 3 4 0 0 . it Fills with a fixed number, set by value (default 0)
        #"reflect"==> 3 2 1 2 3 4 3 2. Mirrors the data at the edge, without repeating the edge value
        #"replicate"==> 1 1 1 2 3 4 4 4. Repeats the edge value outward
        #"circular"=> 3 4 1 2 3 4 1 2 .Wraps around: values from the opposite end fill the gap, as if the data repeated in a loop
        #value is used only with "constant"; the other modes ignore it.
        
        
        #extracting non-overlapping 3D patches.
        # Tensor.unfold(dimension, size, step) takes windows along ONE AXIS.
        # if size = step, this means the patches do not overlap.
        patches = padded.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)
        # Reorder axes into [B, nD, nH, nW, C, pD, pH, pW].
        ordered = patches.permute(0, 2, 3, 4, 1, 5, 6, 7)
        vectors = ordered.reshape(batch, self.num_patches, p ** 3)
        
        # SHARED patch embedding layer
        # Feature = sum(weights * voxel values) + bias. we defined 64 features per patch
        return self.projection(vectors)


# DEFINing multi-head attention
#here we defined three numerical representation for each patch as Quer, Key, and Values
#in this baseline version, I kept the number of features in each representation 64 (considering the residual connections)
#but in other versions, I tried more than 64 for researching the impact. in this case we should decrease the dimension of the output 
#of this step dueing projection layer when we eant to add it to the tokens so that they have same size.

# in this baseline version, I sued equal inputs to each head (16 features for each head, number of heads=4) but we can distribute
#then unevenly. in this case, I considered the rescaling after dot product calcualtion and before entering the softmax. here we 
#divied to 4 but this is when the input size for each head is 16. if we change the input size, we should update rescaling
#step for each ehad separately

#calculating the Q,K, and values from the previous tokens
#dividing Q,K, and V into vectors for each head
#dot product between Q of each patch and K (transposed) of all other patches inclsuinf itself (a table of 560 * 560)
#rescaling the weights before softmax
#entering the softmax so we have the attnetion weight for each patch
#multiplying each softmax value to the value of that patch & summing them up to have one final updated value for each patch (this cotains the impact of all other patches as well)
#concatenating the updated values from each head (end-to-end join)
#using the linear projection to create a new value vector
#adding the value to the tokens (first residual connection)
#normalizing (Z-score) with normLayer
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        if num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("Head number must be an integer that is divisible by the feature numbers in embedding")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads #rounding down
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch, patches, features = x.shape
        # Q, K and V weights are shared among patches.
        # Each head reads ALL 64 inputs and produces 16 Q/K/V features. these feature will be concatenated finally
        q = self.query(x).reshape(
            batch, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.key(x).reshape(
            batch, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(x).reshape(
            batch, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # producing dot product for query of each patch with every other keys, including itself.
        scores = q @ k.transpose(-2, -1) #@ is shorthand for matrix multiplication and is= torch.matmul((queries, keys_transposed)
        scaled_scores = scores / (self.head_dim ** 0.5) #rescaling the dot product for each head. the input size for each head is 16 in this version
        # every row, within each head is turned to weight, sums to one.
        weights = torch.softmax(scaled_scores, dim=-1)
        # each weight multiplies the WHOLE source value vector.
        # Sum those weighted vectors feature by feature within each head.
        head_outputs = weights @ v
        # concatenate the head outputs, then mix them with Linear.
        joined = head_outputs.transpose(1, 2).reshape(
            batch, patches, features
        )
        return self.output_projection(joined)


#Defining the multi layer perceptron(MLP): Linear (64 to 128) --> GELU (keep 128) --> Linear (128 to 64)
# this is the multilayer perceptron that uses the normalized (token+ attention-weighted values)
# here I only defined three layers, fully connected
# activation function here is GELU
# if wwe use RELU, we might face dying RELU problems when facing negative inputs
class PatchMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.expand = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.project = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # Linear 1: 64 --> 128,  GELU: 128 --> 128, Linear 3: 128--> 64, separately for each patch.
        hidden = self.expand(x)
        activated = self.activation(hidden)
        output = self.project(activated)
        return output


#the execution order inside one transformer block (attention+MLP)
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim) #where ever there is an update from previous part, we need z-scoring
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = PatchMLP(embedding_dim, hidden_dim)

    def forward(self, x):
        # normalize BEFORE attention; keep original x for the residual.
        normalized = self.norm1(x)
        # M06–M10: all attention operations run inside this call.
        attention_update = self.attention(normalized)
        # M11: First RESIDUAL CONNECTION
        # Add the attention update to the original tokens.
        after_attention = x + attention_update
        # M12: normalize each patch before the MLP.
        normalized_after_attention = self.norm2(after_attention)
        # M13: the MLP computes a feature update.
        feature_update = self.mlp(normalized_after_attention)
        # M14: SECOND RESIDUAL CONNECTION
        output = after_attention + feature_update
        return output


# Defining the complete model; forward shows the actual image path.
class ViT3D(nn.Module):
    def __init__(self, image_shape, num_classes=4, patch_size=8,
                 embedding_dim=64, num_heads=4, hidden_dim=128):
        super().__init__()
        self.patch_embedding = PatchEmbedding3D(
            image_shape, patch_size, embedding_dim
        )
        # Create a learnable vector for each patch location.
        # Position parameters: 560 * 64, not 512 * 64.
        self.position_embedding = nn.Parameter(torch.zeros(
            1, self.patch_embedding.num_patches, embedding_dim
        ))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.block = TransformerBlock(embedding_dim, num_heads, hidden_dim)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images):
        # pad, extract patches and embed.
        embedded = self.patch_embedding(images)
        # Combine patch content and position
        tokens = embedded + self.position_embedding
        # attention, first residual, MLP and second residual.
        processed = self.block(tokens)
        # Final feature normalization and mean pooling
        normalized = self.final_norm(processed)
        # Mean pooling averages patch vectors into one vector per image.
        # Image feature j = (patch1[j] + ... + patch560[j]) / 560.
        image_features = normalized.mean(dim=1)
        #convert 64 image features to four class scores.
        logits = self.classifier(image_features)
        return logits


def build_model(
    image_shape=(58, 80, 54),
    num_classes=4,
    patch_size=8,
    embedding_dim=64,
    num_heads=4,
    hidden_dim=128,
):
    """Create the baseline ViT3D model with explicit configuration."""
    return ViT3D(
        image_shape=image_shape,
        num_classes=num_classes,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
    )
