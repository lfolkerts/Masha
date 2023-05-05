#!/usr/bin/env python

# ### Import required modules

import time
import random
import numpy as np
import torch
import torch.utils
from copy import deepcopy

from scipy.special import softmax
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.datasets as datasets
from concrete.ml.torch.compile import compile_brevitas_qat_model
import brevitas.nn as qnn
# And some helpers for visualization.
#get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt


# ### Load the data-set and visualize it

# In[2]:
words = open('babynames.txt', 'r').read().splitlines()

#Define Lambda
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?
vocab_size = 27
train_network = True

def build_dataset(words, vocab_size):
  X, Y = [], []
  for w in words:
    #print(w)
    context = np.zeros((block_size,vocab_size), dtype=int) #float) #np.double)
    for ch in w + '.':
      ix = np.zeros((vocab_size), dtype=int) #np.double)
      ix[stoi[ch]] = 1
      X.append(context)
      Y.append(stoi[ch])
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = np.concatenate((context[1:],ix.reshape(1,27)), axis=0) # crop and append

  X = torch.tensor(np.array(X))
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

x_train, y_train = build_dataset(words[:n1], vocab_size) #train
x_val, y_val = build_dataset(words[n1:n2], vocab_size) #validation
x_test, y_test = build_dataset(words[n2:], vocab_size) #test

class TinyMLP(nn.Module):
    """A very small CNN to classify the sklearn digits data-set.

    This class also allows pruning to a maximum of 10 active neurons, which
    should help keep the accumulator bit width low.
    """

    def __init__(self, n_neurons, n_blocks, n_vocab, n_bits) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        a_bits = n_bits
        w_bits = n_bits
        self.n_vocab = n_vocab
        self.n_embed = 10
        self.n_blocks = n_blocks
        
        self.q1 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(self.n_vocab, self.n_embed, bias=False, weight_bit_width=w_bits)

        self.q2 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(self.n_embed*self.n_blocks, n_neurons, bias=True, weight_bit_width=w_bits)
        self.q4 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.fc4 = qnn.QuantLinear(n_neurons, self.n_vocab, bias=True, weight_bit_width=w_bits)

        # Enable pruning, prepared for training
        self.toggle_pruning(True)

    def toggle_pruning(self, enable):
        """Enables or removes pruning."""

        # Maximum number of active neurons (i.e. corresponding weight != 0)
        n_active = 12

        # Go through all the convolution layers
        for layer in [self.fc2]: #self.fc2): #, self.fc3):
            s = layer.weight.shape

            # Compute fan-in (number of inputs to a neuron)
            # and fan-out (number of neurons in the layer)
            st = [s[0], np.prod(s[1:])]

            # The number of input neurons (fan-in) is the product of
            # the kernel width x height x inChannels.
            if st[1] > n_active:
                if enable:
                    # This will create a forward hook to create a mask tensor that is multiplied
                    # with the weights during forward. The mask will contain 0s or 1s
                    prune.l1_unstructured(layer, "weight", (st[1] - n_active) * st[0])
                else:
                    # When disabling pruning, the mask is multiplied with the weights
                    # and the result is stored in the weights member
                    prune.remove(layer, "weight")

    def forward(self, x):
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output."""
        #x_split = 
        #print(x.shape)
        x_pre = self.q1(x[:, 0, :])
        x_pre = self.fc1(x_pre)
        for i in range(1,self.n_blocks):
            x_pre = torch.cat((x_pre, self.fc1(self.q1(x[:,i,:]))), axis = 1)
            #print(f"x_pre: {x_pre.shape}")
        x = self.q2(x_pre)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.q4(x)
        x = self.fc4(x)
        return x

################### Training ######################################
torch.manual_seed(112)

def train_one_epoch(net, optimizer, train_loader):
    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()

    net.train()
    avg_loss = 0
    count = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = net(data)
        loss_net = loss(output, target.long())
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()
        count += 1
        if count % 1000 == 1:
            print(f"Loss: {avg_loss/count}")

    return avg_loss / len(train_loader)

def test_torch(net, n_bits, test_loader):
    """Test the network: measure accuracy on the test set."""

    loss = nn.CrossEntropyLoss(label_smoothing = 1/27/1000)
    # Freeze normalization layers
    net.eval()
    # Iterate over the batches
    idx = 0
    avg_loss = 0
    for data, target in test_loader:
        # Run forward and get the predicted class id
        output = net(data)
        loss_net = loss(output, target.long())
        avg_loss += loss_net.item()

    # Print out the accuracy as a percentage
    print(f"Average Validation Loss: {avg_loss / len(test_loader)}")


# Create the tiny CNN with 10 output classes
N_EPOCHS = 10

# Create a train data loader
train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=64)

# Create a test data loader to supply batches for network evaluation (test)
test_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
test_dataloader = DataLoader(test_dataset)

nets = []
bit_range = range(3,4)
bits = 3
net = TinyMLP(100, block_size, vocab_size, bits)
test_torch(net, bits, test_dataloader)
# Train the network with Adam, output the test set accuracy every epoch
losses = []
for n_bits in bit_range:
    net = TinyMLP(100, block_size, vocab_size, n_bits)
    losses_bits = []
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    if True: 
        for _ in tqdm(range(N_EPOCHS), desc=f"Training with {n_bits} bit weights and activations"):
            losses_bits.append(train_one_epoch(net, optimizer, train_dataloader))
        losses.append(losses_bits)
        net.toggle_pruning(False)
    nets.append(net)

print("Loss vs Epoch")
i = 3
for losses_bits in losses:
    print("\nBits: ", i)
    for i in range(0, int(len(losses_bits)), int(len(losses_bits)/10)):
        print(losses_bits[i], end=' ')

    i+=1
# ### Test the torch network in fp32

for idx, net in enumerate(nets):
    test_torch(net, bit_range[idx], test_dataloader)


# ### Define the Concrete ML testing function
# 
# We introduce the `test_with_concrete` function which allows us to test a Concrete ML model in one of two modes:
# - in FHE
# - in the clear, using simulated FHE execution
# 
# Note that it is trivial to toggle between between the two modes. 

# In[6]:


def test_with_concrete(quantized_module, test_loader, use_sim):
    """Test a neural network that is quantized and compiled with Concrete ML."""
    loss = nn.CrossEntropyLoss()

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    avg_loss = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()#.astype(dtype=np.int32)
        target = target.numpy()

        fhe_mode = "simulate" if use_sim else "execute"

        # Quantize the inputs and cast to appropriate data type
        y_pred = quantized_module.forward(data.astype(float), fhe=fhe_mode)
        loss_net = loss(torch.from_numpy(y_pred), torch.from_numpy(target).long())
        avg_loss += loss_net.item()

    # Compute and report results
    return avg_loss / len(test_loader)

# 
# Note that this is not a test in FHE. The simulated FHE mode gives 
# insight into the number of accumulator bits that are needed and the 
# impact of FHE execution on the accuracy.
# 
# The torch/brevitas neural network is quantized during training and, for inference, it is converted 
# to FHE by Concrete ML using a dedicated function, `compile_brevitas_qat_model`.
# 
# In this test we determine the accuracy and accumulator bit-widths for the various quantization settings
# that are trained above.

# In[7]:


accs = []
accum_bits = []
sim_time = []

for idx in range(len(bit_range)):
    print(x_train.shape)
    q_module = compile_brevitas_qat_model(nets[idx], x_train.float(), n_bits=idx+3)

    accum_bits.append(q_module.fhe_circuit.graph.maximum_integer_bit_width())

    start_time = time.time()
    accs.append(
        test_with_concrete(
            q_module,
            test_dataloader,
            use_sim=True,
        )
    )
    sim_time.append(time.time() - start_time)

for idx, vl_time_bits in enumerate(sim_time):
    print(
        f"Simulated FHE execution for {bit_range[idx]} bit network: {vl_time_bits:.2f}s, "
        f"{len(test_dataloader) / vl_time_bits:.2f}it/s"
    )



# ### Analysis of quantized results
# 
# We plot the accuracies obtained for various levels of quantization of weights and activations. 
# In addition, we plot the maximum accumulator bit width required to run inference of the network for
# each weight and activation bit width. This is shown as the numbers next to the graph markers. 
# 
# This accumulator bit width is determined by the compiler and is an important quantity in designing FHE-compatible neural networks.

# In[8]:


#fig = plt.figure(figsize=(12, 8))
#plt.rcParams["font.size"] = 14
#plt.plot(bit_range, accs, "-x")
#for bits, acc, accum in zip(bit_range, accs, accum_bits):
#    plt.gca().annotate(str(accum), (bits - 0.1, acc + 0.01))
#plt.ylabel("Accuracy on test set")
#plt.xlabel("Weight & activation quantization")
#plt.grid(True)
#plt.title(
#    "Accuracy for varying quantization bit width. Accumulator bit-width shown on graph markers"
#)
#plt.show()
print("BIts vs Accuracy")
for bits, acc, accum in zip(bit_range, accs, accum_bits):
    print(bits, accum, acc)


# ### Test the CNN in FHE
# 
# We identify 3 bit weights and activations as a good compromise for which the maximum accumulator size
# is low but the accuracy is acceptable. We can now compile to FHE and execute on encrypted data.

# ### 1. Compile to FHE

bits_for_fhe = 3
idx_bits_fhe = bit_range.index(bits_for_fhe)

accum_bits_required = accum_bits[idx_bits_fhe]

q_module_fhe = None

net = nets[idx_bits_fhe]

q_module_fhe = compile_brevitas_qat_model(
    net,
    x_train.float(),
    n_bits = bits_for_fhe
)


# ### 2. Generate Keys

# Generate keys first, this may take some time (up to 30min)
t = time.time()
q_module_fhe.fhe_circuit.keygen()
print(f"Keygen time: {time.time()-t:.2f}s")


# ### 3. Execute in FHE on encrypted data


# Run inference in FHE on a single encrypted example
mini_test_dataset = TensorDataset(torch.Tensor(x_test[[0], :]), torch.Tensor(y_test[[0]]))
mini_test_dataloader = DataLoader(mini_test_dataset)

t = time.time()
test_with_concrete(
    q_module_fhe,
    mini_test_dataloader,
    use_sim=False,
)
print(f"Time per inference in FHE: {(time.time() - t) / len(mini_test_dataset):.2f}")


# ### Conclusion
# 
# We see that quantization with **3** bit weight and activations is the best viable FHE configuration,
# as the accumulator bit width for this configuration is between **7 and 8** bits (can vary due to the final 
# distribution of the weights). The accuracy in this setting, 92% is a few percentage points 
# under the maximum accuracy achievable with larger accumulator bit widths (97-98%). 
# 
# Compiling the higher bit-width networks is also possible, but in this example, to ensure FHE execution is fast
# we used the lower bit-width quantization setting.
# 
