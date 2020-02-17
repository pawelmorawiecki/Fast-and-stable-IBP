import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
#from utils import epoch, epoch_robust_bound, epoch_calculate_robust_err, Flatten, generate_kappa_schedule_CIFAR, generate_epsilon_schedule_CIFAR

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1) 



def epoch(loader, model, device, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp,_ = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def bound_propagation(model, initial_bound):
    l, u = initial_bound
    bounds = []
    bounds.append(initial_bound)
    list_of_layers = list(model.children())
    
    for i in range(len(list_of_layers)):
        layer = list_of_layers[i]
        
        if isinstance(layer, Flatten):
            l_ = Flatten()(l)
            u_ = Flatten()(u)

        elif isinstance(layer, nn.Linear):
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t() 
                  + layer.bias[:,None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t() 
                  + layer.bias[:,None]).t()
            
        elif isinstance(layer, nn.Conv2d):
            l_ = (nn.functional.conv2d(l, layer.weight.clamp(min=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(u, layer.weight.clamp(max=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None,:,None,None])
            
            u_ = (nn.functional.conv2d(u, layer.weight.clamp(min=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(l, layer.weight.clamp(max=0), bias=None, 
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) + 
                  layer.bias[None,:,None,None])
            
        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)
            
        bounds.append((l_, u_))
        l,u = l_, u_
    return bounds


def interval_based_bound(model, c, bounds, idx):
    # requires last layer to be linear
    cW = c.t() @ model.last_linear.weight
    cb = c.t() @ model.last_linear.bias
    
    l,u = bounds[-2]
    return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:,None]).t()


def epoch_robust_bound(loader, model, epsilon_schedule, device, kappa_schedule, batch_counter, opt=None):
    robust_err = 0
    total_robust_loss = 0
    total_mse_loss = 0
    total_combined_loss = 0
    
    C = [-torch.eye(10).to(device) for _ in range(10)]
    for y0 in range(10):
        C[y0][y0,:] += 1
    
    for i,data in enumerate(loader,0):
      
        #if i>299:  #calculate only for 100 batches
        #break      
        
        mse_loss_list = []
        lower_bounds = []
        upper_bounds = []
        
        
        X,y = data
        X,y = X.to(device), y.to(device)
        
        ###### fit loss calculation ######
        yp,_ = model(X)
        fit_loss = nn.CrossEntropyLoss()(yp,y)
    
        ###### robust loss calculation ######
        initial_bound = (X - epsilon_schedule[batch_counter], X + epsilon_schedule[batch_counter])
        bounds = bound_propagation(model, initial_bound)
        robust_loss = 0
        for y0 in range(10):
            if sum(y==y0) > 0:
                lower_bound = interval_based_bound(model, C[y0], bounds, y==y0)
                robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound, y[y==y0]) / X.shape[0]
                
                robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item() #increment when true label is not winning       
        
        total_robust_loss += robust_loss.item() * X.shape[0]  
        
        ##### MSE Loss #####
        
        #indices_of_layers = [2,4,7,8] #CNN_small
        indices_of_layers = [2,4,6,8,11,13,14] #CNN_medium
        
        
        for i in range(len(indices_of_layers)):
            lower_bounds.append(Flatten()(bounds[indices_of_layers[i]][0])) #lower bounds 
            upper_bounds.append(Flatten()(bounds[indices_of_layers[i]][1])) #upper bounds 
            mse_loss_list.append(nn.MSELoss()(lower_bounds[i], upper_bounds[i]))
            
        
        mse_loss = mse_loss_list[0] + mse_loss_list[1] + mse_loss_list[2] + mse_loss_list[3] + mse_loss_list[4] + mse_loss_list[5] + mse_loss_list[6]
        total_mse_loss += mse_loss.item()
        
        ###### combined losss ######
        combined_loss = kappa_schedule[batch_counter]*fit_loss + (1-kappa_schedule[batch_counter])*robust_loss + mse_loss

        total_combined_loss += combined_loss.item()
        
        batch_counter +=1
         
        if opt:
            opt.zero_grad()
            combined_loss.backward()
            opt.step() 
        
    return robust_err / len(loader.dataset), total_combined_loss / len(loader.dataset), total_mse_loss/ len(loader.dataset)

        
def epoch_calculate_robust_err (loader, model, epsilon, device):
    robust_err = 0.0
    
    C = [-torch.eye(10).to(device) for _ in range(10)]
    for y0 in range(10):
        C[y0][y0,:] += 1


    for X,y in loader:
        X,y = X.to(device), y.to(device)
        
        initial_bound = (X - epsilon, X + epsilon)
        bounds = bound_propagation(model, initial_bound)

        for y0 in range(10):
            if sum(y==y0) > 0:
                lower_bound = interval_based_bound(model, C[y0], bounds, y==y0)                
                robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item() #increment when true label is not winning       
        
    return robust_err / len(loader.dataset)
        
        


def generate_kappa_schedule_MNIST():

    kappa_schedule = 2000*[1] # warm-up phase
    kappa_value = 1.0
    step = 0.5/58000
    
    for i in range(58000):
        kappa_value = kappa_value - step
        kappa_schedule.append(kappa_value)
    
    return kappa_schedule

def generate_epsilon_schedule_MNIST(epsilon_train):
    
    epsilon_schedule = []
    step = epsilon_train/10000
            
    for i in range(10000):
        epsilon_schedule.append(i*step) #ramp-up phase
    
    for i in range(50000):
        epsilon_schedule.append(epsilon_train)
        
    return epsilon_schedule


def generate_kappa_schedule_CIFAR():

    kappa_schedule = 10000*[1] # warm-up phase
    kappa_value = 1.0
    step = 0.5/340000
    
    for i in range(340000):
        kappa_value = kappa_value - step
        kappa_schedule.append(kappa_value)
    
    return kappa_schedule

def generate_epsilon_schedule_CIFAR(epsilon_train):
    
    epsilon_schedule = []
    step = epsilon_train/150000
            
    for i in range(150000):
        epsilon_schedule.append(i*step) #ramp-up phase
    
    for i in range(200000):
        epsilon_schedule.append(epsilon_train)
        
    return epsilon_schedule 
  
  
def pgd_linf_rand(model, X, y, epsilon, alpha, num_iter, restarts):
    """ Construct PGD adversarial examples on the samples X, with random restarts"""
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(X)
    
    for i in range(restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
        
        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(X + delta)[0], y)
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        
        all_loss = nn.CrossEntropyLoss(reduction='none')(model(X+delta)[0],y)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
        
    return max_delta


def epoch_adversarial(model, loader, attack, *args):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, *args)
        yp = model(X+delta)[0]
        loss = nn.CrossEntropyLoss()(yp,y)
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

#Root MSE

def RMSELoss(yp,y):
    return torch.sqrt(nn.MSELoss()(yp, y)+1e-6) #small epsilon is added to avoid NaN when mse=0  
  
#Log (Product of squared errors)

def Log_Product_Loss(yp,y):
  return torch.mean(torch.log((yp-y)**2+1e-6))

#Log (Product of absolute errors)

def Log_Product_MAE(yp,y):
  return torch.sum(torch.log(torch.abs(yp-y)+1e-6))

# Mean Absolute Error

def MAELoss(yp,y):
  return torch.mean(torch.abs((yp-y)))

def DiffVolume(yp,y):
  vol_lower_bound = torch.sum(torch.log(yp+1e-6))
  vol_upper_bound = torch.sum(torch.log(y+1e-6))
  diff = vol_upper_bound - vol_lower_bound
  return diff

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

BATCH_SIZE = 50
dataset_path = './svhn'

trainset = datasets.SVHN(root=dataset_path, split='train', download=True)

train_mean = trainset.data.mean(axis=(0,2,3))/255  
train_std = trainset.data.std(axis=(0,2,3))/255

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])
kwargs = {'num_workers': 1, 'pin_memory': True}


train_loader = torch.utils.data.DataLoader(datasets.SVHN(
    root=dataset_path, split='train', download=True,
    transform=transform_train),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.SVHN(root=dataset_path, split='test', download=True,
    transform=transform_test),
    batch_size=BATCH_SIZE, shuffle=False, **kwargs)


class CNN_medium(torch.nn.Module):
    def __init__(self):

        super(CNN_medium, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=0, stride=1)
        self.relu1 = nn.ReLU() 
        self.conv2 = nn.Conv2d(32, 32, 4, padding=0, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, padding=0, stride=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 4, padding=0, stride=2)
        self.relu4 = nn.ReLU()
        self.flat = Flatten()
        self.linear1 = nn.Linear(64*5*5, 512)
        self.relu5 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu6 = nn.ReLU()
        self.last_linear = nn.Linear(512, 10)                
        
    def forward(self, x):
        
        hidden_activations = []
        
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.flat(x)

        x = self.linear1(x)
        x = self.relu5(x)

        x = self.linear2(x)
        x = self.relu6(x)

        
        out = self.last_linear(x)
        
        return out, hidden_activations

model = CNN_medium().to(device)

opt = optim.Adam(model.parameters(), lr=1e-3)

EPSILON = 8/255
EPSILON_TRAIN = 8/255
epsilon_schedule = generate_epsilon_schedule_CIFAR(EPSILON_TRAIN)
kappa_schedule = generate_kappa_schedule_CIFAR()
batch_counter = 0

test_err_tab = []
pgd_test_err_tab = []
ver_test_err_tab = []

print("Training started...")
for t in range(240):
    _, combined_loss, mse_loss = epoch_robust_bound(train_loader, model, epsilon_schedule, device, kappa_schedule, batch_counter, opt)

    # check loss and accuracy on test set
    #test_err, _ = epoch(test_loader, model, device)
    #robust_err = epoch_calculate_robust_err(test_loader, model, EPSILON, device)

    batch_counter += 1455

    if t == 137:  #decrease learning rate
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-4

    if t == 171:  #decrease learning rate after 250 epochs
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-5

    if t == 206:  #decrease learning rate after 300 epochs
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-6

    print("Epoch " + str(t) + " done\n")

    test_err, _ = epoch(test_loader, model, device)
    verified = epoch_calculate_robust_err(test_loader, model, EPSILON, device)
    
    test_err_tab.append(test_err)
    ver_test_err_tab.append(verified)
    
    np.savetxt("./SVHN_medium/SVHN_medium_EPSILON_"+str(EPSILON)+"EPSILON_TRAIN_"+ str(EPSILON_TRAIN)+"_Test_Error_Rate.txt", test_err_tab)
    #np.savetxt("./SVHN_small/MNIST_large_EPSILON_"+str(EPSILON)+"EPSILON_TRAIN_"+ str(EPSILON_TRAIN)+"_PGD_Error_Rate.txt", pgd_test_err_tab)
    np.savetxt("./SVHN_medium/SVHN_medium_EPSILON_"+str(EPSILON)+"EPSILON_TRAIN_"+ str(EPSILON_TRAIN)+"_Verified.txt", ver_test_err_tab)

#SAVING RESULTS
with open('./SVHN_medium/results_extra_loss_SVHN_medium.txt', 'w') as f:

  test_err, _ = epoch(test_loader, model, device)
  print ("Test Error Rate: " + str(test_err) + "\n")
  f.write("Test Error Rate: " + str(test_err) + "\n")

  pgd_result = epoch_adversarial(model, test_loader, pgd_linf_rand, EPSILON, 1e-2, 200, 10)[0]
  print ("PGD Error Rate: " + str(pgd_result) + "\n")
  f.write("PGD Error Rate: " + str(pgd_result) + "\n")

  verified = epoch_calculate_robust_err(test_loader, model, EPSILON, device)
  print ("Verified: " + str(verified) + "\n")
  f.write("Verified: " + str(verified) + "\n")

f.close()

test_err_tab.append(test_err)
ver_test_err_tab.append(verified)
pgd_test_err_tab.append(pgd_result)
np.savetxt("./SVHN_medium/SVHN_medium_EPSILON_"+str(EPSILON)+"EPSILON_TRAIN_"+ str(EPSILON_TRAIN)+"_Test_Error_Rate.txt", test_err_tab)
np.savetxt("./SVHN_medium/SVHN_medium_EPSILON_"+str(EPSILON)+"EPSILON_TRAIN_"+ str(EPSILON_TRAIN)+"_PGD_Error_Rate.txt", pgd_test_err_tab)
np.savetxt("./SVHN_medium/SVHN_medium_EPSILON_"+str(EPSILON)+"EPSILON_TRAIN_"+ str(EPSILON_TRAIN)+"_Verified.txt", ver_test_err_tab)

#SAVING WEIGHTS
torch.save(model.state_dict(), 'weights_extra_loss_SVHN_medium.pth')





