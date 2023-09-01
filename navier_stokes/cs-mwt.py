import sys
sys.path.append('../')
from mwt_model import *

torch.manual_seed(0)
np.random.seed(0)


print("torch version is ",torch.__version__)
ntrain = 1000
ntest = 5000


learning_rate = 0.001

step_size = 100
gamma = 0.5

ich = 3
initializer = get_initializer('xavier_normal')

parser = argparse.ArgumentParser(description='parse batch_size, epochs and resolution')
parser.add_argument('--bs',  default=10, type = int, help='batch-size')
parser.add_argument('--ep', default=500, type = int, help='epochs')
parser.add_argument('--res', default=63, type = int, help='resolution')
parser.add_argument('--wd', default=1e-4, type = float, help='weight decay')

args = parser.parse_args()

batch_size = args.bs #100
epochs = args.ep #500
res = args.res + 1#32#sys.argv[1]
wd = args.wd
print("\nUsing batchsize = %s, epochs = %s, and resolution = %s\n"%(batch_size, epochs, res))
params = {}
params["xmin"] = 0
params["ymin"] = 0
params["xmax"] = 1
params["ymax"] = 1

################################################################
# load data and data normalization
################################################################
PATH = "../../../../../../localdata/Derick/stuart_data/Darcy_421/NavierStokes_TrainData=1000_TestData=5000_Resolution=64X64_Domain=[0,1]X[0,1].hdf5"
resultPATH = ""
#Read Datasets
X_train, Y_train, X_test, Y_test = readtoArray(PATH, 1024, 5000, Nx = 64, Ny = 64)

print ("Converting dataset to numpy array.")
tt = time.time()
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test  = np.array(X_test )
Y_test  = np.array(Y_test )
print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))

print ("Subsampling dataset to the required resolution.")
tt = time.time()
X_train = SubSample(X_train, res, res)
Y_train = SubSample(Y_train, res, res)
X_test  = SubSample(X_test , res, res)
Y_test  = SubSample(Y_test , res, res)

new_res = closest_power(res) #Closest Power of 2. MWT takes only resolutions which are a power of 2

X_train = CubicSpline3D(X_train, new_res, new_res)
Y_train = CubicSpline3D(Y_train, new_res, new_res)
X_test  = CubicSpline3D(X_test , new_res, new_res)
Y_test  = CubicSpline3D(Y_test , new_res, new_res)

print ("    Subsampling completed after %.2f minutes"%((time.time()-tt)/60))

print ("Taking out the required train/test size.")
tt = time.time()
x_train = torch.from_numpy(X_train[:ntrain, :, :]).float()
y_train = torch.from_numpy(Y_train[:ntrain, :, :]).float()
x_test  = torch.from_numpy(X_test[ :ntest,  :, :]).float()
y_test  = torch.from_numpy(Y_test[ :ntest,  :, :]).float()
print ("    Taking completed after %s seconds"%(time.time()-tt))
print("...")


x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

grids = []
grids.append(np.linspace(0, 1, new_res))
grids.append(np.linspace(0, 1, new_res))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,new_res,new_res,2)
grid = torch.tensor(grid, dtype=torch.float)
x_train = torch.cat([x_train.reshape(ntrain,new_res,new_res,1), grid.repeat(ntrain,1,1,1)], dim=3)
x_test = torch.cat([x_test.reshape(ntest,new_res,new_res,1), grid.repeat(ntest,1,1,1)], dim=3)
# x_train = x_train.reshape(ntrain,res,res,1)
# x_test = x_test.reshape(ntest,res,res,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
model = MWT2d(ich, 
            alpha = 12, # number of modes
            c = 4, # c*k**d (where d is the dimension) will be the input and output channels of the spectral convolution as introduced in FNo
            k = 4, # we use polynomials of degree upto k = 4
            base = 'legendre', # 'chebyshev'
            nCZ = 4, # number of MWT layers
            L = 0, # 2^L gives the coarsest scale ie how much downampling we should arrive at
            initializer = initializer,
            ).to(device)
print("Model has %s parameters"%(count_params(model)))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
TIMESTAMP = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')#time.strftime("%Y%m%d-%H%M%S-%f")
if os.path.isfile(resultPATH+'files/lossData_'+TIMESTAMP+'.txt'):
    os.remove(resultPATH+'files/lossData_'+TIMESTAMP+'.txt')  
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0


    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, new_res, new_res)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    abs_err = 0.0
    with torch.no_grad():

        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, new_res, new_res)
            out = y_normalizer.decode(out)
            if ep == epochs-1:
                print("Returning to orignial resolution to get test error based on it")
                out = CubicSpline3D(out.detach().cpu().numpy(), res, res)
                out = torch.from_numpy(out).float().to(device)
                y   = CubicSpline3D(y.detach().cpu().numpy(), res, res)
                y   = torch.from_numpy(y).float().to(device)


            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_l2/= ntrain
    abs_err /= ntest
    test_l2 /= ntest

    t2 = default_timer()
    print("epoch: %s, completed in %.4f seconds. Training Loss: %.4f and Test Loss: %.4f"%(ep+1, t2-t1, train_l2, test_l2))

    file = open(resultPATH+'files/lossData_'+TIMESTAMP+'.txt',"a")
    file.write(str(ep+1)+" "+str(train_l2)+" "+str(test_l2)+"\n")

ModelInfos = "_cs_%03d"%(res)+"~res_"+str(np.round(test_l2,6))+"~RelL2TestError_"+str(ntrain)+"~ntrain_"+str(ntest)+"~ntest_"+str(batch_size)+"~BatchSize_"+str(learning_rate)+\
            "~LR_"+str(wd)+"~Reg_"+str(gamma)+"~gamma_"+str(step_size)+"~Step_"+str(epochs)+"~epochs_"+time.strftime("%Y%m%d-%H%M%S")  
                            
torch.save(model.state_dict(), resultPATH+"files/last_model"+ModelInfos+".pt")
os.rename(resultPATH+'files/lossData_'+TIMESTAMP+'.txt', resultPATH+'files/lossData'+ModelInfos+'.txt')

dataLoss = np.loadtxt(resultPATH+'files/lossData'+ModelInfos+'.txt')

stepTrain = dataLoss[:,0] #Reading Epoch                  
errorTrain = dataLoss[:,1] #Reading erros
errorTest  = dataLoss[:,2]

print("Ploting Loss VS training step...")
fig = plt.figure(figsize=(15, 10))
plt.yscale('log')
plt.plot(stepTrain, errorTrain, label = 'Training Loss')
plt.plot(stepTrain , errorTest , label = 'Test Loss')
plt.xlabel('epochs')#, fontsize=16, labelpad=15)
plt.ylabel('Loss')
plt.legend(loc = 'upper right')
plt.title("lr = %s test error = %s"%(learning_rate, str(np.round(test_l2,6))))
plt.savefig(resultPATH+"figures/Error_VS_TrainingStep"+ModelInfos+".png", dpi=500)




model.load_state_dict(torch.load(resultPATH+"files/last_model"+ModelInfos+".pt"))
model.eval()


print()
print()

#Just a file containing data sampled in same way as the training and test dataset
fileName = "NavierStokes_TrainData=1_TestData=1_Resolution=64X64_Domain=[0,1]X[0,1].hdf5"
F_train, U_train, F_test, U_test= readtoArray(fileName, 1, 1, 64, 64)

F_train = SubSample(np.array(F_train), res, res)
F_train_cs = CubicSpline3D(F_train, new_res, new_res)

print("Starting the Verification with Sampled Example")
tt = time.time()
U_FDM = SubSample(np.array(U_train), res, res)[0]
#U_FDM = CubicSpline3D(U_FDM, new_res, new_res)

print("      Doing MWT on Example...")
tt = time.time()
ff = torch.from_numpy(F_train_cs).float()
ff = x_normalizer.encode(ff)
ff = torch.cat([ff.reshape(1,new_res,new_res,1), grid.repeat(1,1,1,1)], dim=3).cuda() #ff.reshape(1,res,res,1).cuda()#

U_MWT = model(ff)
U_MWT = U_MWT.reshape(1, new_res, new_res)
U_MWT = y_normalizer.decode(U_MWT)
U_MWT = U_MWT.detach().cpu().numpy()
U_MWT = CubicSpline3D(U_MWT, res, res)
U_MWT = U_MWT[0] 
print("            MWT completed after %s"%(time.time()-tt))

myLoss = LpLoss(size_average=False)
print()
print("Ploting comparism of FDM and MWT Simulation results")
fig = plt.figure(figsize=((5+2)*4, 5))

#fig.suptitle("Plot of $- \Delta u = f(x, y)$ on $\Omega = ]0,1[ x ]0,1[$ with $u|_{\partial \Omega}  = 0.$")

colourMap = plt.cm.magma # parula() #plt.cm.jet #plt.cm.coolwarm

plt.subplot(1, 4, 1)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Input")
plt.imshow(F_train[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 2)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM")
plt.imshow(U_FDM, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 3)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("MWT")
plt.imshow(U_MWT, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 4)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM-MWT, RelL2Err = "+str(round(myLoss.rel_single(U_MWT, U_FDM).item(), 3)))
plt.imshow(np.abs(U_FDM - U_MWT), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto', norm=matplotlib.colors.LogNorm())#)#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.savefig(resultPATH+'figures/compare'+ModelInfos+'.png',dpi=500)

plt.show()