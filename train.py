from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from include import *
from models import *
NETPARAMS = 'netparams.dat'

# should run the file like "python train.py ./path/to/folder/"
def main():

    # load hyper parameters
    folder = sys.argv[1]
    with open(folder + 'hyperparameters.json', 'r') as f:
        params = json.load(f)

    # load traininig data
    S = pickle.load(open(params["train_pkl"], "rb"))
    S.computeParameters()
    S.toDevice(params["device"])
    
    # load validation set
    T = pickle.load(open(params["valid_pkl"], "rb"))    
    T.computeParameters()
    T.toDevice(params["device"])

    # initialize network 
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
    net = SubdNet(params)
    net = net.to(params['device'])
    net.apply(init_weights)

    # loss function
    lossFunc = torch.nn.MSELoss().to(params['device'])

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=params['lr'])

    # training
    trainLossHis = []
    validLossHis = []

    trainP2pLossHis = []
    validP2pLossHis = []
    
    trainConsistLossHis = [] 
    validConsistLossHis = []
    
    trainSparLossHis = [] 
    validSparLossHis = []

    
    bestLoss = np.inf
    nSub = params["numSubd"]
    for epoch in range(params['epochs']):      
        ts = time.time()

        # loop over training shapes 
        trainErr = 0.0
        trainp2pErr = 0.0
        trainConsistErr = 0.0 
        trainSparErr = 0.0

        
        for mIdx in range(S.nM):
            # forward pass
            x = S.getInputData(mIdx)
            
            # 传入训练集的 groupedLocalData
            outputs, consist_loss, sparse_loss = net(x, mIdx, S.hfList, S.poolMats, S.dofs, S.Us, S.Lambdas, S.groupedLocalData)
            
            # target mesh
            Vt = S.meshes[mIdx][params["numSubd"]].V.to(params['device'])
            Ft = S.meshes[mIdx][params["numSubd"]].F

            # compute loss function
            loss = 0.0
            p2p_loss = 0.0
            for ii in range(params["numSubd"]+1): #错误原因，在此位置少加1
                nV = outputs[ii].size(0)
                V = S.meshes[mIdx][ii].V.to(params['device'])
                p2p_loss += lossFunc(outputs[ii], V[:nV,:])#
                
            loss = p2p_loss+0.1*consist_loss+0.01*sparse_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record training error
            trainErr += loss.cpu().data.numpy()
            trainp2pErr += p2p_loss.item()
            trainConsistErr += consist_loss.item()
            trainSparErr += sparse_loss.item()

        trainLossHis.append(trainErr / S.nM)
        trainP2pLossHis.append(trainp2pErr / S.nM)
        trainConsistLossHis.append(trainConsistErr / S.nM) 
        trainSparLossHis.append(trainSparErr/S.nM)


        # loop over validation shapes 
        validErr = 0.0
        validp2pErr = 0.0
        validConsistErr = 0.0 
        validSparErr = 0.0
        
        for mIdx in range(T.nM):
            x = T.getInputData(mIdx)
            
            outputs, consist_loss, sparse_loss = net(x, mIdx, T.hfList, T.poolMats, T.dofs, T.Us, T.Lambdas, T.groupedLocalData)
            
            # target mesh
            Vt = T.meshes[mIdx][params["numSubd"]].V.to(params['device'])
            Ft = T.meshes[mIdx][params["numSubd"]].F

            # compute loss function
            loss = 0.0
            p2p_loss = 0.0
            for ii in range(params["numSubd"]+1):
                nV = outputs[ii].size(0)
                V = T.meshes[mIdx][ii].V.to(params['device']) 
                p2p_loss += lossFunc(outputs[ii], V[:nV,:])
                
            loss = p2p_loss+0.1*consist_loss+0.01*sparse_loss    
            
            # record validation error
            validErr += loss.cpu().data.numpy()
            
            validp2pErr += p2p_loss.item()
            c_loss_val = consist_loss.item() if torch.is_tensor(consist_loss) else consist_loss
            validConsistErr += c_loss_val
            validSparErr += sparse_loss.item()
            
        validLossHis.append(validErr / T.nM)
        validP2pLossHis.append(validp2pErr / T.nM)
        validConsistLossHis.append(validConsistErr / T.nM) 
        validSparLossHis.append(validSparErr / T.nM)


        # save the best model
        if validErr < bestLoss:
            bestLoss = validErr
            torch.save(net.state_dict(),params['output_path'] + NETPARAMS)

        print("epoch %d, train loss %.6e, valid loss %.6e, remain time: %s" % (epoch, trainLossHis[-1], validLossHis[-1], int(round((params['epochs']-epoch) * (time.time() - ts)))))

    # save loss history
    np.savetxt(params['output_path']+'train_loss.txt', np.array(trainLossHis), delimiter=',')
    np.savetxt(params['output_path']+'valid_loss.txt', np.array(validLossHis), delimiter=',')

    np.savetxt(params['output_path']+'train_p2p_loss.txt', np.array(trainP2pLossHis), delimiter=',')
    np.savetxt(params['output_path']+'valid_p2p_loss.txt', np.array(validP2pLossHis), delimiter=',')
    
    np.savetxt(params['output_path']+'train_consist_loss.txt', np.array(trainConsistLossHis), delimiter=',')
    np.savetxt(params['output_path']+'valid_consist_loss.txt', np.array(validConsistLossHis), delimiter=',')

    np.savetxt(params['output_path']+'train_spar_loss.txt', np.array(trainSparLossHis), delimiter=',')
    np.savetxt(params['output_path']+'valid_spar_loss.txt', np.array(validSparLossHis), delimiter=',')
    # ========================================================================

    # write output shapes (validation set)
    mIdx = 0
    x = T.getInputData(mIdx)
    
    outputs,_ = net(x, mIdx, T.hfList, T.poolMats, T.dofs, T.Us, T.Lambdas, T.groupedLocalData)

    # write unrotated outputs
    tgp.writeOBJ(params['output_path'] + str(mIdx) +'_oracle.obj',\
        T.meshes[mIdx][len(outputs)-1].V.to('cpu'),\
        T.meshes[mIdx][len(outputs)-1].F.to('cpu'))
    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        tgp.writeOBJ(params['output_path'] + str(mIdx) +'_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))

    # write rotated output shapes (validation set)
    mIdx = 0
    x = T.getInputData(mIdx)
    outputs,_,_ = net(x, mIdx, T.hfList, T.poolMats, T.dofs, T.Us, T.Lambdas, T.groupedLocalData)

    dV = torch.rand(1, 3).to(params['device'])
    R = random3DRotation().to(params['device'])
    x[:,:3] = x[:,:3].mm(R.t())
    x[:,3:6] = x[:,3:6].mm(R.t())
    x[:,6:] = x[:,6:].mm(R.t())
    x[:,:3] += dV
    
    outputs,_,_ = net(x, mIdx, T.hfList, T.poolMats, T.dofs, T.Us, T.Lambdas, T.groupedLocalData)

    # write rotated outputs
    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        tgp.writeOBJ(params['output_path'] + str(mIdx) +'_rot_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))
        
if __name__ == '__main__':
    main()
