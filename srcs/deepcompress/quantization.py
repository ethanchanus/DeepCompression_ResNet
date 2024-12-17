import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix
from .netprune import PruningModule, MaskedLinear, MaskedConv2d

def apply_weight_sharing(model, bits=5):
    """
    Applies weight sharing to the given model
    """
    print("applying...")
    # for module in model.children():
    for name, module in model.named_modules():
        if not type(module) is MaskedLinear and not type(module) is MaskedConv2d:
        # if not hasattr(module, 'weight'):
            print("Ignore module:", name)
            continue

        # if (type(module) is not MaskedLinear) and (type(module) is not MaskedConv2d):
        #         print("Ignore module:", name)
        
        dev = module.weight.device
        weight = module.weight.data.cpu() #.numpy()
        shape = weight.shape
        # to 2 dim
        weightview = weight.contiguous().view(weight.shape[0], -1)
        shapeview = weightview.shape
        print(name, module, shape, shapeview)
        # print("weight", weight)
        # print("weightview", weightview)


        mat = csr_matrix(weightview) if shapeview[0] < shapeview[1] else csc_matrix(weightview)
        # mat = torch.tensor(weight).to_sparse_csr()
        print("mat.shape =", mat.shape)
        
        if mat.nnz == 0:
            print("mat", mat)
            print("ignore this all zero csr matrix")
            continue
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2**bits)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = new_weight
        # print(new_weight.shape)
        # print("mat.shape =", mat.shape)
        
        module.weight.data = torch.from_numpy(mat.toarray()).contiguous().view(shape).to(dev)
        # print (module.weight.data.size())

        # exit()


