import numpy as np
import scipy.io as sio

# Taken from http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
def loadmat(filename):
    '''
    this function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


if __name__ == '__main__':
    model_file = 'LOCATION OF dictnet.mat from http://www.robots.ox.ac.uk/~vgg/research/text/#sec-models NIPS
                 DLW 2014 models'
    mat_contents = loadmat(model_file)

    L1 = _todict(mat_contents['layers'][0])
    conv1b = np.array(L1['biases'], dtype=np.float32)
    conv1W = np.array(L1['filters'], dtype=np.float32)
    conv1W = np.reshape(conv1W,(5,5,1,64))
    conv1W = np.array(conv1W, dtype=np.float32).transpose((3,2,0,1))[:,:,::-1,::-1]
 
    L1 = _todict(mat_contents['layers'][3])
    conv2b = np.array(L1['biases'], dtype=np.float32)
    conv2W = np.array(L1['filters'], dtype=np.float32).transpose((3,2,0,1))[:,:,::-1,::-1]

    L1 = _todict(mat_contents['layers'][6])
    conv3b = np.array(L1['biases'], dtype=np.float32)
    conv3W = np.array(L1['filters'], dtype=np.float32).transpose((3,2,0,1))[:,:,::-1,::-1]

    L1 = _todict(mat_contents['layers'][8])
    conv35b = np.array(L1['biases'], dtype=np.float32)
    conv35W = np.array(L1['filters'], dtype=np.float32).transpose((3,2,0,1))[:,:,::-1,::-1]

    L1 = _todict(mat_contents['layers'][11])
    conv4b = np.array(L1['biases'], dtype=np.float32)
    conv4W = np.array(L1['filters'], dtype=np.float32).transpose((3,2,0,1))[:,:,::-1,::-1]

    L1 = _todict(mat_contents['layers'][13])
    dense1b = np.array(L1['biases'], dtype=np.float32)
    dense1W = np.array(L1['filters'],dtype=np.float32)
    dense1W = np.array(L1['filters'], dtype=np.float32).transpose((1,0,2,3))
    dense1W = dense1W.reshape(4*13*512, 4096, order="F").copy() 
    
    L1 = _todict(mat_contents['layers'][15])
    dense2b = np.array(L1['biases'], dtype=np.float32)
    dense2W = np.array(L1['filters'],dtype=np.float32)

    L1 = _todict(mat_contents['layers'][17])
    classb = L1['biases']
    classW = L1['filters']

    np.savez_compressed('matlab_dictnet_weights',conv1W=conv1W,conv1b=conv1b,conv2W=conv2W,conv2b=conv2b,
                        conv3W=conv3W,conv3b=conv3b,conv35W=conv35W,conv35b=conv35b,conv4W=conv4W,
                        conv4b=conv4b,dense1W=dense1W,dense1b=dense1b,dense2W=dense2W,dense2b=dense2b,
                        classW=classW,classb=classb)

