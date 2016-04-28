import numpy as np

if __name__ == '__main__':
    initial_classes = np.load('initial_classes.npy')
    classes_5000 = np.load('classes_5000.npy')
    classes_10000 = np.load('classes_10000.npy')
    classes_15000 = np.load('classes_15000.npy')
    classes_20000 = np.load('classes_20000.npy')
    classes_30000 = np.load('classes_30000.npy')
    classes_40000 = np.load('classes_40000.npy')
    classes_50000 = np.load('classes_50000.npy')
    classes_60000 = np.load('classes_60000.npy')
    classes_70000 = np.load('classes_70000.npy')
    classes_80000 = np.load('classes_80000.npy')
    classes_88172 = np.load('classes_88172.npy')

    mapping = classes_88172 + classes_80000 + classes_70000 + classes_60000 + /
              classes_50000 + classes_40000 + classes_30000) + /
              classes_20000 + classes_15000 + classes_10000 + /
              classes_5000 + initial_classes

    np.save('trained_mapping_lexicon',mapping)
