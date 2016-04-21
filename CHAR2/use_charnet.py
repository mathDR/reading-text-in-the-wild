'''
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python use_charnet.py
'''

from __future__ import print_function
from numpy import argmax, array, dot, float32, mean, std, zeros

class CharNet():
    def __init__(self, architecture_file=None, weight_file=None, optimizer=None):
        # Generate mapping for softmax layer to characters
        output_str = '0123456789abcdefghijklmnopqrstuvwxyz '
        self.output = [x for x in output_str]
        self.L = len(self.output)

        # Load model and saved weights
        from keras.models import model_from_json
        if architecture_file is None:
            self.model = model_from_json(open('char2_architecture.json').read())
        else:
            self.model = model_from_json(open(architecture_file).read())

        if weight_file is None:
            self.model.load_weights('char2_weights.h5')
        else:
            self.model.load_weights(weight_file)

        if optimizer is None:
            from keras.optimizers import SGD
            optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def _rgb2gray(self,rgb):
        return dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def _preprocess(self,img):
        from skimage.transform import resize
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = self._rgb2gray(img)
        img = resize(img, (32,100), order=1, preserve_range=True)
        img = array(img,dtype=float32) # convert to single precision
        img = (img - mean(img)) / ( (std(img) + 0.0001) / 128.0 )
        return img

    def classify_image(self,img):    
        img = self._preprocess(img)
        xtest = zeros((1,1,32,100))
        xtest[0,0,:,:] = img
        z = self.model.predict(xtest,verbose=1)[0,:]
        output_text = []
        for i in range(23):
            ind = argmax(z[i*self.L:(i+1)*self.L])
            output_text.extend(self.output[ind])
        return ''.join(output_text).rstrip(' ')


if __name__ == '__main__':
    import matplotlib.image as mpimg

    filename = 'Chevron.jpg'
    #filename = 'CondoleezzaRice.jpg'
    #filename = 'CMA_CGM.jpg'
    cnn_model = CharNet()

    img = mpimg.imread(filename)        
    print (cnn_model.classify_image(img))
