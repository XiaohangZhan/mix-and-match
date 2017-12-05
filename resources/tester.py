import caffe

class Tester(caffe.Net):
    def __init__(self, net, weights, gpu=-1):
        if gpu != -1:
            caffe.set_mode_gpu()
            caffe.set_device(gpu)
        else:
            caffe.set_mode_cpu()
        caffe.Net.__init__(self, net, weights, caffe.TEST)
        
    def predict(self, inputs, params, query):
        """
        Parameters
        inputs: dict: key (blobs name), value (data)
        query: list of queried blobs string

        Returns
        qblobs: (k x N x c x h x w) queried blobs
        """
        for key in inputs.keys():
            self.blobs[key].data[...] = inputs[key]
        for key in params.keys():
            self.params[key][0].data[:,:,0,0] = params[key]
        self.forward()
        qblobs = {}
        for x in query:
            qblobs[x] = self.blobs[x].data
        
        # test
        if False:
            for key in self.blobs.keys():
                h=self.blobs[key].data.shape[2]/2
                w=self.blobs[key].data.shape[3]/2
                print key, self.blobs[key].data.min(), self.blobs[key].data.max(), self.blobs[key].data[0,0,h,w]
        return qblobs
