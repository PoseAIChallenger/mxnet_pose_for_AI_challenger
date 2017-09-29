import sys
sys.path.append('/data/guest_users/liangdong/liangdong/practice_demo')
from modelCPMWeight import *
from config.config import config

class AIChallengerIterweightBatch:
    def __init__(self, datajson,
                 data_names, data_shapes, label_names,
                 label_shapes, batch_size = 1):

        self._data_shapes = data_shapes
        self._label_shapes = label_shapes
        self._provide_data = zip([data_names], [data_shapes])
        self._provide_label = zip(label_names, label_shapes) * 6
        self._batch_size = batch_size

        with open(datajson, 'r') as f:
            data = json.load(f)

        self.num_batches = len(data)

        self.data = data
        
        self.cur_batch = 0

        self.keys = data.keys()

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            
            transposeImage_batch = []
            heatmap_batch = []
            pagmap_batch = []
            heatweight_batch = []
            vecweight_batch = []
            
            for i in range(batch_size):
                if self.cur_batch >= 45174:
                    break
                image, mask, heatmap, pagmap = getImageandLabel(self.data[self.keys[self.cur_batch]])
                maskscale = mask[0:368:8, 0:368:8, 0]
                heatweight = np.ones((numofparts, 46, 46))
                vecweight = np.ones((numoflinks*2, 46, 46))

                for i in range(numofparts):
                    heatweight[i,:,:] = maskscale

                for i in range(numoflinks*2):
                    vecweight[i,:,:] = maskscale
                
                transposeImage = np.transpose(np.float32(image), (2,0,1))/256 - 0.5
            
                self.cur_batch += 1
                
                transposeImage_batch.append(transposeImage)
                heatmap_batch.append(heatmap)
                pagmap_batch.append(pagmap)
                heatweight_batch.append(heatweight)
                vecweight_batch.append(vecweight)
                
            return DataBatchweight(mx.nd.array(transposeImage_batch),
                                   mx.nd.array(heatmap_batch),
                                   mx.nd.array(pagmap_batch),
                                   mx.nd.array(heatweight_batch),
                                   mx.nd.array(vecweight_batch))
        else:
            raise StopIteration

start_prefix = 0
class poseModule(mx.mod.Module):

    def fit(self, train_data, num_epoch, batch_size, carg_params=None, begin_epoch=0):
        
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=[('data', (batch_size, 3, 368, 368))], label_shapes=[
        ('heatmaplabel', (batch_size, numofparts, 46, 46)),
        ('partaffinityglabel', (batch_size, numoflinks*2, 46, 46)),
        ('heatweight', (batch_size, numofparts, 46, 46)),
        ('vecweight', (batch_size, numoflinks*2, 46, 46))])
   
        
        # self.init_params(mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=1))
        # mx.initializer.Uniform(scale=0.07),
        # mx.initializer.Uniform(scale=0.01)
        # mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=0.01)
        self.init_params(arg_params = carg_params, aux_params={}, allow_missing = True)
        #self.set_params(arg_params = carg_params, aux_params={},
        #                allow_missing = True)
        self.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.00004), ))
        losserror_list = []

        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            i=0
            sumerror=0
            while not end_of_batch:
                data_batch = next_data_batch
                cmodel.forward(data_batch, is_train=True)       # compute predictions  
                prediction=cmodel.get_outputs()
                i=i+1
                sumloss=0
                numpixel=0
                print 'iteration: ', i
                
                '''
                print 'length of prediction:', len(prediction)
                for j in range(len(prediction)):
                    
                    lossiter = prediction[j].asnumpy()
                    cls_loss = np.sum(lossiter)
                    print 'loss: ', cls_loss
                    sumloss += cls_loss
                    numpixel +=lossiter.shape[0]
                    
                '''
                
         
                
                lossiter = prediction[1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print 'start heat: ', cls_loss
                    
                lossiter = prediction[0].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print 'start paf: ', cls_loss
                
                lossiter = prediction[-1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print 'end heat: ', cls_loss
                
                lossiter = prediction[-2].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print 'end paf: ', cls_loss   
               
         
                cmodel.backward()   
                self.update()           
                
                if i > 10:
                    break
                    
                try:
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True
                nbatch += 1
            
                    
            print '------Error-------'
            print sumerror/i
            losserror_list.append(sumerror/i)
            
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)
            #self.save_checkpoint(config.TRAIN.output_model, epoch)
            
            train_data.reset()
        print losserror_list
        text_file = open("OutputLossError.txt", "w")
        text_file.write(' '.join([str(i) for i in losserror_list]))
        text_file.close()
        
sym = ''
if config.TRAIN.head == 'vgg':
    sym = CPMModel() 

## Load parameters from vgg
warmupModel = '/data/guest_users/liangdong/liangdong/practice_demo/mxnet_CPM/model/vgg19'
testsym, arg_params, aux_params = mx.model.load_checkpoint(warmupModel, 0)
newargs = {}
for ikey in config.TRAIN.vggparams:
    newargs[ikey] = arg_params[ikey]

batch_size = 10
aidata = AIChallengerIterweightBatch('pose_io/AI_data_val.json', # 'pose_io/COCO_data.json',
                          'data', (batch_size, 3, 368, 368),
                          ['heatmaplabel','partaffinityglabel','heatweight','vecweight'],
                          [(batch_size, numofparts, 46, 46),
                           (batch_size, numoflinks*2, 46, 46),
                           (batch_size, numofparts, 46, 46),
                           (batch_size, numoflinks*2, 46, 46)])

cmodel = poseModule(symbol=sym, context=mx.cpu(),
                    label_names=['heatmaplabel',
                                 'partaffinityglabel',
                                 'heatweight',
                                 'vecweight'])
starttime = time.time()

'''
output_prefix = config.TRAIN.output_model
testsym, newargs, aux_params = mx.model.load_checkpoint(output_prefix, start_prefix)
'''
iteration = 3
cmodel.fit(aidata, num_epoch = iteration, batch_size = batch_size, carg_params = newargs)
cmodel.save_checkpoint(config.TRAIN.output_model, start_prefix + iteration)
endtime = time.time()

print 'cost time: ', (endtime-starttime)/60

