import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

SUPERPARAMSDESCKEY = "superParams";

layerTypeMap = {"Conv2d": "Convolution",
                "LSTM": "TorchLstm",
                "Linear": "Linear",
                "Input": "Input",
                "BatchNorm2d": "TorchBnFixedParam",
                "ReLU": "ReLU",
                "ReLU6": "ReLUX",
                "LeakyReLU":"ReLU",
                "Scale": "Scale",
                "AvgPool2d": "Pooling",
                "MaxPool2d": "Pooling",
                "L2Norm": "L2Norm",
                "Eltwise":"Eltwise",
                "Upsample":"Bilinearupsampling",
                "Concat":"Concat",
                "Power":"Power"}


class LayerDesc:
    def __init__(self, layer, consumers):
        self.layer = layer
        self.type = ""
        self.name = ""
        self.bottoms = []
        self.top = None
        self.superParam = ""
        self.params = []

    def dump(self):
        txt = "%s %s %d %d " % (self.type, self.name, len(self.bottoms), 1)     
        for iter in self.bottoms:
            txt = txt + "%s " % (iter)
        txt = txt + "%s" % (self.top)
        txt = txt + " %s" % (self.superParam)
        return txt


class FakeLayer:
    def __init__(self, type, superparams, params):
        self.type = type
        self.superParams = superparams
        self.params = params
        self.bidirectional = False


def createReLU(slope):
    return FakeLayer("ReLU", "%d" % (slope), [])


def createInput():
    return FakeLayer("Input", "0 0 0", [])


def createScale(weight, bias):
    return FakeLayer("Scale", "%d %d" % (weight.data.shape[0], 1 if bias.data.shape[0] > 0 else 0), [weight, bias])

def createPower(pow, scale, shift):
    return FakeLayer("Power", "%.4f %.4f %.4f"%(pow, scale, shift), [])

def createEltWise(op):
    return FakeLayer("Eltwise", "%d 0" % (op), [])

def createConcat():
    return FakeLayer("Concat", "", [])

class NcnnNet:
    def __init__(self):
        self.layerRelative = OrderedDict()

    def dumpBnSuperParam(self, bn):
        num_features = bn.num_features
        return "%d %d" % (num_features, 0)

    def dumpConvolution2DSuperParam(self, conv):
        kernelSize = conv.kernel_size[0]
        stride = conv.stride[0]
        padding = conv.padding[0]
        group = conv.groups
        num_output = conv.out_channels
        num_input = conv.in_channels;
        weightDataSize = (num_output / group) * (num_input) * (kernelSize * kernelSize)
        if (not conv.bias is None) and conv.bias.data.shape[0] > 0:
            hasBias = 1
        else:
            hasBias = 0
        return "%d %d %d %d %d %d %d" % (num_output, kernelSize, stride, padding, hasBias, group, weightDataSize)

    def dumpConvolutionDilatedSuperParam(self, conv):
        kernelSize = conv.kernel_size[0]
        stride = conv.stride[0]
        padding = conv.padding[0]
        group = conv.groups
        num_output = conv.out_channels
        num_input = conv.in_channels;
        weightDataSize = weightDataSize = (num_output / group) * (num_input) * (kernelSize * kernelSize)
        dilated = conv.dilation[0]
        if (not conv.bias is None) and conv.bias.data.shape[0] > 0:
            hasBias = 1
        else:
            hasBias = 0
        return "%d %d %d %d %d %d %d %d" % (
        num_output, kernelSize, stride, padding, hasBias, group, dilated, weightDataSize)

    def dumpLinearSuperParam(self, line):
        return "%d %d" % (line.in_features, line.out_features)

    def dumpReLUSuperParam(self, relu):
        if isinstance(relu, nn.LeakyReLU):
            return "%.4f"%(relu.negative_slope)
        return "0.0"
    

    def dumpPoolSuperParam(self, pool):
        if str(type(pool).__name__) == "AvgPool2d":
            operationType = 1
        elif str(type(pool).__name__) == "MaxPool2d":
            operationType = 0
        stride = pool.stride
        padding = pool.padding
        kernnel_size = pool.kernel_size
        return "%d %d %d %d 0" % (operationType, kernnel_size, stride, padding)

    def dumpL2NormSuperParam(self, l2norm):
        return "%d" % (l2norm.n_channels)

    def dumpRelUXSuperParam(self, reluX):
        return "%.2f" % (reluX.max_val)
    
    def dumpBilinearUpsampleParam(self, up):
        return "%d"%(up.scale_factor)

    def dumpLayerSuperParam(self, layerDesc):
        layerType = layerDesc.type
        if layerType == "Convolution":
            return self.dumpConvolution2DSuperParam(layerDesc.layer)
        elif layerType == "ConvolutionDilated":
            return self.dumpConvolutionDilatedSuperParam(layerDesc.layer)
        elif layerType == "Linear":
            return self.dumpLinearSuperParam(layerDesc.layer)
        elif layerType == "TorchBnFixedParam":
            return self.dumpBnSuperParam(layerDesc.layer)
        elif layerType == "ReLU":
            return self.dumpReLUSuperParam(layerDesc.layer)
        elif layerType == "ReLUX":
            return self.dumpRelUXSuperParam(layerDesc.layer)
        elif layerType == "Pooling":
            return self.dumpPoolSuperParam(layerDesc.layer)
        elif layerType == "L2Norm":
            return self.dumpL2NormSuperParam(layerDesc.layer)
        elif layerType == "Bilinearupsampling":
            return self.dumpBilinearUpsampleParam(layerDesc.layer)
        else:
            return layerDesc.layer.superParams

    def dumpLayerParam(self, layerDesc):
        if layerDesc.type == "TorchLstm":
            layerDesc.params = layerDesc.layer.params
        elif layerDesc.type == "Linear":
            layerDesc.params.append(layerDesc.layer.weight)
            layerDesc.params.append(layerDesc.layer.bias)
        elif layerDesc.type == "Convolution" or layerDesc.type == "ConvolutionDilated":
            layerDesc.params.append(layerDesc.layer.weight)
            if ((not layerDesc.layer.bias is None) and layerDesc.layer.bias.data.shape[0] > 0):
                layerDesc.params.append(layerDesc.layer.bias)
        elif layerDesc.type == "TorchBnFixedParam":
            layerDesc.params.append(layerDesc.layer.running_mean)
            layerDesc.params.append(layerDesc.layer.running_var)
        elif layerDesc.type == "Scale":
            layerDesc.params = layerDesc.layer.params
        elif layerDesc.type == "L2Norm":
            layerDesc.params.append(layerDesc.layer.weight)

    def getLastKey(self, orderedDict):
        lastkey = ""
        for k in orderedDict.keys():
            lastkey = k
        return lastkey

    def addModuleList(self, moduleList):
        input = createInput()
        self.addLayer(input, [])

        length = len(moduleList._modules)
        for i in range(0, length):
            if str(type(moduleList[i]).__name__) == "Sequential":
                self.addSequential(moduleList[i])
            else:
                if len(self.layerRelative) > 0:
                    lastkey = self.getLastKey(self.layerRelative)
                    self.layerRelative[lastkey] = [moduleList[i]]
                self.addLayer(moduleList[i], [])

    def addSequential(self, sequential, consumers = [], notConnect = False):
        length = len(sequential)
        
        if notConnect == False:
            if len(self.layerRelative) > 0:
                lastkey = self.getLastKey(self.layerRelative)
                print(lastkey)
                self.layerRelative[lastkey] += [sequential[0]]

        
        for i in range(0, length):
            if i != length - 1:
                self.addLayer(sequential[i], [sequential[i + 1]])
            else:
                self.addLayer(sequential[i], consumers)

    def addBatchNormLayer(self, bn, consumers):
        if bn.affine:
            fakeScale = createScale(bn.weight, bn.bias)
            self.layerRelative[bn] = [fakeScale]
            self.layerRelative[fakeScale] = consumers
        else:
            self.layerRelative[bn] = consumers;

    def add(self, lstm, consumers):
        layerCnt = lstm.num_layers
        lstms = []
        isBidirection = lstm.bidirectional
        if isBidirection is True:
            direct = 2
        else:
            direct = 1
        for i in range(0, layerCnt):
            if i == 0:
                tmp = FakeLayer("LSTM", "%d %d %d" % (lstm.hidden_size, lstm.input_size, direct), [])
                for key in self.layerRelative:
                    for i in range(len(self.layerRelative[key])):
                        if self.layerRelative[key][i] == lstm:
                            self.layerRelative[key][i] = tmp
            else:
                tmp = FakeLayer("LSTM", "%d %d %d" % (lstm.hidden_size, lstm.hidden_size * 2, direct), [])

            tmp.bidirectional = isBidirection
            if isBidirection is False:
                for paramList in lstm.all_weights[i]:
                    tmp.params.append(paramList.data)
            else:
                for paramList in lstm.all_weights[i * 2]:
                    tmp.params.append(paramList.data)
                for paramList in lstm.all_weights[i * 2 + 1]:
                    tmp.params.append(paramList.data)
            lstms.append(tmp)

        for i in range(0, layerCnt):
            if (i < layerCnt - 1):
                self.addLayer(lstms[i], [lstms[i + 1]])
            elif (i == layerCnt - 1):
                self.addLayer(lstms[i], consumers)


    def addLayer(self, layer, consumers):
        if str(type(layer).__name__) == "LSTM":
            self.addLstmLayers(layer, consumers)
        elif str(type(layer).__name__) == "BatchNorm2d":
            self.addBatchNormLayer(layer, consumers)
        else:
            self.layerRelative[layer] = consumers;
        pass

    def writeToFile(self, txt, file):
        file.write(txt)

    def writeLayerParam2File(self, layerDesc, file):
        if layerDesc.type == "TorchLstm":
            paramLists = layerDesc.params
            if paramLists == None:
                return

            isBidirection = layerDesc.layer.bidirectional
            if isBidirection is True:
                direct = 2
            else:
                direct = 1

            for t in range(0, direct):
                for i in range(0 + t * 4, 2 + t * 4):
                    paramList = paramLists[i]
                    for g in paramList:
                        self.writeToFile('%d\n' % (len(g)), file)
                        for v in g:
                            self.writeToFile("%.16f " % (v), file)
                        self.writeToFile('\n', file)

                for i in range(2 + t * 4, 4 + t * 4):
                    paramList = paramLists[i]
                    self.writeToFile('%d\n' % (len(paramList)), file)
                    for v in paramList:
                        self.writeToFile('%.16f ' % (v), file)
                    self.writeToFile('\n', file)
        elif layerDesc.type == "Linear":
            self.writeToFile('%d\n' % (layerDesc.layer.in_features * layerDesc.layer.out_features), file)
            for i in layerDesc.params[0]:
                for j in i:
                    self.writeToFile("%.16f " % (j), file)
            self.writeToFile('\n', file)

            self.writeToFile('%d\n' % (len(layerDesc.params[1])), file)
            for i in layerDesc.params[1]:
                self.writeToFile('%.16f ' % (i), file)
            self.writeToFile('\n', file)
        elif layerDesc.type == "Convolution" or layerDesc.type == "ConvolutionDilated":
            weight = layerDesc.params[0].data
            weightDataSize = weight.shape[0] * weight.shape[1] * weight.shape[2] * weight.shape[3];
            self.writeToFile("%d\n" % weightDataSize, file);
            for oc in weight.numpy():
                for ic in oc:
                    for row in ic:
                        for v in row:
                            self.writeToFile("%.16f " % (v), file)
            self.writeToFile("\n", file)

            if (len(layerDesc.params) > 1):
                biases = layerDesc.params[1].data
                self.writeToFile("%d\n" % (layerDesc.params[1].shape[0]), file)
                for v in biases:
                    self.writeToFile("%.16f " % (v), file)
                self.writeToFile("\n", file)
        elif layerDesc.type == "TorchBnFixedParam":
            self.writeToFile("%d\n" % (layerDesc.params[0].shape[0]), file)
            for v in layerDesc.params[0].numpy():
                self.writeToFile("%.16f " % (v), file);
            self.writeToFile("\n", file)

            self.writeToFile("%d\n" % (layerDesc.params[1].shape[0]), file)
            for v in layerDesc.params[1].numpy():
                self.writeToFile("%.16f " % (v), file)
            self.writeToFile("\n", file)
        elif layerDesc.type == "Scale":
            self.writeToFile("%d\n" % (layerDesc.params[0].data.shape[0]), file)
            for v in layerDesc.params[0].data.numpy():
                self.writeToFile("%.16f " % (v), file);
            self.writeToFile("\n", file)

            self.writeToFile("%d\n" % (layerDesc.params[1].data.shape[0]), file)
            for v in layerDesc.params[1].data.numpy():
                self.writeToFile("%.16f " % (v), file)
            self.writeToFile("\n", file)
        elif layerDesc.type == "L2Norm":
            self.writeToFile("%d\n" % layerDesc.layer.n_channels, file)
            for v in layerDesc.layer.weight.data:
                self.writeToFile("%.16f " % (v), file)
            self.writeToFile("\n", file)

    def getLayerTypeName(self, layer):
        result = ""
        if not isinstance(layer, FakeLayer):
            result = layerTypeMap[str(type(layer).__name__)]
            if (str(type(layer).__name__) == "Conv2d" and layer.dilation[0] > 1):
                result = "ConvolutionDilated"
        else:
            result = layerTypeMap[layer.type]
        return result

    def dumpLayerToFile(self, paramFilePath, binFilePath):
        paramFile = open(paramFilePath, "w")
        binFile = open(binFilePath, "w")
        if paramFile == None or binFile == None:
            print("illegal file Path!!!!!!")
            return

        layerIdx = 1
        blobRefCnt = {}
        layerDescriptions = []
        for layer in self.layerRelative:
            layerDesc = LayerDesc(layer, self.layerRelative[layer]);
            layerDesc.type = self.getLayerTypeName(layer)
            layerDesc.name = "%s_%d" % (layerDesc.type, layerIdx)
            layerDesc.top = layerDesc.name
            for otherLayer in self.layerRelative:
                if otherLayer != layer:
                    if otherLayer in self.layerRelative[layer]:
                        if blobRefCnt.get(layerDesc.top) == None:
                            blobRefCnt[layerDesc.top] = 1
                        else:
                            blobRefCnt[layerDesc.top] += 1
            layerDesc.name = "%s_%d" % (layerDesc.type, layerIdx)
            layerDesc.top = layerDesc.name
            layerDesc.superParam = self.dumpLayerSuperParam(layerDesc)
            self.dumpLayerParam(layerDesc)

            layerDescriptions.append(layerDesc)
            layerIdx += 1

        for layerDesc in layerDescriptions:
            layer = layerDesc.layer
            for otherLayerDesc in layerDescriptions:
                if otherLayerDesc != layerDesc and (layer in self.layerRelative[otherLayerDesc.layer]):
                    layerDesc.bottoms.append(otherLayerDesc.top)

        for key in blobRefCnt.keys():
            refCnt = blobRefCnt[key];
            if refCnt > 1:
                idx = 1
                for layerDesc in layerDescriptions:
                    bottoms = layerDesc.bottoms
                    for i in range(0, len(bottoms)):
                        if bottoms[i] == key:
                            bottoms[i] = key + "_split_%d" % (idx)
                            idx = idx + 1

        totalLayerCnt = 0
        totalBlobCnt = 0
        paramFileContents = []
        for layerDesc in layerDescriptions:
            paramFileContents.append(layerDesc.dump())
            totalLayerCnt += 1
            totalBlobCnt += 1
            if blobRefCnt.get(layerDesc.top) != None and blobRefCnt[layerDesc.top] > 1:
                totalLayerCnt += 1
                content = "Split split_%s 1 %d %s" % (layerDesc.top, blobRefCnt[layerDesc.top], layerDesc.top)
                idx = 1
                for i in range(blobRefCnt[layerDesc.top]):
                    totalBlobCnt += 1
                    t = " %s_split_%d" % (layerDesc.top, idx)
                    content += t
                    idx += 1
                paramFileContents.append(content)

        self.writeToFile("%d %d\n" % (totalLayerCnt, totalBlobCnt), paramFile)
        for content in paramFileContents:
            self.writeToFile(content + "\n", paramFile)

        for layerDesc in layerDescriptions:
            self.writeLayerParam2File(layerDesc, binFile)


if __name__ == "__main__":
    print("hello")
