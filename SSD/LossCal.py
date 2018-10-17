import tensorflow as tf

def lossMultiLayers(MultiLayers):
    predictions = []
    logits = []
    for i,lays in enumerate(MultiLayers):
        p,l = lossSingleLay(lays)
        predictions.append(p)
        logits.append(l)
    return predictions, logits

def lossSingleLay(Inputs):
    p = []
    l = []
    return p,l


