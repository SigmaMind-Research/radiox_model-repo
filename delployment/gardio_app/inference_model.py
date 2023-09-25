
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CNN, MVCNN, TNN, Classifier, Generator, ClsGen, ClsGenInt


class RadioXModel(nn.Module):
    MODEL_NAME = 'ClsGenInt' # ClsGen / ClsGenInt 
    if MODEL_NAME in ['ClsGen', 'ClsGenInt']:
        SOURCES = ['image','caption','label','history']
        TARGETS = ['caption','label']
        KW_SRC = ['image','caption','label','history']
        KW_TGT = None
        KW_OUT = None
            
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2
        
        VOCAB_SIZE = 1000
        POSIT_SIZE = 1000

    # --- Choose a Backbone ---
        
    BACKBONE_NAME = 'DenseNet121'
    backbone = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=True)
    FC_FEATURES = 1024

    # --- Choose a Model --
        
    MODEL_NAME == 'ClsGenInt'
    NUM_EMBEDS = 256
    FWD_DIM = 256
    DROPOUT = 0.1
    NUM_HEADS = 8
    NUM_LAYERS = 1
    
    cnn = CNN(backbone, BACKBONE_NAME)
    cnn = MVCNN(cnn)
    tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
    
    # Not enough memory to run 8 heads and 12 layers, instead 1 head is enough
    NUM_HEADS = 1
    NUM_LAYERS = 12
    
    cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
    gen_model = Generator(num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS)
    
    clsgen_model = ClsGen(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
    clsgen_model = nn.DataParallel(clsgen_model)
    
    # Initialize the Interpreter module
    NUM_HEADS = 8
    NUM_LAYERS = 1
    
    tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
    int_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=None, tnn=tnn, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
    int_model = nn.DataParallel(int_model)
        
    model = ClsGenInt(clsgen_model.module.cpu(), int_model.module.cpu(), freeze_evaluator=True)
    model = nn.DataParallel(model)
