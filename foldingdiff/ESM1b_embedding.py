import torch
import esm
import numpy as np
def add_esm1b_embedding(structures_no_embedding,batch_size):
    
    structures = structures_no_embedding
    sequence_list = []
    sequence_acid_embedding = []
    i = 0
    for structure in structures:
        temp =  "".join(structure['seq_temp'])
        sequence_list.append(("protein"+str(i),temp))
        i = i+1
    model, alphabet = esm.pretrained.load_model_and_alphabet('../premodel/esm2_8M/esm2_t6_8M_UR50D.pt')
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    num_batches = int(np.ceil(len(sequence_list) / batch_size))
    
    batches = [sequence_list[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    
    for k, batch in enumerate(batches):
        _, _, batch_tokens = batch_converter(batch)
        with torch.no_grad():
             results = model(batch_tokens, repr_layers=[6], return_contacts=True)
        for  k,(_,seq1) in enumerate(batch):
            acid_representations = results["representations"][6][0, 1:len(seq1)+1]
            acid_representations = acid_representations.numpy()
            sequence_acid_embedding.append(acid_representations)

    j = 0
  
    for structure in structures: 
        structure.update({'acid_embedding':sequence_acid_embedding[j]})
        j = j + 1
        del structure['seq_temp']

    return structures
        
        
    #batch_labels, batch_strs, batch_tokens = batch_converter(data_1)
    #batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    # Extract per-residue representations (on CPU)
    #with torch.no_grad():
    #$    results = model(batch_tokens, repr_layers=[6], return_contacts=True)
 #   t =  results["representations"][33]
   # t1 = t.shape
    ##acid_representations = results["representations"][6][0, 1:len(seq1)+1]
    #acid_representations = acid_representations.numpy()