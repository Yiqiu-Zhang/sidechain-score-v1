import torch
import esm
import numpy as np
from PDB_processing import get_torsion_seq

#mapping_acid = {"A":0,"C":1,"D":2,"E":3,"F":4,"G":5,"H":6,"I":7,"K":8,"L":9,"M":10,"N":11,"P":12,"Q":13,"R":14,"S":15,"T":16,"V":17,"W":18,"Y":19 }

#def acid_to_number(seq, mapping):
#    num_list = []
#    for acid in seq:
#        if acid in mapping:
#            num_list.append(mapping[acid])
#    return num_list
'''
def acid_embedding(dict_struct):
    
    structures = dict_struct
    
    
    angle_list = structures['angles']
    X = structures['coords']
    pdb_path = structures['fname']
    seq_single = structures['seq']
    
    num_acid_seq = acid_to_number(seq_single, mapping_acid)
    num_acid_seq = np.array(num_acid_seq)
    
    
    seq1 = "".join(seq_single)
    model, alphabet = esm.pretrained.load_model_and_alphabet('../premodel/esm2_8M/esm2_t6_8M_UR50D.pt')
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    data_1 = [("protein1",seq1)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data_1)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
 #   t =  results["representations"][33]
   # t1 = t.shape
    acid_representations = results["representations"][6][0, 1:len(seq1)+1]
    acid_representations = acid_representations.numpy()
   
    structures_d = {'angles':angle_list,'coords': X, 'seq': num_acid_seq, 'acid_embedding': acid_representations, 'fname':pdb_path}
    
    return structures_d
def acd_no_embedding(dict_struct):
    structures = dict_struct
    
    angle_list = structures['angles']
    X = structures['coords']
    pdb_path = structures['fname']
    seq_single = structures['seq']
    
    num_acid_seq = acid_to_number(seq_single, mapping_acid)
    num_acid_seq = np.array(num_acid_seq)
    
   
    structures_d = {'angles':angle_list,'coords': X, 'seq': num_acid_seq, 'seq_temp': seq_single, 'fname':pdb_path}
    if angle_list.shape[0]!= X.shape[0]:
        print("======================pdb_path=====================",pdb_path)
        print("======================angle_list.shape[0]=====================",angle_list.shape[0])
        print("======================X.shape[0]=====================",X.shape[0])
    
    return structures_d


def dict_structure_pdb(pdb_path):
    dict_struct = get_torsion_seq(pdb_path) #
   # structures = acid_embedding(dict_struct)
    structures = acd_no_embedding(dict_struct)
    return structures
'''
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
    
    for batch in batches:
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
             results = model(batch_tokens, repr_layers=[6], return_contacts=True)
        for  k,(_,seq1) in enumerate(batch):
            acid_representations = results["representations"][6][0, 1:len(seq1)+1]
            acid_representations = acid_representations.numpy()
            sequence_acid_embedding.append(acid_representations)
            print("=======================num seq=================",k)
    j = 0
    print("=======================sequence_acid_embedding len=================",len(sequence_acid_embedding))
    print("=======================sequence_acid_embedding[0] shape=================",sequence_acid_embedding[0].shape)
    for structure in structures: 
        structure.update({'acid_embedding':sequence_acid_embedding[j]})
        j = j + 1
        del structure['seq_temp']
    print("===========structures len===============",len(structures))
    print("===========structures key===============",structures[0].keys())
    print("===========structures len===============",structures[0]['seq'].shape)
    print("===========structures len===============",structures[0]['acid_embedding'].shape)
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