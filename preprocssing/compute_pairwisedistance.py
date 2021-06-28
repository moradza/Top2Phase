from __future__ import print_function, division

import numpy as np
from utils import PeriodicCKDTree, distance_pbc
from tqdm import tqdm
from pymatgen.core.structure import IStructure

import mdtraj as md 
import numpy as np
import os, sys
import argparse

parser = argparse.ArgumentParser('The traj.xtc convertor to pairwise distance npz')
parser.add_argument('--xtc_file', help='path to the xtc file')
parser.add_argument('--gro_file', help='path to gro file')
parser.add_argument('--npz_file', help='path to save pairwise distances as a npz file')
parser.add_argument('--radius', type=float,  default=0.7,  help='radius for neigh list')
parser.add_argument('--skip', type=int, default=10, help='skip frames')


args = parser.parse_args()
print("topology file: ", args.gro_file)
print("processing trajectory: ", args.xtc_file) 
print("with radius: ", args.radius)
print("saving to file: ", args.npz_file)
print("skip steps: ", args.skip)

skip = args.skip
if not os.path.exists(args.xtc_file):
    raise NameError(args.xtc_file + 'file does not exit')
if not os.path.exists(args.gro_file):
    raise NameError(args.gro_file + 'file does not exit')

traj = md.load(args.xtc_file, top=args.gro_file)
hbonds =  md.wernet_nilsson(traj,False,True)[1::skip]
print(traj.xyz.shape)

traj_coords = traj.xyz[1::skip,::4,:]
lattices = traj.unitcell_vectors[1::skip]

atom_types = np.array([8 for i in range(traj_coords.shape[1])]).astype(int)
target_index = np.array([i for i in range(int(traj_coords.shape[1]))]).astype(int)

#np.savez_compressed(args.npz_file, traj_coords=traj_coords, lattices=lattices, atom_types=atom_types, target_index=target_index)

nbr_lists, diag_lattices = [], []
radius = args.radius 
cnt_frame = 0
Edges = {}
for coord, lattice in tqdm(zip(traj_coords, lattices),
                    total=len(traj_coords)):
    # take the diagonal part of the lattice matrix
    lattice = np.diagonal(lattice)
    diag_lattices.append(lattice)
    pkdt = PeriodicCKDTree(lattice, coord)
    all_nbrs_idxes = pkdt.query_ball_point(coord, radius)
    nbr_list = []
    for idx, nbr_idxes in enumerate(all_nbrs_idxes):
        cnt_atom_idx = 4 * idx
        nbr_dists = distance_pbc(coord[idx],
                coord[nbr_idxes],
                lattice)
        cnt_atom_idy = 4 * nbr_idxes
        nbr_idx_dist = sorted(zip(nbr_idxes, nbr_dists),key=lambda x: x[1]) 
        
        all_idx_dist = []
        all_idx_hbond = np.zeros((len(nbr_idx_dist), len(nbr_idx_dist), 3))
        all_idx_hbond[:,:,-1] = 1.0
        #print(all_idx_hbond.shape, len(nbr_idx_dist))
        assert nbr_idx_dist[0][1] == 0 and\
                nbr_idx_dist[0][0] == idx 

        for idy, _ in nbr_idx_dist:
            nbr_dists = distance_pbc(coord[idy],
                coord[[ i for  i, _ in nbr_idx_dist]],
                lattice)
            all_idx_dist.append(list(zip([i for  i, _ in nbr_idx_dist], nbr_dists)))
        #print("#####################")
        hbond = hbonds[cnt_frame][:,::2]#.tolist()
        #print(hbond)
        cnt1 = 0
        nbr_idx = [id1 for id1, _ in nbr_idx_dist]
        #print("####################################")
        for id1, _ in nbr_idx_dist:
            for id2 in np.where(hbond[:,0] == 4*id1)[0]:
                mol2 = int(hbond[id2,1]/4)
                for cnt2 in np.where(np.array(nbr_idx) ==mol2)[0]:
                    all_idx_hbond[cnt1, cnt2,0] = 1.
                    all_idx_hbond[cnt1, cnt2,2] = 0.
                    all_idx_hbond[cnt2, cnt1,1] = 1.
                    all_idx_hbond[cnt2, cnt1,2] = 0.
            #for id2 in np.where(hbond[:,1] == 4*id1)[0]:
            #    mol2 = int(hbond[id2,0]/4)
            #    for cnt2 in np.where(np.array(nbr_idx) ==mol2)[0]:
            #        all_idx_hbond[cnt2, cnt1, 0] = 1.
            #        all_idx_hbond[cnt2, cnt1, 2] = 0.
            cnt1 += 1
        '''
        cnt1 = 0
        for id1, _ in nbr_idx_dist:
            cnt2 = 0
            for id2, _ in nbr_idx_dist:
                #print(id1, id2)
                if [4*id1, 4*id2] in  hbond:
                    #print(np.where(hbonds[cnt_frame][:,0] == 4*id1 and hbonds[cnt_frame][:,2] == 4*id2))
                    #print("Donor: ", 4*id1, " Acceptor : ", 4*id2)
                    all_idx_hbond[cnt1,cnt2,0] = 1
                    all_idx_hbond[cnt1, cnt2,2] = 0
                if [4*id2, 4*id1] in hbond:
                    #print(np.where(hbonds[cnt_frame][:,0] == 4*id2 and hbonds[cnt_frame][:,2] == 4*id1))
                    #print("Donor: ", 4*id2, " Acceptor : ", 4*id1)
                    all_idx_hbond[cnt1, cnt2,1] = 1
                    all_idx_hbond[cnt1, cnt2,2] = 0
                #else:
                #    all_idx_hbond[cnt1, cnt2, 2] = 1
                cnt2 += 1
            cnt1 += 1
        '''
        #print(all_idx_hbond)
        #break
        L = [[[all_idx_dist[i][j][1], all_idx_hbond[i,j,0], all_idx_hbond[i,j,1], all_idx_hbond[i,j,2]] for j in range(len(all_idx_dist))] for i in range(len(all_idx_dist)) ]
        L = np.array(L)
        for i in range(L.shape[0]):
            L[i,i,:] =0.0
        #print(np.array(L).shape,cnt_atom_idx, \
        #         np.where(hbonds[cnt_frame][:,0] == cnt_atom_idx)[0],\
        #         np.where(hbonds[cnt_frame][:,0] == cnt_atom_idx)[0],\
        #         np.where(hbonds[cnt_frame][:,0] == cnt_atom_idy))#hbonds[cnt_frame])
        #print(np.array(L).shape, all_idx_hbond.shape)
        np.save('all_idx_hbond.npz', all_idx_hbond)
        Edges["atom_"+str(cnt_frame)+"_"+str(idx)] = L
        Edges["xyz_atom_"+str(cnt_frame)+"_"+str(idx)] = coord[idx] 
    cnt_frame += 1
Edges["n_frames"] = cnt_frame
Edges["n_atoms"] = coord.shape[0]
Edges["radius"] = radius

np.savez_compressed(args.npz_file, **Edges)
print("Successfully saved file: ", args.npz_file)
'''
   return {'traj_coords': traj_coords,
                    'lattices': diag_lattices,
                    'atom_types': atom_types,
                    'target_index': target_index,
                    'nbr_lists': nbr_lists}
'''
#diag_lattices.append(lattices[0])
#pkdt = PeriodicCKDTree(np.diag(lattices[0]), traj_coords[0])
#all_nbrs_idxes = pkdt.query_ball_point(traj_coords[0], 0.35)
#nbr_list = []
#print(len(all_nbrs_idxes))
