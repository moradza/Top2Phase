from __future__ import print_function, division
import numpy as np
from Top2Phase.utils import PeriodicCKDTree, distance_pbc
from tqdm import tqdm
from pymatgen.core.structure import IStructure

import mdtraj as md 
import numpy as np
import os, sys
import argparse
import boo

def main():
    parser = argparse.ArgumentParser('The traj.xtc convertor to order parameters npz')
    parser.add_argument('--xtc_file', help='path to the xtc file')
    parser.add_argument('--gro_file', help='path to gro file')
    parser.add_argument('--npz_file', help='path to save order parmeters as a npz file')
    parser.add_argument('--radius', type=float,  default=0.7,  help='radius for neigh list')
    parser.add_argument('--skip', type=int, default=10, help='skip frames')



    args = parser.parse_args()
    print("topology file: ", args.gro_file)
    print("processing trajectory: ", args.xtc_file) 
    print("with radius: ", args.radius)
    print("saving to file: ", args.npz_file)


    if not os.path.exists(args.xtc_file):
        raise NameError(args.xtc_file + 'file does not exit')
    if not os.path.exists(args.gro_file):
        raise NameError(args.gro_file + 'file does not exit')

    traj = md.load(args.xtc_file, top=args.gro_file)
    traj_coords = traj.xyz[:,::4,:]
    lattices = traj.unitcell_vectors
    atom_types = np.array([8 for i in range(traj_coords.shape[1])]).astype(int)
    target_index = np.array([i for i in range(int(traj_coords.shape[1]))]).astype(int)

    nbr_lists, diag_lattices = [], []
    radius = args.radius 
    cnt_frame = 0
    Qall_dict ={}
    for coord, lattice in tqdm(zip(traj_coords, lattices),
                        total=len(traj_coords)):
        # take the diagonal part of the lattice matrix
        lattice = np.diagonal(lattice)
        diag_lattices.append(lattice)
        pkdt = PeriodicCKDTree(lattice, coord)
        all_nbrs_idxes = pkdt.query_ball_point(coord, radius)
        nbr_list = []
        lsis = np.zeros(coord.shape[0])
        qtet = np.zeros(coord.shape[0])
        for idx, nbr_idxes in enumerate(all_nbrs_idxes):
            nbr_dists = distance_pbc(coord[idx],
                    coord[nbr_idxes],
                    lattice)
            nbr_idx_dist = sorted(zip(nbr_idxes, nbr_dists),key=lambda x: x[1]) 
            #print(idx, nbr_idx_dist[1:7])
            list_lsi = []
            for idy in nbr_idx_dist[1:]:
                if idy[1] <0.37:
                    list_lsi.append(idy[1])
            lsis[idx] = np.mean((np.array(list_lsi) -np.mean(list_lsi))**2)
            if len(nbr_idx_dist) > 7:
                for idy in nbr_idx_dist[1:7]:
                    nbr_list.append([idx, idy[0]])
            if len(nbr_idx_dist) > 5:
                for idj in  range(3):
                    for idk in range(idj+1,4):
                        d1 = nbr_idx_dist[idj+1][1]
                        d2 = nbr_idx_dist[idk+1][1]
                        d3 = distance_pbc(coord[nbr_idx_dist[idj+1][0]],
                           coord[[nbr_idx_dist[idk+1][0]]],
                           lattice)
                        qtet[idx] += ((d1**2 + d2**2 - d3**2)/(2 * d1 * d2) + 1/3)**2
                qtet[idx] = 1 - (3/8)*qtet[idx]
        nbr_list = np.array(nbr_list).astype(int)
        q4m = boo.bonds2qlm(coord, nbr_list, l=4, periods=lattice)
        Q4m , _ = boo.coarsegrain_qlm(q4m, nbr_list, np.array([True for i in range(coord.shape[0])]))
        Q4_m = boo.ql(Q4m)
        q6m =  boo.bonds2qlm(coord, nbr_list, l=6, periods=lattice)
        Q6m , _ = boo.coarsegrain_qlm(q6m, nbr_list, np.array([True for i in range(coord.shape[0])]))
        Q6_m = boo.ql(Q6m)
        q8m =  boo.bonds2qlm(coord, nbr_list, l=8, periods=lattice)
        Q8m , _ = boo.coarsegrain_qlm(q8m, nbr_list, np.array([True for i in range(coord.shape[0])]))
        Q8_m = boo.ql(Q8m)
        q10m =  boo.bonds2qlm(coord, nbr_list, l=10, periods=lattice)
        Q10m , _ = boo.coarsegrain_qlm(q10m, nbr_list, np.array([True for i in range(coord.shape[0])]))
        Q10_m = boo.ql(Q10m)
        q12m =  boo.bonds2qlm(coord, nbr_list, l=12, periods=lattice)
        Q12m , _ = boo.coarsegrain_qlm(q12m, nbr_list, np.array([True for i in range(coord.shape[0])]))
        Q12_m = boo.ql(Q12m)

        Qall_dict['frame_'+str(cnt_frame)+'_q4'] = Q4_m
        Qall_dict['frame_'+str(cnt_frame)+'_q6'] = Q6_m
        Qall_dict['frame_'+str(cnt_frame)+'_q8'] = Q8_m
        Qall_dict['frame_'+str(cnt_frame)+'_q10'] = Q10_m
        Qall_dict['frame_'+str(cnt_frame)+'_q12'] = Q12_m
        Qall_dict['frame_'+str(cnt_frame)+'_lsi'] = lsis
        Qall_dict['frame_'+str(cnt_frame)+'_qtet'] = qtet
        cnt_frame += 1

    Qall_dict["n_frames"] = cnt_frame
    Qall_dict["n_atoms"] = coord.shape[0]
    print("Saving file: ", args.npz_file)
    np.savez_compressed(args.npz_file, **Qall_dict)
    print("Successfully saved file: ", args.npz_file)
if __name__ == "__main__":
    main()
