from __future__ import print_function, division

import numpy as np
from Top2Phase.utils import PeriodicCKDTree, distance_pbc
from tqdm import tqdm
from pymatgen.core.structure import IStructure

import mdtraj as md 
import numpy as np
import os, sys
import argparse
def main():
    parser = argparse.ArgumentParser('The traj.xtc convertor to pairwise distnace npz')
    parser.add_argument('--xtc_file', help='path to the xtc file')
    parser.add_argument('--gro_file', help='path to gro file')
    parser.add_argument('--npz_file', help='path to save pairwise distance as a npz file')
    parser.add_argument('--radius', type=float,  default=0.5,  help='radius for neigh list')
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
            nbr_dists = distance_pbc(coord[idx],
                    coord[nbr_idxes],
                    lattice)
            nbr_idx_dist = sorted(zip(nbr_idxes, nbr_dists),key=lambda x: x[1]) 
            
            all_idx_dist = []
            assert nbr_idx_dist[0][1] == 0 and\
                    nbr_idx_dist[0][0] == idx 

            for idy, _ in nbr_idx_dist:
                nbr_dists = distance_pbc(coord[idy],
                    coord[[ i for  i, _ in nbr_idx_dist]],
                    lattice)
                all_idx_dist.append(list(zip([i for  i, _ in nbr_idx_dist], nbr_dists)))
            
            L = [[all_idx_dist[i][j][1] for i in range(len(all_idx_dist))] for j in range(len(all_idx_dist)) ]
            Edges["atom_"+str(cnt_frame)+"_"+str(idx)] = np.array(L)
            Edges["xyz_atom_"+str(cnt_frame)+"_"+str(idx)] = coord[idx] 
        cnt_frame += 1
    Edges["n_frames"] = cnt_frame
    Edges["n_atoms"] = coord.shape[0]
    Edges["radius"] = radius

    np.savez_compressed(args.npz_file, **Edges)
    print("Successfully saved file: ", args.npz_file)

if __name__ == "__main__":
    main()
