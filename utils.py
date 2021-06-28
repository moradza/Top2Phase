import numpy as np
import os
from spektral.data import Dataset, Graph
import timeit
import requests
from zipfile import ZipFile
from tqdm import tqdm
from spektral.transforms.normalize_adj import NormalizeAdj


################################################################################
# LOAD DATA
################################################################################
class Top2PhaseDataset(Dataset):
    """
    dataset loader for graph neural network training
    """
    def __init__(self, n_samples, list_of_graphs, permute=True, rseed=12345, download_bool=False,data_path=None,max_neighs=28, **kwargs):
        '''
        initilize loader 
        args :
        
        '''
        self.n_samples = n_samples
        self.list_of_graphs = list_of_graphs
        self.rseed = rseed
        self.data_path = data_path
        self.permute = permute
        self.max_neighs = max_neighs
        if not download_bool:
           Dataset.path = os.path.join(os.getcwd(),'data')
        super().__init__(**kwargs)
    def download(self):
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        url = 'https://www.dropbox.com/sh/c64upw5d068fdf2/AABOKFMketvU4DuYv-tMU2tva?dl=0'
        r = requests.get(url, stream=True, headers=headers)
        filepath = 'data_graph.zip'
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk:
                    f.write(chunk)
        # specifying the zip file name
        file_name = "data_graph.zip"
        # opening the zip file in READ mode
        with ZipFile(file_name, 'r') as zipper:
            # printing all the contents of the zip file
            zipper.printdir()
          
            # extracting all the files
            print('Extracting all the files now...')
            zipper.extractall('data')
            print('Done!')
        self.data_path = os.path.join(os.getcwd(),'data')
        Dataset.path = self.data_path

    def read(self):
        #self.download()
        np.random.seed(self.rseed)
        list_of_graphs = self.list_of_graphs
        graphs = [ ]
        print(list_of_graphs)
        number_of_graphs = sum([len(list_of_graphs[i]) for  i in range(len(list_of_graphs))])
        print(number_of_graphs)
        num_per_phase = int(self.n_samples/ number_of_graphs)
        for phase in range(len(list_of_graphs)):
            for phase_graph in range(len(list_of_graphs[phase])):
                # Check if graph file exists in the data path
                if Dataset.path is None:
                    dataset = np.load(list_of_graphs[phase][phase_graph])
                else:
                    dataset = np.load(os.path.join(Dataset.path, list_of_graphs[phase][phase_graph]))
                #print(dataset.files[0])
                cnt_sample = 0
                data = []
                names = []
                frame_i = 1
                def read_atom(name=None, dataset=dataset):
                    #print("name : " ,name)
                    data = dataset[name]
                    if self.permute:
                        y_ = dataset['xyz_'+name]
                    else:
                        if len(list_of_graphs) >2 :
                            y_ = np.array([1 if k == phase  else 0  for k in range(len(list_of_graphs))])
                        else:
                            y_ = [phase]
                    #if self.permute:
                    if not self.max_neighs is None :
                        if data.shape[0] >= self.max_neighs:
                            e = data[:self.max_neighs,:self.max_neighs,:].reshape((self.max_neighs, self.max_neighs,4))
                        else:
                            e = data.reshape((data.shape[0],data.shape[1],4))
                    else:
                        e = data.reshape((data.shape[0],data.shape[1],4))
                    #e = e.reshape((e.shape[0],e.shape[1],1))
                     
                    # Node features
                    x = np.zeros((e.shape[0],2))
                    x[0,0] = 1.0
                    x[1:,1] = 1.0
                    # Edges
                    a = np.ones((x.shape[0], x.shape[0])) - np.eye(x.shape[0])
                    a = a.astype(int)
                  
                    return x, a, e, y_
                a_i = 0
                print("phase ; ", phase, " name of datset: ", list_of_graphs[phase][phase_graph])
                for cnt_sampel in tqdm(range(num_per_phase), position=0, leave=True):
                  name = 'atom_'+str(frame_i)+'_'+str(a_i)
                  #print("Name : " ,name)
                  x, a, e, y = read_atom(name=name)
                  graphs.append(Graph(x=x,a=a,e=e,y=y))
                  a_i += 1    
                  if a_i == dataset['n_atoms']:
                    frame_i += 1
                    a_i = 0
        num_l =  np.random.permutation(len(graphs))
        return [graphs[i] for i in num_l]
