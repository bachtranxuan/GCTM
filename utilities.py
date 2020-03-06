import torch
import numpy as np
import torch.utils.data.dataloader
import scipy.sparse as sp
from scipy.special import digamma
def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return (digamma(alpha)-digamma(sum(alpha)))
    return (digamma(alpha)-digamma(np.sum(alpha, axis=1))[:, np.newaxis])

class Save_Model:
    def __init__(self, folder, num_top_words, vocab):
        self.model_folder = folder
        self.num_top_words = num_top_words
        self.vocab =vocab

    def get_top_word(self, beta):
        num_tops = beta.shape[0]
        list_tops = list()
        for k in range(num_tops):
            top = list()
            arr = np.array(beta[k, :], copy=True)
            for t in range(self.num_top_words):
                index = arr.argmax()
                top.append(index)
                arr[index] = -1.
            list_tops.append(top)
        return (list_tops)

    def write_top_word(self, beta):
        tops_words = self.get_top_word(beta)
        with open('%s/tops_words.txt'%self.model_folder, 'w') as fp:
            for top_k in tops_words:
                for id in top_k:
                    fp.write('%s ' % (self.vocab[id]))
                fp.write('\n')

    def write_beta(self, beta):
        K, V = beta.shape
        with open('%s/beta_final.dat' % (self.model_folder), 'w') as fp:
            for k in range(K):
                for v in range(V):
                    fp.write('%f ' % (beta[k, v]))
                fp.write('\n')

    def save_model(self, per, beta, minibatch):
        with open('%s/perplexities.csv' % (self.model_folder), 'a') as fp:
            fp.write('%f, ' % (per))
        with open('%s/tops%d_%d_%d.dat' % (self.model_folder, self.num_top_words,1, minibatch), 'w') as fp:
            tops_words = self.get_top_word(beta)
            for top_k in tops_words:
                for id in top_k:
                    fp.write('%d ' % (id))
                fp.write('\n')

class Load_Data:
    def __init__(self, fp_train,  file_vocab, batch_size,  folder_test, num_tests):
        self.fp_train = fp_train
        self.folder_test = folder_test
        self.num_tests = num_tests
        self.file_vocab = file_vocab
        self.batch_size = batch_size

    def read_vocab(self):
        vocab = {}
        with open(self.file_vocab, 'r') as fp:
            index = 0
            while True:
                temp = fp.readline().strip()
                if len(temp)<1:
                    break
                vocab[index] = temp
                index+=1
            self.V = index
        return vocab

    def load_data_gcn(self, path, graph):
        """Load citation network dataset (cora only for now)"""

        features = sp.csr_matrix(np.identity(self.V), dtype=np.float32)
        # build graph
        idx_map = {j: i for i, j in enumerate(range(self.V))}
        edges_unordered = np.genfromtxt("{}/{}.txt".format(path, graph),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),
                            shape=(self.V, self.V),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = self.normalize(features)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))

        features = torch.FloatTensor(np.array(features.todense()))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        return adj, features

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def read_vocab(self):
        vocab = {}
        with open(self.file_vocab, 'r') as fp:
            index = 0
            while True:
                temp = fp.readline().strip()
                if len(temp)<1:
                    break
                vocab[index] = temp
                index+=1
            self.V = index
        return vocab

    def read_data_test(self):
        cnts1 = []
        cnts2 = []
        for i in range(self.num_tests):
            with open('%s/data_test_%d_part_1.txt'%(self.folder_test,i+1),'r') as fp:
                while True:
                    ct = np.zeros(self.V, dtype=np.int8)
                    temp = fp.readline().strip()
                    if len(temp)<1:
                        break
                    temp = temp.split(" ")
                    for term in range(int(temp[0])):
                        x,y = temp[term+1].split(':')
                        ct[int(x)] = int(y)
                    cnts1.append(ct)
            with open('%s/data_test_%d_part_2.txt'%(self.folder_test,i+1),'r') as fp:
                while True:
                    ct = np.zeros(self.V, dtype=np.int8)
                    temp = fp.readline().strip()
                    if len(temp)<1:
                        break
                    temp = temp.split(" ")
                    for term in range(int(temp[0])):
                        x,y = temp[term+1].split(':')
                        ct[int(x)] = int(y)
                    cnts2.append(ct)
        tensor_cnts1 = torch.stack([torch.tensor(x) for x in cnts1])
        tensor_cnts2 = torch.stack([torch.tensor(x) for x in cnts2])
        return (tensor_cnts1.to_sparse(), tensor_cnts2.to_sparse())

    def read_data_train_frequent(self):
        d = 0
        cnts = []
        flagF = 0
        cnts_part1 = []
        cnts_part2 = []
        while True:
            if d == self.batch_size:
                break
            temp = self.fp_train.readline().strip()
            if len(temp) < 1:
                flagF = 1
                break
            ct = np.zeros(self.V, dtype=np.int8)
            temp = temp.split(" ")
            doc_id=[]
            doc_cnt=[]
            for i in range(int(temp[0])):
                x,y = temp[i+1].split(':')
                ct[int(x)] = int(y)
                doc_id.append(int(x))
                doc_cnt.append(int(y))
            l = len(doc_id)
            ct1 = np.zeros(self.V, dtype=np.int8)
            ct2 = np.zeros(self.V, dtype=np.int8)
            if (l < 5):
                ct1[np.split(doc_id, [-1])[0]] = np.split(doc_cnt, [-1])[0]
                ct2[np.split(doc_id, [-1])[1]] = np.split(doc_cnt, [-1])[1]
            else:
                pivot = int(np.floor(l * 4.0 / 5))
                ct1[np.split(doc_id, [pivot])[0]] = np.split(doc_cnt, [pivot])[0]
                ct2[np.split(doc_id, [pivot])[1]] = np.split(doc_cnt, [pivot])[1]
            cnts_part1.append(ct1)
            cnts_part2.append(ct2)
            cnts.append(ct)
            d+=1
        tensor_cnts1 = torch.stack([torch.tensor(x) for x in cnts_part1])
        tensor_cnts2 = torch.stack([torch.tensor(x) for x in cnts_part2])
        tensor_cnts = torch.stack([torch.tensor(x) for x in cnts])
        return (flagF,  tensor_cnts.to_sparse(), tensor_cnts1.to_sparse(), tensor_cnts2.to_sparse())

    def read_data_train_month(self):
        d = 0
        cnts = []
        flagF = 0
        check = ""
        while True:
            temp = self.fp_train.readline().strip()
            if len(temp) < 1:
                flagF = 1
                break
            ct = np.zeros(self.V, dtype=np.int8)
            temp = temp.split(" ")
            month = temp[0][4:6]
            if d==0:
                check=month
            if check==month:
                pass
            else:
                break
            for i in range(len(temp)-2):
                x,y = temp[i+2].split(':')
                ct[int(x)] = int(y)
            cnts.append(ct)
            d+=1
        print("Num doc per month %s: %d"%(check, d))
        tensor_cnts = torch.stack([torch.tensor(x) for x in cnts])
        return (flagF,  tensor_cnts.to_sparse())

    def read_data_test_month(self, month):
        cnts1 = []
        cnts2 = []
        d=0
        with open('%s/part1_%d.txt'%(self.folder_test,month),'r') as fp:
            while True:
                ct = np.zeros(self.V, dtype=np.int8)
                temp = fp.readline().strip()
                if len(temp)<1:
                    break
                temp = temp.split(" ")
                d+=1
                for term in range(int(temp[0])):
                    x,y = temp[term+1].split(':')
                    ct[int(x)] = int(y)
                cnts1.append(ct)
        with open('%s/part2_%d.txt'%(self.folder_test,month),'r') as fp:
            while True:
                ct = np.zeros(self.V, dtype=np.int8)
                temp = fp.readline().strip()
                if len(temp)<1:
                    break
                temp = temp.split(" ")
                for term in range(int(temp[0])):
                    x,y = temp[term+1].split(':')
                    ct[int(x)] = int(y)
                cnts2.append(ct)
        tensor_cnts1 = torch.stack([torch.tensor(x) for x in cnts1])
        tensor_cnts2 = torch.stack([torch.tensor(x) for x in cnts2])
        return (tensor_cnts1.to_sparse(), tensor_cnts2.to_sparse(), d)

class Evaluate():
    def __init__(self, beta, num_topics, alpha, batchsize, iterate):
        self.K = num_topics
        self.alpha = alpha
        self.batchsize = batchsize
        self.beta = beta.clone().detach().numpy()
        self.iterate = iterate

    def computer_lpp(self, part1, part2):
        gamma = self.inference(part1)
        return self.LPP(gamma, part2)

    def inference(self, cnts):
        gamma = torch.ones(self.batchsize,self.K)*self.alpha + torch._sparse_sum(cnts,dim=1, dtype=torch.float32).to_dense().view(-1,1)/self.K
        gamma = gamma.numpy()
        ExpElogtheta = np.exp(dirichlet_expectation(gamma))
        betat = self.beta.transpose()
        for d in range(self.batchsize):
            cnt = cnts.narrow_copy(dim=0,start=d, length=1).to_dense()
            ids = torch.nonzero(cnt,as_tuple=True)[1].numpy()
            count = cnt.numpy()[0][ids]
            gammad = gamma[d]
            ExpElogthetad = ExpElogtheta[d]
            betatd = betat[ids,:]
            for i in range(self.iterate):
                phi = ExpElogthetad*betatd +1e-10
                phi /=np.sum(phi,axis=1, keepdims=True)
                gammad = self.alpha + np.dot(count,phi)
                ExpElogthetad = np.exp(dirichlet_expectation(gammad))
            gamma[d] = gammad/sum(gammad)
        return gamma

    def LPP(self,  gamma, part2):
        lpp = 0.0
        for d in range(self.batchsize):
            cnt = part2.narrow_copy(dim=0,start=d, length=1).to_dense()
            id = torch.nonzero(cnt,as_tuple=True)[1].numpy()
            count = cnt.numpy()[0][id]
            gammad = gamma[d]
            p = np.dot(gammad,self.beta[:,id])
            if np.sum(count)<1:
                continue
            lpp+=np.dot(count,np.log(p))/np.sum(count)
        return lpp/self.batchsize



