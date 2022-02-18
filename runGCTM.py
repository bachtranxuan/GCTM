import argparse
import modelGCTM
import time
import shutil
import os
from utilities import *
import torch.optim as optim
torch.set_num_threads(2)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='data', help='Data folder.')
parser.add_argument('--iteration', type=int, default=100,
                    help='Number of loop inference for each mini-batch.')
parser.add_argument('--num_topics', type=int, default=50,
                    help='Number of topics.')
parser.add_argument('--batch_size', type=int, default=500,
                    help='Number of documents in each mini-batch.')
parser.add_argument('--opt', type=str, default='adam',
                    help='Method optimize')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.01,
                    help='Hyperparameter of LDA.')
parser.add_argument('--sigma', type=float, default=1,
                    help='Initial sigma.')
parser.add_argument('--num_tests', type=int, default=1,
                    help='Number of test files.')
parser.add_argument('--top', type=int, default=10,
                    help='Number of top words in each topic.')
parser.add_argument('--hidden', type=int, default=200,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
model_folder = '%s/models/Opt%s-lr%s-K%s-B%s-S%s-D%s'%(args.folder, args.opt, args.lr, args.num_topics, args.batch_size, args.sigma, args.dropout)
if os.path.exists(model_folder):
    shutil.rmtree(model_folder)
os.makedirs(model_folder)
data_loader = Load_Data(fp_train=open('%s/train.txt'%(args.folder),'r'),  file_vocab="%s/vocab.txt"%(args.folder),
                      batch_size=args.batch_size, folder_test=args.folder, num_tests=args.num_tests)
vocab = data_loader.read_vocab()
test_part1,test_part2 = data_loader.read_data_test()
V = len(vocab)
batch_size_test = test_part1.size()[0]
adj, X = data_loader.load_data_gcn(path=args.folder,graph='edgesw')
save_model = Save_Model(model_folder,args.top, vocab)

print('Num topics: %d, Batchsize: %d, Sigma: %s, Iterate: %d\nOpt: %s, Lr: %s, Dropout: %s'%(args.num_topics, args.batch_size,args.sigma,args.iteration, args.opt, args.lr, args.dropout))
gctm = modelGCTM.GCTM(V=V, num_topics=args.num_topics, alpha=args.alpha, sigma=args.sigma, batchsize=args.batch_size, iterate=args.iteration, hidden=args.hidden, nfeat=V, dropout=args.dropout)
if args.opt =="adam":
    print("Adam")
    optimizer = optim.Adam(gctm.parameters(),  lr=args.lr)
elif args.opt == "adadelta":
    print("Adadelta")
    optimizer = optim.Adadelta(gctm.parameters(), lr=args.lr, rho=0.9)
elif args.opt == "adagrad":
    print("adagrad")
    optimizer = optim.Adagrad(gctm.parameters(), lr=args.lr)
else:
    print("SGD")
    optimizer = optim.SGD(gctm.parameters(), lr=args.lr)

for name, param in gctm.named_parameters():
    if param.requires_grad:
        print(name, param.size())
minibatch = 1
weightgc1 = gctm.weightgc1.clone().detach()
weightgc2 = gctm.weightgc2.clone().detach()
betat = gctm.betat.clone().detach()

while True:
    print('Minibatch: %d'%minibatch)
    try:
        (flag, inputs, _1, _2) = data_loader.read_data_train_frequent()
        if flag == 1:
            break
    except:
        break
    start_infer = time.time()
    (logbeta, total_phi) = gctm(inputs, X, adj, weightgc1, weightgc2, betat)
    for _ in range(1):
        optimizer.zero_grad()
        loss = gctm.mELBO(total_phi, logbeta, weightgc1, weightgc2,betat)
        loss.backward(retain_graph = True)
        optimizer.step()
    weightgc1 = gctm.weightgc1.clone().detach()
    weightgc2 = gctm.weightgc2.clone().detach()
    betat = gctm.betat.clone().detach()
    end_infer = time.time()
    with torch.no_grad():
        print('Computing perplexites...')
        start_predict = time.time()
        beta = torch.exp(gctm.updatebeta(X, adj))
        per = Evaluate(beta=beta,num_topics=args.num_topics,  alpha=args.alpha, batchsize=batch_size_test, iterate=args.iteration)
        lp = per.computer_lpp(part1=test_part1, part2=test_part2)
        end_predict = time.time()
        save_model.save_model(lp, beta.clone().detach(), minibatch)
        print('Times infer: %d, Times predict: %d, Perplexities: %f'
              %((end_infer-start_infer), (end_predict-start_predict), lp))
    minibatch += 1
save_model.write_beta(torch.exp(gctm.updatebeta(X, adj)).clone().detach())
save_model.write_top_word(torch.exp(gctm.updatebeta(X, adj)).clone().detach())
