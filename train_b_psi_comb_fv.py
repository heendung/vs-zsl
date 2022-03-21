import torch
import argparse
import os
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import DataLoader
import logging


class Discrimination(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args):    
        super(Discrimination, self).__init__()
        
        if args.type == 'pre_trained':
            layers=[
                    weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
                    nn.RReLU(),
                    weight_norm(nn.Linear(hid_dim, hid_dim), dim=None),
                    nn.RReLU(),
                    weight_norm(nn.Linear(hid_dim, hid_dim), dim=None),
                    nn.RReLU(),
                    weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
                    nn.RReLU()
            ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # (N, S, D) -> (N,S,1)
        mlp_out = self.main(x)
        # softmax: (N,S,1) -> (N,S,1)
        mlp_soft_out = F.softmax(mlp_out, dim=1)
        # attended output: (N,S,1) -> (N,S,D)
        att_out = mlp_soft_out.mul(x)
        # semantic representions: (N,S,D) -> (N,D)
        sem_rep = torch.sum(att_out, dim=1)
        sem_rep = F.normalize(sem_rep, dim=1)
        
        return sem_rep
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, avg_features, mode):
        'Initialization'
        self.feat = features
        self.avg_feat = avg_features
        self.train = mode
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.feat)
    def __getitem__(self, index):
        if self.train:
            return self.feat[index], self.avg_feat[index]
        else: 
            return self.feat[index]

def parse_args():
    parser = argparse.ArgumentParser()
   
    # Arguments we added
    parser.add_argument('--discri', default='sim_after', help='discriminate classes') 
    parser.add_argument('--num_hid', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--num_cls', type=int, default=15833, help='num_classes')
    parser.add_argument('--output', type=str, default='fine_tune_b_psi')
    parser.add_argument('--tau', type=float, default=0.96, help='tau')
    parser.add_argument('--train', type=str2bool, default=False, help='train')
    parser.add_argument('--sem_rep_after', type=str, default='/semantic_rep_after_train/', help='directory_of_semantic_representations_after_train')
    parser.add_argument('--class_name', type=str, default='class_name_having_wiki_pages_w2v.pt', help='directory_of_semantic_representations_after_train')
    parser.add_argument('--split', type=str, default='all_data', help='2hop/3hop/all_data')
    parser.add_argument('--preprocess', type=str2bool, default=False, help='preprocess sentences')
    parser.add_argument('--type', type=str, default='pre_trained', help='pretrain or finetuned') 
    parser.add_argument('--eps', type=float, default=0.95, help='eps')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--beta', type=float, default=0.1, help='threshold between similarator and discriminator')
    parser.add_argument('--sent_rep', type=str, default='pre_trained_all_data_sent_embed_fv.pt', help='sentence representations')
    parser.add_argument('--avg_rep', type=str, default='vis_sec_clu_avg_sem_pre_trained_fv/bert_sem.pt', help='averaged semantic representations')

    args = parser.parse_args()


    return args

def get_logger(args):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    if not os.path.exists(args.output + '_eps_' + str(args.eps) + '_tau_'  + str(args.tau) + '_' + args.type + '_fv'):
        os.makedirs(args.output + '_eps_' + str(args.eps) + '_tau_'  + str(args.tau) + '_' + args.type + '_fv')
 
    fileHandler = logging.FileHandler(args.output + '_eps_' + str(args.eps) + '_tau_'  + str(args.tau) + '_' + args.type + '_fv' +  '/log.txt')
    logger.addHandler(fileHandler)
    return logger

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    global logger
    args=parse_args()
    logger = get_logger(args)
    
    if args.discri =='discriminate':
        
        sent_c = torch.load(args.sent_rep).cuda()

        if args.type == 'pre_trained':                                                   
            sem = torch.load(args.avg_rep)
            avg_sent_c = sem              
        
        d_set = CustomDataset(sent_c, avg_sent_c, args.train)
        tr_set = DataLoader(d_set, args.batch_size, shuffle=True, num_workers=0)
        ev_set =  DataLoader(d_set, args.batch_size, shuffle=False, num_workers=0)

        if args.train:
            print("Start training...")
            model = Discrimination(sent_c.size()[2], args.num_hid, 1, args).cuda()
            train(model, tr_set, args)
        else:
            print("Start evaluating...")
            model = Discrimination(sent_c.size()[2], args.num_hid, 1, args).cuda()
            checkpoint = torch.load(args.output + '_eps_' + str(args.eps) + '_tau_' + str(args.tau) + '_'  + args.type + '_fv/' + args.split + '_epoch_'  + str(args.epochs) +'.pth')
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}'".format(args.output + '_eps_' + str(args.eps) + '_tau_' + str(args.tau) + '_'  + args.type + '_fv/' + args.split + '_epoch_'  + str(args.epochs) +'.pth'))
            eval(model, ev_set, args)

def train(model, loader_set, args):
    num_epochs = args.epochs
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = args.output
    total_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss=0

        for i, (x, avg_x) in tqdm(enumerate(loader_set), ncols=100, desc="Epoch %d" % (epoch + 1), total=len(loader_set)):
            total_step += 1
            optim.zero_grad()
            sem_rep = model(x.cuda())
            #sem_rep = F.normalize(sem_rep, dim=1)

            #Similarator
            avg_x = F.normalize(avg_x, dim=1)
            bs = avg_x.size()[0]
            n_hid = avg_x.size()[1]

            dot_prod = torch.bmm(avg_x.view(bs, 1, n_hid), sem_rep.view(bs, n_hid,1))
            dot_prod = dot_prod.squeeze(1).squeeze(1)
            zeros = torch.zeros_like(dot_prod).cuda()
            loss_similar = torch.max(zeros, args.eps - dot_prod)
            loss_similar = loss_similar.sum()
            #loss_similar = loss_similar.mean()

            # Discriminator
            # calculate Similarity
            # copy tensor
            sem_rep_d = sem_rep.clone().detach()
            sem_rep_d = sem_rep_d.transpose(1,0)
            sem_rep = sem_rep.unsqueeze(0)
            sem_rep_d = sem_rep_d.unsqueeze(0)
            # sim_mat: (N, N)
            sim_mat = sem_rep.bmm(sem_rep_d).squeeze(0)
            # assign a negative value to the diagonal of sem_rep matrix
            sim_mat[torch.eye(sim_mat.size()[0]).bool()] = -1

            margin = torch.ones(sim_mat.size()[0],1) * args.tau
            margin = margin.cuda()
            zeros = torch.zeros_like(margin).cuda()
            loss_discri= torch.max(zeros, sim_mat-margin)
            loss_discri = loss_discri.sum()
            #loss_discri = loss_discri.mean()

            loss = loss_similar + args.beta * loss_discri

            loss.backward()
            optim.step()
            total_loss += loss.item()
            
        logger.info('epoch %d: ' % epoch)
        logger.info('\ttrain_loss: %.5f' % (total_loss))
        filename = output + '_eps_' + str(args.eps) + '_tau_' + str(args.tau) + '_' + args.type + '_fv/' + args.split +'_epoch_' + str(epoch+1) + '.pth'
        print('Saving checkpoint to: ' + filename)
        torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optim.state_dict()}, filename)


def eval(model, loader_set, args):
    model.eval()

    with torch.no_grad():
        for i, x in tqdm(enumerate(loader_set)):
            sem_rep = model(x.cuda())
            sem_rep = sem_rep.data.cpu()
            fname = args.output +  '_eps_' + str(args.eps) + '_tau_' + str(args.tau) + '_'   + args.type + '_fv/' + args.split + '_semantic_rep_after_train_epochs_' + str(args.epochs)

            if i==0:
                sem_tot = sem_rep
            else:
                sem_tot = torch.cat((sem_tot,sem_rep))

        torch.save(sem_tot, fname + '.pt')


# call the main
if __name__ == "__main__":
    main()
