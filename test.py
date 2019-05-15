import numpy as np
import codecs
import operator
import json
from transE import data_loader,entity2id,relation2id

def dataloader(entity_file,relation_file,test_file):
    # entity_file: entity \t embedding
    entity_dict = {}
    relation_dict = {}
    test_triple = []

    with codecs.open(entity_file) as e_f:
        lines = e_f.readlines()
        for line in lines:
            entity,embedding = line.strip().split('\t')
            embedding = json.loads(embedding)
            entity_dict[entity] = embedding

    with codecs.open(relation_file) as r_f:
        lines = r_f.readlines()
        for line in lines:
            relation,embedding = line.strip().split('\t')
            embedding = json.loads(embedding)
            relation_dict[relation] = embedding

    with codecs.open(test_file) as t_f:
        lines = t_f.readlines()
        for line in lines:
            triple = line.strip().split('\t')
            if len(triple) != 3:
                continue
            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]

            test_triple.append(tuple((h_,t_,r_)))

    return entity_dict,relation_dict,test_triple

def distance(h,r,t):
    h = np.array(h)
    r=np.array(r)
    t = np.array(t)
    s=h+r-t
    return np.linalg.norm(s)

class Test:
    def __init__(self,entity_dict,relation_dict,test_triple,train_triple,isFit = True):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.train_triple = train_triple
        self.isFit = isFit

        self.hits10 = 0
        self.mean_rank = 0

        self.relation_hits10 = 0
        self.relation_mean_rank = 0

    def rank(self):
        hits = 0
        rank_sum = 0
        step = 1

        for triple in self.test_triple:
            rank_head_dict = {}
            rank_tail_dict = {}

            for entity in self.entity_dict.keys():
                corrupted_head = [entity,triple[1],triple[2]]
                if self.isFit:
                    if corrupted_head not in self.train_triple:
                        h_emb = self.entity_dict[corrupted_head[0]]
                        r_emb = self.relation_dict[corrupted_head[2]]
                        t_emb = self.entity_dict[corrupted_head[1]]
                        rank_head_dict[tuple(corrupted_head)]=distance(h_emb,r_emb,t_emb)
                else:
                    h_emb = self.entity_dict[corrupted_head[0]]
                    r_emb = self.relation_dict[corrupted_head[2]]
                    t_emb = self.entity_dict[corrupted_head[1]]
                    rank_head_dict[tuple(corrupted_head)] = distance(h_emb, r_emb, t_emb)

                corrupted_tail = [triple[0],entity,triple[2]]
                if self.isFit:
                    if corrupted_tail not in self.train_triple:
                        h_emb = self.entity_dict[corrupted_tail[0]]
                        r_emb = self.relation_dict[corrupted_tail[2]]
                        t_emb = self.entity_dict[corrupted_tail[1]]
                        rank_tail_dict[tuple(corrupted_tail)] = distance(h_emb, r_emb, t_emb)
                else:
                    h_emb = self.entity_dict[corrupted_tail[0]]
                    r_emb = self.relation_dict[corrupted_tail[2]]
                    t_emb = self.entity_dict[corrupted_tail[1]]
                    rank_tail_dict[tuple(corrupted_tail)] = distance(h_emb, r_emb, t_emb)

            rank_head_sorted = sorted(rank_head_dict.items(),key = operator.itemgetter(1))
            rank_tail_sorted = sorted(rank_tail_dict.items(),key = operator.itemgetter(1))

            #rank_sum and hits
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i<10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            for i in range(len(rank_tail_sorted)):
                if triple[1] == rank_tail_sorted[i][0][1]:
                    if i<10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            step += 1
            if step % 5000 == 0:
                print("step ", step, " ,hits ",hits," ,rank_sum ",rank_sum)
                print()

        self.hits10 = hits / (2*len(self.test_triple))
        self.mean_rank = rank_sum / (2*len(self.test_triple))

    def relation_rank(self):
        hits = 0
        rank_sum = 0
        step = 1

        for triple in self.test_triple:
            rank_dict = {}
            for r in self.relation_dict.keys():
                corrupted_relation = (triple[0],triple[1],r)
                if self.isFit and corrupted_relation in self.train_triple:
                    continue
                h_emb = self.entity_dict[corrupted_relation[0]]
                r_emb = self.relation_dict[corrupted_relation[2]]
                t_emb = self.entity_dict[corrupted_relation[1]]
                rank_dict[r]=distance(h_emb, r_emb, t_emb)

            rank_sorted = sorted(rank_dict.items(),key = operator.itemgetter(1))

            rank = 1
            for i in rank_sorted:
                if triple[2] == i[0]:
                    break
                rank += 1
            if rank<10:
                hits += 1
            rank_sum = rank_sum + rank + 1

            step += 1
            if step % 5000 == 0:
                print("relation step ", step, " ,hits ", hits, " ,rank_sum ", rank_sum)
                print()

        self.relation_hits10 = hits / len(self.test_triple)
        self.relation_mean_rank = rank_sum / len(self.test_triple)

if __name__ == '__main__':
    _, _, train_triple = data_loader("FB15k\\")

    entity_dict, relation_dict, test_triple = \
        dataloader("entity_50dim_batch400","relation50dim_batch400",
                   "FB15k\\test.txt")


    test = Test(entity_dict,relation_dict,test_triple,train_triple,isFit=False)
    test.rank()
    print("entity hits@10: ", test.hits10)
    print("entity meanrank: ", test.mean_rank)

    test.relation_rank()
    print("relation hits@10: ", test.relation_hits10)
    print("relation meanrank: ", test.relation_mean_rank)

    f = open("result.txt",'w')
    f.write("entity hits@10: "+ str(test.hits10) + '\n')
    f.write("entity meanrank: " + str(test.mean_rank) + '\n')
    f.write("relation hits@10: " + str(test.relation_hits10) + '\n')
    f.write("relation meanrank: " + str(test.relation_mean_rank) + '\n')
    f.close()

