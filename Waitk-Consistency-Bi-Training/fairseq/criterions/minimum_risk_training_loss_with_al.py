from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.minimum_risk_training_loss import Minimum_Risk_Training_Loss
import torch

@register_criterion('minimum_risk_training_loss_with_al')
class Minimum_Risk_Training_Loss_With_Al(Minimum_Risk_Training_Loss):
    def __init__(self, 
                 task,
                 mrt_beam_size=5,
                 mrt_sampling=False,
                 mrt_sampling_topk=-1,
                 mrt_sampling_topp=-1.0,
                 mrt_seq_max_len_a=0,
                 mrt_seq_max_len_b=200,
                 mrt_length_penalty=0.25,
                 mrt_temperature = 1.0,
                 mrt_greedy = False,
                 mrt_alpha = 0.05,
    ):
        super().__init__(
            task = task,
            mrt_beam_size=mrt_beam_size,
            mrt_sampling=mrt_sampling,
            mrt_sampling_topk=mrt_sampling_topk,
            mrt_sampling_topp=mrt_sampling_topp,
            mrt_seq_max_len_a=mrt_seq_max_len_a,
            mrt_seq_max_len_b=mrt_seq_max_len_b,
            mrt_length_penalty=mrt_length_penalty,
            mrt_temperature=mrt_temperature,
            mrt_greedy = mrt_greedy,
        )
        self.mrt_alpha = mrt_alpha
    @staticmethod
    def add_args(parser):
        super(Minimum_Risk_Training_Loss_With_Al, Minimum_Risk_Training_Loss_With_Al).add_args(parser)
        parser.add_argument('--mrt-alpha', default=0.05, type=float, metavar='N',
                       help='alpha for generation')
    def prepare_sample_and_hypotheses(self, sample=None, hypos=None,model=None):
        #sample['includes_bleu'] = True
        scale_scores = lambda x : x
        hypos = self.add_bleu_to_hypotheses(sample, hypos)
        
        source_lengths = sample['net_input']['src_lengths'].tolist()
        cost = torch.FloatTensor([
            scale_scores([(
                (1.0-self.mrt_alpha)*(100 - h['bleu'])+
                self.mrt_alpha*get_al(ctxs = h['context'], src_len=source_lengths[i])
            ) for h in hypos_i])
            for (i,hypos_i) in enumerate(hypos)
        ])
        
        sample['cost'] = cost
        return sample, hypos

def get_al(ctxs, src_len):
    hyp_len = len(ctxs)
    gamma = hyp_len / src_len
    als = []
    tg = []
    for t, c in enumerate(ctxs):
        if c < src_len:
            als.append(c - t / gamma)
            tg.append(t/gamma)
        else:
            als.append(c - t / gamma)
            tg.append(t/gamma)
            break
    return sum(als)/float(len(als))