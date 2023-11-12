# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


from fairseq import search,data
import torch
import torch.nn.functional as F
import numpy as np
from fairseq.scoring import bleu
#from memory_profiler import profile
@register_criterion('minimum_risk_training_loss')
class Minimum_Risk_Training_Loss(FairseqCriterion):

    def __init__(self, 
                 task,
                 mrt_beam_size=5,
                 mrt_sampling=False,
                 mrt_sampling_topk=-1,
                 mrt_sampling_topp=-1.0,
                 mrt_seq_max_len_a=0,
                 mrt_seq_max_len_b=200,
                 mrt_length_penalty=0.0,
                 mrt_temperature=1.0,
                 mrt_greedy = "false",
                 mrt_waitk = 1,
    ):
        super().__init__(task)
        #beam generator
        self.mrt_temperature = mrt_temperature
        self.mrt_beam_size = mrt_beam_size
        self.mrt_sampling = mrt_sampling
        self.mrt_sampling_topk = mrt_sampling_topk
        self.mrt_sampling_topp = mrt_sampling_topp
        self._generator = None
        self._scorer = None

        #greedy generator
        self.mrt_greedy = mrt_greedy == "true"
        self._greedy_generator = None


        #generator
        self.mrt_seq_max_len_a = mrt_seq_max_len_a
        self.mrt_seq_max_len_b = mrt_seq_max_len_b
        self.mrt_waitk = mrt_waitk
        
        #add al or pl
        self.mrt_length_penalty = mrt_length_penalty

        self.target_dictionary = task.target_dictionary
        self.pad_idx = self.target_dictionary.pad()
        self.eos_idx = self.target_dictionary.eos()
        self.unk_idx = self.target_dictionary.unk()
        
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
        #                     help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--mrt-beam-size', default=5, type=int,
                            help='minimum_risk_training_loss beam size')
        parser.add_argument('--mrt-sampling', action='store_true',  default=False,
                            help='minimum_risk_training_loss sampling')
        parser.add_argument('--mrt-sampling-topk', default=-1, type=int,
                            help='minimum_risk_training_loss beam size')
        parser.add_argument('--mrt-sampling-topp', default=-1.0, type=float,
                            help='minimum_risk_training_loss beam size')
        parser.add_argument('--mrt-seq-max-len-a', default=0, type=float, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))
        parser.add_argument('--mrt-seq-max-len-b', default=200, type=int, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))
        
        parser.add_argument('--mrt-length-penalty', type=float, default=0.0, metavar='D',
                       help='weight of length penalty on BLEU.')
        
        parser.add_argument('--mrt-temperature', default=1., type=float, metavar='N',
                       help='temperature for generation')
        
        #add greedy search generator
        parser.add_argument('--mrt-greedy', type=str, default='false',
                    help='greedy generator (True or False)')
        
        #add itst_train_threshold
        parser.add_argument('--mrt-waitk', type=float, default=1.0,)

        


        # fmt: on
    def _generate_hypotheses(self, model, sample):
        # initialize generator
        if self._generator is None:
            if self.mrt_sampling:
                search_strategy = search.Sampling(
                    self.task.target_dictionary, self.mrt_sampling_topk, self.mrt_sampling_topp
                )
            else:
                search_strategy = search.BeamSearch(self.task.target_dictionary)
            from fairseq.sequence_generator import SequenceGenerator
            self._generator = SequenceGenerator(
                [model],
                self.task.target_dictionary,
                beam_size=self.mrt_beam_size,
                search_strategy=search_strategy,
                max_len_a=self.mrt_seq_max_len_a,
                max_len_b=self.mrt_seq_max_len_b,
                min_len=getattr(self.task.args, "min_len", 1),
                normalize_scores=(not getattr(self.task.args, "unnormalized", False)),
                len_penalty=getattr(self.task.args, "lenpen", 1),
                unk_penalty=getattr(self.task.args, "unkpen", 0),
                temperature=self.mrt_temperature,
                match_source_len=getattr(self.task.args, "match_source_len", False),
                no_repeat_ngram_size=getattr(self.task.args, "no_repeat_ngram_size", 0),
                lm_model=None,
                lm_weight=0.0,
                
            )
            self._generator.cuda()
        
        if self._greedy_generator is None and self.mrt_greedy:
            from fairseq.sequence_generator import SequenceGenerator
            self._greedy_generator = SequenceGenerator(
                [model],
                self.task.target_dictionary,
                beam_size=1,
                search_strategy=search.BeamSearch(self.task.target_dictionary),
                max_len_a=self.mrt_seq_max_len_a,
                max_len_b=self.mrt_seq_max_len_b,
                min_len=getattr(self.task.args, "min_len", 1),
                normalize_scores=(not getattr(self.task.args, "unnormalized", False)),
                len_penalty=getattr(self.task.args, "lenpen", 1),
                unk_penalty=getattr(self.task.args, "unkpen", 0),
                temperature=1.0,
                match_source_len=getattr(self.task.args, "match_source_len", False),
                no_repeat_ngram_size=getattr(self.task.args, "no_repeat_ngram_size", 0),
                lm_model=None,
                lm_weight=0.0,
                
            )
            self._greedy_generator.cuda()
        # generate hypotheses
        with torch.no_grad():
            hypos, _, _, _, _, =  self._generator.generate(
                models = [model], 
                sample = sample, 
                prefix_tokens=None,
                bos_token=None,
                test_waitk_lagging = self.mrt_waitk
            )
            torch.cuda.empty_cache()
            if self.mrt_greedy:
                greedy_hypos, _, _, _, _, = self._greedy_generator.generate(
                    models = [model], 
                    sample = sample, 
                    prefix_tokens=None,
                    bos_token=None,
                    test_waitk_lagging = self.mrt_waitk
                )
                hypos = [sublist1 + sublist2 for sublist1, sublist2 in zip(greedy_hypos, hypos)]
                torch.cuda.empty_cache()
        return hypos
    def create_sequence_scorer(self):
        self._scorer = BleuScorer(self.pad_idx, self.eos_idx, self.unk_idx)
    
    def add_bleu_to_hypotheses(self, sample, hypos):
        """
        Add BLEU scores to the set of hypotheses.
        This can be called from prepare_sample_and_hypotheses.
        """
        if 'includes_bleu' in sample:
            return hypos
        sample['includes_bleu'] = True

        if self._scorer is None:
            self.create_sequence_scorer()

        target = sample['target'].data.int().cpu()
        target_lengths = torch.count_nonzero(target!= 1, dim=1)
        for i, hypos_i in enumerate(hypos):
            r = target[i, :]
            
            for hypo in hypos_i:
                h = hypo['tokens'].int().cpu()
                hypo['bleu'] = self._scorer.score(r, h)
                #cal lp
                ref_l = target_lengths[i].item()
                hyp_l = len(h)
                lp = np.exp(1 - max(ref_l, hyp_l) / float(min(ref_l, hyp_l)))
                hypo['bleu'] = lp ** self.mrt_length_penalty * hypo['bleu']
        return hypos
    
    def prepare_sample_and_hypotheses(self, sample=None, hypos=None,model=None):
        #sample['includes_bleu'] = True
        scale_scores = lambda x : x
        hypos = self.add_bleu_to_hypotheses(sample, hypos)
        cost = torch.FloatTensor([
            scale_scores([100 - h['bleu'] for h in hypos_i])
            for hypos_i in hypos
        ])
        sample['cost'] = cost
        return sample, hypos
    
    def _update_sample_with_hypos(self, sample, hypos):
        num_hypos_per_batch = len(hypos[0])
        assert all(len(h) == num_hypos_per_batch for h in hypos)

        def repeat_num_hypos_times(t):
            return t.repeat(1, num_hypos_per_batch).view(num_hypos_per_batch*t.size(0), t.size(1))
        def repeat_src_lengths_num_hypos_times(t):
            repeated_lengths = t.repeat(num_hypos_per_batch)
            return repeated_lengths
        input = sample['net_input']
        bsz = input['src_tokens'].size(0)
        input['src_tokens'].data = repeat_num_hypos_times(input['src_tokens'].data)
        input['src_lengths'].data = repeat_src_lengths_num_hypos_times(input['src_lengths'].data)
        input_hypos = [h['tokens'] for hypos_i in hypos for h in hypos_i]
        sample['hypotheses'] = collate_tokens(
            input_hypos, 
            self.pad_idx, 
            self.eos_idx, 
            left_pad=False, 
            move_eos_to_beginning=False
        )
        input['prev_output_tokens'].data = collate_tokens(
            input_hypos, 
            self.pad_idx, 
            self.eos_idx, 
            left_pad=False, 
            move_eos_to_beginning=True
        )
        sample['target'].data = repeat_num_hypos_times(sample['target'].data)
        sample['ntokens'] = sample['target'].data.ne(self.pad_idx).sum()
        sample['bsz'] = bsz
        sample['num_hypos_per_batch'] = num_hypos_per_batch
        return sample
    
    def prepare_main(self,model,sample):
        with torch.no_grad():
            # 1. generate in beam search
            hypos = self._generate_hypotheses(model, sample)
            
            # 2. calculate cost by BLEU
            sample, hypos = self.prepare_sample_and_hypotheses(sample, hypos,model)

            # 3. create a new sample out of the hypotheses
            sample = self._update_sample_with_hypos(sample, hypos)
            del hypos
            return sample
    
    def forward(self, model, sample, update_num = None,reduce=True):
        """Compute the loss for the given sample.
        1. generate in beam search
            1.1 add generator
            1.2 add beam search
        2. prepare data for net_input
            2.1 change hypos to batch_size * beam_size * max_length
            2.2 change source to batch_size * beam_size * max_length
            2.3 change target to batch_size * beam_size * max_length
            2.4 calcuate cost by BLEU add to sample
        3. training teacher forcing decoding with prefix as hypos 
            3.1 model forward get prob
        4. calculate loss 

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample = self.prepare_main(model,sample)
        
        net_output = model(**sample['net_input'],train_waitk_lagging = self.mrt_waitk)
        loss, sample_size, logging_output = self.compute_mrt_loss(model,net_output, sample)
        #del tmp_sample
        return loss, sample_size, logging_output

    def get_hypothesis_scores(self, lprobs, sample, score_pad=False):
        """Return a tensor of model scores for each hypothesis.

        The returned tensor has dimensions [bsz, nhypos, hypolen]. This can be
        called from sequence_forward.
        """
        bsz, nhypos, hypolen ,_= lprobs.size()
        hypotheses = sample['hypotheses'].view(bsz, nhypos, hypolen, 1)
        scores = lprobs.gather(dim=-1, index=hypotheses)
        if not score_pad:
            scores = scores * hypotheses.ne(self.pad_idx).float()
        return scores.squeeze(dim=-1)
    def get_hypothesis_lengths(self, lprobs, sample):
        """Return a tensor of hypothesis lengths.
        The returned tensor has dimensions [bsz, nhypos]. This can be called
        from sequence_forward.
        """
        bsz, nhypos, hypolen, _ = lprobs.size()
        lengths = sample['hypotheses'].view(bsz, nhypos, hypolen).ne(self.pad_idx).sum(2).float()
        return lengths
    def compute_mrt_loss(self, model,net_output, sample):
        bsz = sample['bsz']
        num_hypos_per_batch = sample['num_hypos_per_batch']
        hypolen = sample['hypotheses'].size(1)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(bsz, num_hypos_per_batch, hypolen, -1)
        
        scores = self.get_hypothesis_scores(lprobs, sample)
        lengths = self.get_hypothesis_lengths(lprobs, sample)
        avg_scores = scores.sum(dim=-1) / lengths
        probs = F.softmax(torch.exp(avg_scores),dim=-1)
        loss = (probs * sample['cost'].type_as(probs)).sum()
        sample_size = lprobs.size(0)  # bsz 
        logging_output = {
            'loss': loss.data,
            'sample_size': sample_size,
            'sum_cost': sample['cost'].sum(),
            'num_cost': sample['cost'].numel(),
            "ntokens": sample['ntokens'].data,
            "nsentences": sample['nsentences'],
        }
        return loss, sample_size, logging_output
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sum_cost = sum(log.get('sum_cost', 0) for log in logging_outputs)
        sum_num_cost = sum(log.get('num_cost', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('mrt_loss', loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('avg_cost', sum_cost / sum_num_cost, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
class BleuScorer(object):

    def __init__(self, pad, eos, unk):
        from fairseq.dataclass.configs import FairseqConfig
        cfg = FairseqConfig()
        cfg.pad = pad
        cfg.eos = eos
        cfg.unk = unk
        self._scorer = bleu.Scorer(cfg=cfg)

    def score(self, ref, hypo):
        self._scorer.reset(one_init=True)
        self._scorer.add(ref, hypo)
        return self._scorer.score()
@staticmethod
def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning):
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            if left_pad:
                copy_tensor(v, res[i][size-len(v):])
            else:
                copy_tensor(v, res[i][:len(v)])
        return res