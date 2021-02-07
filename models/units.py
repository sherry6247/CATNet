import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# def load_data(training_file, validation_file, testing_file):
#     train = np.array(pickle.load(open(training_file, 'rb')))
#     validate = np.array(pickle.load(open(validation_file, 'rb')))
#     test = np.array(pickle.load(open(testing_file, 'rb')))
#     return train, validate, test

def cut_data(training_file, validation_file, testing_file):
    train = list(pickle.load(open(training_file, 'rb')))
    validate = list(pickle.load(open(validation_file, 'rb')))
    test = list(pickle.load(open(testing_file, 'rb')))
    for dataset in [train, validate, test]:
        dataset[0] = dataset[0][0: len(dataset[0]) // 18]
        dataset[1] = dataset[1][0: len(dataset[1]) // 18]
        dataset[2] = dataset[2][0: len(dataset[2]) // 18]
    return train, validate, test

def pad_time(seq_time_step, options):
    lengths = np.array([len(seq) for seq in seq_time_step])
    maxlen = np.max(lengths)
    for k in range(len(seq_time_step)):
        while len(seq_time_step[k]) < maxlen:
            seq_time_step[k].append(100000)

    return seq_time_step

def pad_matrix(seq_diagnosis_codes, seq_labels, options):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    n_diagnosis_codes = options['all_input_dim']
    maxlen = np.max(lengths)
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    maxlen = np.max(lengths)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + n_diagnosis_codes
    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)
    batch_mask_code = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask_code[bid, pid, tid] = 1


    for i in range(n_samples):
        batch_mask[i, 0:lengths[i]-1] = 1
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1

    batch_labels = np.array(seq_labels, dtype=np.int64)

    return batch_diagnosis_codes, batch_labels, batch_mask, batch_mask_final, batch_mask_code

def adjust_input(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes):
    batch_time_step = copy.deepcopy(batch_time_step)
    batch_diagnosis_codes = copy.deepcopy(batch_diagnosis_codes)
    for ind in range(len(batch_diagnosis_codes)):
        if len(batch_diagnosis_codes[ind]) > max_len:
            batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
        batch_time_step[ind].append(0)
        batch_diagnosis_codes[ind].append([n_diagnosis_codes-1])
    return batch_diagnosis_codes, batch_time_step


def adjust_input_new(batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, max_len, n_med_codes, n_labtest_codes, n_diag_codes, n_proc_codes, options):
    batch_time_step = copy.deepcopy(batch_time_step)
    batch_med_codes = copy.deepcopy(batch_med_codes)
    batch_labtest_codes = copy.deepcopy(batch_labtest_codes)
    batch_diag_codes = copy.deepcopy(batch_diag_codes)
    batch_proc_codes = copy.deepcopy(batch_proc_codes)
    for ind in range(len(batch_diag_codes)):
        if len(batch_diag_codes[ind]) > max_len:
            batch_diag_codes[ind] = batch_diag_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
        if len(batch_med_codes[ind]) > max_len:
            batch_med_codes[ind] = batch_med_codes[ind][-(max_len):]
        if len(batch_labtest_codes[ind]) > max_len:
            batch_labtest_codes[ind] = batch_labtest_codes[ind][-(max_len):]
        if options['dataset'] == 'mimc_dataset':
            if len(batch_proc_codes[ind]) > max_len:
                batch_proc_codes[ind] = batch_proc_codes[ind][-(max_len):]
            batch_proc_codes[ind].append([n_proc_codes-1])
        batch_time_step[ind].append(0)
        batch_med_codes[ind].append([n_med_codes-1])
        batch_labtest_codes[ind].append([n_labtest_codes-1])
        batch_diag_codes[ind].append([n_diag_codes-1])
        
    return batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step


def get_maxCode(seqs):
    length_code = []
    for seq in seqs:
        for code_set in seq:
            length_code.append(len(code_set))
    length_code = np.array(length_code)
    maxcode = np.max(length_code)
    return maxcode

def pad_matrix_new(seq_med_codes, seq_labtest_codes, seq_diag_codes, seq_proc_codes, seq_labels, options):
    lengths = np.array([len(seq) for seq in seq_diag_codes])
    n_samples = len(seq_diag_codes)
    n_med_codes = options['n_med_codes']
    n_labtest_codes = options['n_labtest_codes']
    n_diag_codes = options['n_diag_codes']
    n_proc_codes = options['n_proc_codes']
    maxlen = np.max(lengths)
    lengths_code = []
    maxcode_med = get_maxCode(seq_med_codes)
    maxcode_labtest = get_maxCode(seq_labtest_codes)
    maxcode_diag = get_maxCode(seq_diag_codes)

    batch_med_codes = np.zeros((n_samples, maxlen, maxcode_med), dtype=np.int64) + n_med_codes
    batch_med_mask_code = np.zeros((n_samples, maxlen, maxcode_med), dtype=np.float32)
    batch_labtest_codes = np.zeros((n_samples, maxlen, maxcode_labtest), dtype=np.int64) + n_labtest_codes
    batch_labtest_mask_code = np.zeros((n_samples, maxlen, maxcode_labtest), dtype=np.float32)
    batch_diag_codes = np.zeros((n_samples, maxlen, maxcode_diag), dtype=np.int64) + n_diag_codes
    batch_diag_mask_code = np.zeros((n_samples, maxlen, maxcode_diag), dtype=np.float32)

    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)


    for bid, seq in enumerate(seq_med_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_med_codes[bid, pid, tid] = code
                batch_med_mask_code[bid, pid, tid] = 1

    for bid, seq in enumerate(seq_labtest_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_labtest_codes[bid, pid, tid] = code
                batch_labtest_mask_code[bid, pid, tid] = 1

    for bid, seq in enumerate(seq_diag_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diag_codes[bid, pid, tid] = code
                batch_diag_mask_code[bid, pid, tid] = 1
    if options['dataset'] == 'mimic_data':
        maxcode_proc = get_maxCode(seq_proc_codes)
        batch_proc_codes = np.zeros((n_samples, maxlen, maxcode_proc), dtype=np.int64) + n_proc_codes
        batch_proc_mask_code = np.zeros((n_samples, maxlen, maxcode_proc), dtype=np.float32)
        for bid, seq in enumerate(seq_proc_codes):
            for pid, subseq in enumerate(seq):
                for tid, code in enumerate(subseq):
                    batch_proc_codes[bid, pid, tid] = code
                    batch_proc_mask_code[bid, pid, tid] = 1
    elif options['dataset'] == 'eicu_data':
        batch_proc_codes = []
        batch_proc_mask_code = []


    for i in range(n_samples):
        batch_mask[i, 0:lengths[i]-1] = 1
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1

    batch_labels = np.array(seq_labels, dtype=np.int64)

    return batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_labels, batch_mask, batch_mask_final, batch_med_mask_code, batch_labtest_mask_code, batch_diag_mask_code, batch_proc_mask_code


def calculate_cost_tran(model, data, options, max_len, loss_function=F.cross_entropy):
    model.eval()
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0
    
    total_pred = []
    total_true = []
    for index in range(n_batches):
        batchX = data[0][batch_size * index: batch_size * (index + 1)]
        batchY = data[1][batch_size * index: batch_size * (index + 1)]
        batchS = data[-1][batch_size * index: batch_size * (index + 1)]
        batchT = data[2][batch_size * index: batch_size * (index + 1)]        
        # batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        # batch_time_step = data[2][batch_size * index: batch_size * (index + 1)]
        # batch_diagnosis_codes, batch_time_step = adjust_input(batch_diagnosis_codes, batch_time_step, max_len, options['n_diagnosis_codes'])
        # batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        # lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
        # maxlen = np.max(lengths)
        if options['model_choice'] == 'TransformerTime' or options['model_choice'] == 'TransformerTime_addsefatt_gated_auxattencode':
            batch_diagnosis_codes, batch_time_step, batch_labels, batch_original_y = padInputWithTime(batchX, batchY, batchT, options)
            batch_diagnosis_codes, batch_time_step = adjust_input(batch_diagnosis_codes, batch_time_step, max_len, options['all_input_dim'])
            lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
            maxlen = np.max(lengths)

            logit, labels, self_attention = model(batch_diagnosis_codes, batch_time_step, batch_labels, options, maxlen, batchS)
        else:
            batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, \
                batch_labels, batch_original_y = padInputWithTime_new(batchX, batchY, batchT, options)
            batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step = adjust_input_new(batch_med_codes, \
                batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, max_len, options['n_med_codes'], options['n_labtest_codes'], \
                options['n_diag_codes'], options['n_proc_codes'], options)
            lengths = np.array([len(seq) for seq in batch_diag_codes])
            maxlen = np.max(lengths)
            logit, labels, self_attention = model(batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_labels, options, maxlen, batchS)
            
        loss = loss_function(logit, labels)
        cost_sum += loss.cpu().data.numpy()

        pred_score = logit.cpu().detach().numpy()
        y_true = labels.cpu().detach().numpy()
        total_pred.append(pred_score)
        total_true.append(y_true)
    total_pred_value = np.concatenate(total_pred)
    total_true_value = np.concatenate(total_true)
    total_avg_auc_micro = roc_auc_score(total_true_value, total_pred_value, average='micro')
    model.train()
    return cost_sum / n_batches, total_avg_auc_micro

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        # if alpha is None:
        #     self.alpha = Variable(torch.ones(class_num))
        # else:
        #     if isinstance(alpha, Variable):
        #         self.alpha = alpha
        #     else:
        #         self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, options, alpha=None,):
        N = inputs.size(0)
        C = inputs.size(1)
        P = nn.functional.softmax(inputs)

        # class_mask = inputs.data.new(N, C).fill_(0)
        # class_mask = Variable(class_mask)
        # ids = targets.view(-1, 1)

        # class_mask.scatter_(1, ids.data, 1.)
        # # print(class_mask)

        # if inputs.is_cuda and not self.alpha.is_cuda:
        #     self.alpha = self.alpha.cuda()
        # alpha = self.alpha[ids.data.view(-1)]

        # probs = (P * class_mask).sum(1).view(-1, 1)

        # log_p = probs.log()
        # # print('probs size= {}'.format(probs.size()))
        # # print(probs)

        # batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # # print('-----bacth_loss------')
        # # print(batch_loss)
        if alpha is None:
            self.alpha = Variable(torch.ones(N,C))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.alpha = self.alpha.to(options['device'])
        log_p = P.log()
        batch_loss = -self.alpha * (torch.pow((1. - P), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class CrossEntropy(nn.Module):
    def __init__(self, n_labels):
        super(CrossEntropy, self).__init__()
        self.num_class = n_labels

    def forward(self, inputs, targets):
        logEps = 1e-8
        cross_entropy = -(targets * torch.log(inputs + logEps) + (1. - targets) * torch.log(1. - inputs + logEps))

        prediction_loss = torch.sum(cross_entropy, dim=1)
        loss = torch.mean(prediction_loss)
        return loss


def load_data(seqFile, labelFile, timeFile):
    train_set_x = pickle.load(open(seqFile+'.train', 'rb'))
    valid_set_x = pickle.load(open(seqFile+'.valid', 'rb'))
    test_set_x = pickle.load(open(seqFile+'.test', 'rb'))
    train_set_s = pickle.load(open(seqFile+'.static_train', 'rb'))
    valid_set_s = pickle.load(open(seqFile+'.static_valid', 'rb'))
    test_set_s = pickle.load(open(seqFile+'.static_test', 'rb'))
    train_set_y = pickle.load(open(labelFile+'.train', 'rb'))
    valid_set_y = pickle.load(open(labelFile+'.valid', 'rb'))
    test_set_y = pickle.load(open(labelFile+'.test', 'rb'))
    train_set_t = None
    valid_set_t = None
    test_set_t = None

    if len(timeFile) > 0:
        train_set_t = pickle.load(open(timeFile+'.train', 'rb'))
        valid_set_t = pickle.load(open(timeFile+'.valid', 'rb'))
        test_set_t = pickle.load(open(timeFile+'.test', 'rb'))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]
    train_set_s = [train_set_s[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]
    valid_set_s = [valid_set_s[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]
    test_set_s = [test_set_s[i] for i in test_sorted_index]

    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_t, train_set_s)
    valid_set = (valid_set_x, valid_set_y, valid_set_t, valid_set_s)
    test_set = (test_set_x, test_set_y, test_set_t, test_set_s)

    return train_set, valid_set, test_set

def set_keep_order(sequences):
    set_seq = list(set(sequences))
    set_seq.sort(key=sequences.index)
    return set_seq

def padInputWithTime(seqs, labels, times, options):
    lengths = np.array([len(seq) for seq in seqs])
    n_sample = len(seqs)
    maxlen = np.max(lengths)
    numClass = options['n_labels']
    x = []
    y = np.zeros((n_sample, numClass))
    original_y = []
    max_len = []
    if options['dataset'] == 'mimic_data':
        for idx, seq in enumerate(seqs):
            tmp_x = []
            for subseq in (seq):
                diag_code = subseq[0]
                proc_code = subseq[1]
                med_code = subseq[2]
                labtest_code = subseq[3]
                uniform_code = copy.deepcopy(med_code)
                uniform_code.extend([i+346 for i in labtest_code])
                uniform_code.extend([i+1022 for i in diag_code])
                uniform_code.extend([i+1927 for i in proc_code])
                tmp_x.append(uniform_code)
            x.append(tmp_x)
        for yvec, label in zip(y, labels):
            last_label  = label
            med_code = last_label[2]
            diag_code = last_label[0]
            proc_code = last_label[1]
            labtest_code = last_label[3]
            if options['predDiag']:
                yvec[diag_code] = 1
                original_y.append(diag_code)
            elif options['predProc']:
                yvec[proc_code] = 1.
                original_y.append(proc_code)
            elif options['predLabtest']:
                yvec[labtest_code] = 1.
                original_y.append(labtest_code)
            else:
                yvec[med_code] = 1.
                original_y.append(med_code)
    elif options['dataset'] == 'eicu_data':
        for idx, seq in enumerate(seqs):
            tmp_x = []
            for index, subseq in enumerate(seq):
                med_seqs = set_keep_order(subseq[1])
                labtest_seqs = set_keep_order(subseq[2])
                diag_seqs = set_keep_order(subseq[0])
                uniform_code = copy.deepcopy(med_seqs)
                uniform_code.extend([i+1374 for i in labtest_seqs])
                uniform_code.extend([i+1529 for i in diag_seqs])
                tmp_x.append(uniform_code)
            x.append(tmp_x)
        for yvec , label in zip(y, labels):
            last_label = label
            med_seqs = set_keep_order(last_label[1])
            labtest_seqs = set_keep_order(last_label[2])
            diag_seqs = set_keep_order(last_label[0])
            if options['predDiag']:
                yvec[diag_seqs] = 1
                original_y.append(diag_seqs)
            elif options['predLabtest']:
                yvec[labtest_seqs] = 1
                original_y.append(labtest_seqs)
            else:
                yvec[med_seqs] = 1
                original_y.append(med_seqs)


    lengths = np.array(lengths)
    return x, times, y, original_y

def padInputWithTime_new(seqs, labels, times, options):
    lengths = np.array([len(seq) for seq in seqs])
    n_sample = len(seqs)
    maxlen = np.max(lengths)
    numClass = options['n_labels']
    med_x = []
    labtest_x = []
    diag_x = []
    proc_x = []
    y = np.zeros((n_sample, numClass))
    original_y = []
    max_len = []
    if options['dataset'] == 'mimic_data':
        for idx, seq in enumerate(seqs):
            tmp_med_x = []
            tmp_labtest_x = []
            tmp_diag_x = []
            tmp_proc_x = []
            for subseq in (seq):
                diag_code = subseq[0]
                proc_code = subseq[1]
                med_code = subseq[2]
                labtest_code = subseq[3]
                tmp_med_x.append(med_code)
                tmp_labtest_x.append(labtest_code)
                tmp_diag_x.append(diag_code)
                tmp_proc_x.append(proc_code)
            med_x.append(tmp_med_x)
            labtest_x.append(tmp_labtest_x)
            diag_x.append(tmp_diag_x)
            proc_x.append(tmp_proc_x)
        for yvec, label in zip(y, labels):
            last_label  = label
            med_code = last_label[2]
            diag_code = last_label[0]
            proc_code = last_label[1]
            labtest_code = last_label[3]
            if options['predDiag']:
                yvec[diag_code] = 1
                original_y.append(diag_code)
            elif options['predProc']:
                yvec[proc_code] = 1.
                original_y.append(proc_code)
            elif options['predLabtest']:
                yvec[labtest_code] = 1.
                original_y.append(labtest_code)
            else:
                yvec[med_code] = 1.
                original_y.append(med_code)
        
    elif options['dataset'] == 'eicu_data':
        for idx, seq in enumerate(seqs):
            tmp_med_x = []
            tmp_labtest_x = []
            tmp_diag_x = []
            for index, subseq in enumerate(seq):
                med_code = set_keep_order(subseq[1])
                labtest_code = set_keep_order(subseq[2])
                diag_code = set_keep_order(subseq[0])
                tmp_med_x.append(med_code)
                tmp_labtest_x.append(labtest_code)
                tmp_diag_x.append(diag_code)
            med_x.append(tmp_med_x)
            labtest_x.append(tmp_labtest_x)
            diag_x.append(tmp_diag_x)
        for yvec , label in zip(y, labels):
            last_label = label
            med_code = set_keep_order(last_label[1])
            labtest_code = set_keep_order(last_label[2])
            diag_code = set_keep_order(last_label[0])
            if options['predDiag']:
                yvec[diag_code] = 1
                original_y.append(diag_code)
            elif options['predLabtest']:
                yvec[labtest_code] = 1
                original_y.append(labtest_code)
            else:
                yvec[med_code] = 1
                original_y.append(med_code)
    
    lengths = np.array(lengths)
    return med_x, labtest_x, diag_x, proc_x, times, y, original_y


