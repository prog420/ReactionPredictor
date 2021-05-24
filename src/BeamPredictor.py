import numpy as np


class Prediction:
    def __init__(self, model, vocab, tokenizer, max_reg=150, max_prod=150, beam_size=5, reduce=True, alpha=0.6):
        self.alpha = alpha
        self.model = model
        self.vocab = vocab
        self.smiles = ""
        self.max_reg = max_reg
        self.max_prod = max_prod

        self.beam_size = beam_size
        self.reduce = reduce
        self.smiles_tokenizer = tokenizer

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.EOS_ind = [ind for ind, t in self.inv_vocab.items() if t == 'pad'][0]
        self.SOS_ind = [ind for ind, t in self.inv_vocab.items() if t == 'SOS'][0]
        self.data = list()
        self.num_values = {}
        for ind, tok in self.inv_vocab.items():
            if "[" in tok:
                try:
                    self.num_values[ind] = int(tok.strip("[]"))
                except ValueError:
                    continue

    def get_data(self):
        reg_batch = np.zeros((1, self.max_reg))
        prod_batch = np.zeros((1, self.max_prod))
        reg_batch[:, 0] = self.vocab['SOS']
        prod_batch[:, 0] = self.vocab['SOS']
        tokens_prod = list()
        if ">>" in self.smiles:
            reg = str(self.smiles.split('>>')[0])
            prod = str(self.smiles.split('>>')[1])
            tokens_prod = [token for token in self.smiles_tokenizer.split(prod) if token]
        else:
            reg = str(self.smiles)
        tokens_reg = [token for token in self.smiles_tokenizer.split(reg) if token]
        if self.reduce:
            tokens_reg = self.reduce_array(tokens_reg)
            tokens_reg = [token for token in self.smiles_tokenizer.split(tokens_reg) if token]

        if len(tokens_reg) > self.max_reg - 2 or len(tokens_prod) > self.max_prod - 2:
            raise ValueError("Not in reg / prod max lengths!")
        for i in range(len(tokens_reg)):
            reg_batch[0][i + 1] = self.vocab[tokens_reg[i]]
        for i in range(len(tokens_prod)):
            prod_batch[0][i + 1] = self.vocab[tokens_prod[i]]
        self.data = [reg_batch, prod_batch]
        return [self.data[0][:], self.data[1][:, :-1]], self.data[1][:, 1:]

    def reduce_array(self, array):
        indices = [0]
        counter = 0
        for i in range(len(array)):
            try:
                next_token = array[i + 1]
            except IndexError:
                if counter:
                    indices.append(i + 1)
                break
            if array[i] == next_token:
                if counter == 0:
                    indices.append(i)
                counter += 1
            else:
                if counter:
                    indices.append(i + 1)
                    counter = 0
        new_array = []
        for i in range(0, len(indices), 2):
            try:
                new_array += array[indices[i]:indices[i + 1]]
                new_array.append(f"{array[indices[i + 1]]}[{indices[i + 2] - indices[i + 1]}]")
            except IndexError:
                try:
                    new_array += array[indices[i]:]
                except IndexError:
                    break
        new_seq = "".join(new_array)
        return new_seq

    def get_full_smi(self, array):
        indices = []
        new_ind = []
        values = []
        for i in range(len(array)):
            try:
                num = self.num_values[array[i]] - 1
            except KeyError:
                continue
            indices.append([int(i)] * num)
            new_ind.append([int(i) - len(indices)] * num)
            values.append([array[i - 1]] * num)
        if indices:
            new_arr = np.delete(array, np.concatenate(indices))
            new_arr = np.insert(new_arr, np.concatenate(new_ind), np.concatenate(values))
        else:
            new_arr = array
        restored = ''.join([self.inv_vocab[tok] for tok in new_arr])
        return restored

    def beam_pred(self, data, k):
        vocab_size = len(self.inv_vocab)

        inputs, out = data
        enc_inp, dec_inp = inputs
        batch_size, seq_len = dec_inp.shape

        def get_len_penalty(length):
            return ((5 + length) ** self.alpha) / (6 ** self.alpha)

        def get_sos_tensor(bt, seq):
            z = np.zeros((bt, seq), dtype=np.int32)
            z[:, 0] = self.SOS_ind
            return z

        sos = get_sos_tensor(batch_size, seq_len)

        k_beam = [[np.zeros(batch_size), get_sos_tensor(batch_size, seq_len + 1)]]

        for l in range(seq_len):
            candidates_prob = []
            candidates_seq = []
            for beam in k_beam:
                pr = self.model.predict([enc_inp, beam[1][:, :-1], beam[1][:, :-1]])
                pr_k = pr[:, l].argsort(axis=-1)[:, -k:][:, ::-1]
                prob_k = np.sort(pr[:, l], axis=-1)[:, -k:][:, ::-1]
                pr_k = np.split(pr_k, k, axis=-1)
                prob_k = np.split(prob_k, k, axis=-1)

                for i, p in enumerate(pr_k):
                    mask = (beam[1][:, :l + 1] == self.EOS_ind).sum(axis=-1)
                    mask = (mask > 0).astype(np.int32)

                    candidates_prob.append(
                        (beam[0] * get_len_penalty(l) + np.log(prob_k[i][:, 0] + 1e-7)) / get_len_penalty(l + 1) * (
                                    1 - mask) + \
                        beam[0] * mask
                    )
                    candidates_seq.append(
                        np.concatenate([beam[1][:, :l + 1], p, np.zeros((batch_size, seq_len - l - 1))], axis=-1)
                        # """-2"""
                    )

            candidates_prob = np.array(candidates_prob).transpose([1, 0])
            candidates_seq = np.array(candidates_seq, dtype=np.int32).transpose([1, 2, 0])  # (b,seq,k)

            # if the probabilities of several branches coincide, then masking all of them but the last as np.log(1e-7)
            for i in range(candidates_prob.shape[-1] - 1):
                mask = (candidates_prob[:, i + 1] == candidates_prob[:, i]).astype(np.int32)
                candidates_prob[:, i] -= mask * 100000

            top_ids = candidates_prob.argsort(axis=-1)[:, -k:][:, np.newaxis, ::-1]  # tr
            top_ids = np.tile(top_ids, (1, seq_len + 1, 1))  # tr

            candidates_prob = np.sort(candidates_prob, axis=-1)[:, -k:][:, ::-1].transpose([1, 0])
            candidates_seq = np.take_along_axis(candidates_seq, top_ids, axis=-1).transpose([2, 0, 1])  # (k,b,seq)

            tmp = []
            for i in range(len(candidates_prob)):
                tmp.append([candidates_prob[i], candidates_seq[i]])
            k_beam = tmp

        return candidates_seq.transpose([1, 0, 2])[:, :, 1:]  # (b,k,seq)

    def prediction(self, smiles):
        if not isinstance(smiles, str):
            raise TypeError('Only "str" type can be used as SMILES')
        self.smiles = smiles
        matrix = self.get_data()
        pred_list = self.beam_pred(matrix, self.beam_size)[0]
        true_smiles = ''.join([self.inv_vocab[i] for i in matrix[1][0]]).split('pad')[0]
        pred_smiles = []
        for rxn in pred_list:
            if self.reduce:
                pred_smi = self.get_full_smi(rxn)
            else:
                pred_smi = ''.join([self.inv_vocab[i] for i in rxn])

            if 'pad' in pred_smi and len(pred_smi.split('pad')[0]) > 0:
                pred_smi = pred_smi.split('pad')[0]
                pred_smiles.append(pred_smi)
            else:
                continue

        return pred_smiles[0]