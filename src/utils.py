import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def generate_batch(data_batch):
    q_batch, ans_batch = [], []
    for (q_item, ans_item) in data_batch:
        q_batch.append(q_item)
        ans_batch.append(ans_item)
    q_batch = pad_sequence(q_batch)
    ans_batch = pad_sequence(ans_batch)
    return q_batch, ans_batch

def generate_square_subsequent_mask(sz, mydevice):
    mask = (torch.triu(torch.ones((sz, sz), device=mydevice)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, src_vocab, mydevice):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, mydevice)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=mydevice).type(torch.bool)

    src_padding_mask = (src == src_vocab.pad_id).transpose(0, 1)
    tgt_padding_mask = (tgt == src_vocab.pad_id).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, train_data_loader, src_vocab, optimizer, mydevice, loss_fn, save_dir):
    model.train()
    losses = 0
    for idx, (src, tgt) in enumerate(train_data_loader):
        src = src.to(mydevice)
        tgt = tgt.to(mydevice)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, src_vocab, mydevice)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if idx % 10 == 0:
            optimizer.step()
        
        losses += loss.item()
    torch.save(model.state_dict(), save_dir)

    return losses / len(train_data_loader)

def evaluate(model, valid_data_loader, src_vocab, mydevice, loss_fn):
    model.eval()
    losses = 0
    for idx, (src, tgt) in (enumerate(valid_data_loader)):
        src = src.to(mydevice)
        tgt = tgt.to(mydevice)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, src_vocab, mydevice)
        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(valid_data_loader)

def greedy_decode(model, src_vocab, src, src_mask, max_len, start_symbol, mydevice):
    src = src.to(mydevice)
    src_mask = src_mask.to(mydevice)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(mydevice)
    for i in range(max_len-1):
        memory = memory.to(mydevice)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(mydevice).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), mydevice)
                                    .type(torch.bool)).to(mydevice)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == src_vocab.eos_id:
          break
    return ys

def translate(model, src, src_vocab, tgt_vocab, tgt_length, mydevice):
    model.eval()
    tokens = [src_vocab.string_to_id[i] for i in src]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src_vocab, src, src_mask, max_len=tgt_length + 2, start_symbol=src_vocab.sos_id, mydevice=mydevice).flatten()
    return "".join([tgt_vocab.id_to_string[int(tok)] for tok in tgt_tokens]).replace("<sos>", "").replace("<eos>", "")

def batch_greedy(model, src_vocab, tgt_vocab, data : DataLoader, num_batches: int = 10):
    score = 0
    for idx, batch in enumerate(data):
        questions = batch[0].T
        answers = batch[1].T
        for jdx,(question,answer) in enumerate(zip(questions,answers)):
            question_text = [src_vocab.id_to_string[int(i)] for i in question]
            question_text = "".join(question_text).replace("<pad>", "")
            answer_text = [tgt_vocab.id_to_string[int(i)] for i in answer]
            answer_text = "".join(answer_text).replace("<sos>", "").replace("<eos>", "").replace("<pad>","")

            soln = translate(model, question_text, src_vocab, tgt_vocab, len(answer_text))
            if (soln == answer_text):
                score += 1
                print(question_text)
                print(soln)
        if idx >= num_batches:
            break

def Accuracy_Computation(model, src_vocab, tgt_vocab, mydevice, data : DataLoader):
    score = 0
    for idx, batch in enumerate(data):
        questions = batch[0].T
        answers = batch[1].T
        for jdx,(question,answer) in enumerate(zip(questions,answers)):
            question_text = [src_vocab.id_to_string[int(i)] for i in question]
            question_text = "".join(question_text).replace("<pad>", "")
            answer_text = [tgt_vocab.id_to_string[int(i)] for i in answer]
            answer_text = "".join(answer_text).replace("<sos>", "").replace("<eos>", "").replace("<pad>","")

            soln = translate(model, question_text, src_vocab, tgt_vocab, len(answer_text), mydevice)
            if (soln == answer_text):
                score += 1
        if idx >= 100:
            break
    accuracy = 100 * score/(jdx+1)
    return accuracy
