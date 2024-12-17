import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F


def test_sequence_shift(model, sequence: np.ndarray, shifts: list, test_seq_len: int = 256, batch_size: int = 10):
    test_seq_len = len(sequence) - 1 if test_seq_len >= len(sequence) or test_seq_len < 0 else test_seq_len

    shifted_seqs = np.tile(sequence, (len(shifts), 1))
    shifted_seqs += np.array(shifts)[:, None]

    results = []
    for batch_start in tqdm(range(0, len(shifts), batch_size)):
        inputs_batch = torch.tensor(shifted_seqs[batch_start:batch_start + batch_size], dtype=torch.int32)

        # predicts, shifted_logits = run_sequence_through_model(model, inputs_ids)
        with torch.no_grad():
            outputs = model(inputs_batch)
            logits = outputs.logits

        shifted_logits = logits[:, :-1, :]
        predicts = torch.argmax(shifted_logits, dim=2).squeeze()
        test_seq = inputs_ids[:, -test_seq_len:].contiguous().squeeze(0)
        shifted_inputs = inputs_ids[:, 1:].contiguous().squeeze(0)

        actual_log_probs = F.log_softmax(shifted_logits, dim=-1)[torch.arange(shifted_inputs.shape[0]), shifted_inputs]
        mean_log_likelyhood = actual_log_probs.mean().numpy()
        acc = (predicts[-test_seq_len:] == test_seq).sum().numpy() / test_seq_len
        median_dist = torch.median(predicts[-test_seq_len:] - test_seq).numpy()

        for matching_start in range(shifted_inputs.shape[0]):
            if (predicts[matching_start:] == shifted_inputs[matching_start:]).all():
                break

        results.append((acc, mean_log_likelyhood, median_dist, matching_start))

    return results
