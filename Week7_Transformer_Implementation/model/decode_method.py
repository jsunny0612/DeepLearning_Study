import torch


def greedy_search(outputs, tokenizer):
    generated_sequences = []
    batch_size, seq_len, vocab_size = outputs.size()

    for batch_idx in range(batch_size):
        sequence = []
        for step in range(seq_len):
            token_id = torch.argmax(outputs[batch_idx, step], dim=-1).item()
            if token_id == tokenizer.eos_token_id:
                break
            sequence.append(token_id)
        generated_sequences.append(sequence)
    return generated_sequences

'''
def beam_search(outputs, k, eos_token_id=None):
    # 초기 빈 시퀀스와 점수
    sequences = [[[], 0.0]]
    batch_size = outputs.size(1)  # 배치 크기 확인

    for t in range(outputs.size(0)):  # 각 타임스텝 반복
        new_sequences = []

        for batch_idx in range(batch_size):  # 각 배치 샘플 처리
            candidates = []
            for seq, score in sequences:
                # 시퀀스가 종료 토큰으로 끝났으면 그대로 저장
                if seq and eos_token_id is not None and seq[-1] == eos_token_id:
                    candidates.append((seq, score))
                    continue

                # 현재 배치의 확률 분포 계산
                step_probs = torch.softmax(outputs[t, batch_idx], dim=-1)

                # 각 토큰에 대해 새로운 후보 시퀀스 생성
                for token_id in range(step_probs.size(0)):
                    token_prob = step_probs[token_id].item()  # Python 스칼라 값으로 변환
                    new_seq = seq + [token_id]
                    new_score = score - log(token_prob + 1e-8)  # log-prob 추가
                    candidates.append((new_seq, new_score))

            # 점수를 기준으로 상위 k개의 후보만 선택
            top_candidates = sorted(candidates, key=lambda x: x[1])[:k]
            new_sequences.extend(top_candidates)

        # 현재 타임스텝의 최종 상위 시퀀스만 유지
        sequences = sorted(new_sequences, key=lambda x: x[1])[:k]

    return sequences
'''


def beam_search(outputs, k):

    # Initialize sequences: each sequence is a pair [sequence (list of indices), score (log probability)].
    sequences = [[[], 0.0]]  # Log probability starts at 0 (log(1) = 0)

    for t in range(outputs.shape[0]):  # Iterate over time steps
        all_candidates = []

        for seq, score in sequences:  # Expand each sequence in the current beam
            for j in range(outputs.shape[1]):  # Iterate over vocab size
                # Extend sequence with token j and update score
                new_seq = seq + [j]
                new_score = score + torch.log(outputs[t, j] + 1e-10).item()  # Use log probability
                all_candidates.append([new_seq, new_score])

        # Sort candidates by score (descending) and keep top-k
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:k]

    return sequences