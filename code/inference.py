from preprocessing import read_test
from tqdm import tqdm
import numpy as np
import heapq
from preprocessing import represent_input_with_features

# def memm_viterbi(sentence, pre_trained_weights, feature2id):
#     """
#     Write your MEMM Viterbi implementation below
#     You can implement Beam Search to improve runtime
#     Implement q efficiently (refer to conditional probability definition in MEMM slides)
#     """
#     pass


def memm_viterbi(sentence, pre_trained_weights, feature2id, beam_size=8, t=False):
    possible_tags = feature2id.feature_statistics.tags
    n = len(sentence) - 3 # subtrack the padders - * * ~
    #print(len(pre_trained_weights))

    viterbi = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]

    # Initialization
    for tag in possible_tags:
        prev_prev_tag = '*'
        prev_tag = '*'
        feats = represent_input_with_features((sentence[2], tag, '*', prev_tag, '*', prev_prev_tag, sentence[3]), feature2id.feature_to_idx) #tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        #print(feats)
        score = 0
        for f in feats:
            score += pre_trained_weights[f]
        viterbi[0][(prev_tag, tag)] = score
        backpointer[0][(prev_tag, tag)] = prev_prev_tag

    # Recursion with beam search
    for i in range(1, n):
        current_beam = []
        prev_beam = sorted(viterbi[i - 1].items(), key=lambda x: -x[1])[:beam_size]

        for (prev_prev_tag, prev_tag), prev_score in prev_beam:
            for curr_tag in possible_tags:
                #print(sentence[i+2])
                feats = represent_input_with_features((sentence[i+2], curr_tag, sentence[i+1], prev_tag, sentence[i], prev_prev_tag, sentence[i+3]), feature2id.feature_to_idx)
                score = 0
                for f in feats:
                    score += pre_trained_weights[f]
                total_score = prev_score + score

                key = (prev_tag, curr_tag)
                if key not in viterbi[i] or total_score > viterbi[i][key]:
                    viterbi[i][key] = total_score
                    backpointer[i][key] = prev_prev_tag

        # Keep top beam_size entries for next step
        viterbi[i] = dict(heapq.nlargest(beam_size, viterbi[i].items(), key=lambda x: x[1]))

    # Termination
    # print("here3")
    # print(viterbi[n - 1].items())
    # print(viterbi)

    max_score = -np.inf
    last_tags = None
    for (prev_tag, last_tag), score in viterbi[n - 1].items():
        if score > max_score:
            max_score = score
            last_tags = (prev_tag, last_tag)

    # Backtracking
    best_tags = [None] * n
    best_tags[n - 1] = last_tags[1]
    best_tags[n - 2] = last_tags[0]

    for i in range(n-3, -1, -1):
        prev_prev_tag = backpointer[i + 2][(best_tags[i + 1], best_tags[i + 2])]
        best_tags[i] = prev_prev_tag

    if t:
        return transform_tags(sentence,best_tags)
    return best_tags

def transform_tags(sentence, tags):
    """
    takes the tags from viterbi and makes sure that there aren't any common mistakes.
    """

    symbols = [',', '.',':']
    for i in range(2, len(sentence) - 3):
        c_word = sentence[i]
        c_tag = tags[i-2]
        if all(char.isdigit() or char == '.'  for char in c_word):
            tags[i-2] = 'CD'
        if c_word[0] == '-' and all(char.isdigit() or char == '.'  for char in c_word[1:]):
            tags[i - 2] = 'CD'
        if c_tag == 'NN' and c_word[-1] == 's':
            tags[i-2] == 'NNS'
        if c_word in symbols:
            tags[i-2] = c_word
    return tags



def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, transform=False):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, t=transform)[0:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()


