import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from collections import Counter, defaultdict
import pandas as pd
from preprocessing2 import preprocess_train2
import random
from inference2 import tag_all_test2

def compare_files(true_file, pred_file):
    with open(true_file, 'r') as f:
        true_data = [x.strip() for x in f.readlines() if x != '']
    with open(pred_file, 'r') as f:
        pred_data = [x.strip() for x in f.readlines() if x != '']
    if len(pred_data) != len(true_data):
        if len(pred_data) > len(true_data):
            pred_data = pred_data[:len(true_data)]
        else:
            raise KeyError
    num_correct, num_total = 0, 0
    prob_sent = set()
    predictions, true_labels = [], []
    for idx, sen in enumerate(true_data):
        pred_sen = pred_data[idx]
        if pred_sen.endswith('._.') and not pred_sen.endswith(' ._.'):
            pred_sen = pred_sen[:-3] + ' ._.'
        true_words = [x.split('_')[0] for x in sen.split()]
        true_tags = [x.split('_')[1] for x in sen.split()]
        true_labels += true_tags
        pred_words = [x.split('_')[0] for x in pred_sen.split()]
        try:
            pred_tags = [x.split('_')[1] for x in pred_sen.split()]
            predictions += pred_tags
        except IndexError:
            prob_sent.add(idx)
            pred_tags = []
            for x in pred_sen.split():
                if '_' in x:
                    pred_tags.append(x.split('_'))
                else:
                    pred_tags.append(None)
        if pred_words[-1] == '~':
            pred_words = pred_words[:-1]
            pred_tags = pred_tags[:-1]
        if pred_words != true_words:
            prob_sent.add(idx)
        elif len(pred_tags) != len(true_tags):
            prob_sent.add(idx)
        for i, (tt, tw) in enumerate(zip(true_tags, true_words)):
            num_total += 1
            if len(pred_words) > i:
                pw = pred_words[i]
                pt = pred_tags[i]
            else:
                prob_sent.add(idx)
                continue
            if pw != tw:
                continue
            if tt == pt:
                num_correct += 1
        pass
    labels = sorted(list(set(true_labels)))
    if len(prob_sent) > 0:
        print(prob_sent)

    return num_correct / num_total, prob_sent
def split_train_file(wtag_file_path, k, i, seed=42):
    """
    Splits the .wtag file into two files:
    - one containing the i-th fold (validation set)
    - one containing the remaining folds (training set)

    Parameters:
        wtag_file_path (str): Path to the input .wtag file
        k (int): Number of folds
        i (int): Index of the validation fold (0-based)
        seed (int): Random seed for shuffling

    Output:
        - val_i.wtag : contains sentences from the i-th fold
        - train_i.wtag : contains the rest
    """
    # Read sentences
    with open(wtag_file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(sentences)

    # Split into k groups
    fold_size = len(sentences) // k
    folds = [sentences[j * fold_size: (j + 1) * fold_size] for j in range(k)]

    # Handle remainder
    remainder = sentences[k * fold_size:]
    for j, sent in enumerate(remainder):
        folds[j % k].append(sent)

    # Split into validation and training sets
    val_sentences = folds[i]
    train_sentences = [s for j, fold in enumerate(folds) if j != i for s in fold]

    # Write to files
    with open(f'test_{i}.wtag', 'w', encoding='utf-8') as val_out:
        val_out.write('\n'.join(val_sentences) + '\n')

    with open(f'train_{i}.wtag', 'w', encoding='utf-8') as train_out:
        train_out.write('\n'.join(train_sentences) + '\n')

    # print(f"Created 'val_{i}.wtag' with {len(val_sentences)} sentences.")
    # print(f"Created 'train_{i}.wtag' with {len(train_sentences)} sentences.")

#TODO: make k-folds alg for the 2nd model. hopefully be done by tmrw.
# get all sentences
# mix it and divide to k groups
# one group is val, rest is train
# train on train group, and validate on val group
# iterate, so that each time a different group is the val
# get avg score of the model.

def k_folds_cv(data_path, k, threshold, lam=1):
    acc = []
    check_for_me = []
    for i in range(k):
        split_train_file(data_path, k, i)
        train_path = f"train_{i}.wtag"
        test_path = f"test_{i}.wtag"
        weights_path = f'weights_model2_{i}.pkl'
        predictions_path = f'predictions_model2_{i}.wtag'
        statistics, feature2id = preprocess_train2(train_path, threshold)
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

        with open(weights_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]

        tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, transform=True)
        accuracy, check = eval_k_folds(test_path, predictions_path)
        acc.append(accuracy)
        check_for_me.append(check)
        print(f"Fold {i+1}: Accuracy = {accuracy:.2f}%")
    avg_acc = sum(acc) / k
    print(f"\nAverage Accuracy: {avg_acc:.2f}%")
    # for line in check_for_me:
    #     for str in line:
    #         print(str)

def get_stats(test_file, pred_file):
    correct = 0
    total = 0
    mistakes = Counter()
    confusion = defaultdict(lambda: defaultdict(int))
    tag_set = set()
    test = []
    val = []
    with open(test_file) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                cur_word, cur_tag = split_words[word_idx].split('_')
                test.append((cur_word, cur_tag))

    with open(pred_file) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                cur_word, cur_tag = split_words[word_idx].split('_')
                val.append((cur_word, cur_tag))

    print(f"Length of test file: {len(test)}")
    print(f"Length of prediction file: {len(val)}")
    n = min(len(val), len(test))
    total = 0
    for i in range(n):
        true_word, true_tag = test[i]
        pred_word, pred_tag = val[i]
        if true_word != pred_word:
            print(f"Warning: token mismatch - true: {true_word}, pred: {pred_word}")
            continue
        tag_set.update([true_tag, pred_tag])
        confusion[true_tag][pred_tag] += 1
        if true_tag == pred_tag:
            correct += 1
        else:
            mistakes[(true_tag, pred_tag)] += 1
        total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nTop 10 Most Common Mistaken Tags (true → predicted):")
    for (true_tag, pred_tag), count in mistakes.most_common(10):
        print(f"{true_tag} → {pred_tag}: {count} times")

    # Build confusion matrix as DataFrame
    tags = sorted(tag_set)
    matrix_data = {
        true_tag: [confusion[true_tag][pred_tag] for pred_tag in tags]
        for true_tag in tags
    }

    # df = pd.DataFrame(matrix_data, index=tags, columns=tags)
    # print("\nConfusion Matrix (rows = true tags, columns = predicted tags):")
    # print(df)

def eval_k_folds(test_file, pred_file):
    correct = 0
    total = 0
    mistakes = Counter()
    confusion = defaultdict(lambda: defaultdict(int))
    tag_set = set()
    test = []
    val = []
    for_me = []
    with open(test_file) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                cur_word, cur_tag = split_words[word_idx].split('_')
                test.append((cur_word, cur_tag))

    with open(pred_file) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                cur_word, cur_tag = split_words[word_idx].split('_')
                val.append((cur_word, cur_tag))

    # print(f"Length of test file: {len(test)}")
    # print(f"Length of prediction file: {len(val)}")
    n = min(len(val), len(test))
    total = 0
    for i in range(n):
        true_word, true_tag = test[i]
        pred_word, pred_tag = val[i]
        if true_word != pred_word:
            print(f"Warning: token mismatch - true: {true_word}, pred: {pred_word}")
            continue
        tag_set.update([true_tag, pred_tag])
        confusion[true_tag][pred_tag] += 1
        if true_tag == pred_tag:
            correct += 1
        else:
            if i > 0:
                for_me.append(f'word: {true_word} true tag: {true_tag} pred tag: {pred_tag}, '
                              f'previous word & tag: {test[i-1]}')
            mistakes[(true_tag, pred_tag)] += 1
        total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print("\nTop 10 Most Common Mistaken Tags (true → predicted):")
    for (true_tag, pred_tag), count in mistakes.most_common(10):
        print(f"{true_tag} → {pred_tag}: {count} times")
    return accuracy , for_me


def main():

    threshold = 14
    lam = 1

    train_path = "data/train1.wtag"
    test_path = "data/comp1.words"

    weights_path = 'weights_1.pkl'
    predictions_path = 'comp_m1_322440140.wtag'

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    #tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
    tag_all_test("data/test1.wtag", pre_trained_weights, feature2id, 'predictions_test1.wtag',transform=True)
    get_stats("data/test1.wtag", 'predictions_test1.wtag')
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, transform=True)

    tt, prob = compare_files("data/test1.wtag",'predictions_test1.wtag' )
    print(tt)
    print(prob)

    # MODEL 2
    print('________________________________________________________________')
    print('Model 2 check: ')
    threshold = 2
    lam = 1
    k = 7
    train_path = "data/train2.wtag"
    test_path = "data/comp2.words"
    k_folds_cv(train_path, k, threshold)

    weights_path = 'weights_2.pkl'
    predictions_path = 'comp_m2_322440140.wtag'

    statistics, feature2id = preprocess_train2(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    tag_all_test2(test_path, pre_trained_weights, feature2id, predictions_path, transform=True)
    #get_stats(test_path, predictions_path)


if __name__ == '__main__':
    main()
