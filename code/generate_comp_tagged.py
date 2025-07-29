import pickle
from inference import tag_all_test
from inference2 import tag_all_test2



def main():
    test_path = 'data/comp1.words'
    weights_path = 'trained_models/weights_1.pkl'
    predictions_path = 'comp_m1_322440140.wtag'


    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]


    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, transform=True)

    # MODEL 2

    test_path = "data/comp2.words"

    weights_path = 'trained_models/weights_2.pkl'
    predictions_path = 'comp_m2_322440140.wtag'


    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]


    tag_all_test2(test_path, pre_trained_weights, feature2id, predictions_path, transform=True)



if __name__ == '__main__':
    main()
