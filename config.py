class DataConfig(object):
    data_path = "/home/ubuntu/rambo/dataset/"
    data_name = "hsv_gray_diff_ch4"
    img_height = 192
    img_width = 256
    num_channels = 2
    
class TrainConfig(DataConfig):
    model_name = "comma_prelu"
    batch_size = 32
    num_epoch = 10
    val_part = 4
    X_train_mean_path = "data/X_train_gray_diff2_mean.npy"
    
class TestConfig(TrainConfig):
    model_path = "dataset/models/weights_hsv_gray_diff_ch4_comma_prelu-03-0.07953.hdf5"
    angle_train_mean = -0.004179079
	#processors = {"CenterImage": preprocess_test_data:make_hsv_grayscale_diff_data(4)}}

class VisualizeConfig(object):
    pred_path = "dataset/models/submissions/komanda.csv"
    #pred_path = "data/CH2_final_evaluation.csv"
    true_path = "data/CH2_final_evaluation.csv"
    img_path = "dataset/round2/test/center/*.jpg"
