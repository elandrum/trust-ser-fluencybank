import json
import yaml
import numpy as np
import pandas as pd
import pickle, pdb, re

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


# added this because I had some type conversion errors reading in the csv for dataframe(EL)
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

if __name__ == '__main__':

    # Read data path
    with open("../config/config.yml", "r") as stream:
        config = yaml.safe_load(stream)
    data_path   = Path(config["data_dir"]["crema_d"])
    output_path = Path(config["project_dir"])


    ####################BEGIN WORKSHOPPING MSP PODCAST#######################
    ###############################################################
     # iterate over the labels
    label_df = pd.read_csv(Path(data_path).joinpath("fluencybank_labels.csv"), index_col=0)
    clipNum = len(label_df)
    train_list, dev_list, test_list = list(), list(), list()
    # print(clipNum)
    # print(clipNum/10, (clipNum/10)*2)
    currFile = 0
    for idx, firstcol in tqdm(enumerate(list(label_df.index)[:]), ncols=100, miniters=100):
        # Show,EpId,ClipId,Start,Stop,Unsure,PoorAudioQuality,Prolongation,Block,SoundRep,WordRep,DifficultToUnderstand,Interjection,NoStutteredWords,NaturalPause,Music,NoSpeech
        # basically storing everything just-in-case you could delete a lot of this
        ep_id = label_df.iloc[idx]["EpId"]
        clip_id = label_df.iloc[idx]["ClipId"]
        start = label_df.iloc[idx]["Start"]
        stop = label_df.iloc[idx]["Stop"]
        unsure = label_df.iloc[idx]["Unsure"]
        poor_audio_quality = label_df.iloc[idx]["PoorAudioQuality"]
        prolongation = label_df.iloc[idx]["Prolongation"]
        block = label_df.iloc[idx]["Block"]
        soundrep = label_df.iloc[idx]["SoundRep"]
        wordrep = label_df.iloc[idx]["WordRep"]
        difficult_to_understand = label_df.iloc[idx]["DifficultToUnderstand"]
        interjection = label_df.iloc[idx]["Interjection"]
        no_stuttered_words = label_df.iloc[idx]["NoStutteredWords"]
        natural_pause = label_df.iloc[idx]["NaturalPause"]
        music = label_df.iloc[idx]["Music"]
        no_speech = label_df.iloc[idx]["NoSpeech"]
        file_name = firstcol+ "_"+ str(ep_id).zfill(3)+"_"+ str(clip_id)+ ".wav"
        file_path = Path(data_path).joinpath("AudioWAV", file_name)
        if Path.exists(file_path) is False: 
            # print(file_path)
            continue

        #############################################################################################
        #the following is an old chunk from msp-podcast.py but it might be important and I just don't know (EL)
        #############################################################################################
        # # skip condictions, unknown speakers, other emotions, no agreement emotions
        # if "known" in speaker_id or "known" in gender: continue
        # session_id = label_df.iloc[idx, "Split_Set"]
        
        # # [key, speaker id, gender, path, label]
        # gender_label = "female" if gender == "Female" else "male"
        # sentence_file = file_path.parts[-1].split('.wav')[0]
        # sentence_part = sentence_file.split('_')
        #############################################################################################
        
        
        #change this when we want to train on different labels?
        file_data = [file_name, str(file_path), unsure, poor_audio_quality, soundrep]

        session_id="Test1"
        # determine which dataset to put it in
        if (currFile<(8*clipNum/10)):
            session_id = 'Train'
        elif (currFile<(9*clipNum/10)):
            session_id = 'Validation'

        # append data
        if session_id == 'Test1': test_list.append(file_data)
        # elif session_id == 'Test2': test2_list.append(file_data)
        elif session_id == 'Validation': dev_list.append(file_data)
        elif session_id == 'Train': train_list.append(file_data)
        currFile += 1
    ####################END OF WORKSHOPPING#######################
    ###############################################################


    ####################CREMA D FROM BEFORE STARTS AGAIN HERE#######################
    ### parts of this might be useful/neccesary to not cause some later problem so I kept in comments (EL)#########
    
    # kf = KFold(n_splits=5, random_state=None, shuffle=False)
    # for fold_idx, (train_index, test_index) in enumerate(kf.split(np.arange(1001, 1092, 1))):
    #     Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    #     train_list, dev_list, test_list = list(), list(), list()
        
    #     # crema-d
    #     file_list = [x for x in Path(data_path).joinpath("AudioWAV").iterdir() if '.wav' in x.parts[-1]]
    #     file_list.sort()
    #     # read demographics and ratings
    #     # demo_df = pd.read_csv(str(Path(data_path).joinpath('fluencybank_labels.csv')), index_col=0)
    #     demo_df = pd.read_csv(str(Path(data_path).joinpath('VideoDemographics.csv')), index_col=0)
    #     rating_df = pd.read_csv(str(Path(data_path).joinpath('processedResults', 'summaryTable.csv')), index_col=1)
    #     train_index, dev_index = train_index[:-len(train_index)//5], train_index[-len(train_index)//5:]
    #     # read speakers
    #     train_speakers = [np.arange(1001, 1092, 1)[idx] for idx in train_index]
    #     dev_speakers = [np.arange(1001, 1092, 1)[idx] for idx in dev_index]
    #     test_speakers = [np.arange(1001, 1092, 1)[idx] for idx in test_index]
        
    #     for idx, file_path in enumerate(file_list):
    #         # read basic information
    #         if '1076_MTI_SAD_XX.wav' in str(file_path): continue
    #         sentence_file = file_path.parts[-1].split('.wav')[0]
    #         sentence_part = sentence_file.split('_')
    #         speaker_id = int(sentence_part[0])
    #         gender = 'male' if demo_df.loc[int(speaker_id), 'Sex'] == 'Male' else 'female'
    #         label = rating_df.loc[sentence_file, 'MultiModalVote']
            
    #         # [key, speaker id, gender, path, label]
    #         file_data = [sentence_file, speaker_id, gender, str(file_path), label]

    #         # append data
    #         if speaker_id in test_speakers: test_list.append(file_data)
    #         elif speaker_id in dev_speakers: dev_list.append(file_data)
    #         else: train_list.append(file_data)
        
    return_dict = dict()
    return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
    logging.info(f'-------------------------------------------------------')
    # kept this so we don't forget to change naming stuff because otherwise it's very confusing to someone outside project (EL)
    logging.info(f'Split distribution for CREMA-D[secret FluencyBank] dataset')
    for split in ['train', 'dev', 'test']:
        logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')
    # dump the dictionary
    jsonString = json.dumps(return_dict, cls=NumpyEncoder, indent=4)
    # it has to be named fold1 even though we aren't doing folds because they use the fold name syntax over and over (EL)
    jsonFile = open(str(output_path.joinpath('train_split', f'crema_d_fold1.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    ####################################################
     ################END OF WAS HERE BEFORE ################