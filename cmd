 CUDA_VISIBLE_DEVICES=0 nohup python -u train_MoCoDAD.py --config config/DogLeg/mocodad_train_no_nose_dup_withers.yaml > log_tr_dog_no_nose &
 CUDA_VISIBLE_DEVICES=0 nohup python -u train_MoCoDAD.py --config config/DogLeg/mocodad_train_2trans.yaml > log_tr_dog_2trans &
 CUDA_VISIBLE_DEVICES=1 nohup python -u train_MoCoDAD.py --config config/DogLeg/mocodad_train_kpthr_07.yaml > log_tr_dog_kpthr_07 &
 CUDA_VISIBLE_DEVICES=1 nohup python -u train_MoCoDAD.py --config config/DogLeg/mocodad_train_kpthr_08.yaml > log_tr_dog_kpthr_08 &
 CUDA_VISIBLE_DEVICES=1 nohup python -u train_MoCoDAD.py --config config/DogLeg/mocodad_train_kpthr_09.yaml > log_tr_dog_kpthr_09 &
 CUDA_VISIBLE_DEVICES=0 nohup python -u train_MoCoDAD.py --config config/DogLeg/mocodad_train_win4.yaml > log_tr_dog_win4 &
 CUDA_VISIBLE_DEVICES=0 nohup python -u train_MoCoDAD.py --config config/DogLeg/mocodad_train_win8.yaml > log_tr_dog_win8 &