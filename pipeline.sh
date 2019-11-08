#!/bin/bash

# Author: Nanxin Chen, Cheng-I Lai
# Pipeline for preprocessing + training + postprocessing neural speaker embeddings. This includes:
# step 0:  create VoxCeleb1+2 data directories
# step 1:  make FBanks + VADs (based on MFCCs) for clean data
# step 2:  data augmentation 
# step 3:  make FBanks for noisy data
# step 4:  applies CM and removes silence (for training data)
# step 5:  filter by length, split to train/cv, and (optional) save as pytorch tensors
# step 6:  nn training 
# step 7:  applies CM and removes silence (for decoding data)
# step 8:  decode with the trained nn
# step 9:  get train and test embeddings 
# step 10: compute mean, LDA and PLDA on decode embeddings
# step 11: scoring
# step 12: EER & minDCF results
# (This script is modified from Kaldi egs/)

. ./cmd.sh
. ./path.sh
set -e

# Change this to your Kaldi voxceleb egs directory
kaldi_voxceleb=/data/sls/scratch/clai24/nii/sid/template

# The trials file is downloaded by local/make_voxceleb1.pl.
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=/data/sls/scratch/clai24/data/voxceleb1
voxceleb2_root=/data/sls/scratch/clai24/data/voxceleb2
musan_root=/data/sls/scratch/clai24/data/musan

stage=0

if [ $stage -le -1 ]; then
    # link necessary Kaldi directories
    rm -fr utils steps sid
    ln -s $kaldi_voxceleb/v2/utils ./
    ln -s $kaldi_voxceleb/v2/steps ./
    ln -s $kaldi_voxceleb/v2/sid ./
fi


if [ $stage -le 0 ]; then
  log=exp/make_voxceleb
  $train_cmd $log/make_voxceleb2_dev.log local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  $train_cmd $log/make_voxceleb2_test.log local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test
  # This script creates data/voxceleb1_test and data/voxceleb1_train.
  # Our evaluation set is the test portion of VoxCeleb1.
  $train_cmd $log/make_voxceleb1.log local/make_voxceleb1.pl $voxceleb1_root data
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7325 speakers and 1277344 utterances.
  $train_cmd $log/combine_voxceleb1+2.log local/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train
fi


if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train voxceleb1_test; do
    local/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank.conf \
      --nj 40 --cmd "$train_cmd" data/${name} exp/make_fbank fbank
    local/fix_data_dir.sh data/${name}
    local/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad fbank
    local/fix_data_dir.sh data/${name}
  done

  # Make MFCCs and compute the energy-based VAD for each dataset
  # NOTE: Kaldi VAD is based on MFCCs, so we need to additionally extract MFCCs
  # (https://groups.google.com/forum/#!msg/kaldi-help/-REizujqa5k/u_FJnGokBQAJ)
  for name in train voxceleb1_test; do
    local/copy_data_dir.sh data/${name} data/${name}_mfcc
    local/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf \
      --nj 40 --cmd "$train_cmd" data/${name}_mfcc exp/make_mfcc mfcc
    local/fix_data_dir.sh data/${name}_mfcc
    local/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name}_mfcc exp/make_vad mfcc
    local/fix_data_dir.sh data/${name}_mfcc
  done

  # correct the right vad.scp
  for name in train voxceleb1_test; do
    cp data/${name}_mfcc/vad.scp data/${name}/vad.scp
    local/fix_data_dir.sh data/$name
  done 
fi


# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ]; then
  log=exp/augmentation
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  $train_cmd $log/reverberate_data_dir.log python2 steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  local/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  $train_cmd $log/make_musan.log local/make_musan.sh $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  $train_cmd $log/augment_musan_noise.log python2 steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  $train_cmd $log/augment_musan_music.log python2 steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  $train_cmd $log/augment_musan_speech.log python2 steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  local/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble
fi


if [ $stage -le 3 ]; then
  # Take a random subset of the augmentations
  local/subset_data_dir.sh data/train_aug 1000000 data/train_aug_1m
  local/fix_data_dir.sh data/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  local/make_fbank.sh --fbank-config conf/fbank.conf --nj 40 --cmd "$train_cmd" \
    data/train_aug_1m exp/make_fbank fbank

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  local/combine_data.sh data/train_combined data/train_aug_1m data/train
fi


# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" --compress false \
    data/train_combined data/train_combined_no_sil exp/train_combined_no_sil
  local/fix_data_dir.sh data/train_combined_no_sil
fi


# Now we split all data into two parts: training and cv
if [ $stage -le 5 ]; then
  log=exp/processed
  mkdir -p $log

  # filter out utterances w/ < 800 frames 
  awk 'NR==FNR{a[$1]=$2;next}{if(a[$1]>=800)print}' data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > $log/utt2spk 
  # create spk2num_frames, this will be useful for balancing training
  awk '{if(!($2 in a))a[$2]=0;a[$2]+=1;}END{for(i in a)print i,a[i]}' $log/utt2spk > $log/spk2num 
  # create train (90%) and cv (10%) utterance list
  awk -v seed=$RANDOM 'BEGIN{srand(seed);}NR==FNR{a[$1]=$2;next}{if(a[$2]<10)print $1>>"exp/processed/train.list";else{if(rand()<=0.1)print $1>>"exp/processed/cv.list";else print $1>>"exp/processed/train.list"}}' $log/spk2num $log/utt2spk 

  # get the feats.scp for train and cv based on train.list and cv.list
  awk 'NR==FNR{a[$1]=1;next}{if($1 in a)print}' $log/train.list data/train_combined_no_sil/feats.scp | shuf > $log/train_orig.scp
  awk 'NR==FNR{a[$1]=1;next}{if($1 in a)print}' $log/cv.list data/train_combined_no_sil/feats.scp | shuf > $log/cv_orig.scp

  # maps speakers to labels (spkid)
  awk 'BEGIN{s=0;}{if(!($2 in a)){a[$2]=s;s+=1;}print $1,a[$2]}' $log/utt2spk > $log/utt2spkid
  
  # save the uncompressed, preprocessed pytorch tensors
  # Note: this is optional!  
  mkdir -p $log/py_tensors
  python scripts/prepare_data.py --feat_scp $log/train_orig.scp --save_dir $log/py_tensors
  python scripts/prepare_data.py --feat_scp $log/cv_orig.scp --save_dir $log/py_tensors
fi 


log=exp/processed
expname=test # chance the experiment name to your liking
expdir=$log/$expname/
mkdir -p $expdir
num_spk=`awk 'BEGIN{s=0;}{if($2>s)s=$2;}END{print s+1}' $log/utt2spkid`
echo "There are "$num_spk" number of speakers."

# Network Training
if [ $stage -le 6 ]; then  
  $cuda_cmd $expdir/train.log python scripts/main.py \
                       --train $log/train_orig.scp --cv $log/cv_orig.scp \
                       --utt2spkid $log/utt2spkid --spk_num $num_spk \
                       --min-chunk-size 300 --max-chunk-size 800 \
                       --model 'resnet34' \
                       --input-dim 30 --hidden-dim 512 --D 32 \
                       --pooling 'mean' --network-type 'lde' \
                       --distance-type 'sqr' --asoftmax True --m 2 \
                       --log-dir $expdir
  exit 0
fi


# !!!note that we also need to apply the same pre-processing to decode data!!!
if [ $stage -le 7 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" --compress false \
    data/train data/train_no_sil exp/train_no_sil
  local/fix_data_dir.sh data/train_no_sil

  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" --compress false \
    data/voxceleb1_test data/voxceleb1_test_no_sil exp/voxceleb1_test_no_sil
  local/fix_data_dir.sh data/voxceleb1_test_no_sil

  cat data/train_no_sil/feats.scp > $log/decode_fixed.scp
  cat data/voxceleb1_test_no_sil/feats.scp >> $log/decode_fixed.scp
fi


log=exp/processed
mkdir -p $log/ivs/
ivs=$log/ivs/$expname/
mkdir -p $ivs
chmod 777 $expdir/*

# Network Decoding; do this for all your data
if [ $stage -le 8 ]; then
  model=$expdir/model_best.pth.tar # get best model
  $cuda_cmd $expdir/decode.log python scripts/decode.py \
                       --spk_num $num_spk --model 'resnet34' \
                       --input-dim 30 --hidden-dim 512 --D 32 \
                       --pooling 'mean' --network-type 'lde' \
                       --distance-type 'sqr' --asoftmax True --m 2 \
                       --model-path $model --decode-scp $log/decode_fixed.scp \
                       --out-path $ivs/embedding.ark
fi


train_utt2spk=data/train/utt2spk
train_spk2utt=data/train/spk2utt
test_utt2spk=data/voxceleb1_test/utt2spk
mkdir -p exp/backend/
backend_log=exp/backend/$expname/
mkdir -p $backend_log

# get train and test embeddings 
decode=$ivs/embedding.ark
if [ $stage -le 9 ]; then
    awk 'NR==FNR{a[$1]=1;next}{if($1 in a)print}' $test_utt2spk $decode > $backend_log/test.iv
    awk 'NR==FNR{a[$1]=1;next}{if($1 in a)print}' $train_utt2spk $decode > $backend_log/train.iv
fi


if [ $stage -le 10 ]; then
    # Compute the mean vector for centering the evaluation ivectors.
	$train_cmd $backend_log/compute_mean.log \
		ivector-mean ark:$backend_log/train.iv\
		$backend_log/mean.vec || exit 1;

    # This script uses LDA to decrease the dimensionality prior to PLDA.
    lda_dim=200
    $train_cmd $backend_log/lda.log \
        ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
        "ark:ivector-subtract-global-mean ark:$backend_log/train.iv ark:- |" \
        ark:$train_utt2spk $backend_log/transform.mat || exit 1;

    # Train the PLDA model.
    $train_cmd $backend_log/plda.log \
        ivector-compute-plda ark:$train_spk2utt \
        "ark:ivector-subtract-global-mean ark:$backend_log/train.iv ark:- | transform-vec $backend_log/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
        $backend_log/plda || exit 1;
fi


if [ $stage -le 11 ]; then
    $train_cmd $backend_log/voxceleb1_test_scoring.log \
        ivector-plda-scoring --normalize-length=true \
        "ivector-copy-plda --smoothing=0.0 $backend_log/plda - |" \
        "ark:ivector-subtract-global-mean $backend_log/mean.vec ark:$backend_log/test.iv ark:- | transform-vec $backend_log/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $backend_log/mean.vec ark:$backend_log/test.iv ark:- | transform-vec $backend_log/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $backend_log/scores_voxceleb1_test || exit 1;
fi


if [ $stage -le 12 ]; then
    eer=`compute-eer <(python local/prepare_for_eer.py $voxceleb1_trials $backend_log/scores_voxceleb1_test) 2> /dev/null`
    mindcf1=`python local/compute_min_dcf.py --p-target 0.01 $backend_log/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
    mindcf2=`python local/compute_min_dcf.py --p-target 0.001 $backend_log/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
    echo "EER: $eer%"
    echo "minDCF(p-target=0.01): $mindcf1"
    echo "minDCF(p-target=0.001): $mindcf2"
    # EER: 3.043%
    # minDCF(p-target=0.01): 0.3129
    # minDCF(p-target=0.001): 0.4291
fi
