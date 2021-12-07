#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Install PaddleSeg
get_ipython().system('pip install paddleseg')


# In[ ]:


# Download PaddleSeg codes
get_ipython().system('git clone https://gitee.com/PaddlePaddle/PaddleSeg.git')
# unzip the file
get_ipython().system('unzip PaddleSeg.zip')


# In[ ]:


##########
# fast track test
##########
# all command under PaddleSeg/contrib/HumanSeg
get_ipython().run_line_magic('cd', 'PaddleSeg/contrib/PP-HumanSeg')
# Download Inference Model
get_ipython().system('python export_model/download_export_model.py')
# Download test data
get_ipython().system('python data/download_data.py')


# In[ ]:


# human segmentation- with build-in camera realtime stream as the data source
get_ipython().system('python bg_replace.py --config ')
export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml


# In[ ]:


# human segmentation- with a mp4 file as the data source
get_ipython().system('python bg_replace.py --config export_model/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax/deploy.yaml --video_path data/video_test.mp4')


# In[6]:


get_ipython().run_line_magic('cd', 'PaddleSeg/contrib/PP-HumanSeg')
# background replacement
# replace the background for the realtime stream from the buildin cam, '--background_video_path' provides the background image.
# !python bg_replace.py \
# --config export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml \
# --input_shape 224 398 \
# --bg_img_path data/background.jpg

# # # replace the background for a mp4 file, '--background_video_path' provides the background image.
# !python bg_replace.py \
# --config export_model/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax/deploy.yaml \
# --bg_img_path data/background.jpg \
# --video_path data/video_test.mp4

# replace the background for a JPG pic, '--background_video_path' provides the background image.
get_ipython().system('python bg_replace.py --config export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml --input_shape 224 398 --img_path data/human_image.jpg --bg_img_path data/background.jpg')


# In[7]:


# download pre-trained models
get_ipython().system('python pretrained_model/download_pretrained_model.py')


# In[8]:


# Training
# finetune the model with data/mini_supervisely
get_ipython().system('export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡')
#===============================
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
#===============================
get_ipython().system('python train.py --config configs/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml --save_dir saved_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely --save_interval 100 --do_eval --use_vdl')


# In[9]:


# more help about training are available via following command
get_ipython().system('python train.py --help')


# In[10]:


# evaluation
get_ipython().system('python val.py --config configs/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml --model_path saved_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely/best_model/model.pdparams')


# In[ ]:





# In[11]:


# prediction, the output result is defaut to put in ./oputput/result/

get_ipython().system('python predict.py --config configs/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml --model_path saved_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely/best_model/model.pdparams --image_path data/human_image.jpg')


# In[12]:


# model export
get_ipython().system('export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡')
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
get_ipython().system('python ../../export.py --config configs/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml --model_path saved_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely/best_model/model.pdparams --save_dir export_model/fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely_with_softmax --without_argmax --with_softmax')
# command for ppseg-lite
get_ipython().system('python ../../export.py --config ../../configs/ppseg_lite/ppseg_lite_export_398x224.yml --save_dir export_model/ppseg_lite_portrait_398x224_with_softmax --model_path pretrained_model/ppseg_lite_portrait_398x224/model.pdparams --without_argmax --with_softmax')


# In[ ]:





# In[ ]:





# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
