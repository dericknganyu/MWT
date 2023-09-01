#runnung fno.py for different reolutions on different gpus

import subprocess

resolutions = [40]#[16, 32, 64, 128, 256, 512]
batch_size =  [10]#[10, 10, 10, 10 , 10, 5]
gpu_infos =   [2]#[ 0,  0,  1,  2 , 4, 7]

for res, gpu_in, bs  in zip(resolutions, gpu_infos, batch_size):

    screen_name = 'mwt-structuralMechanics_'+str(res)#'inv_fno_'+str(res)
    command =  'python cs-mwt.py --res %s --bs %s'%(res, bs)
    subprocess.run('screen -dmS '+screen_name, shell=True)
    
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'conda activate base'), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'export CUDA_VISIBLE_DEVICES=%s'%(gpu_in)), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, command), shell=True)

gpu_infos =   [2]#[ 0,  0,  1,  2 , 3, 7]

for res, gpu_in, bs  in zip(resolutions, gpu_infos, batch_size):

    screen_name = 'inv-mwt-structuralMechanics_'+str(res)#'inv_fno_'+str(res)
    command =  'python cs-inv-mwt.py --res %s --bs %s'%(res, bs)
    subprocess.run('screen -dmS '+screen_name, shell=True)
    
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'conda activate base'), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'export CUDA_VISIBLE_DEVICES=%s'%(gpu_in)), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, command), shell=True)