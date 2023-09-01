#runnung fno.py for different reolutions on different gpus

import subprocess

resolutions = [128, 256, 512]
batch_size =  [10 , 10, 5]
gpu_infos =   [0, 2, 5]

for res, gpu_in, bs  in zip(resolutions, gpu_infos, batch_size):

    screen_name = 'mwt-darcyLN_'+str(res)#'inv_fno_'+str(res)
    command =  'python mwt.py --res %s --bs %s'%(res, bs)
    subprocess.run('screen -dmS '+screen_name, shell=True)
    
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'conda activate base'), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'export CUDA_VISIBLE_DEVICES=%s'%(gpu_in)), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, command), shell=True)

gpu_infos =   [ 5, 3, 6]

for res, gpu_in, bs  in zip(resolutions, gpu_infos, batch_size):

    screen_name = 'inv-mwt-darcyLN_'+str(res)#'inv_fno_'+str(res)
    command =  'python inv-mwt.py --res %s --bs %s'%(res, bs)
    subprocess.run('screen -dmS '+screen_name, shell=True)
    
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'conda activate base'), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'export CUDA_VISIBLE_DEVICES=%s'%(gpu_in)), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, command), shell=True)