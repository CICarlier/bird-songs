# Supervised and unsupervised analysis of xeno-canto's bird songs dataset


## How to get started
Here is the workflow to follow if you want to re-run this experiment from scratch:
1. **Files Download**

Start by opening the xeno-canto-download-py file. This file queries the xeno-canto.org API by querying for all birds listed in the United States and Canada with recordings of quality A or B 
(this will also returns files for which quality is unknown). It then filters the dataset of features to only keep the six birds we had selected for our analysis (Red-winged Blackbird, 
Common Yellowthroat, Northern Cardinal, Carolina Wren, Red Crossbill, Spotted Towhee). 
Modify the API queries if you are interested in learning about different species or want to restrict your queries to different file length or quality and save your changes. 
Refer to the [xeno-canto API documentation](https://www.xeno-canto.org/explore/api) for help on how to write your query. 
Then Run the download file in a command line shell or IDE interpreter shell:

`python xeno-canto-download-py`

Depending on your filters and internet speed, this may take a little while. 

2. **Noise Reduction and Resampling**

This step varies depending on your environment. We are providing the steps to run the denoising and resampling steps on an Ubuntu docker image running on Windows 10. 
If you are using Ubuntu, open a command shell and go directly to step e).
 a) Prerequisite for Windows 10 enviornment: install Docker and WSL2. 
 b) Open PowerShell
 c) Optional: It happens sometimes that if your computer goes to sleep the WSL clock gets misalign. To make sure you do not run into an issue, start WSL by typing `wsl` in your powershell command, then type `date`. 
 If the UTC date is not correct, use `hwclock -s` to reset time. 
 d) Start your docker container by typing `docker run --rm --entrypoint /bin/bash -v "{windowspath_to_your_repository}:/tmp/bird" -it ubuntu`.
 e) Validate that your files are in the tmpt/bird folder with `ls /tmp/bird`
 f) Optional: update ubuntu `apt update`
 g) Install sox `apt install -y sox libsox-fmt-mp3`
 h: Execute the noise reduction and resampling script. This script uses two sox functions for noise reduction: `noiseprof` create the noise profile of a file that will be used as a baseling to remove the noise, 
 `noisered` attenuate any signal below the threshold established by the moise profile. The noise profile is set at .20 here, but you can filter it further up or down. 
 Then the `-r` resampling code will harmonize all the files to a common rate and resolution of 22,050Htz. 
 As files may come from different media, they may have been recorded at different rates and this technique harmonizes them.
 ```
 mkdir -p /tmp/bird/audio_noise_reduction/
 for file in $(ls /tmp/bird/audio)
 do 
	sox /tmp/bird/audio/${file} -n noiseprof /tmp/noise.prof
	sox /tmp/bird/audio/${file} /tmp/clean-${file} noisered /tmp/noise.prof 0.20
	sox /tmp/clean-${file} -r 22050 /tmp/bird/audio_noise_reduction/resampled-clean-${file}
 done```
 
 Note that the SoX utility tool can be downloaded and installed on any platform (Windows, Mac, Linux). For more information on [Sound eXchange (SoX)](http://sox.sourceforge.net/), refer to [SoX documentation](http://sox.sourceforge.net/Docs/Documentation).

3. **Cutting and Mel-spectrograms generation**

The cutting and mel-spectrogram generation is done with the preprocessing.py file.
This step will perform two different transformations on the resampled files created at the precedent step:
- ***8-second files***: resampled files will be cut at 8 seconds. Files with a duration lesser than 8 seconds will be looped till they reach this length. This allows us to harmonize the lenght of all our files. 
- ***2-second clips***: resampled files are run through our clipping function which cuts each file into multiple sections where the amplitude is highest, indicating the likelyhood of birds singing at this precise moments. 
Clips are no longer than 2 seconds as most birds songs do not last longer than that. We augment our data through this process by selecting multiple clips within the same audio file.

Once our files all have the same sample rate and length, we generate their mel-spectrograms using the librosa library. As humans do not perceive frequencies on a linear scale, it is common to use a mel scale to represent pitches.
A mel-spectrogram is a time-frequency spectrogram where the frequencies are converted to the mel scale. This images represent what will be fed to our supervised and unsupervised learning models. 
Our 