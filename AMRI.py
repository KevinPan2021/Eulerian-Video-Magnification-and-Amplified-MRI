import cv2
import scipy.signal as signal           
from skimage import img_as_ubyte     
import numpy as np
import pyrtools as pt
from tqdm import tqdm


# read mp4 to 2d grayscale numpy array
def mp4read(input_video):
    # reading input mp4
    cap = cv2.VideoCapture(input_video)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    input_data = np.array(frames, dtype=np.uint8)
    
    # only keep 1 channel for grayscale
    input_data = input_data[...,0]
    return input_data


# write 2d grayscale numpy array to .mp4
def mp4write(data, path, fps):
    # Define video properties
    frames, h, w = data.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
    
    # Create a VideoWriter object
    out = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=False)
    
    # Generate and write frames to the video file
    for i in range(frames):
        out.write(data[i,...])
    out.release()
    

# Eulerian Video Magnification for grayscale video using Butterworth filter
class AMRI():
    def __init__(self, input_video, output_video, 
                 alpha_c, lambda_c, fl, fh, samplingRate, exaggeration_factor=2):
        self.input_video = input_video # input video path
        self.output_video = f'{output_video}_alpha_c{alpha_c}_lambda_c{lambda_c}_fl{fl}_fh{fh}.mp4' # output video path
        self.alpha_c = alpha_c # alpha cutoff for magnifying the motion
        self.lambda_c = lambda_c # lambda cutoff for Laplacian EVM
        self.fl = fl # low omega
        self.fh = fh # high omega
        self.samplingRate = samplingRate # sampling rate in HZ
        self.exaggeration_factor = exaggeration_factor
    

    # build Laplacian pyramid
    def buildLpyr(self, img):
        pyramid = pt.pyramids.LaplacianPyramid(img)
        lpyr = np.concatenate([im.flatten() for im in pyramid.pyr_coeffs.values()])
        pind = np.array(list(pyramid.pyr_size.values()))
        return lpyr, pind
    
    
    # reconstruct Laplacian pyramid
    def reconLpyr(self, pyr, pind):
        levels = len(pind)
        # Precompute cumulative sums of sizes in pind for efficient slicing
        cumulative_sizes = np.cumsum(np.array([np.prod(j) for j in pind]))
        # Reverse loop to populate reshaped_pyr
        reshaped_pyr = [
            pyr[cumulative_sizes[k - 1] if k > 0 else 0 : cumulative_sizes[k]].reshape(pind[k])
            for k in range(levels - 1, -1, -1)
        ]
        
        # reconstruct
        maxLev = len(reshaped_pyr)
        levs = range(0,maxLev)  # The levels is range(0,maxLev)
        filt2 = pt.binomial_filter(5)  #The named Filter filt2 . This has been finalized here. 
        res = []
        
        reshaped_pyr = reshaped_pyr[::-1]
        
        for lev in range(maxLev-1, -1, -1):
            if lev in levs and len(res) == 0:
                res = reshaped_pyr[lev]
            elif len(res) != 0:
                res_sz = res.shape
                new_sz = reshaped_pyr[lev].shape
                if res_sz[0] == 1:
                    hi2 = pt.upConv(image=res, filt=filt2, step=(2,1), stop=(new_sz[1], new_sz[0])).T
                elif res_sz[1] == 1:
                    hi2 = pt.upConv(image=res, filt=filt2.T, step=(1,2), stop=(new_sz[1], new_sz[0])).T
                else:
                    hi = pt.upConv(image=res, filt=filt2, step=(2,1), stop=(new_sz[0], res_sz[1]))
                    hi2 = pt.upConv(image=hi, filt=filt2.T, step=(1,2), stop=(new_sz[0], new_sz[1]))
                
                if lev in levs:
                    bandIm =  reshaped_pyr[lev]
                    res = hi2 + bandIm
                else:
                    res = hi2
        return res    
    
    
    def run(self,):
        # input_data is RGB uint8 np array
        input_data = mp4read(self.input_video)
        
        # construct the output_data to store the results
        output_data = np.pad(input_data, ((0,0), (0,0), (0,input_data.shape[2])), mode='constant', constant_values=0)
        
        # convert range to float ranged [0, 1]
        input_data = input_data.astype(np.float32)/255
        
        ## Low_a, Low_B is being used. 
        [low_a,low_b] = signal.butter(1,self.fl/self.samplingRate,'low')   
        [high_a,high_b] = signal.butter(1,self.fh/self.samplingRate,'low')
    
        frame_count, width, height = input_data.shape
        
        # Setting video for writing. 
        rgbframe = input_data[0,...].copy()

        # building pyramid
        pyr, pind = self.buildLpyr(rgbframe)

        nLevels = len(pind) # number of pyramid levels
        
        # Lowpass1, Lowpass2, pyr_prev. 
        lowpass1, lowpass2, pyr_prev = pyr, pyr, pyr
        
        output = rgbframe
        
        # store the output data
        output_data[0,:,rgbframe.shape[1]:] = img_as_ubyte(rgbframe)
        
        # Printing the frames from startIndex+1 to end. 
        for i in tqdm(range(1, frame_count)):
            # extracting the current frame into YIQ color space
            rgbframe = input_data[i,...].copy()

            pyr, _ = self.buildLpyr(rgbframe)
            
        	# Filtering the signal.
            lowpass1 = (-high_b[1]*lowpass1 + high_a[0]*pyr + high_a[1]*pyr_prev)/high_b[0]
            lowpass2 = (-low_b[1]*lowpass2 + low_a[0]*pyr + low_a[1]*pyr_prev)/low_b[0]
            filtered = lowpass1-lowpass2 
        	
            pyr_prev = pyr
           
        	# for all levels we try to obtain the chrom attenutation. 
            delta = self.lambda_c/8/(1+self.alpha_c)
            lambd = np.sqrt(width**2 + height**2)/3
            for l in range(nLevels-1,-1,-1):
                # calculate the indices correspounding to current pyramid level
                startIndex = sum(np.prod(j) for j in pind[:l])
                endIndex = len(pyr) - sum(np.prod(j) for j in pind[l+1:])
                indices = range(startIndex, endIndex)   
                
                # compute the modified alpha for this level
                currAlpha = (lambd/delta/8 - 1)*self.exaggeration_factor
                                
                # if it is the first or the last level, set the filter to 0
                if l in [0, nLevels-1]:
                    filtered[indices] = 0 
                # amplification the temporal filter                                                               
                else:
                    filtered[indices] = min(currAlpha, self.alpha_c)*filtered[indices]
                
                # representative lambda will reduce by factor of 2
                lambd /= 2
           
        	# Reconstrion of the signal. 
            output = self.reconLpyr(filtered,pind)
            
        	# Rgbframe is being obtained here.
            output_final = np.clip(rgbframe + output, a_min=-1, a_max=1)
            
            # add to output_data
            output_data[i, :, output_final.shape[1]:] = img_as_ubyte(output_final)
            
        # writing to mp4 file
        mp4write(output_data, self.output_video, self.samplingRate)
    
    
def main():
    
    # alpha, lambda_c, fl, fh, samplingRate
    AMRI('data\\brain_MRI.mp4', 'output\\brain_MRI', 200, 100, 0.6, 1.8, 60).run()
    
    
if __name__ == '__main__':
    main()
