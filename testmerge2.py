import numpy as np
import scipy.fftpack as spf
from numpy.fft import fft2, fftshift, ifft2,ifftshift
import mrcfile
import cv2 as cv
from scipy import misc
import matplotlib.pyplot as plt

class ExposureFilter:
    def __init__(self,img, acceleration_voltage, critical_dose_scaling, critical_dose_power, critical_dose_a, critical_dose_b,critical_dose_c, voltage_scaling_factor):
        self.img = img
        self.acceleration_volatage = acceleration_voltage
        self.critical_dose_scaling = critical_dose_scaling
        self.critical_dose_power = critical_dose_power
        self.critical_dose_a = critical_dose_a
        self.critical_dose_b = critical_dose_b
        self.critical_dose_c = critical_dose_c
        self.voltage_scaling_factor = voltage_scaling_factor
    def DoseFilter(self,dose_at_end_of_frame, critical_dose):
        #compute dose filter, which is the signal attenuation factor due to radiation damage
        dose_filter = np.exp((-0.5*dose_at_end_of_frame)/critical_dose)
        return dose_filter
    def CriticalDose(self, spatial_frequency):
        #given a spatial frequency, return the critical dose in electrons per square angstroms
        critical_dose = (self.critical_dose_a*(spatial_frequency**self.critical_dose_b)+self.critical_dose_c)*self.voltage_scaling_factor
        return critical_dose
    def SignalToNoiseFromDoseGivenCriticalDose(self, dose, critical_dose):
        #given a number of electrons and a critical dose, return the snr
        if dose == 0.0:
            snr = 0.0
        else:
            snr = (1-np.exp(-dose*0.5/critical_dose))**2/dose
        return snr
    def OptimalDoseGiven(self,critical_dose):
        #given the critical dose, return an estimate of the optimal dose
        optimal_dose = 2.51284 * critical_dose
        return optimal_dose
    def GetDoseFilter(self, dose_start, dose_finish):
        #apply dose filter to the image
        num_of_img = self.img.allimg().shape[0]
        dose_per_frame = (dose_finish-dose_finish)/num_of_img
        current_critical_dose = 0
        critical_dose_at_dc = 10e35
        all_filters = np.zeros([num_of_img, self.img.allimg().shape[1],self.img.allimg().shape[2]])
        all_fft = np.zeros([num_of_img, self.img.allimg().shape[1],self.img.allimg().shape[2]], dtype=np.complex_)
        for i in range(num_of_img):
            print(i)
            dose_finish += dose_per_frame
            #do FFT of the image
            single_frame = self.img.allimg()[i]
            F = fft2(single_frame)
            F = fftshift(F)
            xfreq = np.fft.fftfreq(single_frame.shape[0],d = self.img.pixel_distance())
            xfreq = fftshift(xfreq)
            yfreq = np.fft.fftfreq(single_frame.shape[1],d = self.img.pixel_distance())
            yfreq = fftshift(yfreq)
            yfreq = yfreq*(-1)
            for j in range(len(xfreq)):
                for k in range(len(yfreq)):
                    if xfreq[j] == 0.0 and yfreq[k] == 0.0:
                        current_critical_dose = critical_dose_at_dc
                    else:
                        current_critical_dose = self.CriticalDose(np.sqrt(xfreq[j]**2+yfreq[k]**2))
                    current_optimal_dose = self.OptimalDoseGiven(current_critical_dose)
                    #print(current_critical_dose)
                    #if (abs(dose_finish-current_optimal_dose) < abs(dose_start-current_optimal_dose)):
                    if True:
                        dose_filter = self.DoseFilter(dose_finish, current_critical_dose)
                        #print(dose_filter)
                        all_filters[i,j,k] = dose_filter
                        new_val = dose_filter * F[j,k]
                        all_fft[i,j,k] = new_val
                    else:
                        all_filters[i,j,k] = 0
                        all_fft[i,j,k] = 0
            dose_start += dose_per_frame
        return all_filters, all_fft
class MovieImage:
    def __init__(self, mrcfile, pixel_spacing, acceleration_voltage):
        self.mrcfile = mrcfile
        self.pixel_spacing = pixel_spacing
        self.acceleration_voltage = acceleration_voltage
    def allimg(self):
        with mrcfile.open(self.mrcfile) as f:
            movie = f.data
        return movie
    def pixel_distance(self):
        return self.pixel_spacing


W = 52          # window size is WxW
C_Thr = 0.43    # threshold for coherency
LowThr = 35     # threshold1 for orientation, it ranges from 0 to 180
HighThr = 57    # threshold2 for orientation, it ranges from 0 to 180
def calcGST(inputIMG, w):
    img = inputIMG.astype(np.float32)
    # GST components calculation (start)
    # J =  (J11 J12; J12 J22) - GST
    imgDiffX = cv.Sobel(img, cv.CV_32F, 1, 0, 3)
    imgDiffY = cv.Sobel(img, cv.CV_32F, 0, 1, 3)
    imgDiffXY = cv.multiply(imgDiffX, imgDiffY)
    
    imgDiffXX = cv.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv.multiply(imgDiffY, imgDiffY)
    J11 = cv.boxFilter(imgDiffXX, cv.CV_32F, (w,w))
    J22 = cv.boxFilter(imgDiffYY, cv.CV_32F, (w,w))
    J12 = cv.boxFilter(imgDiffXY, cv.CV_32F, (w,w))
    return J11, J22, J12
def CalcBase(img, std):
    lowpass = ndimage.gaussian_filter(img, std)
    gauss_highpass = img - lowpass
    return gauss_highpass

def ReadMovement(file, num_of_frames, num_of_patches, num_of_row, num_of_col):
    movement_mat_x = np.zeros([num_of_frames, num_of_row, num_of_col])
    movement_mat_y = np.zeros([num_of_frames, num_of_row, num_of_col])
    error_mat = np.zeros([num_of_frames, num_of_row, num_of_col])
    patches = -1
    patch_row_col = np.sqrt(num_of_patches)
    patch_row_width = num_of_row//patch_row_col
    patch_col_width = num_of_col//patch_row_col
    with open(file,'r') as f:
        for l in f:
            if not l.startswith('#'):
                l = l.split()
                if len(l) == 0:
                    patches += 1
                else:
                    frame = int(l[0])-1
                    #print(frame)
                    #center_coord = np.array([float(l[1]), float(l[2])])
                    #print(center_coord)
                    measured = np.array([float(l[3]),float(l[4])])
                    interpolated = np.array([float(l[5]),float(l[6])])
                    #print(int(float(l[2])))
                    #print(int(float(l[2]))-patch_row_width//2)
                    movement_mat_x[frame, int(int(float(l[2]))*2-patch_row_width//2):int(int(float(l[2]))*2+patch_row_width//2), int(int(float(l[1])*2)-patch_col_width//2):int(int(float(l[1]))*2+patch_col_width//2)]= float(l[5])
                    movement_mat_y[frame, int(float(l[2])*2-patch_row_width//2):int(float(l[2])*2+patch_row_width//2), int(float(l[1])*2-patch_col_width//2):int(float(l[1])*2+patch_col_width//2)]= float(l[6])
                    distance = np.linalg.norm(measured-interpolated)
                    #errors[frame,patches] = distance
                    error_mat[frame, int(float(l[2])*2-patch_row_width//2):int(float(l[2])*2+patch_row_width//2), int(float(l[1])*2-patch_col_width//2):int(float(l[1])*2+patch_col_width//2)] = distance
    f.close()
    return movement_mat_x, movement_mat_y, error_mat

def MovementThres(movement_mat_x, movement_mat_y, error, s1, s2):
    s_val = error.copy()
    total_mov = np.sqrt(np.power(movement_mat_x,2)+np.power(movement_mat_y,2))
    num_of_frames= movement_mat_x.shape[0]
    for i in range(num_of_frames):
        mean_mov = np.mean(total_mov[i,:,:])
        std_mov = np.std(total_mov[i,:,:])
        upper = mean_mov + 2*std_mov
        lower = mean_mov - 2*std_mov
        mean_error = np.mean(error[i,:,:])
        std_error = np.std(error[i,:,:])
        for j in range(movement_mat_x.shape[1]):
            for k in range(movement_mat_x.shape[2]):
                if total_mov[i,j,k] > upper or total_mov[i,j,k] < lower:
                    if error[i,j,k] > mean_error+std_error or error[i,j,k] < mean_error-std_error:
                        s_val[i,j,k] = s1
                    else:
                        s_val[i,j,k] = s2
                else:
                    s_val[i,j,k] = s2
    return s_val
class PixelOperation:
    def __init__(self, J11, J12, J22):
        self.J11 = J11
        self.J12 = J12
        self.J22 = J22

    def gst(self):
        gst_mat = np.array([[self.J11, self.J12],[self.J12, self.J22]])
        return gst_mat
    def DirectionVector(self, gst):
        #each column is an eigenvector
        w,v = np.linalg.eig(gst)
        #print(w)
        return w, v
    def ComputeKs(self, w, v, kdetail, kshrink, kstretch, knoise, Dth, Dtr):
        A = 1+np.sqrt(w[0]/w[1])
        D = clamp(1-np.sqrt(w[0])/Dtr,0,1)
        k1hat = kdetail*(kstretch*A)
        k2hat = kdetail/(kshrink*A)
        k1 = ((1-D)*k1hat+D*kdetail*knoise)**2
        k2 = ((1-D)*k2hat+D*kdetail*knoise)**2
        return k1, k2
    def ComputeO(self, k1, k2, v):
        kmat = np.diag([k1, k2])
        o1 = np.matmul(v, kmat)
        o2 = np.matmul(o1, np.transpose(v))

        invo2 = np.linalg.pinv(o2)
        return invo2

class ImageOperation:
    def __init__(self, img_stack):
        self.img_stack = img_stack
    def SlidingWindow(self,image, stepSize, windowSize):
    # slide a window across the image
        for y in range(1, image.shape[0]-1, stepSize):
            for x in range(1, image.shape[1]-1, stepSize):
            # yield the current window
                yield (x, y, image[y-windowSize[1]//2:y + windowSize[1]//2+1, x-windowSize[0]//2:x + windowSize[0]//2+1])
    def CalcMeanStd(self, stepSize, windowSize, guide_image):
        num_of_img = self.img_stack.shape[0]
        pixel_mean = self.img_stack.copy()
        pixel_std = self.img_stack.copy()
        guide_mean = guide_image.copy()
        guide_std = guide_image.copy()
        guide_image = np.pad(guide_image, ((1,1),(1,1)), 'constant')
        
        for (x, y, window) in self.SlidingWindow(guide_image, stepSize=stepSize, windowSize= windowSize):
            window_mean = np.mean(window)
            window_std = np.std(window)
            guide_mean[y-1, x-1] = window_mean
            guide_std[y-1, x-1] = window_std
        for i in range(num_of_img):
            single_img = self.img_stack[i]
            single_img = np.pad(single_img,((1,1),(1,1)), 'constant')
            #print(single_img)
            for (x, y, window) in self.SlidingWindow(single_img, stepSize=stepSize, windowSize= windowSize):
                #print(window)
                #print(x)
                #print(y)
                window_mean = np.mean(window)
                window_std = np.std(window)
                pixel_mean[i, y-1, x-1] = window_mean
                if guide_std[y-1, x-1] > window_std:
                    pixel_std[i, y-1, x-1] = guide_std[y-1,x-1]
                else:
                    pixel_std[i, y-1, x-1] = window_std


                
        


        return pixel_mean, pixel_std, guide_mean
    def CalcDiff(self, pixel_mean, pixel_std, guide_mean):
        num_of_img = pixel_mean.shape[0]
        base_mean = guide_mean
        dms = pixel_mean.copy()
        dmd = pixel_mean.copy()
        for i in range(num_of_img):
            current_mean = pixel_mean[i,:,:]
            other_means = pixel_mean[np.arange(num_of_img)!=i,:,:]
            #print(other_means.shape)
            other_mean_mean = np.mean(other_means, axis = 0)
            dmd[i,:,:] = current_mean - other_mean_mean
            dms[i,:,:] = current_mean - guide_mean
        return dms, dmd
    
    def FinalD(self, dms, dmd):
        dms2 = np.power(dms,2)
        dmd2 = np.power(dmd, 2)
        frac = dms2/(dms2+dmd2)
        d = dms*frac
        return d
    
    
    def CalcR(self, s, d, sig, t):
        R = s*np.exp(-(np.power(d, 2)/np.power(sig, 2)))-t
        return R

def Merge(window_dis,img_stack, R, J11, J12, J22,kdetail, kshrink, kstretch, knoise, Dth, Dtr):
    #padding image and R for boundary cases
    num_of_img = img_stack.shape[0]
    merged_img = np.zeros([img_stack.shape[1],img_stack.shape[2]])
    r = img_stack.shape[1]
    c = img_stack.shape[2]
    padded_img = np.pad(img_stack,((0,0),(1,1),(1,1)),'constant')
    padded_R = np.pad(R, ((0,0),(1,1),(1,1)),'constant')
    for i in range(1, padded_img.shape[1]-1):
        for j in range(1, padded_img.shape[2]-1):
            numerator = 0
            denom = 0
            for k in range(num_of_img):
                #print(k)
                pix_J11 = J11[k,i-1,j-1]
                pix_J12 = J12[k,i-1,j-1]
                pix_J22 = J22[k,i-1,j-1]
                pix_op = PixelOperation(pix_J11, pix_J12, pix_J22)
                pix_gst = pix_op.gst()
                w,v = pix_op.DirectionVector(pix_gst)
                k1, k2 = pix_op.ComputeKs(w,v,kdetail, kshrink, kstretch, knoise, Dth, Dtr)
                invo = pix_op.ComputeO(k1, k2, v)
                ws = np.zeros(9)
                for it, dis in enumerate(window_dis):
                    mult = np.matmul(np.matmul(dis, invo), dis)
                    w_i = np.exp(-0.5*mult)
                    ws[it] = w_i
                ws = ws.reshape([3,3])
                #print(invo)
                #print(ws)
                img_win = padded_img[k,i-1:i+2, j-1:j+2]
                R_win = padded_R[k,i-1:i+2, j-1:j+2]
                #print(R_win)
                num = ws*img_win*R_win
                den = ws*R_win
                num = np.sum(num)
                den = np.sum(den)
                numerator += num
                denom += den
            #print(denom)
            merged_img[i-1,j-1] = numerator/denom
    return merged_img

def ReadError(file, num_of_frames, num_of_patches):
    errors = np.zeros([num_of_frames, num_of_patches])
    patches = -1
    with open(file,'r') as f:
        for l in f:
            if not l.startswith('#'):
                l = l.split()
                if len(l) == 0:
                    patches += 1
                else:
                    frame = int(l[0])-1
                    measured = np.array([float(l[3]),float(l[4])])
                    interpolated = np.array([float(l[5]),float(l[6])])
                    distance = np.linalg.norm(measured-interpolated)
                    errors[frame,patches] = distance
    f.close()
    return errors

def clamp(x, lower, upper):
    if x < lower:
        x = lower
    elif x > upper:
        x = upper
    return x

def main():
    error_file = '/home/home2/whuang/research/test.log0-Patch-Patch.log'
    img_file = '/home/home2/whuang/research/test2_Stk.mrc'
    window_dis = np.array([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[1,1],[1,0],[1,1]])
    with mrcfile.open('test2_Stk.mrc',permissive=True) as f1:
        dat = f1.data
    test_img = dat[:,dat.shape[1]//4:dat.shape[1]//2, dat.shape[2]//4:dat.shape[2]//2]
    interpolated_test_img = np.zeros([test_img.shape[0],test_img.shape[1]*2, test_img.shape[2]*2])
    for i in range(test_img.shape[0]):
        interpolated = cv.resize(test_img[i], (test_img.shape[2]*2, test_img.shape[1]*2))
        #print(interpolated.shape)
        interpolated_test_img[i,:,:] = interpolated
    print('done interpolating image')
    guide_img = interpolated_test_image[test_img.shape[0]//2,:,:]
    guide_img = CalcBase(guide_img, 2)
    
    op = ImageOperation(interpolated_test_img)
    J11 = interpolated_test_img.copy()
    J12 = interpolated_test_img.copy()
    J22 = interpolated_test_img.copy()
    for i in range(interpolated_test_img.shape[0]):
        j11s,j22s,j12s = calcGST(interpolated_test_img[i,:,:], 5)
        #print(j11s.shape)
        J11[i,:,:] = j11s
        J12[i,:,:] = j12s
        J22[i,:,:] = j22s
    print('done calculate gradient structure tensor')
    kdetail = 0.25
    kshrink = 1
    kstretch = 4
    knoise = 6.0
    Dth = 0.0005
    Dtr = 0.0003
    pixel_mean, pixel_std, guide_mean = op.CalcMeanStd(1, [3,3], guide_img)
    dms, dmd = op.CalcDiff(pixel_mean,pixel_std, guide_mean)
    d = op.FinalD(dms, dmd)
    x, y, e = ReadMovement(error_file,50,25, 1200*2, 1240*2)
    sval = MovementThres(x, y, e, 12, 2)
    sval = sval[:,sval.shape[1]//4:sval.shape[1]//2, sval.shape[2]//4:sval.shape[2]//2]
    #interpolated_sval = np.zeros([sval.shape[0], sval.shape[1]*2, sval.shape[2]*2])

    R = op.CalcR(sval, d, pixel_std, 0)
    print('start merging')
    merged = Merge(window_dis, interpolated_test_img, R, J11, J12, J22,kdetail, kshrink, kstretch, knoise, Dth, Dtr)
    plt.imsave('/home/home2/whuang/research/merged2.png',merged)
    np.save('/home/home2/whuang/research/merged2.npy', merged)
    merged_mrc = '/home/home2/whuang/researh/merged2.mrc'
    with mrcfile.new(merged_mrc) as mrc:
        mrc.set_data(np.zeros([merged.shape[0],merged.shape[1]], dtype = np.float32))
        mrc.data[:,:] = merged

if __name__== "__main__":
    
    main()





