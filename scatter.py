
import numpy as np 
import tensorflow as tf
from math import pi as pi
from math import log2 as log2
from scipy.fft import fft2, ifft2



class Scatter2D():
    
    def __init__(self, J, L, M, slant):

        self.J = J
        self.L = L 
        self.M = M 
        self.slant = slant

        if 2 ** self.J > self.M:
            print ('ERROR: 2^J should be greater than M!')


    #deprecated
    '''
    def get_js(self):
        mx = int(log2(self.M))
        self.js = list(range(mx+1))


    def get_thetas(self):
        self.thetas = []
        for l in range(self.L):
            self.thetas += [l/self.L * pi]
    '''


    ###################################
    ### Wavelet functions #############
    ###################################


    def gabor_2d(self, xi, j, theta):


        #rotations
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float64)
        R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float64)


        #covariances
        sigma = 0.8 * 2. ** j
        Sigma_base = np.array([[sigma ** 2, 0], [0, sigma ** 2 / self.slant ** 2]])
        Sigma = np.dot(R, np.dot(Sigma_base, R_inv))
        #Sigma_inv = np.linalg.inv(Sigma)


        #k_0_vector
        k_0= xi * np.dot(R,np.array([0,1]))

        #normalising factor
        norm = np.linalg.det(Sigma)

        #gabor on the grid
        gab = np.zeros((self.M, self.M), dtype = np.complex64)
        k_0 =  k_0 / (2 * pi)
        for kx in range(self.M):
            for ky in range(self.M):
                k = np.array([kx,ky]) / (2. * pi)
                exponent = - np.dot((k - k_0).T, np.dot(Sigma,(k-k_0))) /2
                gab[kx,ky] = np.exp(exponent)
   

        return gab 



    def morlet_2d(self, xi, j, theta):

        
        val = self.gabor_2d(xi, j, theta)
        modulus = self.gabor_2d(0, j, theta)
        K = np.sum(val) / np.sum(modulus)
  

        mor = val - K * modulus

        return mor 



    def periodize_filter_fft(self, x, res):
        #taken directly from kymatio filter_bank.py
        """
            Parameters
            ----------
            x : numpy array
                signal to periodize in Fourier
            res :
                resolution to which the signal is cropped.

            Returns
            -------
            crop : numpy array
                It returns a crop version of the filter, assuming that
                 the convolutions will be done via compactly supported signals.
        """
        M = x.shape[0]
        N = x.shape[1]

        crop = np.zeros((M // 2 ** res, N // 2 ** res), x.dtype)

        mask = np.ones(x.shape, np.float32)
        len_x = int(M * (1 - 2 ** (-res)))
        start_x = int(M * 2 ** (-res - 1))
        len_y = int(N * (1 - 2 ** (-res)))
        start_y = int(N * 2 ** (-res - 1))
        mask[start_x:start_x + len_x,:] = 0
        mask[:, start_y:start_y + len_y] = 0
        x = np.multiply(x,mask)

        for k in range(int(M / 2 ** res)):
            for l in range(int(N / 2 ** res)):
                for i in range(int(2 ** res)):
                    for j in range(int(2 ** res)):
                        crop[k, l] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res)]

        return crop


    def filter_bank(self):
        filters = {}
        filters['psi'] = []

        for j in range(self.J):
            for theta in range(self.L):
                psi = {'levels': [], 'j': j, 'theta': theta}
                psi_signal = self.morlet_2d(3.0 / 4.0 * np.pi /2**j, j, (int(self.L-self.L/2-1)-theta) * np.pi / self.L)
                psi_signal_fourier = np.real(fft2(psi_signal))
                # drop the imaginary part, it is zero anyway
                psi_levels = []
                for res in range(min(j + 1, max(self.J - 1, 1))):
                    psi_levels.append(self.periodize_filter_fft(psi_signal_fourier, res))
                psi['levels'] = psi_levels
                filters['psi'].append(psi)

        phi_signal = self.gabor_2d(0, self.J-1, 0)
        phi_signal_fourier = np.real(fft2(phi_signal))
        # drop the imaginary part, it is zero anyway
        filters['phi'] = {'levels': [], 'j': self.J}
        for res in range(self.J):
            filters['phi']['levels'].append(
                self.periodize_filter_fft(phi_signal_fourier, res))

        return filters

    #this is my own version to get dimensions right
    def padded_filter_bank(self):
        #this is adopted from base_frontend.py in kymatio
        M = self.M
        M_padded, N_padded = self.compute_padding()
        self.M = M_padded
        filters = self.filter_bank()
        #reset M after computing the filters with the correct padding
        self.M = M

        return filters


    ###################################
    ### Utility functions #############
    ###################################



    #utility function
    def subsample_fourier(self, data_batch, k):
        #k will be 2^j and and x will be (batch_size, M, M)
        batch_size = np.shape(data_batch)[0]
        y = tf.reshape(data_batch, shape = (batch_size, k, data_batch.shape[1] // k, k, data_batch.shape[2] // k))
        return tf.math.reduce_mean(y, axis = (1,3))


    #utility function
    def compute_padding(self):
        #taken from the kymatio utils.py except I am assuming M = N
        M_padded = ((self.M + 2 ** self.J) // 2 ** self.J + 1) * 2 ** self.J
        N_padded = ((self.M + 2 ** self.J) // 2 ** self.J + 1) * 2 ** self.J
        return M_padded, N_padded


    #utility function
    def unpad(self, x):
        #taken from numpy_backend.py in kymatio
        return x[..., 1:-1, 1:-1]


    #utility function

    ''' 
    #this numpy version doesn't work
    def pad(self, x, pad_size):
        #adopted from numpy backend
        paddings = ((0, 0),)      
        paddings += ((pad_size[0], pad_size[1]), (pad_size[2], \
            pad_size[3]))
        return tf.pad(x, paddings, mode ='reflect')
    '''

    #utility function
    def pad(self, x, pad_size, input_size):
        pad_size = list(pad_size)

        # Clone to avoid passing on modifications.
        new_pad_size = list(pad_size)

        # This handles the case where the padding is equal to the image size.
        if pad_size[0] == input_size[0]:
            new_pad_size[0] -= 1
            new_pad_size[1] -= 1
        if pad_size[2] == input_size[1]:
            new_pad_size[2] -= 1
            new_pad_size[3] -= 1

        paddings = [[0, 0]] * len(x.shape[:-2])
        paddings += [[new_pad_size[0], new_pad_size[1]], [new_pad_size[2], new_pad_size[3]]]

        x_padded = tf.pad(x, paddings, mode="REFLECT")

        # Again, special handling for when padding is the same as image size.
        if pad_size[0] == input_size[0]:
            x_padded = tf.concat([tf.expand_dims(x_padded[..., 1, :], axis=-2), x_padded, tf.expand_dims(x_padded[..., x_padded.shape[-2] -2, :], axis=-2)], axis=-2)
        if pad_size[2] == input_size[1]:
            x_padded = tf.concat([tf.expand_dims(x_padded[..., :, 1], axis=-1), x_padded, tf.expand_dims(x_padded[..., :,  x_padded.shape[-1]-2], axis=-1)], axis=-1)

        return x_padded


    #utility function
    def rfft(self, x):
        return tf.signal.fft2d(tf.cast(x, tf.complex64))

    #utility function
    def irfft(self, x):
        return tf.math.real(tf.signal.ifft2d(x))

    #utility function
    def ifft(self, x):
        return tf.signal.ifft2d(x)

    def cdgmm(self, A, B):
        return A * B 

    #utility function
    #not sure I will need this
    def stack(self, arrays):
        return tf.stack(arrays, axis=-3)

    def modulus(self, x):
        return tf.abs(x)


    ###################################
    ###compute the final output########
    ###################################


    def compute_coefs(self, x, phi, psi, max_order, out_type = 'array'):
        
        #taken from /core/scattering2d.py


        # Define lists for output.
        out_S_0, out_S_1, out_S_2 = [], [], []

        #this is mine...
        M_padded, N_padded = self.compute_padding()
        pad_size = [(M_padded - self.M) // 2, (M_padded - self.M+1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2]
        input_size = [self.M, self.M]

        #back to code
        U_r = self.pad(x, pad_size, input_size)
        #print ('U_r:', np.shape(U_r))

        U_0_c = self.rfft(U_r)

        #print ('U_0_c:', np.shape(U_r))
        #print ('phi:', np.shape(phi['levels'][0]))

        #first low pass filter
        U_1_C = self.cdgmm(U_0_c, phi['levels'][0])
        U_1_C = self.subsample_fourier(U_1_C, 2 ** self.J)

        S_0 = self.irfft(U_1_C)
        S_0 = self.unpad(S_0)

        out_S_0.append({'coef': S_0,
                'j': (),
                'n': (),
                'theta': ()})


        for n1 in range(len(psi)):
            j1 = psi[n1]['j']
            theta1 = psi[n1]['theta']

            U_1_c = self.cdgmm(U_0_c, psi[n1]['levels'][0])
            if j1 > 0:
                U_1_c = self.subsample_fourier(U_1_c, k=2 ** j1)
            U_1_c = self.ifft(U_1_c)
            U_1_c = self.modulus(U_1_c)
            U_1_c = self.rfft(U_1_c)

            # Second low pass filter
            S_1_c = self.cdgmm(U_1_c, phi['levels'][j1])
            S_1_c = self.subsample_fourier(S_1_c, k=2 ** (self.J - j1))

            S_1_r = self.irfft(S_1_c)
            S_1_r = self.unpad(S_1_r)

            out_S_1.append({'coef': S_1_r,
                            'j': (j1,),
                            'n': (n1,),
                            'theta': (theta1,)})

            if max_order < 2:
                continue
            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                theta2 = psi[n2]['theta']

                if j2 <= j1:
                    continue

                U_2_c = self.cdgmm(U_1_c, psi[n2]['levels'][j1])
                U_2_c = self.subsample_fourier(U_2_c, k=2 ** (j2 - j1))
                U_2_c = self.ifft(U_2_c)
                U_2_c = self.modulus(U_2_c)
                U_2_c = self.rfft(U_2_c)

                # Third low pass filter
                S_2_c = self.cdgmm(U_2_c, phi['levels'][j2])
                S_2_c = self.subsample_fourier(S_2_c, k=2 ** (self.J - j2))

                S_2_r = self.irfft(S_2_c)
                S_2_r = self.unpad(S_2_r)

                out_S_2.append({'coef': S_2_r,
                                'j': (j1, j2),
                                'n': (n1, n2),
                                'theta': (theta1, theta2)})

        out_S = []
        out_S.extend(out_S_0)
        out_S.extend(out_S_1)
        out_S.extend(out_S_2)

        if out_type == 'array':
            out_S = self.stack([x['coef'] for x in out_S])

        return out_S

    #deprecated
    '''
    def generate_filters(self):
        filter_dict = {}
        self.get_js()
        self.get_thetas()


        for j in self.js:
            #hard-code this for now
            xi_j = 3. * pi / 4. / 2. ** j
            for l in range(self.L):

                filter_dict['psi,j:%s,l:%s' %(j,l)] = self.morlet_2d(xi_j, j, l)
                filter_dict['phi,j:%s,l:%s' %(j,l)] = self.gabor_2d(0., self.J-1, l)

        return filter_dict


    def get_filters_tensor(self):
        
        filters_np_phi = np.zeros((self.J, self.L, self.M, self.M), dtype=np.complex64)
        filters_np_psi = np.zeros((self.J, self.L, self.M, self.M), dtype=np.complex64)
        for j in range(self.J):
            for l in range(self.L):
                filters_np_phi[j,l,:,:] = self.generate_filters()['phi,j:%s,l:%s' %(j,l)]
                filters_np_psi[j,l,:,:] = self.generate_filters()['psi,j:%s,l:%s' %(j,l)]


        phi = tf.convert_to_tensor(filters_np_phi, dtype= tf.complex64)
        psi = tf.convert_to_tensor(filters_np_psi, dtype= tf.complex64)

        
        return phi, psi
    '''



    
    '''
    def compute_coefs(self, data_batch, filters):

        #filter has dimesnions (J,L,M,M), data_batch has dimesnions \
        (batch_size,M,M)


        #convert input to complex tensor
        dat = tf.convert_to_tensor(data_batch, dtype= tf.complex64)
        batch_size = np.shape(data_batch)[0]
        del data_batch

        #Fourier transform
        dat_f = tf.signal.fft2d(dat) 

        #compute 
        M1 = tf.zeros((batch_size, self.J, self.L, self.M, self.M), dtype = tf.complex64)
        S1 = tf.zeros((batch_size, self.J, self.L), dtype = tf.complex64)

        M1 = tf.signal.ifft2d(dat_f[:,None,None,:,:] * filters[None,:,:,:,:])
        S1 = tf.math.reduce_mean(M1, axis = (3,4))

        return S1
    '''
    






class Scatter3D():
    
    def __init__(self, J, L, M, slant1, slant2, max_order = 2):

        self.J = J
        self.L = L 
        self.M = M 
        self.slant1 = slant1
        self.slant2 = slant2
        self.max_order = max_order

        if 2 ** self.J > self.M:
            print ('ERROR: 2^J should be greater than M!')


    def get_js(self):
        mx = int(log2(self.M))
        self.js = list(range(mx+1))


    def get_thetas(self):
        self.thetas = []
        for l in range(self.L):
            self.thetas += [l/self.L * pi]

    def get_phis(self):
        self.phis = []
        for l in range(self.L):
            self.phis += [l/self.L * pi]



    def gabor_3d(self, xi, j, l_theta, l_phi):

        theta = self.thetas[l_theta]
        phi = self.phis[l_phi]


        #rotations
        R_1 = np.array([[np.cos(theta), -np.sin(theta), 0.],\
            [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]], np.float64)
        R_2 = np.array([[np.cos(phi), 0., -np.sin(phi)], \
            [0., 1., 0.], [np.sin(phi), 0., np.cos(phi)]], np.float64)
        R = np.dot(R_1, R_2)
        R_T = R.T

        #covariances
        sigma = 0.8 * 2. ** j
        Sigma_base = np.array([[sigma ** 2, 0, 0], [0, sigma ** 2 / self.slant1 ** 2, 0], [0, 0, sigma ** 2 / self.slant2 ** 2]])
        Sigma = np.dot(R, np.dot(Sigma_base, R_T))
        #Sigma_inv = np.linalg.inv(Sigma)


        #k_0_vector
        k_0= xi * np.dot(R,np.array([1,0,0]))

        #normalising factor
        norm = np.linalg.det(Sigma)

        #gabor on the grid
        gab = np.zeros((self.M, self.M, self.M), dtype = np.complex64)
        k_0 =  k_0 / (2 * pi)
        for kx in range(self.M):
            for ky in range(self.M):
                for kz in range(self.M):
                    k = np.array([kx,ky,kz]) / (2. * pi)
                    exponent = - np.dot((k - k_0).T, np.dot(Sigma,(k-k_0))) / 2.
                    gab[kx,ky,kz] = np.exp(exponent)
   

        return gab 



    def morlet_3d(self, xi, j, l_theta, l_phi):

        
        val = self.gabor_3d(xi, j, l_theta, l_phi)
        modulus = self.gabor_3d(0., j, l_theta, l_phi)
        K = np.sum(val) / np.sum(modulus)
  

        mor = val - K * modulus

        return mor 


    def generate_filters(self):
        filter_dict = {}
        self.get_js()
        self.get_thetas()
        self.get_phis()


        for j in self.js:
            #hard-code this for now
            xi_j = 3. * pi / 4. / 2. ** j
            for l_theta in range(self.L):
                for l_phi in range(self.L):


                    filter_dict['psi,j:%s,l_theta:%s,l_phi:%s' %(j,l_theta,l_phi)] = self.morlet_3d(xi_j, j, l_theta, l_phi)
                    filter_dict['phi,j:%s,l_theta:%s,l_phi:%s' %(j,l_theta,l_phi)]  = self.gabor_2d(0., self.J-1, l_theta, l_phi)

        return filter_dict




    def compute_coeffecient(self):
        pass








