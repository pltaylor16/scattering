
import numpy as np 
import tensorflow as tf
from math import pi as pi
from math import log2 as log2



class Scatter2D():
    
    def __init__(self, J, L, M, slant, max_order = 2):

        self.J = J
        self.L = L 
        self.M = M 
        self.slant = slant
        self.max_order =max_order

        if 2 ** self.J > self.M:
            print ('ERROR: 2^J should be greater than M!')


    def get_js(self):
        mx = int(log2(self.M))
        self.js = list(range(mx+1))


    def get_thetas(self):
        self.thetas = []
        for l in range(self.L):
            self.thetas += [l/self.L * pi]



    def gabor_2d(self, xi, j, l):

        theta = self.thetas[l]

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



    def morlet_2d(self, xi, j, l):

        
        val = self.gabor_2d(xi, j, l)
        modulus = self.gabor_2d(0, j, l)
        K = np.sum(val) / np.sum(modulus)
  

        mor = val - K * modulus

        return mor 


    def generate_filters(self):
        filter_dict = {}
        self.get_js()
        self.get_thetas()


        for j in self.js:
            #hard-code this for now
            xi_j = 3. * pi / 4. / 2. ** j
            for l in range(self.L):

                filter_dict['j:%s,l:%s' %(j,l)] = self.morlet_2d(xi_j, j, l)

        return filter_dict


    def get_filters_tensor(self):
        
        filters_np = np.zeros((self.J, self.L, self.M, self.M), dtype=np.complex64)
        for j in range(self.J):
            for l in range(self.L):
                filters_np[j,l,:,:] = self.generate_filters()['j:%s,l:%s' %(j,l)]

        filters = tf.convert_to_tensor(filters_np, dtype= tf.complex64)
        
        return filters



    def compute_coefs(self, data_batch, filters):

        '''filter has dimesnions (J,L,M,M), data_batch has dimesnions \
        (batch_size,M,M)'''


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


                    filter_dict['j:%s,l_theta:%s,l_phi:%s' %(j,l_theta,l_phi)] = self.morlet_3d(xi_j, j, l_theta, l_phi)

        return filter_dict




    def compute_coeffecient(self):
        pass



