
import numpy as np 
import tensorflow as tf
from math import pi as pi
from math import log2 as log2
from scipy.fft import fft2, ifft2, fftn
import cv2
import random
import multiprocessing



class Scatter2D():
    
    def __init__(self, J, L, M, slant):

        self.J = J
        self.L = L 
        self.M = M 
        self.slant = slant

        if 2 ** self.J > self.M:
            print ('ERROR: 2^J should be greater than M!')




    ###################################
    ### Wavelet functions #############
    ###################################

   

    
    def gabor_2d(self, xi, j, theta):
        """
            Computes a 2D Gabor filter.
            A Gabor filter is defined by the following formula in space:
            psi(u) = g_{sigma}(u) e^(i xi^T u)
            where g_{sigma} is a Gaussian envelope and xi is a frequency.

            Parameters
            ----------
            M, N : int
                spatial sizes
            sigma : float
                bandwidth parameter
            xi : float
                central frequency (in [0, 1])
            theta : float
                angle in [0, pi]
            slant : float, optional
                parameter which guides the elipsoidal shape of the morlet
            offset : int, optional
                offset by which the signal starts

            Returns
            -------
            morlet_fft : ndarray
                numpy array of size (M, N)
        """

        M, N = self.M, self.M
        sigma = 0.8 * 2. ** j
        slant = self.slant
        offset = 0.


        #print ('params:', M, N, sigma, theta, xi, slant, offset)
        gab = np.zeros((M, N), np.complex64)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
        R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
        D = np.array([[1, 0], [0, slant * slant]])
        curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

        # xx just is all the x valus in the grip ends up running from 64 to 95 for M = 32 for example
        for ex in [-2, -1, 0, 1, 2]:
            for ey in [-2, -1, 0, 1, 2]:
                [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
                arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                    1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
                gab += np.exp(arg)

        norm_factor = (2 * 3.1415 * sigma * sigma / slant)
        gab /= norm_factor

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
                #print ('psi signal:', psi_signal)
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


	def compute_psi(j, theta, J, L):
	    psi = {'levels': [], 'j': j, 'theta': theta}
	    psi_signal = self.morlet_2d(3.0 / 4.0 * np.pi / 2**j, j, (int(L - L / 2 - 1) - theta) * np.pi / L)
	    psi_signal_fourier = np.real(fft2(psi_signal))
	    psi_levels = []
	    for res in range(min(j + 1, max(J - 1, 1))):
	        psi_levels.append(self.periodize_filter_fft(psi_signal_fourier, res))
	    psi['levels'] = psi_levels
	    return psi

	def compute_phi(J):
	    phi_signal = self.gabor_2d(0, J - 1, 0)
	    phi_signal_fourier = np.real(fft2(phi_signal))
	    phi_levels = []
	    for res in range(J):
	        phi_levels.append(self.periodize_filter_fft(phi_signal_fourier, res))
	    return phi_levels

	def filter_bank(self):
	    filters = {}
	    filters['psi'] = []

	    with multiprocessing.Pool() as pool:
	        results = []

	        for j in range(self.J):
	            for theta in range(self.L):
	                results.append(pool.apply_async(compute_psi, args=(j, theta, self.J, self.L)))

	        phi_result = pool.apply_async(compute_phi, args=(self.J,))

	        for j in range(self.J):
	            for theta in range(self.L):
	                psi = results.pop(0).get()
	                filters['psi'].append(psi)

	        filters['phi'] = {'levels': phi_result.get(), 'j': self.J}

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
        batch_size = tf.shape(data_batch)[0]
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

    def real_part(self, x):
        return tf.math.real(x)


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

        U_0_c = self.rfft(U_r)


        #first low pass filter
        U_1_C = self.cdgmm(U_0_c, phi['levels'][0])
        U_1_C = self.subsample_fourier(U_1_C, 2 ** self.J)

        S_0 = self.irfft(U_1_C)
        S_0 = self.unpad(S_0)

        out_S_0.append({'coef': S_0,
                'j': (),
                'n': (),
                'theta': ()})

        #print ('out_S_0:', out_S_0) #passed


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

        #print ('our_S:', len(out_S))

        if out_type == 'array':
            out_S = self.stack([x['coef'] for x in out_S])

        return out_S

    

class Scatter3D():
    
    def __init__(self, J, L, M, slant1, slant2):

        self.J = J
        self.L = L 
        self.M = M 
        self.slant1 = slant1
        self.slant2 = slant2





    # Next step is to code up the Gabor wavelets in https://www.sciencedirect.com/science/article/pii/S0262885605000934
    def gabor_3d(self, xi, j, theta, phi, prefactor):

        sigma = prefactor * 2. ** j
        M = self.M
        slant1 = self.slant1
        slant2 = self.slant2

        gab = np.zeros((M, M, M), np.complex64)

        D = np.array([[1, 0, 0], [0, slant1 * slant1, 0], [0, 0, slant2 * slant2]])

        R = np.array([
        [np.cos(phi) * np.cos(theta), -np.sin(phi), np.cos(phi) * np.sin(theta)],
        [np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

        R_inv = np.linalg.inv(R)
        curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)


        for ex in [-2, -1, 0, 1, 2]:
            for ey in [-2, -1, 0, 1, 2]:
                for ez in [-2, -1, 0, 1, 2]:
                    [xx, yy, zz] = np.mgrid[ex * M:M + ex * M, ey * M:M + ey * M, ez * M:M + ez * M]
                    vec = np.array([xx, yy, zz])
                    arg_real = -(  curv[0, 0] * np.multiply(xx, xx) + curv[1, 1] * np.multiply(yy, yy) + curv[2, 2] * np.multiply(zz, zz) \
                        + (curv[0,1] + curv[1,0]) * np.multiply(xx, yy) \
                        + (curv[0,2] + curv[2,0]) * np.multiply(xx, zz) \
                        + (curv[1,2] + curv[2,1]) * np.multiply(yy, zz) )
                    arg_im = 1.j * (xx * xi * np.sin(theta) * np.cos(phi) + yy * xi * np.sin(theta) * np.sin(phi) + zz * xi * np.cos(theta))
                    arg = arg_real + arg_im
                    gab += np.exp(arg)

        norm_factor = (2 * 3.14159 * sigma * sigma / (slant1 * slant2))

        return gab




    def morlet_3d(self, xi, j, theta, phi, prefactor = 0.8):

        
        val = self.gabor_3d(xi, j, theta, phi, prefactor)
        modulus = self.gabor_3d(0, j, theta, phi, prefactor)
        K = np.sum(val) / np.sum(modulus)
  

        mor = val - K * modulus

        return mor 


    def periodize_filter_fft(self, x, res):
        M = x.shape[0]
        N, L = M, M

        crop = np.zeros((M // 2 ** res, N // 2 ** res, L // 2 ** res), x.dtype)

        mask = np.ones(x.shape, np.float32)
        len_x = int(M * (1 - 2 ** (-res)))
        start_x = int(M * 2 ** (-res - 1))
        len_y = int(N * (1 - 2 ** (-res)))
        start_y = int(N * 2 ** (-res - 1))
        len_z = int(L * (1 - 2 ** (-res)))
        start_z = int(L * 2 ** (-res - 1))
        mask[start_x:start_x + len_x,:,:] = 0
        mask[:, start_y:start_y + len_y,:] = 0
        mask[:, :, start_z:start_z + len_z] = 0

        x = np.multiply(x,mask)

        for k in range(int(M / 2 ** res)):
            for l in range(int(N / 2 ** res)):
                for l2 in range(int(L / 2 ** res)):
                    for i in range(int(2 ** res)):
                        for j in range(int(2 ** res)):
                            for j2 in range(int(2 ** res)):
                                crop[k, l, l2] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res), l2 + j2 * int(L / 2 ** res)]

        return crop




    def filter_bank(self, prefactor = 0.8):
        filters = {}
        filters['psi'] = []

        for j in range(self.J):
            for theta in range(self.L):
                for phi in range(int(self.L)):
                    psi = {'levels': [], 'j': j, 'theta': theta, 'phi': phi}
                    #you are here....
                    #psi_signal = self.morlet_2d(3.0 / 4.0 * np.pi /2**j, j, (int(self.L-self.L/2-1)-theta) * np.pi / self.L)
                    psi_signal = self.morlet_3d(3.0 / 4.0 * np.pi /2**j, j, (int(self.L-self.L/2-1)-theta) * np.pi / self.L, (int(self.L-self.L/2-1)-phi) * np.pi / self.L, prefactor = 0.8)


                    psi_signal_fourier = np.real(fftn(psi_signal))

                    psi_levels = []
                    for res in range(min(j + 1, max(self.J - 1, 1))):
                        psi_levels.append(self.periodize_filter_fft(psi_signal_fourier, res))
                    psi['levels'] = psi_levels
                    filters['psi'].append(psi)


        phi_signal = self.gabor_3d(0, self.J-1, 0, 0, prefactor)
        phi_signal_fourier = np.real(fftn(phi_signal))
        # drop the imaginary part, it is zero anyway
        filters['phi'] = {'levels': [], 'j': self.J}
        for res in range(self.J):
            filters['phi']['levels'].append(
                self.periodize_filter_fft(phi_signal_fourier, res))

        return filters



    def rotate_image_numpy(self, image, theta):
        # Get the image dimensions
        height, width = image.shape[:2]
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), theta, 1)
        # Apply the rotation to the image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated_image



    def filter_bank_helix(self, filters_2d, filters_3d, rot_rate):
        #filers must be stored as a dictionary output from the 2d code
        # we fill the j,theta,psi from a 3d array 

        new_filters = filters_3d.copy()

        #let's do the phi filers first
        new_filters['phi'] = filters_3d['phi']

        #Get the dimension right for the new filters
        for i in range (len(filters_3d['psi'])):
            new_filters['psi'][i]['levels'][0] = np.zeros_like(filters_3d['psi'][i]['levels'][0])

        #now fill the empty filters
        for k in range(len(filters_2d)):
            for j in range(self.M):
                my_set = {1, 2, 3}
                random_number = random.choice(list(my_set))
                if random_number == 1:
                    new_filters['psi'][k]['levels'][0][:,:,j]= self.rotate_image_numpy(np.fft.fftshift(filters_2d[k]), rot_rate * j)
                elif random_number == 2:
                    new_filters['psi'][k]['levels'][0][:,j,:] = self.rotate_image_numpy(np.fft.fftshift(filters_2d[k]), rot_rate * j)
                elif random_number == 3:
                    new_filters['psi'][k]['levels'][0][j,:,:] = self.rotate_image_numpy(np.fft.fftshift(filters_2d[k]), rot_rate * j)




        return new_filters



    #still needs to be tested
    def filter_bank_mpi(self, prefactor=0.8):
        import mpi4py
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        filters = {}
        filters['psi'] = []

        J_range = range(rank, self.J, size)
        for j in J_range:
            for theta in range(self.L):
                for phi in range(int(L)):
                    psi = {'levels': [], 'j': j, 'theta': theta, 'phi': phi}

                    # Calculate psi_signal as before
                    # ...

                    psi_signal_fourier = np.real(fftn(psi_signal))

                    psi_levels = []
                    for res in range(min(j + 1, max(self.J - 1, 1))):
                        psi_levels.append(periodize_filter_fft(psi_signal_fourier, res))
                    psi['levels'] = psi_levels

                    comm.barrier()  # Wait for all processes to finish psi_levels computation

                    if rank == 0:
                        filters['psi'].append(psi)

        # Additional processing for phi_signal as before
        # ...

        # Perform the necessary collective operations to gather filter information
        all_filters = comm.gather(filters, root=0)

        if rank == 0:
            combined_filters = {'psi': [], 'phi': all_filters[0]['phi']}
            for f in all_filters:
                combined_filters['psi'].extend(f['psi'])
            return combined_filters
        else:
            return None



    #this is my own version to get dimensions right
    def padded_filter_bank(self):
        #this is adopted from base_frontend.py in kymatio
        M = self.M
        M_padded, N_padded, L_padded = self.compute_padding()
        self.M = M_padded
        filters = self.filter_bank()
        #reset M after computing the filters with the correct padding
        self.M = M

        return filters




    ###################################
    ### Utility functions #############
    ###################################


    def subsample_fourier(self, data_batch, k):
        #k will be 2^j and and x will be (batch_size, M, M, M)
        batch_size = tf.shape(data_batch)[0]
        y = tf.reshape(data_batch, shape = (batch_size, k, data_batch.shape[1] // k, k, data_batch.shape[2] // k, k, data_batch.shape[3] // k))
        return tf.math.reduce_mean(y, axis = (1,3,5))


    def compute_padding(self):
        #taken from the kymatio utils.py except I am assuming M = N
        M_padded = ((self.M + 2 ** self.J) // 2 ** self.J + 1) * 2 ** self.J
        N_padded = ((self.M + 2 ** self.J) // 2 ** self.J + 1) * 2 ** self.J
        L_padded = ((self.M + 2 ** self.J) // 2 ** self.J + 1) * 2 ** self.J
        return M_padded, N_padded, L_padded


    #utility function
    def unpad(self, x):
        #taken from numpy_backend.py in kymatio
        return x[..., 1:-1, 1:-1, 1:-1]

    #utility function
    def pad(self, x, pad_size, input_size):
        pad_size = list(pad_size)

        # Clone to avoid passing on modifications.
        new_pad_size = list(pad_size)

        print (tf.shape(x))
        print (pad_size)
        print (input_size)


        '''
        # This handles the case where the padding is equal to the image size.
        if pad_size[0] == input_size[0]:
            new_pad_size[0] -= 1
            new_pad_size[1] -= 1
        if pad_size[2] == input_size[1]:
            new_pad_size[2] -= 1
            new_pad_size[3] -= 1
        '''

        paddings = [[0, 0]] * len(x.shape[:-2])
        paddings += [[new_pad_size[0], new_pad_size[1]], [new_pad_size[2], new_pad_size[3]], [new_pad_size[4], new_pad_size[5]] ]

        x_padded = tf.pad(x, paddings, mode="REFLECT")

        '''
        # Again, special handling for when padding is the same as image size.
        if pad_size[0] == input_size[0]:
            x_padded = tf.concat([tf.expand_dims(x_padded[..., 1, :], axis=-2), x_padded, tf.expand_dims(x_padded[..., x_padded.shape[-2] -2, :], axis=-2)], axis=-2)
        if pad_size[2] == input_size[1]:
            x_padded = tf.concat([tf.expand_dims(x_padded[..., :, 1], axis=-1), x_padded, tf.expand_dims(x_padded[..., :,  x_padded.shape[-1]-2], axis=-1)], axis=-1)
        '''

        if pad_size[0] == input_size[0]:
            print ('Cuation: There is some corner case when pad size equal to image size')

        return x_padded

    #utility function
    def rfft(self, x):
        return tf.signal.fft3d(tf.cast(x, tf.complex64))

    #utility function
    def irfft(self, x):
        return tf.math.real(tf.signal.ifft3d(x))

    #utility function
    def ifft(self, x):
        return tf.signal.ifft3d(x)

    def cdgmm(self, A, B):
        return A * B 

    #utility function
    #not sure I will need this
    def stack(self, arrays):
        return tf.stack(arrays, axis=-3)

    def modulus(self, x):
        return tf.abs(x)

    def real_part(self, x):
        return tf.math.real(x)

    def real_part(self, x):
        return tf.math.real(x)
    

    ###################################
    ###compute the final output########
    ###################################


    def compute_coefs(self, x, phi, psi, max_order, out_type = 'array'):
            
            #taken from /core/scattering2d.py


            # Define lists for output.
            out_S_0, out_S_1, out_S_2 = [], [], []

            #this is mine...
            M_padded, N_padded, L_padded = self.compute_padding()
            pad_size = [(M_padded - self.M) // 2, (M_padded - self.M+1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2]
            input_size = [self.M, self.M, self.M]

            #back to code
            U_r = self.pad(x, pad_size, input_size)

            U_0_c = self.rfft(U_r)




            #first low pass filter
            U_1_C = self.cdgmm(U_0_c, phi['levels'][0])
            U_1_C = self.subsample_fourier(U_1_C, 2 ** self.J)

            S_0 = self.irfft(U_1_C)
            S_0 = self.unpad(S_0)

            out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': (),
                    'phi': ()})

            #you are here

            for n1 in range(len(psi)):
                j1 = psi[n1]['j']
                theta1 = psi[n1]['theta']
                phi1 = psi[n1]['phi']

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
                                'theta': (theta1,),
                                'phi': (phi1,)})


                if max_order < 2:
                    continue
                for n2 in range(len(psi)):
                    j2 = psi[n2]['j']
                    theta2 = psi[n2]['theta']
                    phi2 = psi[n2]['phi']


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
                                    'theta': (theta1, theta2),
                                    'phi': (phi1, phi2)})

            out_S = []
            out_S.extend(out_S_0)
            out_S.extend(out_S_1)
            out_S.extend(out_S_2)

            #print ('our_S:', len(out_S))

            if out_type == 'array':
                print ('Caution: stack may not be doing the right thing!')
                out_S = self.stack([x['coef'] for x in out_S])

            return out_S



    def compute_coefs_no_pad(self, x, phi, psi, max_order, out_type = 'array'):
            
            #taken from /core/scattering2d.py


            # Define lists for output.
            out_S_0, out_S_1, out_S_2 = [], [], []

            #this is mine...
            #M_padded, N_padded, L_padded = self.compute_padding()
            #pad_size = [(M_padded - self.M) // 2, (M_padded - self.M+1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2]
            input_size = [self.M, self.M, self.M]

            #back to code
            #U_r = self.pad(x, pad_size, input_size)
            U_r = x

            U_0_c = self.rfft(U_r)


            #first low pass filter
            U_1_C = self.cdgmm(U_0_c, phi['levels'][0])
            U_1_C = self.subsample_fourier(U_1_C, 2 ** self.J)

            S_0 = self.irfft(U_1_C)
            #S_0 = self.unpad(S_0)

            out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': (),
                    'phi': ()})

            #you are here

            for n1 in range(len(psi)):
                j1 = psi[n1]['j']
                theta1 = psi[n1]['theta']
                phi1 = psi[n1]['phi']

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
                #S_1_r = self.unpad(S_1_r)

                out_S_1.append({'coef': S_1_r,
                                'j': (j1,),
                                'n': (n1,),
                                'theta': (theta1,),
                                'phi': (phi1,)})


                if max_order < 2:
                    continue
                for n2 in range(len(psi)):
                    j2 = psi[n2]['j']
                    theta2 = psi[n2]['theta']
                    phi2 = psi[n2]['phi']


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
                    #S_2_r = self.unpad(S_2_r)

                    out_S_2.append({'coef': S_2_r,
                                    'j': (j1, j2),
                                    'n': (n1, n2),
                                    'theta': (theta1, theta2),
                                    'phi': (phi1, phi2)})

            out_S = []
            out_S.extend(out_S_0)
            out_S.extend(out_S_1)
            out_S.extend(out_S_2)

            #print ('our_S:', len(out_S))

            if out_type == 'array':
                #print ('Caution: stack may not be doing the right thing!')
                out_S = self.stack([x['coef'] for x in out_S])

            return out_S



    def cast_to_tensor(self, x, phi, psi):
        x = tf.convert_to_tensor(x)
        phi = tf.convert_to_tensor(phi)
        psi = tf.convert_to_tensor(psi)
        return x, phi, psi



    def compute_coefs_no_pad_extend(self, x, phi, psi, max_order, out_type = 'array'):
            
            #taken from /core/scattering2d.py


            # Define lists for output.
            out_S_0, out_S_1, out_S_2, out_S_3 = [], [], [], []

            #this is mine...
            #M_padded, N_padded, L_padded = self.compute_padding()
            #pad_size = [(M_padded - self.M) // 2, (M_padded - self.M+1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2]
            input_size = [self.M, self.M, self.M]

            #back to code
            #U_r = self.pad(x, pad_size, input_size)
            U_r = x

            U_0_c = self.rfft(U_r)


            #first low pass filter
            U_1_C = self.cdgmm(U_0_c, phi['levels'][0])
            U_1_C = self.subsample_fourier(U_1_C, 2 ** self.J)

            S_0 = self.irfft(U_1_C)
            #S_0 = self.unpad(S_0)

            out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': (),
                    'phi': ()})

            #you are here

            for n1 in range(len(psi)):
                j1 = psi[n1]['j']
                theta1 = psi[n1]['theta']
                phi1 = psi[n1]['phi']

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
                #S_1_r = self.unpad(S_1_r)

                out_S_1.append({'coef': S_1_r,
                                'j': (j1,),
                                'n': (n1,),
                                'theta': (theta1,),
                                'phi': (phi1,)})


                if max_order < 2:
                    continue
                for n2 in range(len(psi)):
                    j2 = psi[n2]['j']
                    theta2 = psi[n2]['theta']
                    phi2 = psi[n2]['phi']


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
                    #S_2_r = self.unpad(S_2_r)

                    out_S_2.append({'coef': S_2_r,
                                    'j': (j1, j2),
                                    'n': (n1, n2),
                                    'theta': (theta1, theta2),
                                    'phi': (phi1, phi2)})


                    if max_order < 3:
                        continue
                    for n3 in range(len(psi)):
                        j3 = psi[n3]['j']
                        theta3 = psi[n3]['theta']
                        phi3 = psi[n3]['phi']


                        if j3 <= j1:
                            continue

                        U_3_c = self.cdgmm(U_1_c, psi[n3]['levels'][j1])
                        U_3_c = self.subsample_fourier(U_3_c, k=2 ** (j3 - j1))
                        U_3_c = self.ifft(U_3_c)
                        U_3_c = self.modulus(U_3_c)
                        U_3_c = self.rfft(U_3_c)

                        # Third low pass filter
                        S_3_c = self.cdgmm(U_3_c, phi['levels'][j3])
                        S_3_c = self.subsample_fourier(S_3_c, k=2 ** (self.J - j3))

                        S_3_r = self.irfft(S_3_c)
                        #S_2_r = self.unpad(S_2_r)

                        out_S_3.append({'coef': S_3_r,
                                        'j': (j1, j2, j3),
                                        'n': (n1, n2, n3),
                                        'theta': (theta1, theta2, theta3),
                                        'phi': (phi1, phi2, phi3)})

            out_S = []
            out_S.extend(out_S_0)
            out_S.extend(out_S_1)
            out_S.extend(out_S_2)
            out_S.extend(out_S_3)


            #print ('our_S:', len(out_S))

            if out_type == 'array':
                #print ('Caution: stack may not be doing the right thing!')
                out_S = self.stack([x['coef'] for x in out_S])

            return out_S

    



    def compute_coefs_no_pad_real(self, x, phi, psi, max_order, out_type = 'array'):
            
            #taken from /core/scattering2d.py


            # Define lists for output.
            out_S_0, out_S_1, out_S_2 = [], [], []

            #this is mine...
            #M_padded, N_padded, L_padded = self.compute_padding()
            #pad_size = [(M_padded - self.M) // 2, (M_padded - self.M+1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2]
            input_size = [self.M, self.M, self.M]

            #back to code
            #U_r = self.pad(x, pad_size, input_size)
            U_r = x

            U_0_c = self.rfft(U_r)


            #first low pass filter
            U_1_C = self.cdgmm(U_0_c, phi['levels'][0])
            U_1_C = self.subsample_fourier(U_1_C, 2 ** self.J)

            S_0 = self.irfft(U_1_C)
            #S_0 = self.unpad(S_0)

            out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': (),
                    'phi': ()})

            #you are here

            for n1 in range(len(psi)):
                j1 = psi[n1]['j']
                theta1 = psi[n1]['theta']
                phi1 = psi[n1]['phi']

                U_1_c = self.cdgmm(U_0_c, psi[n1]['levels'][0])
                if j1 > 0:
                    U_1_c = self.subsample_fourier(U_1_c, k=2 ** j1)
                U_1_c = self.ifft(U_1_c)
                U_1_c = self.real_part(U_1_c)
                U_1_c = self.rfft(U_1_c)

                # Second low pass filter
                S_1_c = self.cdgmm(U_1_c, phi['levels'][j1])
                S_1_c = self.subsample_fourier(S_1_c, k=2 ** (self.J - j1))

                S_1_r = self.irfft(S_1_c)
                #S_1_r = self.unpad(S_1_r)

                out_S_1.append({'coef': S_1_r,
                                'j': (j1,),
                                'n': (n1,),
                                'theta': (theta1,),
                                'phi': (phi1,)})


                if max_order < 2:
                    continue
                for n2 in range(len(psi)):
                    j2 = psi[n2]['j']
                    theta2 = psi[n2]['theta']
                    phi2 = psi[n2]['phi']


                    if j2 <= j1:
                        continue

                    U_2_c = self.cdgmm(U_1_c, psi[n2]['levels'][j1])
                    U_2_c = self.subsample_fourier(U_2_c, k=2 ** (j2 - j1))
                    U_2_c = self.ifft(U_2_c)
                    U_2_c = self.real_part(U_2_c)
                    U_2_c = self.rfft(U_2_c)

                    # Third low pass filter
                    S_2_c = self.cdgmm(U_2_c, phi['levels'][j2])
                    S_2_c = self.subsample_fourier(S_2_c, k=2 ** (self.J - j2))

                    S_2_r = self.irfft(S_2_c)
                    #S_2_r = self.unpad(S_2_r)

                    out_S_2.append({'coef': S_2_r,
                                    'j': (j1, j2),
                                    'n': (n1, n2),
                                    'theta': (theta1, theta2),
                                    'phi': (phi1, phi2)})

            out_S = []
            out_S.extend(out_S_0)
            out_S.extend(out_S_1)
            out_S.extend(out_S_2)

            #print ('our_S:', len(out_S))

            if out_type == 'array':
                #print ('Caution: stack may not be doing the right thing!')
                out_S = self.stack([x['coef'] for x in out_S])

            return out_S



    def compute_coefs_no_pad_imag(self, x, phi, psi, max_order, out_type = 'array'):
            
            #taken from /core/scattering2d.py


            # Define lists for output.
            out_S_0, out_S_1, out_S_2 = [], [], []

            #this is mine...
            #M_padded, N_padded, L_padded = self.compute_padding()
            #pad_size = [(M_padded - self.M) // 2, (M_padded - self.M+1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2, (M_padded - self.M) // 2, (M_padded - self.M + 1) // 2]
            input_size = [self.M, self.M, self.M]

            #back to code
            #U_r = self.pad(x, pad_size, input_size)
            U_r = x

            U_0_c = self.rfft(U_r)


            #first low pass filter
            U_1_C = self.cdgmm(U_0_c, phi['levels'][0])
            U_1_C = self.subsample_fourier(U_1_C, 2 ** self.J)

            S_0 = self.irfft(U_1_C)
            #S_0 = self.unpad(S_0)

            out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': (),
                    'phi': ()})

            #you are here

            for n1 in range(len(psi)):
                j1 = psi[n1]['j']
                theta1 = psi[n1]['theta']
                phi1 = psi[n1]['phi']

                U_1_c = self.cdgmm(U_0_c, psi[n1]['levels'][0])
                if j1 > 0:
                    U_1_c = self.subsample_fourier(U_1_c, k=2 ** j1)
                U_1_c = self.ifft(U_1_c)
                U_1_c = tf.math.imag(U_1_c)
                U_1_c = self.rfft(U_1_c)

                # Second low pass filter
                S_1_c = self.cdgmm(U_1_c, phi['levels'][j1])
                S_1_c = self.subsample_fourier(S_1_c, k=2 ** (self.J - j1))

                S_1_r = self.irfft(S_1_c)
                #S_1_r = self.unpad(S_1_r)

                out_S_1.append({'coef': S_1_r,
                                'j': (j1,),
                                'n': (n1,),
                                'theta': (theta1,),
                                'phi': (phi1,)})


                if max_order < 2:
                    continue
                for n2 in range(len(psi)):
                    j2 = psi[n2]['j']
                    theta2 = psi[n2]['theta']
                    phi2 = psi[n2]['phi']


                    if j2 <= j1:
                        continue

                    U_2_c = self.cdgmm(U_1_c, psi[n2]['levels'][j1])
                    U_2_c = self.subsample_fourier(U_2_c, k=2 ** (j2 - j1))
                    U_2_c = self.ifft(U_2_c)
                    U_2_c = tf.math.imag(U_2_c)
                    U_2_c = self.rfft(U_2_c)

                    # Third low pass filter
                    S_2_c = self.cdgmm(U_2_c, phi['levels'][j2])
                    S_2_c = self.subsample_fourier(S_2_c, k=2 ** (self.J - j2))

                    S_2_r = self.irfft(S_2_c)
                    #S_2_r = self.unpad(S_2_r)

                    out_S_2.append({'coef': S_2_r,
                                    'j': (j1, j2),
                                    'n': (n1, n2),
                                    'theta': (theta1, theta2),
                                    'phi': (phi1, phi2)})

            out_S = []
            out_S.extend(out_S_0)
            out_S.extend(out_S_1)
            out_S.extend(out_S_2)

            #print ('our_S:', len(out_S))

            if out_type == 'array':
                #print ('Caution: stack may not be doing the right thing!')
                out_S = self.stack([x['coef'] for x in out_S])

            return out_S

