#from SAD_Guess import SAD
import numpy as np
import pyscf.gto
import time
from math import sqrt
import matplotlib.pyplot as plt

class SCF:
    def __init__(self, mol,guess='atom', name='System',spin=0, method_type='rhf', Etolerance=0.0000001, Dtolerance=0.0000001, max_cycles=1000, verbose=0, plot=False, DIIS=True):
        '''Creates an SCF object with default convergence parameters
        Kwargs:
            mol : SCF mol object
                Very important to pass this in, provides geometry/basis set to SCF
            name : string
                Any output files will be written to the given calculation name.
                Default is just 'system'
            Etolerance : float
                Tolerance for SCF energy convergence
            Dtolerance : float
                Tolerance for SCF density convergence
            max_cycles : integer
                Max number of SCF cycles without convergence
            verbose : integer
                0 : No Output
                1 : Total Converged System Energy
                2 : SCF Cycle Count, Run Time
                3 : dE/dD/total E Per SCF Cycle
                6 : Full Converged Electron Density Matrix             
            plot : bool
                True : generated dE, dD, E_tot vs Cycle Count .png plots
                False : Does not plot
            DIIS : bool
                True : Use DIIS convergence acceleration
                False : Do not use DIIS convergence acceleration
            spin : integer (use with uhf)
                spin state 2S
                0 : singlet state
                2 : triplet state
                4 : double state, etc
            method_type : string
                'rhf' : restricted hartree-focke for closed shell systems (S=0)
                'uhf' : unrestricted HF for open shell systems (S != 0, nelec%2 != 0)
        '''
        self.name = name
        self.mol = mol
        #Define SCF Convergence Parameters
        self.Econvergence = False #starting out unconverged E and DM of course!
        self.Dconvergence = False
        self.Etolerance=Etolerance
        self.Dtolerance = Dtolerance
        self.max_cycles = max_cycles
        self.cycle_count = 0 
        self.method_type=method_type
        #DIIS Parameters
        self.DIIS = DIIS
        if self.DIIS:
            if self.method_type == 'rhf':
                self.focke_matrices = []
                self.error_vectors = []
            if self.method_type == 'uhf':
                self.focke_alpha = []
                self.focke_beta = []
                self.error_alpha = []
                self.error_beta = []

        #Gathering properties of the system 
        
        self.nao = mol.nao_nr()
        self.nelec = mol.nelectron
        
        #Performing 1/2e integrals on Atomic Orbitals
        
        #nuclear repulsion energy
        self.enuc=mol.energy_nuc()
        #overlap_integrals
        self.ovlp = mol.intor('cint1e_ovlp_sph')
        #one-electron kinetic integrals
        self.T = mol.intor('cint1e_kin_sph')
        #one-electron potential integrals
        self.V=mol.intor('cint1e_nuc_sph')
        #Compute 2-electron repulsion integrals
        self.v2e = mol.intor('cint2e_sph').reshape((self.nao,)*4)
        self.F = self.T + self.V #Defining Initial Focke Matrix from 1e Integral
        #Defining Output Parameters for Users to Request After Calculation
        #if guess == 'atom':
        #    guess = SAD(self.mol)
        #    self.F = guess.kernel()
        self.e_elec = 0
        self.e_tot = 0
        self.mo_coeff = None
        self.density = None
        self.Sdensity = None
        self.elapsed_time = 0
        self.verbose = verbose

        #Defining Change Matrices for Plotting SCF
        self.e_tot_change = []
        self.dE_change = []
        self.dD_change = []
        self.plot_convergence = plot
        
        #Handling UHF Calculation Parameters
        self.flag=False #the flag is only true if we are performing a calculation on a system where all spins are the same (ie. H2 Triplet) 
        if self.method_type=='uhf':
            self.F_alpha = self.F #spin up/spin down focke matrices
            self.F_beta = self.F
            self.spin = spin #spin quantum number 2S
            self.N_alpha = (self.nelec - self.spin)/2 #number of spin up/spin down electrons
            self.N_beta = self.nelec-self.N_alpha
            if self.N_alpha == 0: #if all the electrons are one spin basically just run a regular RHF calculation
                self.method_type='rhf'
                self.focke_matrices = []
                self.error_vectors=[]
                self.flag = True
                self.F = self.F_beta
                self.nelec = self.N_beta*2

            self.D_alpha_converge = False
            self.D_beta_converge = False

        #handling bad inputs
        if spin != 0 and method_type == 'rhf':
            raise ValueError("Restricted Hartree-Focke Calculation Requires Closed Shell System. Run UHF Instead") 
        if method_type == 'rhf' and self.nelec%2 != 0:
            raise ValueError("Restricted Hartree-Focke Requires Even Electron Count. Run UHF Instead")
    
    def kernel(self):
        '''Manages UHF/RHF job submissions'''
        start_time = time.perf_counter()
        if self.method_type == 'uhf':
            self.kernel_uhf()
        elif self.method_type == 'rhf':
            self.kernel_rhf()

        #Printing output with verbosity
        end_time = time.perf_counter()
        self.elapsed_time = end_time - start_time
        if self.verbose > 0:
            if self.verbose > 1:
                print('SCF Cycles: ', self.cycle_count)
                print('Run Time: ',  self.elapsed_time, 'Seconds')
            if self.verbose > 5:
                print('Converged Electron Density Matrix: ', self.Sdensity)
        print('Total System Energy: ', self.e_tot)

    def kernel_uhf(self):
        '''Performs UHF SCF Calculation'''
        S_inverse = self.Diagonalize_Ovlp() #obtain S^-1/2
        #While all three convergence parameters have not been met...
        while (not self.Econvergence or not self.D_alpha_converge or not self.D_beta_converge) and self.cycle_count < self.max_cycles:
            C_alpha = self.initial_Focke(S_inverse, self.F_alpha) #calculate coefficients for spin up/down
            C_beta = self.initial_Focke(S_inverse, self.F_beta)
            D_alpha = self.density_matrix(C_alpha, self.N_alpha) #get density matrices
            D_beta = self.density_matrix(C_beta, self.N_beta)
            if self.DIIS and self.cycle_count > 1: #run DIIS speedup algorithm (not well-implemented for UHF)
                self.focke_alpha.append(self.F_alpha)
                self.focke_beta.append(self.F_beta)
                self.F_alpha = self.apply_DIIS(old_d_alpha, S_inverse, self.F_alpha, self.focke_alpha, self.error_alpha)
                self.F_beta = self.apply_DIIS(old_d_beta, S_inverse, self.F_beta, self.focke_beta, self.error_beta)
                C_alpha = self.initial_Focke(S_inverse, self.F_alpha)
                C_beta = self.initial_Focke(S_inverse, self.F_beta)
                D_alpha = self.density_matrix(C_alpha, self.N_alpha)
                D_beta = self.density_matrix(C_beta, self.N_beta)
            self.New_Focke_Matrix_UHF(D_alpha, D_beta) #update self.F_alpha, self.F_beta
            #Calculate new Energy = 1/2 E_alpha + 1/2 E_beta
            NewE =  1/2*self.electronic_energy(D_alpha, self.F_alpha)[0] + 1/2*self.electronic_energy(D_beta, self.F_beta)[0]
            self.e_tot_change += [NewE]
            #Check Convergence for alpha, beta
            if self.cycle_count != 0:
                self.D_alpha_converge = self.check_convergence(OldE, NewE, old_d_alpha, D_alpha, 'alpha')
                self.D_beta_converge = self.check_convergence(OldE, NewE, old_d_beta, D_beta, 'beta')
            OldE = NewE
            old_d_alpha = D_alpha
            old_d_beta = D_beta
            self.cycle_count += 1
            self.e_tot = NewE
        self.density = D_alpha + D_beta
        if self.plot_convergence:
            self.plot('uhf')
    
    def kernel_rhf(self):
        '''Performs Entire RHF SCF Loop Procedure'''
        diagonalize_S=self.Diagonalize_Ovlp() #Diagonalize the S Matrix and Compute S^-1/2
        
        #Begin SCF Iterations!
        self.cycle_count=0
        while (not self.Econvergence or not self.Dconvergence) and self.cycle_count < self.max_cycles:
            C=self.initial_Focke(diagonalize_S)
            if self.mol.atom == 'H':
                D = self.density_matrix(C, 1)
                self.density = D
                return 
            New_Density=self.density_matrix(C, self.nelec/2)
            if self.DIIS and self.cycle_count > 1:
                self.focke_matrices.append(self.F)
                self.F = self.apply_DIIS(Old_Density, diagonalize_S, self.F, self.focke_matrices, self.error_vectors)
                C=self.initial_Focke(diagonalize_S)
                New_Density=self.density_matrix(C, self.nelec/2)
            self.New_Focke_Matrix_RHF(New_Density)
            Etot, NewE=self.electronic_energy(New_Density, self.F) 
            self.e_tot_change += [Etot] #Save for Convergence Plots
            
            if self.cycle_count != 0:
               self.check_convergence(OldE,NewE,Old_Density,New_Density)
            
            #Saving old Matrices/Energies for Comparison to check Convergence
            OldE = NewE
            Old_Density = New_Density
            self.cycle_count+=1
            self.e_elec = NewE #Updating Output With Converged Values
            self.e_tot = Etot
            self.mo_coeff = C
            self.density = New_Density
            self.Sdensity = New_Density@self.ovlp
        
        if self.plot_convergence:
            self.plot('rhf')
    def plot(self, method):
        '''Generates Line Plots for Convergence vs Cycle Count'''
        x = []
        for i in range(self.cycle_count):
            x += [i]
        #plotting cycle_count vs total energy
        plt.plot(x,self.e_tot_change)
        plt.xlabel('SCF Cycle Count')
        plt.ylabel('Total System Energy (Eh)')
        plt.title(self.name + ' Energy vs Cycle Count')
        plt.savefig(self.name + '_E.png')
        
        plt.clf()
        #Plotting Energy Change vs Cycle Count
        plt.plot(x[1:], self.dE_change)
        plt.xlabel('SCF Cycle Count')
        plt.ylabel('Energy Change (Eh)')
        plt.title(self.name + ' Energy Change vs Cycle Count')
        plt.savefig(self.name + '_dE.png')
        plt.clf()
        #Plotting Density Change vs Cycle Count
        if self.method_type == 'rhf':
            plt.plot(x[1:], self.dE_change)
            plt.xlabel('SCF Cycle Count')
            plt.ylabel('Density RMS (e)')
            plt.title(self.name + ' Density RMS vs Cycle Count')
            plt.savefig(self.name + '_dD.png')

    
    def check_convergence(self, Ei, Ef, Di, Df, spin=None):
        '''Checks Whether the Energies and DM RMSD Values Meet Convergence Thresholds'''
        E_status='NO' #Set Initial Status Print Variables
        D_status='NO'
        #Compute dE, dD
        dE = Ef-Ei #Should always be < 0 b/c HF/SCF is Variational, take absolute value! 
        dD = self.RMSD(Di, Df)
        #Saving for Plotting Convergence
        if dE not in self.dE_change:
            self.dE_change += [dE]
        
        self.dD_change += [dD]

        if abs(dE) < self.Etolerance:
                self.Econvergence = True
                E_status='YES'
        if dD < self.Dtolerance:
            D_status = 'YES'
            if spin is None:
                self.Dconvergence = True
            else:
                return True
        if dD > self.Dtolerance:
            return False
        #Outputs:
        if self.verbose > 2:
            print('Energy Change: ', dE, 'Convergence Status ' + E_status, 'Threshold: ', self.Etolerance)
            print('Density Change: ', dD, 'Convergence Status ' + D_status, 'Threshold: ', self.Dtolerance)
        return
    
    def RMSD(self, Di, Df):
        '''Compute RMS between initial DM and final DM'''
        RMSD = 0
        for rowi, rowf in zip(Di, Df):
            for i, f in zip(rowi, rowf):
                RMSD += (f - i)**2
        RMSD = sqrt(RMSD)
        return RMSD

    def Diagonalize_Ovlp(self):
        '''Finds EigenValues + EigenVectors of S'''
        #Get Initial Diagonalization of S
        eigenvalues, eigenvectors = np.linalg.eigh(self.ovlp)
        #Take the 1/sqrt(s) for each eigenvalue of S and construct a diagonal matrix
        inverse_sqrt_eigenvalues = []
        for eigenvalue in eigenvalues:
            inverse_sqrt_eigenvalues += [1/sqrt(eigenvalue)]
        inverse_eigen_matrix = np.diag(inverse_sqrt_eigenvalues) #s^-1/2 matrix

        transpose_eigenvectors = eigenvectors.T #U.T
        #compute S^-1/2 = U@s^-1/2@U.T
        s_inverse_root = eigenvectors @ inverse_eigen_matrix @ transpose_eigenvectors 
        return s_inverse_root

    def initial_Focke(self, s_inverse_root, spin_focke=None):
        '''Compute the Matrix of Coefficients for MOs from AOs and Transform Focke Matrix'''
        if self.method_type=='uhf':
            self.F = spin_focke

        F_new_basis = s_inverse_root.T @ self.F @ s_inverse_root #Transforming F > S^-1/2@F@S^-1/2 to orthogonalize
        eigenvalues, eigenvectors = np.linalg.eigh(F_new_basis) #solve eigenvalue/vector problem S^-1/2@F@S^-1/2 = Cbar@E
        Cbar = eigenvectors #Within S^-1/2 Basis
        C=s_inverse_root@Cbar #obtain coefficient eigenvectors for MOs outside of the S^-1/2 Basis
        return C

    def density_matrix(self, C, n_elec):
        '''Compute the Density Matrix for a set of coefficients C'''
        count=0
        D=np.array([])
        for molecular_orbital in range(int(n_elec)): #summing over occupied orbitals
            Duv = []
            for atomic_orbital_u in C.T[molecular_orbital]: #summing over AOs w/in MOs, columns of C_matrix, rows of C.T
                 row = []
                 for atomic_orbital_v in C.T[molecular_orbital]:
                     row+=[atomic_orbital_u*atomic_orbital_v] #computing Cu*Cv for every uv
                 Duv+=[row]
            np_array_Duv=np.array(Duv)

            
            if count == 0: #if we are in the first MO, construct the density matrix, D
                D=np_array_Duv 
            else: #if not in the first MO, add to existing DM, D
                D=D+np_array_Duv
            count+=1
        return D

    def New_Focke_Matrix_RHF(self,D):
        '''Takes initial Density Matrix and Computes a New Focke Matrix'''
        #One_Electron_Terms
        one_electron_terms=self.T+self.V
        #Manual Tensor Contractions:
        full_focke = [[0 for n in range(self.nao)] for n in range(self.nao)] #Build an empty nxn matrix to store Fuv terms
        #generate the new fock matrix AOs: u, v, rho, sigma
        for u in range(self.nao):
            for v in range(self.nao):
                focke_uv = one_electron_terms[u][v]
                for rho in range(self.nao):
                    for sigma in range(self.nao): 
                        coulomb_integral_J=2*self.v2e[u,v,rho,sigma] #summing up electrostatic interactions between the current two-electrons u,v, and rho,sigma
                        if self.flag: #UHF calculation for all alpha/beta, need to use J not 2J
                            coulomb_integral_J = coulomb_integral_J/2
                        Exchange_Integral_K=self.v2e[u,rho,v,sigma] #exchanging v/sigma to account for indistinguishabillity of electrons (Pauli)
                        Density_Matrix_Element=D[rho,sigma]
                        focke_uv +=  Density_Matrix_Element * (coulomb_integral_J - Exchange_Integral_K)
                full_focke[u][v] = focke_uv #Adding Fuv to the new Focke Matrix
        self.F = np.array(full_focke) #New F matrix Generated
        return 
    
    def New_Focke_Matrix_UHF(self,D_alpha, D_beta):
        '''Takes Alpha/Beta Density Matrices and Computes Two New A/B Focke Matrices'''
        #One_Electron_Terms
        one_electron_terms=self.T+self.V
        #Manual Tensor Contractions:
        focke_alpha = [[0 for n in range(self.nao)] for n in range(self.nao)] #Build an empty nxn matrix to store Fuv terms
        focke_beta = [[0 for n in range(self.nao)] for n in range(self.nao)]
        #generate the new fock matrix AOs: u, v, rho, sigma
        for u in range(self.nao):
            for v in range(self.nao):
                Fuv_alpha = one_electron_terms[u][v] #same for alpha and beta
                Fuv_beta = one_electron_terms[u][v]
                for rho in range(self.nao):
                    for sigma in range(self.nao):
                        J_alpha=D_alpha[rho,sigma]*self.v2e[u,v,rho,sigma] #summing up electrostatic interactions between the current two-electrons u,v, and rho,sigma
                        J_beta=D_beta[rho,sigma]*self.v2e[u,v,rho,sigma]


                        K_alpha=D_alpha[rho,sigma]*self.v2e[u,rho,v,sigma] #exchanging v/sigma to account for indistinguishabillity of electrons (Pauli)
                        K_beta=D_beta[rho,sigma]*self.v2e[u,rho,v,sigma]
                        
                        Fuv_alpha += J_alpha + J_beta - K_alpha
                        Fuv_beta += J_beta + J_alpha - K_beta
                focke_alpha[u][v] = Fuv_alpha
                focke_beta[u][v] = Fuv_beta #Adding Fuv to the new Focke Matrix

        self.F_alpha = np.array(focke_alpha) #New F matrix Generated
        self.F_beta = np.array(focke_beta)
        return
    
    
    def apply_DIIS(self, D, S_inverse, F, F_list, error_vectors):
        '''Applies the DIIS algorithm to the current Focke Matrix/Density Matrix'''
        self.error_vector(D, S_inverse, F, error_vectors)
        if len(error_vectors) >= 8:
            error_vectors.pop(0)
            F_list.pop(0)
        if len(error_vectors) >= 2:
            B = self.compute_error_prod_B(error_vectors)
            coeff = self.solve_error_coeff(B)
            F = self.adjust_F(coeff, F_list)
        return F

    def error_vector(self, D, S_inverse, F, error_vectors):
        '''Computes the Error Vector ei for the Current Focke Matrix'''
        ei =S_inverse.T@(F@D@self.ovlp - self.ovlp@D@F)@S_inverse
        #vectorizing error vector
        ei= ei.flatten()
        error_vectors.append(ei)
        return
    
    def compute_error_prod_B(self, error_vectors):
        '''Computes the Frobenius Inner Product Matrix of the Error Vectors'''
        num_Ei = len(error_vectors)
        B = [[0 for n in range(num_Ei)] for n in range(num_Ei)] #Build an empty nxn matrix of error vector dot products
        for v1 in range(num_Ei):
            for v2 in range(num_Ei):
                B[v1][v2] = error_vectors[v1].T@error_vectors[v2]
        
        return B
    
    def solve_error_coeff(self, B):
        '''Solves the Linear System for Error Vector Coefficients
        Sets up and solves the System of Equations:
        |B11 B12 .. -1| |c1|   |0   |
        |B21 B22 .. -1|@|cn| = |0(n)| 
        |..-1 -1 ..  0| |L |   |-1  |'''

        sys_matrix = [] #modifying B matrix with constraints/lagrange
        bottom_row = [] 
        #constructing the solution vector [0, .., -1]
        solution_vector = [] 
        for row in B:
           solution_vector.append(0)
           row.append(-1)
           bottom_row.append(-1)
           sys_matrix.append(row)
        bottom_row.append(0)
        solution_vector.append(-1)
        sys_matrix.append(bottom_row)
        sys_matrix = np.array(sys_matrix)
        solution = np.array(solution_vector)
        #Solving System of Linear Equations for c1, .., cn, lambda
        coeff = np.linalg.solve(sys_matrix, solution)
        return coeff[:-1]

    def adjust_F(self, C, F_list):
        '''Calculates the linear combination of Focke Matrices F = ciFI + ... cnFn'''
        F = [[0 for n in range(self.nao)] for n in range(self.nao)] #Build an empty nxn matrix to store Fuv terms
        F = np.array(F)
        for coefficient, focke_matrix in zip(C, F_list):
            F = F + coefficient*focke_matrix
        return F

    def electronic_energy(self,D, F):
        '''Takes a Focke Matrix, Density Matrix, and 1-electron Integrals.
        Computes the electronic, nuclear, and total energy of the system'''
        E_elec=0
        E_nuc=self.enuc
        for u in range(self.nao):
            for v in range(self.nao):
                Density = D[u][v]
                one_electron_pe=self.V[u][v]
                one_electron_ke=self.T[u][v]
                two_electron=F[u][v] #exchange + repulsion (exchange lowers energy, repulsion increases it!)
                E_elec += Density*(one_electron_pe + one_electron_ke + two_electron) #total energy scaled by electron density!
        
        if self.flag:
            E_elec = E_elec/2
        E_tot=E_elec + E_nuc
        
        return E_tot, E_elec
            

if __name__ == '__main__':
    mol = pyscf.gto.M(verbose = 1,atom = [ ['N', (0.0, 0.0, 0.0)], ['N', (1.1, 0.0, 0.0)], ], basis = '3-21g', symmetry = False)
    k=SCF(mol, DIIS=True, guess='atom', name='Cl2_631g_No_DIIS', method_type='rhf', spin=0, plot=False, max_cycles=100, verbose=4)                                                                                               
    k.kernel()


