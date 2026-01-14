import pyscf
from SCF import SCF
import numpy as np
class SAD:
    def __init__(self, mol):
        self.atoms = mol.atom
        self.atom_full = []
        
        self.basis = mol.basis 
        self.atom_list = []
        self.offset = mol.ao_labels()
        self.nao = mol.nao_nr()
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
        #print('initial focke :', self.T + self.V)  
        for i in self.atoms:
            self.atom_full.append(i[0])
            if i[0] not in self.atom_list:
                self.atom_list += i[0]
        
        self.atom_dict = { 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10 }
        self.loc_dict = {}
        count = 0
        self.size_list = []
        prev_ao = ''

    def kernel(self):
        densities, naos = self.calculate_atomic_DMs(self.atom_list)
        fullD = self.build_matrix(densities, naos)
        print(np.trace(fullD))
        print(fullD)
        fullFocke = self.fullFocke(fullD)
        #print(fullFocke)
        return fullFocke
    def calculate_atomic_DMs(self, atom_list):
        density_list = {}
        naos = {}
        for i in self.atom_list:
            if self.atom_dict[i]%2 != 0:
                mol = pyscf.gto.M(verbose=0, atom=i, spin=1, basis = self.basis, symmetry=False)
                SCF_calculation = SCF(mol, DIIS=False, verbose=0, spin=1, method_type='uhf')
            else:
                mol = pyscf.gto.M(verbose=0, atom=i, basis = self.basis, symmetry=False)
                SCF_calculation = SCF(mol, DIIS=True, verbose=0, max_cycles=1000)
            SCF_calculation.kernel()
            #print(SCF_calculation.density)
            density_list[i] = SCF_calculation.density
            naos[i] = SCF_calculation.nao
        return density_list, naos

    def build_matrix(self, densities, naos):
        D = np.zeros((self.nao,self.nao))
        #for i in range(self.nao):
        start_index = 0
        for atom in self.atom_full:
            D[start_index:start_index+naos[atom], start_index:start_index+naos[atom]] = densities[atom]
            start_index = start_index+naos[atom]
        return D

    def fullFocke(self, D):
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
                        Exchange_Integral_K=self.v2e[u,rho,v,sigma] #exchanging v/sigma to account for indistinguishabillity of electrons (Pauli)
                        Density_Matrix_Element=D[rho,sigma]
                        focke_uv +=  Density_Matrix_Element * (coulomb_integral_J - Exchange_Integral_K)
                full_focke[u][v] = focke_uv #Adding Fuv to the new Focke Matrix
        F = np.array(full_focke) #New F matrix Generated
        return F

if __name__ == '__main__':
    mol = pyscf.gto.M(verbose = 1, atom = [ ['N', (0.0, 0.0, 0.0)], ['N', (0.74, 0.0, 0.0)]], basis = 'sto-3g', symmetry = False)
    k=SAD(mol)
    k.kernel()


