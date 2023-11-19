from current_tripole import CurrentTripole
import numpy as np

class MuscleFiber:
    def __init__(self, start_point, end_point, nmj_relative_pos, conduction_velocity, current_tripole: CurrentTripole):
        self.start_point = start_point
        self.end_point = end_point
        self.nmj_relative_pos = nmj_relative_pos
        self.conduction_velocity = conduction_velocity
        self.current_tripole = current_tripole

    @staticmethod
    def _get_ortho_basis(v0):
        idx_max = np.argmax(np.abs(v0))

        v1 = np.zeros(v0.shape)

        v1[idx_max] = -v0[(idx_max+1)%3]/v0[idx_max]
        v1[(idx_max+1)%3] = 1
        v1[(idx_max+2)%3] = 0
        
        v2 = np.cross(v0, v1)

        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)
        
        return v1, v2

    @property
    def length(self):
        return np.linalg.norm(self.end_point-self.start_point)
    
    @property
    def nmj_point(self):
        return self.start_point + (self.end_point-self.start_point)*self.nmj_relative_pos
    
    @property
    def direction_vector(self):
        return (self.end_point-self.start_point)/self.length

    @staticmethod
    def create_fibers(start_point, end_point, nmj_relative_pos, start_width, end_width, 
                        nmj_width, radius, n_fibers, conduction_velocity, current_tripole: CurrentTripole):
        fiber_vector = (end_point-start_point)
        fiber_vector = fiber_vector/np.linalg.norm(fiber_vector)
        nmj_point = start_point + nmj_relative_pos * (end_point-start_point)

        start_points = start_point + fiber_vector * start_width * (np.random.rand(n_fibers, 1) - 1/2)
        end_points = end_point + fiber_vector * end_width * (np.random.rand(n_fibers, 1) - 1/2)
        nmj_points = nmj_point + fiber_vector * nmj_width * (np.random.rand(n_fibers, 1) - 1/2)
        angles = np.random.rand(n_fibers, 1) * 2 * np.pi
        radii = np.sqrt(np.random.rand(n_fibers, 1)) * radius
        phasors = radii*np.exp(1j*angles)

        nmj_relative_positions = np.linalg.norm(nmj_points - start_points, axis=1)/np.linalg.norm(end_points - start_points, axis=1)

        v1, v2 = MuscleFiber._get_ortho_basis(fiber_vector)

        ortho_vectors = np.real(phasors)*v1 + np.imag(phasors)*v2

        start_points = start_points + ortho_vectors
        end_points = end_points + ortho_vectors

        fibers = [MuscleFiber(start_point, end_point, nmj_rel_pos, conduction_velocity, current_tripole) for
                  (start_point, end_point, nmj_rel_pos) in zip(start_points, end_points, nmj_relative_positions)]

        return fibers