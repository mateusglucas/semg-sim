from conductor_volume import ConductorVolume
from muscle_fiber import MuscleFiber
from typing import List
import numpy as np
from scipy import signal as sig
import multiprocessing as mp
import functools

class MotorUnit:
    def __init__(self, conductor_volume: ConductorVolume, fibers: List[MuscleFiber] = []):
        self.fibers = fibers
        self.conductor_volume = conductor_volume

    def add_fiber(self, fiber: MuscleFiber):
        self.fibers.append(fiber)
    
    def add_fibers(self, fibers: List[MuscleFiber]):
        self.fibers += fibers
    
    def trigger_job(self, t, xz_sample_points):
        
        ka = self.conductor_volume.sigma_z/self.conductor_volume.sigma_r

        # Matriz com sinais de saida, em que cada linha corresponde
        # ao sinal no i-ésimo eletrodo. O i-ésimo eletrodo é aquele
        # localizado na posição descrita pela i-ésima linha de xz_sample_points.
        # O elemento [i,j] é o valor do potencial elétrico no i-ésimo eletrodo
        # no j-ésimo instante de tempo.
        signals = np.zeros((xz_sample_points.shape[0], len(t)))
        for fiber in self.fibers:
            for j in range(len(t)):
                # variação de posição dos tripolos
                delta_pos = fiber.conduction_velocity * t[j]

                start_to_nmj_length = fiber.length * fiber.nmj_relative_pos
                end_to_nmj_length = fiber.length * (1-fiber.nmj_relative_pos)

                # posições relativas de cada polo de corrente, em relação à zona de inervação
                # [P11, P21, P31, P12, P22, P32], em que Px1 são os tripolos que se propagam no
                # sentido fim-inicio da fibra e Px2 os tripolos que se propagam no sentido
                # inicio-fim, todos se propagando a partir da zona de inervação.
                rel_pos = delta_pos - np.array([0, fiber.current_tripole.d12, fiber.current_tripole.d13])
                rel_pos[rel_pos<0] = 0
                rel_pos = np.concatenate((-rel_pos, rel_pos))
                rel_pos[:3] = np.maximum(rel_pos[:3], -start_to_nmj_length)
                rel_pos[3:] = np.minimum(rel_pos[3:], end_to_nmj_length)

                # se tripolos já se extinguiram, encerrar cálculos para a fibra
                if all(rel_pos[:3] == -start_to_nmj_length) and all(rel_pos[3:] == end_to_nmj_length):
                    break

                # cada linha contém a posição absoluta [x, y, z] de cada polo de corrente
                abs_pos = fiber.nmj_point + fiber.direction_vector * np.reshape(rel_pos, (-1, 1))

                # vetor coluna com intensidade de cada polo de corrente
                p = np.array([fiber.current_tripole.P1, 
                                fiber.current_tripole.P2,
                                fiber.current_tripole.P3])
                p = np.concatenate((p, p))
                p = np.reshape(p, (-1,1))

                # vetores coluna com coordenadas dos eletrodos
                x = xz_sample_points[:,0].reshape((-1,1))
                z = xz_sample_points[:,1].reshape((-1,1))

                # vetores coluna com coordenadas dos polos de corrente
                xp = abs_pos[:,0].reshape((-1,1))
                yp = abs_pos[:,1].reshape((-1,1))
                zp = abs_pos[:,2].reshape((-1,1))

                # plt.figure()
                # ax = plt.axes(projection='3d')
                # plt.plot(xp,yp,zp,'x')
                # ax.set_zlim([-90e-3-120e-3,0])
                # plt.show()

                # cada coluna é o potencial nos eletrodos (a menos de um fator de multiplicação) 
                # devido a cada um dos polos
                aux = p.T/np.sqrt(((x-xp.T)**2 + yp.T**2)*ka + (z-zp.T)**2)
                signals[:, j] += 1/(2*np.pi*self.conductor_volume.sigma_r) * np.sum(aux, 1)

        return signals

    def trigger(self, xz_sample_points, sample_rate, n_points = None, stimuli = None, jobs = None):
        # se ambos são None ou ambos são definidos
        if stimuli is None == n_points is None:
            raise Exception("Just one of period or stimuli parameters should be defined.")

        if stimuli is None:
            # TODO

            dt = 1/sample_rate
            t = np.linspace(0, (n_points-1)*dt, n_points)

            # Matriz com sinais de saida, em que cada linha corresponde
            # ao sinal no i-ésimo eletrodo. O i-ésimo eletrodo é aquele
            # localizado na posição descrita pela i-ésima linha de xz_sample_points.
            # O elemento [i,j] é o valor do potencial elétrico no i-ésimo eletrodo
            # no j-ésimo instante de tempo.
            signals = np.zeros((xz_sample_points.shape[0], len(t)))

            # Multiprocessing no tempo. Pontos intercalados entre os k processos
            # utilizados:
            #
            #       t[0] - process[0]
            #       t[1] - process[1]
            #       ...
            #       t[k-1] - process[k-1]
            #       t[k]   - process[0]
            #       ...
            #
            #       Isso evita com que um processo pegue somente tempos bem iniciais,
            #       em que há movimentação dos tripolos e é necessário o cálculo do potential, 
            #       e outro pegue tempos finais, em que os tripolos já se extinguiram e não
            #       é necessário mais realizar nenhum cálculo.
            #
            if jobs is None:
                signals = self.trigger_job(t, xz_sample_points)   
            else:
                jobs_params = [t[i::jobs] for i in range(jobs)]

                with mp.Pool(jobs) as pool:
                    f = functools.partial(self.trigger_job, xz_sample_points=xz_sample_points)
                    partial_signals = pool.map(f, jobs_params)
                
                for i in range(jobs):
                    signals[:, i::jobs] = partial_signals[i]
                
            return signals
        else:
            n_points = len(stimuli)
            signals = self.trigger(xz_sample_points, sample_rate, n_points, None)

            # Zero-padding do estímulo para obter resposta diretamente por convolução com opção 'same'.
            # É adicionada uma quantidade N-1 de zeros à esquerda do vetor de estímulo, em que
            # N é o tamanho original do vetor de estimulo.
            stimuli = np.reshape(stimuli,(1,-1))
            stimuli = np.concatenate((np.zeros(1, stimuli.size-1), stimuli), axis=1)

            return sig.convolve2d(signals, stimuli, 'same')